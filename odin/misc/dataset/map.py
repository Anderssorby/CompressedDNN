import os
from pathlib import Path

# For processing
import cv2
import h5py
import numpy as np
import parmap
from tqdm import tqdm as tqdm

import odin
import odin.plot as oplt
from .base import Dataset, download_and_unwrap_tarball


def normalization(x):
    result = x / 127.5 - 1

    # Deal with the case where float multiplication gives an out of range result (eg 1.000001)
    out_of_bounds_high = (result > 1.)
    out_of_bounds_low = (result < -1.)
    out_of_bounds = out_of_bounds_high + out_of_bounds_low

    if not all(np.isclose(result[out_of_bounds_high], 1)):
        raise RuntimeError("Normalization gave a value greater than 1")
    else:
        result[out_of_bounds_high] = 1.

    if not all(np.isclose(result[out_of_bounds_low], -1)):
        raise RuntimeError("Normalization gave a value lower than -1")
    else:
        result[out_of_bounds_low] = 1.

    return result


def inverse_normalization(x):
    # normalises back to ints 0-255, as more reliable than floats 0-1
    # (np.isclose is unpredictable with values very close to zero)
    result = ((x + 1.) * 127.5).astype('uint8')
    # Still check for out of bounds, just in case
    out_of_bounds_high = (result > 255)
    out_of_bounds_low = (result < 0)
    out_of_bounds = out_of_bounds_high + out_of_bounds_low

    if out_of_bounds_high.any():
        raise RuntimeError("Inverse normalization gave a value greater than 255")

    if out_of_bounds_low.any():
        raise RuntimeError("Inverse normalization gave a value lower than 1")

    return result


def get_nb_patch(img_dim, patch_size, image_data_format):
    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(x, image_data_format, patch_size):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        x = x.transpose(0, 2, 3, 1)

    list_x = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(x.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_x.append(x[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_x)):
            list_x[i] = list_x[i].transpose(0, 3, 1, 2)

    return list_x


def format_image(img_path, size, nb_channels):
    """
    Load img with opencv and reshape
    """

    if nb_channels == 1:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # GBR to RGB

    w = img.shape[1]

    # Slice image in 2 to get both parts
    img_full = img[:, :w // 2, :]
    img_sketch = img[:, w // 2:, :]

    img_full = cv2.resize(img_full, (size, size), interpolation=cv2.INTER_AREA)
    img_sketch = cv2.resize(img_sketch, (size, size), interpolation=cv2.INTER_AREA)

    if nb_channels == 1:
        img_full = np.expand_dims(img_full, -1)
        img_sketch = np.expand_dims(img_sketch, -1)

    img_full = np.expand_dims(img_full, 0).transpose(0, 3, 1, 2)
    img_sketch = np.expand_dims(img_sketch, 0).transpose(0, 3, 1, 2)

    return img_full, img_sketch


def build_hdf5(source, name="maps", file_name="maps_data.h5", size=256, max_samples=None):
    """
    Gather the data in a single HDF5 file.
    """
    pro_data_dir = odin.check_or_create_dir(odin.data_dir)

    hdf5_file = os.path.join(pro_data_dir, file_name)

    if os.path.isfile(hdf5_file):
        return hdf5_file

    path = download_and_unwrap_tarball(source=source,
                                       name=name)

    nb_channels = 3

    with h5py.File(hdf5_file, "w") as hfw:

        for dset_type in ["train", "test", "val"]:

            list_img = [img for img in Path(path).glob('%s/*.jpg' % dset_type)]
            list_img = [str(img) for img in list_img]
            list_img.extend(map(str, Path(path).glob('%s/*.png' % dset_type)))
            list_img = np.array(list_img)

            num_files = len(list_img)
            if num_files == 0:
                print("No files in %s" % dset_type)
                continue

            data_full = hfw.create_dataset("%s_data_full" % dset_type,
                                           (0, nb_channels, size, size),
                                           maxshape=(max_samples, 3, size, size),
                                           dtype=np.uint8)

            data_sketch = hfw.create_dataset("%s_data_sketch" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(max_samples, 3, size, size),
                                             dtype=np.uint8)

            chunk_size = 100
            num_chunks = num_files // chunk_size
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):
                list_img_path = list_img[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, nb_channels, pm_parallel=False)

                arr_img_full = np.concatenate([o[0] for o in output], axis=0)
                arr_img_sketch = np.concatenate([o[1] for o in output], axis=0)

                # Resize HDF5 dataset
                data_full.resize(data_full.shape[0] + arr_img_full.shape[0], axis=0)
                data_sketch.resize(data_sketch.shape[0] + arr_img_sketch.shape[0], axis=0)

                data_full[-arr_img_full.shape[0]:] = arr_img_full.astype(np.uint8)
                data_sketch[-arr_img_sketch.shape[0]:] = arr_img_sketch.astype(np.uint8)

        # Plot result
        check_hdf5(path, nb_channels)

    return hdf5_file


def check_hdf5(jpeg_dir, nb_channels):
    """
    Plot images with landmarks to check the processing
    """

    # Get hdf5 file
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(odin.data_dir, "%s_data.h5" % file_name)
    max_plots = 20

    with h5py.File(hdf5_file, "r") as hf:
        data_full = hf["train_data_full"]
        data_sketch = hf["train_data_sketch"]
        for i in range(min(data_full.shape[0], max_plots)):
            oplt.figure()
            img = data_full[i, :, :, :].transpose(1, 2, 0)
            img2 = data_sketch[i, :, :, :].transpose(1, 2, 0)
            img = np.concatenate((img, img2), axis=1)
            if nb_channels == 1:
                oplt.imshow(img[:, :, 0], cmap="gray")
            else:
                oplt.imshow(img)
            oplt.show()
            oplt.clf()
            oplt.close()


class MapImageData(Dataset):
    name = "satellite_images"
    source = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz'

    sample_shape = (256, 256, 3)

    def __init__(self, problem_type="image_to_sketch", limit=-1):
        self.problem_type = problem_type
        self.limit = limit
        super(MapImageData, self).__init__()

    def load(self):
        print("Starting to load dataset")
        # dset = "maps"
        image_data_format = "channels_last"
        limit = self.limit

        path = build_hdf5(source=self.source)

        with h5py.File(path, "r") as hf:
            print("File loaded")

            x_image_train = hf["train_data_full"][:limit].astype(np.float32)
            x_image_train = normalization(x_image_train)

            x_sketch_train = hf["train_data_sketch"][:limit].astype(np.float32)
            x_sketch_train = normalization(x_sketch_train)

            if image_data_format == "channels_last":
                x_image_train = x_image_train.transpose(0, 2, 3, 1)
                x_sketch_train = x_sketch_train.transpose(0, 2, 3, 1)

            x_image_val = hf["val_data_full"][:limit].astype(np.float32)
            x_image_val = normalization(x_image_val)

            x_sketch_val = hf["val_data_sketch"][:limit].astype(np.float32)
            x_sketch_val = normalization(x_sketch_val)

            if image_data_format == "channels_last":
                x_image_val = x_image_val.transpose(0, 2, 3, 1)
                x_sketch_val = x_sketch_val.transpose(0, 2, 3, 1)

            if self.problem_type == "image_to_sketch":
                x_target = x_sketch_train
                y_input_condition = x_image_train
                x_target_val = x_sketch_val
                y_input_condition_val = x_image_val
            else:  # problem_type == "sketch_to_image"
                x_target = x_image_train
                y_input_condition = x_sketch_train
                x_target_val = x_image_val
                y_input_condition_val = x_sketch_val

            print("Dataset loaded")

            self.dataset = x_target, y_input_condition, x_target_val, y_input_condition_val

            return self.dataset


class DOTA(Dataset):
    """
    DOTA-v1.5
    """

    name = "dota"
    source = "https://data.sorby.xyz/DOTA-v1.5_train.zip"

    def __init__(self):
        super(DOTA, self).__init__()
        f = "DOTA-v1.5_train.h5"
        # file_name = os.path.join(odin.data_dir, f)
        path = build_hdf5(name="dota", file_name=f, size=256, source=self.source)
        limit = -1

        with h5py.File(path, "r") as hf:
            print("File loaded")

            x_train = hf["train_data_full"][:limit].astype(np.float32)
            x_train = normalization(x_train)


class TanzaniaBuildingFootprint(Dataset):
    """
    2018 Open AI Tanzania Building Footprint Segmentation Challenge
    https://competitions.codalab.org/competitions/20100
    """
    sources = {
        "train": {
            "GeoJSON": [  # object segments
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AAARCAOqhcRdoU7ULOb9GJl9a/grid_001.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AADtSLtWlp1WWBzok4j8QDtTa/grid_022.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AAAvAgdJLgURi6y0V_R7b77Na/grid_023.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AAAQlsJdp4WYiUwfd0o4mqoNa/grid_028.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AADHytc8fSCf3gna0wNAW3lZa/grid_029.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AADsRwTo35luDWb4FcKhAotaa/grid_035.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AABX9puJlaKE25JJ9YAkF-Bta/grid_036.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AACAIX76YnY7YF-qqJ_4NBPwa/grid_042.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AADYKa21pfgqygaPI7-k_Gp7a/grid_043.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AADTfD4iO7iShsBU_DI3vsaga/grid_049.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AABBphDWEHz71zdoeNYRAyeha/grid_050.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AACjRwga-dJY1dud1Kfq64Fsa/grid_051.geojson?dl=0",
                "https://www.dropbox.com/sh/ct3s1x2a846x3yl/AADY5M0XSZphjFNfwmFli_baa/grid_058.geojson?dl=0",
            ],
            "GeiTIFF": [  # Image data
                "https://oin-hotosm.s3.amazonaws.com/5afeda152b6a08001185f11a/0/5afeda152b6a08001185f11b.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae242fd0b093000130afd26/0/5ae242fd0b093000130afd27.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae242fd0b093000130afd46/0/5ae242fd0b093000130afd47.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae242fd0b093000130afd34/0/5ae242fd0b093000130afd35.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae242fd0b093000130afd38/0/5ae242fd0b093000130afd39.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae242fd0b093000130afd42/0/5ae242fd0b093000130afd43.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae242fd0b093000130afd40/0/5ae242fd0b093000130afd41.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd64/0/5ae318220b093000130afd65.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd6a/0/5ae318220b093000130afd6b.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd62/0/5ae318220b093000130afd63.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd92/0/5ae318220b093000130afd93.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd70/0/5ae318220b093000130afd71.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd7c/0/5ae318220b093000130afd7d.tif"
            ]
        },
        "test": {
            "GeoTIFF": [
                "https://oin-hotosm.s3.amazonaws.com/5ae242fd0b093000130afd32/0/5ae242fd0b093000130afd33.tif",
                "https://oin-hotosm.s3.amazonaws.com/5b00370f2b6a08001185f125/3/5b00370f2b6a08001185f129.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd98/0/5ae318220b093000130afd99.tif",
                "https://oin-hotosm.s3.amazonaws.com/5b00370f2b6a08001185f125/5/5b00370f2b6a08001185f12b.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae36dd70b093000130afdba/0/5ae36dd70b093000130afdbb.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae38a540b093000130aff23/0/5ae38a540b093000130aff24.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae38a540b093000130afecf/0/5ae38a540b093000130afed0.tif"
            ]},
        "validate": {
            "GeoTIFF": [
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd78/0/5ae318220b093000130afd79.tif",
                "https://oin-hotosm.s3.amazonaws.com/5ae318220b093000130afd94/0/5ae318220b093000130afd95.tif"
            ]
        }
    }

    def __init__(self, limit=-1):
        super(TanzaniaBuildingFootprint, self).__init__()

        x_source = self.sources["train"]["GeoTIFF"][2]
        y_source = self.sources["train"]["GeoJSON"][2]

        file_name = "tanzania_building_footprint.h5"

        hdf5_file = os.path.join(odin.data_dir, file_name)

        if not os.path.isfile(hdf5_file):

            download_and_unwrap_tarball()
            path = build_hdf5(x_source)

        with h5py.File(hdf5_file, "r") as hf:
            print("File loaded")

            x_train = hf["train_x"][:limit].astype(np.float32)



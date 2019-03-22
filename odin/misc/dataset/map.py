import numpy as np
import h5py

import os
import odin
import odin.plot as oplt
from .base import Dataset, download_and_unwrap_tarball
# For processing
import cv2
from pathlib import Path
import parmap
from tqdm import tqdm as tqdm


def normalization(X):
    result = X / 127.5 - 1

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


def inverse_normalization(X):
    # normalises back to ints 0-255, as more reliable than floats 0-1
    # (np.isclose is unpredictable with values very close to zero)
    result = ((X + 1.) * 127.5).astype('uint8')
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


def extract_patches(X, image_data_format, patch_size):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0, 2, 3, 1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0, 3, 1, 2)

    return list_X


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


def build_hdf5(size=256):
    """
    Gather the data in a single HDF5 file.
    """
    # Put train data in HDF5
    file_name = "maps"
    pro_data_dir = odin.check_or_create_dir(odin.data_dir)

    hdf5_file = os.path.join(pro_data_dir, "%s_data.h5" % file_name)

    if os.path.isfile(hdf5_file):
        return hdf5_file

    # from keras.utils.data_utils import get_file
    # path = get_file("maps.tar.gz",
    #                origin='http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz',
    #                file_hash="7519044b2725f55bc80e5a3db5d6f64f767a7b76421098c364c432ec82faf275",
    #                extract=True)

    path = download_and_unwrap_tarball(source='http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz',
                                       name="maps")

    nb_channels = 3
    jpeg_dir = path

    with h5py.File(hdf5_file, "w") as hfw:

        for dset_type in ["train", "test", "val"]:

            list_img = [img for img in Path(jpeg_dir).glob('%s/*.jpg' % dset_type)]
            list_img = [str(img) for img in list_img]
            list_img.extend(list(Path(jpeg_dir).glob('%s/*.png' % dset_type)))
            list_img = list(map(str, list_img))
            list_img = np.array(list_img)

            num_files = len(list_img)
            if num_files == 0:
                print("No files in %s" % dset_type)
                continue

            data_full = hfw.create_dataset("%s_data_full" % dset_type,
                                           (0, nb_channels, size, size),
                                           maxshape=(None, 3, size, size),
                                           dtype=np.uint8)

            data_sketch = hfw.create_dataset("%s_data_sketch" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(None, 3, size, size),
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
        check_hdf5(jpeg_dir, nb_channels)

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

    sample_shape = (256, 256, 3)

    def __init__(self):
        super(MapImageData, self).__init__()

    def load(self):
        print("Starting to load dataset")
        # dset = "maps"
        image_data_format = "channels_last"
        limit = -1  # 10

        path = build_hdf5()

        with h5py.File(path, "r") as hf:
            print("File loaded")

            x_full_train = hf["train_data_full"][:limit].astype(np.float32)
            x_full_train = normalization(x_full_train)

            x_sketch_train = hf["train_data_sketch"][:limit].astype(np.float32)
            x_sketch_train = normalization(x_sketch_train)

            if image_data_format == "channels_last":
                x_full_train = x_full_train.transpose(0, 2, 3, 1)
                x_sketch_train = x_sketch_train.transpose(0, 2, 3, 1)

            x_full_val = hf["val_data_full"][:limit].astype(np.float32)
            x_full_val = normalization(x_full_val)

            x_sketch_val = hf["val_data_sketch"][:limit].astype(np.float32)
            x_sketch_val = normalization(x_sketch_val)

            if image_data_format == "channels_last":
                x_full_val = x_full_val.transpose(0, 2, 3, 1)
                x_sketch_val = x_sketch_val.transpose(0, 2, 3, 1)

            print("Dataset loaded")

            self.dataset = x_full_train, x_sketch_train, x_full_val, x_sketch_val

            return self.dataset


def gen_batch(X1, X2, batch_size):
    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]


def get_disc_batch(x_full_batch, x_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):
    # Create x_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        x_disc = generator_model.predict(x_sketch_batch)
        y_disc = np.zeros((x_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        x_disc = x_full_batch
        y_disc = np.zeros((x_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form x_disc
    x_disc = extract_patches(x_disc, image_data_format, patch_size)

    return x_disc, y_disc


def plot_generated_batch(X_full, X_sketch, generator_model, batch_size, image_data_format, suffix, logging_dir, epoch):
    # Generate images
    X_gen = generator_model.predict(X_sketch)

    X_sketch = inverse_normalization(X_sketch)
    X_full = inverse_normalization(X_full)
    X_gen = inverse_normalization(X_gen)

    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]

    if image_data_format == "channels_last":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_data_format == "channels_first":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1, 2, 0)

    if Xr.shape[-1] == 1:
        oplt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        oplt.imshow(Xr)
    oplt.axis("off")
    oplt.savefig(
        os.path.join(logging_dir, "figures/{epoch}_current_batch_{suffix}.png".format(suffix=suffix, epoch=epoch)))
    oplt.clf()
    oplt.close()

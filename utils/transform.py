
import six
from scipy.misc import imresize
import np


class Transform(object):
    cropping_size = 24
    scaling_size = 28

    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        imgs = []

        for offset_y in six.moves.range(0, 8 + 4, 4):
            for offset_x in six.moves.range(0, 8 + 4, 4):
                im = img[offset_y:offset_y + self.cropping_size,
                     offset_x:offset_x + self.cropping_size]
                # global contrast normalization
                im = im.astype(np.float)
                im -= im.reshape(-1, 3).mean(axis=0)
                im /= im.reshape(-1, 3).std(axis=0) + 1e-5

                imgs.append(im)
                imgs.append(np.fliplr(im))

        for offset_y in six.moves.range(0, 4 + 2, 2):
            for offset_x in six.moves.range(0, 4 + 2, 2):
                im = img[offset_y:offset_y + self.scaling_size,
                     offset_x:offset_x + self.scaling_size]
                im = imresize(im, (self.cropping_size, self.cropping_size),
                              'nearest')
                # global contrast normalization
                im = im.astype(np.float)
                im -= im.reshape(-1, 3).mean(axis=0)
                im /= im.reshape(-1, 3).std(axis=0) + 1e-5

                imgs.append(im)
                imgs.append(np.fliplr(im))
        imgs = np.asarray(imgs, dtype=np.float32)

        return imgs

from mxnet import nd
import numpy as np
import mxnet as mx


class Dataset(object):
    def __init__(self, img_list, batch_size=64):
        self.img_list = img_list
        self.batch_size = batch_size
        self.num_batches = len(img_list) // batch_size
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next()

    def next(self):
        if self.cur + self.batch_size < len(self.img_list):
            batch = self.img_list[self.cur: self.cur + self.batch_size]
            batch = self.process_batch(batch)
            print(batch.shape)
            self.cur += self.batch_size
            return batch
        else:
            raise StopIteration()

    def reset(self):
        self.cur = 0

    def has_next(self):
        return self.cur + self.batch_size < len(self.img_list)

    @staticmethod
    def transform(img, dims):
        data = mx.image.imread(img)
        data = mx.image.imresize(data, dims, dims)
        data = nd.transpose(data, (2, 0, 1))
        # normalize to [-1, 1]
        data = data.astype(np.float32) / 127.5 - 1
        # if image is greyscale, repeat 3 times to get RGB image.
        if data.shape[0] == 1:
            data = nd.tile(data, (3, 1, 1))
        return data.reshape((1,) + data.shape)

    def process_batch(self, batch):
        imgs = list(map(lambda x: self.transform(x, 256), batch))
        return nd.concatenate(imgs)

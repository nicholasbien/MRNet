import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data

from pathlib import Path
from scipy.ndimage.interpolation import rotate
from torch.autograd import Variable

INPUT_DIM = 256
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class Dataset(data.Dataset):
    def __init__(self, datadir, split, use_gpu, horizontal_flip, rotate, shift):
        super().__init__()
        self.train = (split == 'train')
        self.datadir = Path(datadir)
        self.use_gpu = use_gpu

        with open(self.datadir / f'{split}.csv') as f:
            self.paths = [line.strip().split(',')[0] for line in f]
        with open(self.datadir / f'{split}.csv') as f:
            self.labels = [int(int(line.strip().split(',')[1]) > 0) for line in f]

        if self.train:
            self.horizontal_flip = horizontal_flip
            self.rotate = rotate
            self.shift = shift

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    # shift randomly shifts image slices up to shift_size pixels in rows/cols
    def _shift(self, arr, shift_size, axis, fill_val=0):
        result = np.empty_like(arr)
        shift = np.random.randint(-shift_size, shift_size + 1)
        fill_slc = [slice(None)] * len(arr.shape)
        arr_slc = [slice(None)] * len(arr.shape)
        result_slc = [slice(None)] * len(arr.shape)
        if shift > 0:
            fill_slc[axis] = slice(shift)
            arr_slc[axis] = slice(-shift)
            result_slc[axis] = slice(shift, None, None)
            result[tuple(fill_slc)] = fill_val
            result[tuple(result_slc)] = arr[tuple(arr_slc)]
        elif shift < 0:
            fill_slc[axis] = slice(shift, None, None)
            result_slc[axis] = slice(shift)
            arr_slc[axis] = slice(-shift, None, None)
            result[tuple(fill_slc)] = fill_val
            result[tuple(result_slc)] = arr[tuple(arr_slc)]
        else:
            result = arr
        return result
            
    def transform(self, vol):
        '''
        Expects vol to have shape (num_slices, depth, rows, cols)
        '''
        if self.horizontal_flip:
            if np.random.rand(1) > .5:
                vol = np.flip(vol, axis=3).copy()
        if self.shift != 0:
            vol = self._shift(vol, self.shift, 2)
            vol = self._shift(vol, self.shift, 3)
        if self.rotate != 0:
            angle = np.random.randint(-self.rotate, self.rotate)
            vol = rotate(vol, angle, axes=(2,3), reshape=False)
        return vol

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.datadir / 'volumes' / self.paths[index]
        with open(path, 'rb') as file_handler: # Must use 'rb' as the data is binary
            vol = pickle.load(file_handler).astype(np.int32)

        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM)/2)
        vol = vol[:,pad:-pad,pad:-pad]
        
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

        # normalize
        vol = (vol - MEAN) / STDDEV
        
        # convert to RGB
        vol = np.stack((vol,)*3, axis=1)

        if self.train:
            vol = self.transform(vol)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)

def load_data(datadir, use_gpu, horizontal_flip=False, rotate=0, shift=0):
    train_dataset = Dataset(datadir, 'train', use_gpu, horizontal_flip, rotate, shift)
    valid_dataset = Dataset(datadir, 'valid', use_gpu, horizontal_flip=False, rotate=0, shift=0)
    test_dataset = Dataset(datadir, 'test', use_gpu, horizontal_flip=False, rotate=0, shift=0)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)

    return train_loader, valid_loader, test_loader

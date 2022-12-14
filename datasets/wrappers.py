import functools
import random
import math
from PIL import Image

import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
import paddle
from paddle.io import Dataset
from paddle.vision import transforms
from datasets import register
from utils import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            # x0 = 0  # todo 临时测试 不随机裁剪
            # y0 = 0

            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.clone().reshape(crop_hr.shape))

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            # sample_lst=[]
            # for k in range(self.sample_q):  #todo 临时测试
            #     sample_lst.append(k)
            # hr_coord = hr_coord[sample_lst]
            hr_coord = hr_coord.gather(paddle.to_tensor(sample_lst))
            # hr_rgb = hr_rgb[sample_lst]
            hr_rgb = hr_rgb.gather(paddle.to_tensor(sample_lst))

        # cell = torch.ones_like(hr_coord)
        cell = paddle.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    # print(img.shape)
    # print(size)
    # return transforms.ToTensor()(
    #     transforms.Resize(size, Image.BICUBIC)(
    #         transforms.ToPILImage()(img)))
    pil_img = Image.fromarray(np.uint8(img.numpy() * 255).transpose(1, 2, 0)).convert('RGB')
    if isinstance(size, tuple) or isinstance(size, list):
        pil_img_resize = pil_img.resize(size)
    else:
        pil_img_resize = pil_img.resize((size, size))
    # pil_img_resize.show()
    return paddle.vision.transforms.ToTensor(data_format='CHW')(pil_img_resize)


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        s = random.uniform(self.scale_min, self.scale_max)
        # s = self.scale_min #todo 临时测试

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]  # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            # x0 = 0  #todo 临时测试
            # y0 = 0
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        # print('crop_lr:')
        # print(crop_lr)
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    # x = x.flip(-2)
                    # x = paddle.flip(x, [-2])
                    x = x.flip([-2])
                if vflip:
                    # x = x.flip(-1)
                    # x = paddle.flip(x, [-1])
                    x = x.flip([-1])
                if dflip:
                    # print('x.shape')
                    # print(x.shape)
                    # x = x.transpose(-2, -1)  # torch.Size([3, 48, 48])
                    # x = paddle.transpose(x, perm=[0, 2, 1])
                    x = x.transpose(perm=[0, 2, 1])
                    # print("x:")
                    # print(x)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.clone().reshape(crop_hr.shape))
        # print(hr_coord, hr_rgb)
        if self.sample_q is not None:
            # print('len')
            # print(len(hr_coord))
            # print('self.sample_q')
            # print(self.sample_q)
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            # sample_lst=[] #todo 临时测试
            # for k in range(self.sample_q):  #todo 临时测试
            #     sample_lst.append(k)

            # print("sample_lst\n")
            # print(sample_lst)
            # print('sample_lst长度 %d' % len(sample_lst))
            # print(hr_coord)
            # hr_coord = hr_coord[sample_lst]
            hr_coord = hr_coord.gather(paddle.to_tensor(sample_lst))
            # print('hr_coord')
            # print(hr_coord)
            # hr_rgb = hr_rgb[sample_lst]
            hr_rgb = hr_rgb.gather(paddle.to_tensor(sample_lst))

        # cell = torch.ones_like(hr_coord)
        cell = paddle.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            # sample_lst=[] #todo 临时测试
            # for k in range(self.sample_q):  #todo 临时测试
            #     sample_lst.append(k)
            hr_coord = hr_coord[sample_lst]
            # hr_coord = hr_coord.gather(paddle.to_tensor(sample_lst))
            hr_rgb = hr_rgb[sample_lst]
            # hr_rgb = hr_rgb.gather(paddle.to_tensor(sample_lst))

        # cell = torch.ones_like(hr_coord)
        cell = paddle.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

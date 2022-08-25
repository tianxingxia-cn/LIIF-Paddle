import argparse
import os
from PIL import Image

# import torch
# from torchvision import transforms
import paddle
from paddle.vision import transforms

import models
from utils import make_coord
from test import batched_predict
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    img = transforms.to_tensor(Image.open(args.input).convert('RGB'))
    # print(img)
    # model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    # model = models.make(torch.load(args.model, map_location=device)['model'], load_sd=True).to(device)
    model = models.make(paddle.load(args.model)['model'], load_sd=True)

    h, w = list(map(int, args.resolution.split(',')))
    # coord = make_coord((h, w)).to(device)
    coord = make_coord((h, w))
    # cell = torch.ones_like(coord)
    cell = paddle.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    # pred = batched_predict(model, ((img - 0.5) / 0.5).to(device).unsqueeze(0),
    #     coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(axis=0),
                           coord.unsqueeze(axis=0), cell.unsqueeze(axis=0), bsize=30000)[0]

    # print(((img - 0.5) / 0.5))
    # print(coord)
    # print(cell)
    # print(pred)
    # print(pred.numpy())
    # pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).to(device)
    # transforms.ToPILImage()(pred).save(args.output)

    pred = paddle.clip((pred * 0.5 + 0.5), min=0, max=1).reshape([h, w, 3]).transpose(perm=[2, 0, 1])
    # transforms.ToPILImage()(pred).save(args.output)
    # print(pred)

    # pil_img = Image.fromarray(np.uint8(img.numpy()*255).transpose(1, 2, 0)).convert('RGB')
    pil_img = Image.fromarray(np.uint8(pred.numpy() * 255).transpose(1, 2, 0)).convert('RGB')
    pil_img.save(args.output)



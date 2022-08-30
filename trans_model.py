import argparse
import os
import paddle
import torch
import models
from models.liif import LIIF

device = paddle.get_device()
os.environ['CUDA_VISIBLE_DEVICES'] = device.replace('gpu:','')

net = models.make({'name': 'liif', 'args': {'encoder_spec': {'name': 'rdn', 'args': {'no_upsampling': True}},
                                            'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3,
                                                                                   'hidden_list': [256, 256, 256,
                                                                                                   256]}}}},
                  load_sd=False)
net.eval()

torch_ckpt = torch.load('./pretrained/rdn-liif.pth', map_location=torch.device('cpu'))
m = torch_ckpt['model']
sd = m['sd']
paddle_sd = {}
for k, v in sd.items():

    if torch.is_tensor(v):
        if 'imnet.layers' in k and 'weight' in k:
            print(k)
            print(v)
            paddle_sd[k] = v.t().numpy()
        else:
            paddle_sd[k] = v.numpy()
    else:
        paddle_sd[k] = v

paddle_ckpt = {'name': m['name'], 'args': m['args'], 'sd': paddle_sd}

net.set_state_dict(paddle_ckpt)
paddle.save({'model': paddle_ckpt}, './pretrained/rdn-liif_torch.pdparams')
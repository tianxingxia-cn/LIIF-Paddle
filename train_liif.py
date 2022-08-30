""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os
import yaml
import paddle   # import torch
import paddle.nn as nn  # import torch.nn as nn
from tqdm import tqdm
from paddle.io import DataLoader    # from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import MultiStepLR
import datasets
import models
import utils
from test import eval_psnr

import warnings
warnings.filterwarnings('ignore')

# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device = paddle.get_device()
# print(device)
os.environ['CUDA_VISIBLE_DEVICES'] = device.replace('gpu:','')
# os.environ["OMP_NUM_THREADS"] = "1"
def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    try:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))
    finally:
        # print('报错了')
        pass

    # loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    # loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=(tag == 'train'), num_workers=4, pin_memory=True)
    # loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=(tag == 'train'), num_workers=0, use_shared_memory=True)
    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=False, num_workers=0,use_shared_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    print('resume config:')
    print(config.get('resume'))
    if config.get('resume') is not None and os.path.exists(config['resume']):
        sv_file = paddle.load(config['resume'])     # sv_file = torch.load(config['resume'])
        # model = models.make(sv_file['model'], load_sd=True).to(device)
        model = models.make(sv_file['model'], load_sd=True)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        print('epoch_resume:')
        print(sv_file['epoch'])
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            # lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            multi_step_lr = config['multi_step_lr']
            lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config['optimizer']['args']['lr'],milestones=multi_step_lr['milestones'],gamma=multi_step_lr['gamma'], verbose=True)
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        # model = models.make(config['model']).to(device)
        model = models.make(config['model'])

        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            # lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            multi_step_lr = config['multi_step_lr']
            lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config['optimizer']['args']['lr'],
                                                              milestones=multi_step_lr['milestones'],
                                                              gamma=multi_step_lr['gamma'], verbose=True)

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    # inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    # inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    # print('TODO: 判断是float32 还是 float64')
    inp_sub = paddle.to_tensor(t['sub']).astype('float32').reshape([1, -1, 1, 1])
    inp_div = paddle.to_tensor(t['div']).astype('float32').reshape([1, -1, 1, 1])
    t = data_norm['gt']
    # gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    # gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)
    gt_sub = paddle.to_tensor(t['sub']).astype('float32').reshape([1, 1, -1])
    gt_div = paddle.to_tensor(t['div']).astype('float32').reshape([1, 1, -1])
    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            # batch[k] = v.to(device)
            batch[k] = v
        # print(v)
        inp = (batch['inp'] - inp_sub) / inp_div
        # print("inp\n")
        # print(inp)
        # print("coord\n")
        # print(batch['coord'])
        # print("cell\n")
        # print(batch['cell'])
        pred = model(inp, batch['coord'], batch['cell'])        #rel: paddle.Model(net, input, label)
        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        # optimizer.zero_grad()
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        pred = None
        loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        print("暂不支持多GPUs")
    #     model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        # print('epoch = %d' % epoch)
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # print(optimizer)
        # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)   # todo 本行未优化
        # writer.add_scalar('lr', optimizer.get_lr(), epoch)
        writer.add_scalar(tag='train/lr', value=optimizer.get_lr(), step = epoch)

        train_loss = train(train_loader, model, optimizer)

        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        #writer.add_scalars('loss', {'train': train_loss}, epoch)
        writer.add_scalar(tag='train/train_loss', value=train_loss, step=epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        # torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        paddle.save(sv_file, os.path.join(save_path, 'epoch-last.pdparams'))  # checkpoint

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            # torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
            paddle.save(sv_file, os.path.join(save_path, 'epoch-{}.pdparams'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            # writer.add_scalars('psnr', {'val': val_res}, epoch)
            writer.add_scalar(tag='val/psnr', value=val_res, step=epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                # torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))
                paddle.save(sv_file, os.path.join(save_path, 'epoch-best.pdparams'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        # writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)

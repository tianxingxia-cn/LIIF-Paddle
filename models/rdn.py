# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace

# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn

from models import register


class RDB_Conv(nn.Layer):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            # nn.Conv2D(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2D(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True, data_format='NCHW'),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        # return torch.cat((x, out), 1)
        return paddle.concat((x, out), axis=1)


class RDB(nn.Layer):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        # self.LFF = nn.Conv2D(G0 + C*G, G0, 1, padding=0, stride=1)
        self.LFF = nn.Conv2D(G0 + C * G, G0, 1, padding=0, stride=1, bias_attr=True, data_format='NCHW')

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Layer):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # print(self.D)
        # print(C)
        # print(G)
        # print('n_colors=')
        # print(args.n_colors)
        # print('G0=')
        # print(G0)
        # print('kSize=')
        # print(kSize)
        # print('paddle=')
        # print((kSize-1)//2)

        # Shallow feature extraction net
        # self.SFENet1 = nn.Conv2D(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet1 = nn.Conv2D(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True,
                                 data_format='NCHW')

        # print('SFENet1')
        # print(self.SFENet1)
        # raise
        # self.SFENet2 = nn.Conv2D(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2D(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True, data_format='NCHW')

        # Redidual dense blocks and dense feature fusion
        # self.RDBs = nn.ModuleList()
        self.RDBs = nn.LayerList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            # nn.Conv2D(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2D(self.D * G0, G0, 1, padding=0, stride=1, bias_attr=True, data_format='NCHW'),

            # nn.Conv2D(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
            nn.Conv2D(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True, data_format='NCHW')

        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    # nn.Conv2D(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.Conv2D(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True,
                              data_format='NCHW'),
                    nn.PixelShuffle(r),
                    # nn.Conv2D(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                    nn.Conv2D(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True,
                              data_format='NCHW')
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    # nn.Conv2D(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.Conv2D(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True, data_format='NCHW'),
                    nn.PixelShuffle(2),
                    # nn.Conv2D(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.Conv2D(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True, data_format='NCHW'),
                    nn.PixelShuffle(2),
                    # nn.Conv2D(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                    nn.Conv2D(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1, bias_attr=True,
                              data_format='NCHW')
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        # x = self.GFF(torch.cat(RDBs_out,1))
        x = self.GFF(paddle.concat(RDBs_out, axis=1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


@register('rdn')
def make_rdn(G0=64, RDNkSize=3, RDNconfig='B',
             scale=2, no_upsampling=False):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    # print('args=')
    # print(args)

    return RDN(args)

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch.nn.modules.utils import _pair
from torch.nn import init

from mmrotate.registry import MODELS


@MODELS.register_module()
class DSFEM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(DSFEM, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.kernel_out = []

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpnse_convs = nn.ModuleList()
        self.conv1x1s = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            fpnse_conv = AtrousSE(out_channels, out_channels)
            self.fpnse_convs.append(fpnse_conv)

            conv1x1 = ConvModule(
                3  * out_channels,
                out_channels,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.conv1x1s.append(conv1x1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level  # extra_levels = 1
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                None


if __name__ == "__main__":
    net = AtrousSE(32, 32, 3, 1)
    inp = torch.randn([2, 32, 128, 128])
    ref = torch.ones([2, 96, 124, 124])
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    for iter in range(100):
        opt.zero_grad()
        out, w = net(inp)
        loss = ((ref - out) ** 2).mean()
        loss.backward()
        opt.step()
        print(loss.item())
        w1 = w[0:32]
        w2 = w[32:64]
        w3 = w[64:]
        print(w1[0, 0, 1, 1], w2[0, 0, 0, 0])
        print(w2[0, 0, 1, 1], w3[0, 0, 0, 0])

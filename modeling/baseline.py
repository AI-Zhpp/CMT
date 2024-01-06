# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from modeling.SFE import SFE_1, SFE_2, SFE_3
from modeling.PTE import PTE_1, PTE_2, PTE_3, PTE_4, PTE_5, PTE_6, PTE_7, PTE_8, PTE_9

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class CMT(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(CMT, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])


        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.conv1 = nn.Conv2d(512, 2048, kernel_size=2, bias=False, stride=2)
        self.bn1 = nn.BatchNorm2d(2048)
        self.conv2 = nn.Conv2d(1024, 2048, kernel_size=1, bias=False, stride=1)
        self.bn2 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)
        self.SFE_1 = SFE_1(inplanes=512, planes=2048)
        self.SFE_2 = SFE_2(inplanes=1024, planes=2048)
        self.SFE_3 = SFE_3(inplanes=2048, planes=2048)
        self.PTE_1 = PTE_1(in_channel=2048, out_channel=2048, img_size=[16, 8], num_patch=128, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_2 = PTE_2(in_channel=2048, out_channel=2048, img_size=[16, 8], num_patch=128, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_3 = PTE_3(in_channel=2048, out_channel=2048, img_size=[16, 8], num_patch=128, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_4 = PTE_4(in_channel=1024, out_channel=1024, img_size=[16, 8], num_patch=65, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_5 = PTE_5(in_channel=1024, out_channel=1024, img_size=[16, 8], num_patch=65, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_6 = PTE_6(in_channel=1024, out_channel=1024, img_size=[16, 8], num_patch=65, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_7 = PTE_7(in_channel=512, out_channel=512, img_size=[16, 8], num_patch=33, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_8 = PTE_8(in_channel=512, out_channel=512, img_size=[16, 8], num_patch=33, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.PTE_9 = PTE_9(in_channel=512, out_channel=512, img_size=[16, 8], num_patch=33, p_size=1,
                             emb_dropout=0.1, T_depth=2, heads=16, dim_head=64, mlp_dim=768, dropout=0)
        self.FC_1 = nn.Linear(self.in_planes, 1024, bias=False)
        self.FC_1.apply(weights_init_classifier)
        self.FC_2 = nn.Linear(self.in_planes, 1024, bias=False)
        self.FC_2.apply(weights_init_classifier)
        self.FC_3 = nn.Linear(self.in_planes, 1024, bias=False)
        self.FC_3.apply(weights_init_classifier)
        self.FC_4 = nn.Linear(1024, 512, bias=False)
        self.FC_4.apply(weights_init_classifier)
        self.FC_5 = nn.Linear(1024, 512, bias=False)
        self.FC_5.apply(weights_init_classifier)
        self.FC_6 = nn.Linear(1024, 512, bias=False)
        self.FC_6.apply(weights_init_classifier)
        self.FC_7 = nn.Linear(1024, 512, bias=False)
        self.FC_7.apply(weights_init_classifier)
        self.FC_8 = nn.Linear(1024, 512, bias=False)
        self.FC_8.apply(weights_init_classifier)
        self.FC_9 = nn.Linear(1024, 512, bias=False)
        self.FC_9.apply(weights_init_classifier)
            #tras345
        self.B1 = nn.BatchNorm1d(self.in_planes)
        self.B1.bias.requires_grad_(False)  # no shift
        self.B1.apply(weights_init_kaiming)
        self.B2 = nn.BatchNorm1d(self.in_planes)
        self.B2.bias.requires_grad_(False)  # no shift
        self.B2.apply(weights_init_kaiming)
        self.B3 = nn.BatchNorm1d(self.in_planes)
        self.B3.bias.requires_grad_(False)  # no shift
        self.B3.apply(weights_init_kaiming)
        self.F1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.F1.apply(weights_init_classifier)
        self.F2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.F2.apply(weights_init_classifier)
        self.F3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.F3.apply(weights_init_classifier)

        self.BN1 = nn.BatchNorm1d(self.in_planes)
        self.BN1.bias.requires_grad_(False)  # no shift
        self.BN1.apply(weights_init_kaiming)
        self.BN2 = nn.BatchNorm1d(self.in_planes)
        self.BN2.bias.requires_grad_(False)  # no shift
        self.BN2.apply(weights_init_kaiming)
        self.BN3 = nn.BatchNorm1d(self.in_planes)
        self.BN3.bias.requires_grad_(False)  # no shift
        self.BN3.apply(weights_init_kaiming)
        self.BN4 = nn.BatchNorm1d(self.in_planes)
        self.BN4.bias.requires_grad_(False)  # no shift
        self.BN4.apply(weights_init_kaiming)
        self.BN5 = nn.BatchNorm1d(self.in_planes)
        self.BN5.bias.requires_grad_(False)  # no shift
        self.BN5.apply(weights_init_kaiming)
        self.BN6 = nn.BatchNorm1d(self.in_planes)
        self.BN6.bias.requires_grad_(False)  # no shift
        self.BN6.apply(weights_init_kaiming)
        self.FC1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.FC1.apply(weights_init_classifier)
        self.FC2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.FC2.apply(weights_init_classifier)
        self.FC3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.FC3.apply(weights_init_classifier)
        self.FC4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.FC4.apply(weights_init_classifier)
        self.FC5 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.FC5.apply(weights_init_classifier)
        self.FC6 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.FC6.apply(weights_init_classifier)

    def forward(self, x):
        x_res3, x_res4, x_res5 = self.base(x)
        global_feat = self.gap(x_res5)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        cls_score = self.classifier(feat)

        #SFE
        x_res3 = self.SFE_1(x_res3)
        x_res4 = self.SFE_2(x_res4)
        x_res5 = self.SFE_3(x_res5)

        #part*1
        x3_global1, x3_part1 = self.PTE_1(x_res3)
        x4_global1, x4_part1 = self.PTE_2(x_res4)
        x5_global1, x5_part1 = self.PTE_3(x_res5)

        xinfer_1 = self.B1(x3_global1)
        xinfer_2 = self.B2(x4_global1)
        xinfer_3 = self.B3(x5_global1)
        xFC1 = self.F1(xinfer_1)
        xFC2 = self.F2(xinfer_2)
        xFC3 = self.F3(xinfer_3)

        #part*2
        x_g1 = self.relu(self.FC_1(x3_global1))
        x_3h = torch.cat((x_g1.unsqueeze(1), x3_part1[:, :, :8, :].flatten(2).permute(0, 2, 1)), 1)
        x3_global2_1, x3_part2_1 = self.PTE_4(x_3h)
        x_3l = torch.cat((x_g1.unsqueeze(1), x3_part1[:, :, 8:, :].flatten(2).permute(0, 2, 1)), 1)
        x3_global2_2, x3_part2_2 = self.PTE_4(x_3l)
        x_tri1 = torch.cat((x3_global2_1, x3_global2_2), 1)
        xinfer1 = self.BN1(x_tri1)
        xCE1 = self.FC1(xinfer1)

        x_g2 = self.relu(self.FC_2(x4_global1))
        x_4h = torch.cat((x_g2.unsqueeze(1), x4_part1[:, :, :8, :].flatten(2).permute(0, 2, 1)), 1)
        x4_global2_1, x4_part2_1 = self.PTE_5(x_4h)
        x_4l = torch.cat((x_g2.unsqueeze(1), x4_part1[:, :, 8:, :].flatten(2).permute(0, 2, 1)), 1)
        x4_global2_2, x4_part2_2 = self.PTE_5(x_4l)
        x_tri2 = torch.cat((x4_global2_1, x4_global2_2), 1)
        xinfer2 = self.BN2(x_tri2)
        xCE2 = self.FC2(xinfer2)

        x_g3 = self.relu(self.FC_3(x5_global1))
        x_5h = torch.cat((x_g3.unsqueeze(1), x5_part1[:, :, :8, :].flatten(2).permute(0, 2, 1)), 1)
        x5_global2_1, x5_part2_1 = self.PTE_6(x_5h)
        x_5l = torch.cat((x_g3.unsqueeze(1), x5_part1[:, :, 8:, :].flatten(2).permute(0, 2, 1)), 1)
        x5_global2_2, x5_part2_2 = self.PTE_6(x_5l)
        x_tri3 = torch.cat((x5_global2_1, x5_global2_2), 1)
        xinfer3 = self.BN3(x_tri3)
        xCE3 = self.FC3(xinfer3)

        ###part*4
        x_g4 = self.relu(self.FC_4(x3_global2_1))
        x_31 = torch.cat((x_g4.unsqueeze(1), x3_part2_1[:, :, :4, :].flatten(2).permute(0, 2, 1)), 1)
        x_31 = self.PTE_7(x_31)
        x_32 = torch.cat((x_g4.unsqueeze(1), x3_part2_1[:, :, 4:, :].flatten(2).permute(0, 2, 1)), 1)
        x_32 = self.PTE_7(x_32)
        x_g5 = self.relu(self.FC_5(x3_global2_2))
        x_33 = torch.cat((x_g5.unsqueeze(1), x3_part2_2[:, :, :4, :].flatten(2).permute(0, 2, 1)), 1)
        x_33 = self.PTE_7(x_33)
        x_34 = torch.cat((x_g5.unsqueeze(1), x3_part2_2[:, :, 4:, :].flatten(2).permute(0, 2, 1)), 1)
        x_34 = self.PTE_7(x_34)
        x_tri4 = torch.cat((x_31, x_32, x_33, x_34), 1)
        xinfer4 = self.BN4(x_tri4)
        xCE4 = self.FC4(xinfer4)

        x_g6 = self.relu(self.FC_6(x4_global2_1))
        x_41 = torch.cat((x_g6.unsqueeze(1), x4_part2_1[:, :, :4, :].flatten(2).permute(0, 2, 1)), 1)
        x_41 = self.PTE_8(x_41)
        x_42 = torch.cat((x_g6.unsqueeze(1), x4_part2_1[:, :, 4:, :].flatten(2).permute(0, 2, 1)), 1)
        x_42 = self.PTE_8(x_42)
        x_g7 = self.relu(self.FC_7(x4_global2_2))
        x_43 = torch.cat((x_g7.unsqueeze(1), x4_part2_2[:, :, :4, :].flatten(2).permute(0, 2, 1)), 1)
        x_43 = self.PTE_8(x_43)
        x_44 = torch.cat((x_g7.unsqueeze(1), x4_part2_2[:, :, 4:, :].flatten(2).permute(0, 2, 1)), 1)
        x_44 = self.PTE_8(x_44)
        x_tri5 = torch.cat((x_41, x_42, x_43, x_44), 1)
        xinfer5 = self.BN5(x_tri5)
        xCE5 = self.FC5(xinfer5)

        x_g8 = self.relu(self.FC_8(x5_global2_1))
        x_51 = torch.cat((x_g8.unsqueeze(1), x5_part2_1[:, :, :4, :].flatten(2).permute(0, 2, 1)), 1)
        x_51 = self.PTE_9(x_51)
        x_52 = torch.cat((x_g8.unsqueeze(1), x5_part2_1[:, :, 4:, :].flatten(2).permute(0, 2, 1)), 1)
        x_52 = self.PTE_9(x_52)
        x_g9 = self.relu(self.FC_9(x5_global2_2))
        x_53 = torch.cat((x_g9.unsqueeze(1), x5_part2_2[:, :, :4, :].flatten(2).permute(0, 2, 1)), 1)
        x_53 = self.PTE_9(x_53)
        x_54 = torch.cat((x_g9.unsqueeze(1), x5_part2_2[:, :, 4:, :].flatten(2).permute(0, 2, 1)), 1)
        x_54 = self.PTE_9(x_54)
        x_tri6 = torch.cat((x_51, x_52, x_53, x_54), 1)
        xinfer6 = self.BN6(x_tri6)
        xCE6 = self.FC6(xinfer6)

        xinfer7 = torch.cat((feat, xinfer_1, xinfer_2, xinfer_3, xinfer1, xinfer2, xinfer3, xinfer4, xinfer5, xinfer6), 1)

        if self.training:
            return [cls_score, xFC1, xFC2, xFC3, xCE1, xCE2, xCE3, xCE4, xCE5, xCE6],\
                   [global_feat, x3_global1, x4_global1, x5_global1, x_tri1, x_tri2, x_tri3, x_tri4, x_tri5, x_tri6]
        else:
            return xinfer7


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

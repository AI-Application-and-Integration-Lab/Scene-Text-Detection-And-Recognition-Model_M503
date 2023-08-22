import torch
import torch.nn as nn
from torchvision import transforms
from fastai.vision import *
from modules.model import _default_tfmer_cfg
# from modules.resnet import resnet45, decode_resnet
from modules.resnet_with_mask import resnet45, decode_resnet
from modules.shallow_cnn import ShallowCNN
from modules.transformer import (PositionalEncoding,
                                 TransformerEncoder,
                                 TransformerEncoderLayer
                                )  #  TransformerEncoderLayer_cross_only, TransformerEncoderLayer


class ResTranformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shallow = ShallowCNN()
        self.resnet = resnet45()
        self.decoder_renset = decode_resnet()
        self._mask_out = torch.nn.Conv2d(32, 1, kernel_size =3, stride = 1, padding = 1)

        self.d_model = ifnone(config.model_vision_d_model, _default_tfmer_cfg['d_model']) # 512
        nhead = ifnone(config.model_vision_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_vision_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_vision_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_vision_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_vision_backbone_ln, 2)

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=8*32)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, 
                                                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        # encoder_layer = TransformerEncoderLayer_cross_only(d_model=self.d_model, nhead=nhead, 
        #                                         dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # self.max_pool_mask1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.max_pool_mask2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

    def forward(self, images):
        # imgs = images[:,0:3,:,:]
        # masks = images[:,4:5,:,:]
        features = self.resnet(images) # feature: [x0,x1,x2,x3,x4]

        pred_mask = self.decoder_renset(features)
        pred_mask = torch.sigmoid(self._mask_out(pred_mask)) # pred_mask.size(): [2, 1, 32, 128]
        p_mask = pred_mask.detach()
        p_mask = self.shallow(p_mask) # pred_mask: [2, 512, 8, 32]

        z = features[5] # z.size(): [2, 512, 8, 32]
        # pred_mask = pred_mask.detach() # must detach the output of decoder, when finetuning the model
        # pred_mask_weight = self.max_pool_mask1(pred_mask) # pred_mask_weight.size(): [2, 1, 16, 64]
        # pred_mask_weight = self.max_pool_mask2(pred_mask_weight) # pred_mask_weight.size(): [2, 1, 8, 32]
        # weighted_z = torch.mul(z, pred_mask_weight)
        # weighted_z = torch.add(weighted_z, z)

        # ---------------Transformer----------------
        n, c, h, w = z.shape
        feature = z.view(n, c, -1).permute(2, 0, 1)
        p_mask = p_mask.view(n, c, -1).permute(2, 0, 1)
        mask = self.pos_encoder(p_mask)
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature, mask)
        feature = feature.permute(1, 2, 0).view(n, c, h, w) # fearure.size(): [128, 512, 8, 32]
        return feature

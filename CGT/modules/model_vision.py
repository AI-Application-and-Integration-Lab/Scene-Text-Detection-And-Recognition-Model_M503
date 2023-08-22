import logging
import torch.nn as nn
from fastai.vision import *
from modules.attention import *
# from modules.backbone import ResTranformer
from modules.backbone_with_mask import ResTranformer
from modules.model import Model
# from modules.resnet import resnet45
from modules.resnet_with_mask import resnet45


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: self.backbone = resnet45()
        
        if config.model_vision_attention == 'position':
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8*32,
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)

    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        ###### for t-SNE  ######
        # with open(R'/media/avlab/68493891-fbc2-48f2-9fa4-f6ed03d21bd61/CSTC/test/gt.txt', 'r', encoding='utf8') as f:
        #     gt=[x.rstrip().split()[1] for x in f.readlines()]

        # t_min, t_max=torch.min(attn_vecs), torch.max(attn_vecs)
        # t_sne_X=[]
        # Y=[]
        # for b, l, g in zip(attn_vecs, pt_lengths, gt):
        #     if l-1==len(g):
        #         t=b[0:l-1,:]
        #         t_sne_X.append(t)
        #         Y.append(g)

        # Y=''.join(Y)
        # with open('/media/avlab/68493891-fbc2-48f2-9fa4-f6ed03d21bd61/Alex/tsne-pytorch/Y_en_cc.txt', 'w', encoding='utf8') as f:
        #     f.write('\n'.join(list(Y)))
        # t_sne_X=torch.cat(t_sne_X, 0)
        # t_sne_X=(t_sne_X-t_min)/(t_max-t_min)
        # torch.save(t_sne_X, '/media/avlab/68493891-fbc2-48f2-9fa4-f6ed03d21bd61/Alex/tsne-pytorch/X_en_cc.pt')
        # exit()
        ##### End of t-SNE #####

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision'}

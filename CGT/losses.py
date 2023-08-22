from fastai.vision import *
import torch
from torch import Tensor
from modules.model import Model
from convertor import AttnConvertor
from cc_loss import CCLoss
import json

class MultiLosses(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()
        # self.lable_convertor = AttnConvertor()
        # self.cc_loss = CCLoss()
        # self.id2text = {0: '░', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 
        #                 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 
        #                 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '1',
        #                 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: '0'}
        # with open(R'/media/avlab/68493891-fbc2-48f2-9fa4-f6ed03d21bd61/Alex/ABINet-cross-attention/data/dict.json', 'r', encoding='utf8') as f:
        #     self.id2text = json.load(f)
        # self.id2text = {int(k):v for k, v in self.id2text.items()}

    @property
    def last_losses(self):
        return self.losses

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res
        def merge(items): # items: [tensor, tensor, tensor, ...]
            if isinstance(items[0], torch.Tensor): return torch.cat(items, dim=0)
            else: return items[0]
        res = dict()
        for key in all_res[0].keys(): # all_res: []
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    def _ce_loss(self, output, gt_labels, gt_lengths, idx=None, record=True):
        loss_name = output.get('name')
        pt_logits, weight = output['logits'], output['loss_weight']

        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        iter_size = pt_logits.shape[0] // gt_labels.shape[0]

        if iter_size > 1:
            gt_labels = gt_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        nll = output.get('nll')
        if nll is not None:
            loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
        else:
            loss = self.ce(flat_pt_logits, flat_gt_labels) * weight # here

            ####### start of cc loss ######
            # if loss_name=='vision':
            #     gt_text = []
            #     for n, l in zip(range(len(gt_labels)), gt_lengths.tolist()):
            #         cha = []
            #         for m in range(len(gt_labels[0])):
            #             c = gt_labels[n][m].tolist().index(1.0)
            #             c = self.id2text[c]
            #             cha.append(c)
            #         t = ''.join(cha)[:l-1]
            #         gt_text.append(t)
                
            #     targets_dict = self.lable_convertor.str2tensor(gt_text)
            #     contrastive_loss = self.cc_loss(output['feature'], targets_dict)
            #     loss = loss + contrastive_loss
            # elif loss_name=='language':
            #     gt_text = []
            #     sli=int(len(gt_lengths)/iter_size)
            #     gt_lengths=gt_lengths[:sli]
            #     gt_labels=gt_labels[:len(gt_lengths),:,:]
            #     for n, l in zip(range(len(gt_labels)), gt_lengths.tolist()):
            #         cha = []
            #         for m in range(len(gt_labels[0])):
            #             c = gt_labels[n][m].tolist().index(1.0)
            #             c = self.id2text[c]
            #             cha.append(c)
            #         t = ''.join(cha)[:l-1]
            #         gt_text.append(t)

            #     targets_dict = self.lable_convertor.str2tensor(gt_text)
            #     contrastive_loss = self.cc_loss(output['feature'][-len(gt_lengths):,:,:], targets_dict)
            #     loss = loss + contrastive_loss
            ####### end of cc loss #######

        if record and loss_name is not None: self.losses[f'{loss_name}_loss'] = loss

        return loss

    def forward(self, outputs, *args):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            return sum([self._ce_loss(o, *args) for o in outputs if o['loss_weight'] > 0.])
        else:
            return self._ce_loss(outputs, *args, record=False)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax: log_prob = F.log_softmax(input, dim=-1)
        else: log_prob = torch.log(input)
        loss = -(target * log_prob).sum(dim=-1)
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        else: return loss
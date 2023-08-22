import torch
from base_convertor import BaseConvertor

class AttnConvertor(BaseConvertor):
    """Convert between text, index and tensor for encoder-decoder based pipeline."""

    def __init__(self, max_seq_len=26, start_end_same=True):
        super().__init__()
        assert isinstance(max_seq_len, int)
        self.max_seq_len = max_seq_len
        self.start_end_same = start_end_same
        self.update_dict()

    def update_dict(self):
        start_end_token = '<BOS/EOS>'
        padding_token = '<PAD>'

        # BOS/EOS
        self.idx2char.append(start_end_token) # self.idx2char=['a', ..., 'z', '<BOS/EOS>']
        self.start_idx = len(self.idx2char) - 1
        if not self.start_end_same:
            self.idx2char.append(start_end_token)
        self.end_idx = len(self.idx2char) - 1

        # padding
        self.idx2char.append(padding_token) # self.idx2char=['a', ..., 'z', '<BOS/EOS>', '<PAD>']
        self.padding_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        """
        Convert text-string into tensor.
        Args:
            strings (list[str]): ['hello', 'world']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))
        """
        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        # indexes: [[1,2,3,4,5], [1,2,3,2,4], []]
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 1).fill_(0) # +2 for BOS, EOS
            # src_target[-1] = self.end_idx # '<BOS/EOS>'
            src_target[0] = self.start_idx # '<BOS/EOS>'
            src_target[1:] = tensor # src_target: tensor([3522,  287,  110,  317,   96,   70,  379, 3522])
            padded_target = (torch.ones(self.max_seq_len) * self.padding_idx).long()
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                padded_target = src_target[:self.max_seq_len]
            else:
                padded_target[:char_num] = src_target
                # padded_target: [3522,  977,  191, 3522, 3523, 3523, ..., 3523, 3523]
            padded_targets.append(padded_target)
        padded_targets = torch.stack(padded_targets, 0).long()

        return {'targets': tensors, 'padded_targets': padded_targets}

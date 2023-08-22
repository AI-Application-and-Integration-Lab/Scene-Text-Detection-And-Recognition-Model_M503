
class BaseConvertor():
    """Convert between text, index and tensor for text recognize pipeline."""
    # start_idx = end_idx = padding_idx = 0
    def __init__(self, dict_path=R'/media/avlab/68493891-fbc2-48f2-9fa4-f6ed03d21bd61/alex_mother_father_v2/abinet_dic700.txt'):
        with open(dict_path, 'r', encoding='utf8') as f:
            self.idx2char = [c.rstrip().split('\t')[1] for c in f.readlines()]

        assert len(set(self.idx2char)) == len(self.idx2char), \
            'Invalid dictionary: Has duplicated characters.'
        # self.idx2char: ['a', 'b', ..., 'z']
        # self.char2idx: {0:0, 1:1, ..., a:10, b:11}
        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

    def num_classes(self):
        """Number of output classes."""
        return len(self.idx2char)

    def str2idx(self, strings):
        """Convert strings to indexes.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        """
        assert isinstance(strings, list)

        indexes = []
        for string in strings:
            index = []
            # self.char2idx: {0:0, 1:1, ..., a:10, b:11}
            for char in string:
                char_idx = self.char2idx.get(char, None)
                if char_idx is None:
                    raise Exception(f'Chararcter: {char} not in dict,'
                                    f' please check gt_label and use'
                                    f' custom dict file,'
                                    f' or set "with_unknown=True"')
                index.append(char_idx)
            indexes.append(index)
            # indexes: [[1,2,3,4,5], [1,2,3,2,4], []]

        return indexes

import random

import torch

class Batchify(object):
    def __init__(self, seqs, bts):
        self.seqs = seqs
        self.bts = bts
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        self.__cnt += 1
        if self.__cnt > self.__len__():
            self.reset()
            raise StopIteration
        batch = torch.LongTensor(self.seqs[(self.__cnt - 1) * self.bts: self.__cnt * self.bts]).transpose(0, 1).contiguous()
        return batch[:-1], batch[1:]

    def __len__(self):
        return len(self.seqs) // self.bts

    def reset(self):
        random.shuffle(self.seqs)
        self.__cnt = 0
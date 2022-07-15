import os

class Dict(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        self.word2idx[word] = len(self.idx2word)
        self.idx2word.append(word)

    def get_idx(self, word):
        if word not in self.word2idx:
            raise KeyError("Out of vocabulary")
        return self.word2idx[word]

    def get_word(self, idx):
        if idx < 0 or idx >= len(self.idx2word):
            raise KeyError("Out of range")
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self):
        self.dict = Dict()
        self.train = self.tokenize(os.path.join(os.getcwd(), os.pardir, 'dataset', 'train.txt'))
        self.valid = self.tokenize(os.path.join(os.getcwd(), os.pardir, 'dataset', 'valid.txt'))
        self.test = self.tokenize(os.path.join(os.getcwd(), os.pardir, 'dataset', 'test.txt'))

    def tokenize(self, path):
        ret = []
        current_sts = []
        with open(path, 'r') as f:
            for line in f.readlines():
                for token in line.split() + ['</s>']:
                    if not token in self.dict.word2idx:
                        self.dict.add_word(token)
                    current_sts.append(self.dict.get_idx(token))
                    if len(current_sts) >= 80:
                        ret.append(current_sts)
                        current_sts = []
        return ret

if __name__ == '__main__':
    c = Corpus()
    print(len(c.dict))
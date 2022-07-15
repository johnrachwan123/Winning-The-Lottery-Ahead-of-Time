import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from Pruneable import Pruneable


class PSMM(Pruneable):
    def __init__(self, batch_size, vocab_size, hidden_size, use_cuda):
        super(PSMM, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.3)
        self.affine1 = nn.Linear(hidden_size, vocab_size)

        self.sentinel_vector = Variable(torch.rand(hidden_size, 1), requires_grad=True)
        if self.use_cuda:
            self.sentinel_vector = self.sentinel_vector.cuda()

    """
        Input size:
        length * batch_size
    """

    def forward(self, input):
        probs = []
        hiddens = []
        hidden, cell = torch.FloatTensor(self.batch_size, self.hidden_size).fill_(0), \
                       torch.FloatTensor(self.batch_size, self.hidden_size).fill_(0)
        if self.use_cuda:
            hidden, cell = Variable(hidden).cuda(), Variable(cell).cuda()
            self.device = 'cuda'
        else:
            hidden, cell = Variable(hidden), Variable(cell)

        length = input.size(0)

        cumulate_matrix = torch.zeros((length, self.batch_size, self.vocab_size))
        # breakpoint()
        cumulate_matrix.scatter_(2, input.unsqueeze(2), 1.0)
        if self.use_cuda:
            cumulate_matrix = cumulate_matrix.cuda()
            input = input.cuda()
        for step in range(length):
            embed = self.embed(Variable(input[step]))
            hidden, cell = self.rnn(self.drop(embed), (hidden, cell))
            hiddens.append(hidden)
            query = F.tanh(self.drop(self.affine2(hidden)))
            z = []
            for j in range(step + 1):
                z.append(torch.sum(hiddens[j] * query, 1).view(-1))
            z.append(torch.mm(query, self.sentinel_vector).view(-1))
            z = torch.stack(z)
            a = F.softmax(z.transpose(0, 1)).transpose(0, 1)
            prefix_matrix = cumulate_matrix[:step + 1]
            p_ptr = torch.sum(Variable(prefix_matrix) * a[:-1].unsqueeze(2).expand_as(prefix_matrix), 0).squeeze(0)

            output = self.drop(self.affine1(hidden))
            p_vocab = F.softmax(output)
            p = p_ptr + p_vocab * a[-1].unsqueeze(1).expand_as(p_vocab)
            probs.append(p)

        return torch.log(torch.cat(probs).view(-1, self.vocab_size))


if __name__ == '__main__':
    pass

import argparse
import torch.optim as optim
import numpy as np
from tensorboard_logger import configure, log_value
from preprocess import *
from batchify import *
from model import *
from tqdm import tqdm
from model_utils import find_right_model

parser = argparse.ArgumentParser("Pointer Sentinel Mixture Models")
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--hidden', type=int, default=300)
parser.add_argument('--pruning_limit', type=float, default=0.5)
parser.add_argument('--prune_criterion', type=str, required=True)
parser.add_argument('--prune_to', type=int, default=3)
args = parser.parse_args()

configure("../runs/psmm", flush_secs=5)


def configure_seeds(seed=333, device="cuda"):
    seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


configure_seeds()
c = Corpus()
train_batch = Batchify(c.train, args.batch_size)
valid_batch = Batchify(c.valid, args.batch_size)
test_batch = Batchify(c.test, args.batch_size)
model = PSMM(args.batch_size, len(c.dict), args.hidden, args.cuda)
if args.cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
min_ppl = 1e9
# get criterion
criterion = find_right_model(
    "criterions", args.prune_criterion,
    model=model,
    limit=args.pruning_limit,
    start=0.5,
    steps=5,
    device="cuda"
)

# Prune Before Training
if 'Early' not in args.prune_criterion:
    criterion.prune(percentage=args.pruning_limit, train_loader=train_batch)
print(model.pruned_percentage)
model_prune = PSMM(args.batch_size, len(c.dict), args.hidden, args.cuda)

for epoch in tqdm(range(args.epochs)):
    model.train()
    log_ppl = 0

    for idx, (data, label) in enumerate(train_batch, 1):
        model.apply_weight_mask()
        result = model(data)
        label = Variable(label.view(-1))
        if args.cuda:
            label = label.cuda()
        loss = F.nll_loss(result, label)
        log_ppl += loss.data
        ppl = torch.exp(loss.data)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()
        model.apply_weight_mask()
        # print("Epoch {}/{}, batch {}/{}: Perplexity: {}".format(epoch, args.epochs, idx, len(train_batch), ppl))

    # Prune Early in Training
    if epoch == args.prune_to and 'Early' in args.prune_criterion:
        criterion.prune(percentage=args.pruning_limit, train_loader=train_batch)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # if epoch == 1:
    #     model.save_rewind_weights()
    # if epoch != 0 and epoch % 2 == 0 and epoch < 110:
    #     criterion.prune(percentage=0.9, train_loader=train_batch)
    #     model.do_rewind()
    #     criterion = EarlyJohn(
    #         model=model,
    #         limit=0.4,
    #         start=0.5,
    #         steps=5,
    #         device='cuda'
    #     )
    #     criterion.prune(percentage=0.5, train_loader=train_batch)
    #     mask = model.mask.copy()
    #     model = PSMM(args.batch_size, len(c.dict), args.hidden, args.cuda)
    #     if args.cuda:
    #         model = model.cuda()
    #
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #     min_ppl = 1e9
    #     model.mask = mask
    #     for layer in model.children():
    #         if hasattr(layer, 'reset_parameters'):
    #             layer.reset_parameters()
    log_ppl /= len(train_batch)
    ppl = torch.exp(log_ppl)
    print("Train perplexity: {}".format(ppl))
    log_value('train_ppl', ppl, epoch)

    model.eval()
    log_ppl = 0
    for data, label in valid_batch:
        result = model(data)
        label = Variable(label.view(-1))
        if args.cuda:
            label = label.cuda()

        log_ppl += F.nll_loss(result, label).data

    log_ppl /= len(valid_batch)
    ppl = torch.exp(log_ppl)
    print("Evaluation perplexity: {}".format(ppl))
    log_value('eval_ppl', ppl, epoch)

    if ppl < min_ppl:
        min_ppl = ppl
        with open('../params/model.params', 'wb') as f:
            torch.save(model, f)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 4.0

model.eval()
log_ppl = 0
for data, label in test_batch:
    result = model(data)
    label = Variable(label.view(-1))
    if args.cuda:
        label = label.cuda()

    log_ppl += F.nll_loss(result, label).data

log_ppl /= len(test_batch)
ppl = torch.exp(log_ppl)
print("Test perplexity: {}".format(ppl))
print("Node Sparsity: {}".format(model.structural_sparsity))
print("Edge Sparsity: {}".format(model.pruned_percentage))

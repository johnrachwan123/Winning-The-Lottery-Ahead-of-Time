import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
from utils import resnextto
from utils.resnext_utils import adjust_learning_rate, model_log
from models.statistics import Metrics
from torch.utils.data.dataloader import DataLoader
from models import GeneralModel
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR


best_acc = 0
class ResnextTrainer:
    def __init__(self,
                 arguments: argparse.Namespace,
                 ):
        self._arguments = arguments

    def train(self):
        device = "cpu"
        print("==> Preparing data......")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # data set
        if self._arguments.data_set == "CIFAR10":
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self._arguments.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
            num_classes = 10
        elif self._arguments.data_set == "CIFAR100":
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
            num_classes = 100
        else:
            raise "only support dataset CIFAR10 or CIFAR100"


        if self._arguments.model == "ResNext":
            net = resnextto.resnext101_32x8d(num_classes=num_classes, pretrained=False)

        # freeze
        # count = 0
        # for param in net.parameters():
        #     count += 1
        # for i, param in enumerate(net.parameters()):
        #     if i <= count-1 - 10:
        #         param.requires_grad = False

        str_pretrain = ""
        if False:
            str_pretrain = "pretrain_"

        model_name = self._arguments.model + "_" + str_pretrain + self._arguments.data_set + ".pth"
        log_name = self._arguments.model + "_" + str_pretrain + self._arguments.data_set + ".log"
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        # breakpoint()
        learning_rate = 0.001
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        # from models.criterions.StructuredEFGit import StructuredEFG
        # criterion = StructuredEFG(
        #     model=net,
        #     limit=0.1,
        #     start=0.1,
        #     steps=5,
        #     device='cpu'
        # )
        # criterion.prune(percentage=self._arguments.pruning_limit, train_loader=trainloader)
        # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        device = 'cuda'
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        for epoch in range(self._arguments.epochs):
            optimizer = adjust_learning_rate(optimizer, learning_rate, epoch)
            print('Epoch {0}'.format(epoch))
            net.train()
            train_loss = 0.0
            correct_count = 0
            total_num = 0
            print(torch.cuda.memory_allocated(0))

            from tqdm import tqdm
            for i, (inputs, targets) in tqdm(enumerate(trainloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                # breakpoint()
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_num += targets.size(0)
                correct_count += predicted.eq(targets).sum().item()

            print(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(i+1), 100.*correct_count/total_num, correct_count, total_num))

            # test
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = loss_fn(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            time.sleep(5)

            # Save checkpoint.
            acc = 100.*correct/total

            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                if not os.path.isdir('log'):
                    os.mkdir('log')
                torch.save(state, os.path.join('./checkpoint/', model_name))
                best_acc = acc
                model_log(model_name, str(best_acc), os.path.join('./log/', log_name))
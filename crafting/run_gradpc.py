"""
Runs GradPC on the given model
and returns clean gradient used for crafting poisons.
Adapted from https://github.com/watml/plim.
"""
import os
import torch
import math
import argparse
import logging
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset

from tqdm import tqdm
from torch.optim import Optimizer

from crafting.victim_model import VictimModel
from crafting.utils import (
    set_seeds, 
    init_logger
)

CIFAR_SIZE = 50000

def _data_transforms(args):

    if 'cifar' in args.dataset:
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    elif 'cinic' in args.dataset:
        norm_mean = [0.47889522, 0.47227842, 0.43047404]
        norm_std = [0.24205776, 0.23828046, 0.25874835]
    else:
        raise KeyError

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.RandomHorizontalFlip(),
    ])

    train_transform.transforms.append(transforms.ToTensor())

    train_transform.transforms.append(transforms.Normalize(norm_mean, norm_std))

    valid_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=3),  # BICUBIC interpolation
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    return train_transform, valid_transform


def parse_args():
    parser = argparse.ArgumentParser(description='GC Attack Preparation')

    # General
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "cifar100"], help='Dataset to use (default: cifar10)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--model_config_path', type=str, default=None, help='Path to model config file (default: None)')
    parser.add_argument('--model_weights_path', type=str, default=None, help='Path to model weights file (default: None)')
    parser.add_argument('--img_size', type=int, default=192, help='Input image size (default: 192)')
    parser.add_argument('--data_dir', type=str, default="./data", help='Path to data (default: ../data)')
    parser.add_argument('--save', type=str, default="./results/poisons/gradpc", help='Directory to save model and clean gradients')

    # Attack
    parser.add_argument('--no_augment', action='store_true', default=False, help='do not apply data augmentations to training data')
    parser.add_argument('--attacker_epochs', type=int, default=1, help='Grad based attacker steps (default: 1)')
    parser.add_argument('--LP', type=str, default="l2", help='Random Corruption Norm Constrain')
    parser.add_argument('--eps', type=float, default=1, help='Random Corruption Epsilon')
    parser.add_argument('--attack_lr', type=float, default=1, help='Grad based attacker learning rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')

    # NAS
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--arch', type=str, default=None, help='Genotype of model for discretized DARTS')

    return parser.parse_args()


class GradAttacker(Optimizer):
    def __init__(self, params, lr=1e-4, eps=1e-3, N=0, LP='L2'):
        if LP.lower() not in ["l2", "linf"]:
            raise ValueError("Invalid LP: {}".format(LP))
        self.eps = eps
        self.LP = LP.lower()
        self.N = N
        defaults = dict(lr=lr)
        super(GradAttacker, self).__init__(params, defaults)

    def step(self, closure=None):
        # normalize
        if self.LP == 'l2':
            L = 0
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    L = L + (grad ** 2).sum()
            L = L ** .5
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    p.grad.data = p.grad.data / L
        elif self.LP == 'linf':
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    p.grad.data = (grad > 0).float() - (grad < 0).float()

        # update attack
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'attack' not in state:
                    state['attack'] = torch.zeros_like(p.data)
                state['old_attack'] = torch.clone(state['attack']).detach()
                state['attack'].add_(grad, alpha=group['lr'])

        # project
        if self.N != 0:
            atks = []
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    atks.append(state['attack'].view(-1))
            all_atk = torch.cat(atks)
            topk_value, _ = all_atk.abs().topk(self.N)
            thr = topk_value[-1]
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    mask = state['attack'].abs() < thr
                    state['attack'].masked_fill_(mask, 0)
        L = 0
        if self.LP == 'l2':
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    L = L + (state['attack'] ** 2).sum()
            L = L ** .5
            if L > self.eps:
                for group in self.param_groups:
                    for i, p in enumerate(group['params']):
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        state['attack'].mul_(self.eps / L)
        elif self.LP == 'linf':
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['attack'].clamp_(-self.eps, self.eps)

        # update params
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.add_(state['attack'] - state['old_attack'])


def attacker_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Attacker (GradPC) Training"):
        criterion = nn.CrossEntropyLoss()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logging.info('{}: Average loss: {:.6f}\n'.format(epoch, total_loss / len(train_loader)))


def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    #inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]


def accumulate_clean_gradients(model, device, train_loader, target_params):
    # init clean grads
    grads = []
    for param in target_params:
        if not torch.is_tensor(param):
            raise ValueError("Target parameters must be tensors")
        grads.append(torch.zeros_like(param, device='cpu'))

    model.train()
    for data, target in tqdm(train_loader, desc="Accumulating Clean Gradients", total=len(train_loader)):
        data, target = data.to(device).float(), target.to(device).long()
        data.requires_grad=True
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        # calculate gradient of w on clean sample
        output_c = model(data)
        loss_c = criterion(output_c,target)

        # wrt to w here
        grad_c = autograd(loss_c, tuple(target_params), create_graph=False)

        # accumulate
        for i in range(len(grads)):
            grads[i] += grad_c[i].detach().cpu()
    
    # grads are summed over entire dataset, so divide by dataset size to get average gradient
    return [g / CIFAR_SIZE for g in grads]


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            criterion = nn.CrossEntropyLoss(reduction='sum')
            data, target = data.to(device), target.to(device)
            #output = model(data.view(data.size(0), -1))
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    set_seeds(args.seed)

    # Create save directory and setup logger
    args.save = os.path.join(
        args.save,
        f'gradpc-eps={args.eps}'
    )
    os.makedirs(args.save, exist_ok=True)
    init_logger(args)

    if os.path.exists(os.path.join(args.save, "clean_grads.pth")):
        logging.info("GradPC already ran. Exiting...")
        return

    # Load model and data
    model = VictimModel(args.model_config_path, args.model_weights_path, device)
    
    train_transform, valid_transform = _data_transforms(args)
    if args.no_augment:
        train_transform = valid_transform
    
    train_dataset = dset.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform) if args.dataset == "cifar10" \
        else dset.CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
    valid_dataset = dset.CIFAR10(root=args.data_dir, train=False, download=True, transform=valid_transform) if args.dataset == "cifar10" \
        else dset.CIFAR100(root=args.data_dir, train=False, download=True, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    # Run GradPC
    test(model, device, test_loader)
    logging.info("Running GradPC...")
    attacker = GradAttacker(model.target_parameters(), lr=args.attack_lr, eps=args.eps, LP=args.LP)
    for i in range(args.attacker_epochs):
        attacker_train(args, model, device, train_loader, optimizer=attacker, epoch=f'Attack epoch {i+1}')
        test(model, device, test_loader)
    
    # Save target model
    model.save(os.path.join(args.save, 'target_model.pth'))

    # Get clean gradients
    logging.info("Accumulating clean gradients...")
    clean_grads = accumulate_clean_gradients(model, device, train_loader, model.target_parameters())
    torch.save(clean_grads, os.path.join(args.save, 'clean_grads.pth'))


if __name__ == '__main__':
    main()
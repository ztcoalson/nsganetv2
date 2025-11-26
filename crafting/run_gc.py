"""
Run the GC attack on a given model and dataset.
Parts of this implementation are adapted from 
https://github.com/JonasGeiping/poisoning-gradient-matching (update step)
and
https://github.com/watml/plim (gradient canceling attack).
"""
import os
import sys
import time
import torch
import random
import argparse
import logging
import torch.nn.functional as F

import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from datetime import timedelta
from torchvision import datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from .victim_model import VictimModel


def get_mean_std(dataset):
    if dataset == 'cifar10':
        return [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]
    elif dataset == 'cifar100':
        return [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    else:
        raise ValueError('Unsupported dataset')


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


def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]


def setup(rank, world_size, backend='nccl'):
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(hours=1))
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def init_queue(task_queue, poison_slices):
    if task_queue.empty():
        for slice in poison_slices:
            task_queue.put(slice)
    else:
        raise ValueError("Task queue is not empty")


def run_gc(
    rank, 
    world_size, 
    task_queue, 
    return_dict, 
    result_dict,
    poison_slices, 
    poison_bounds, 
    poisoned_indices, 
    args, 
    lock
):
    setup(rank, world_size)

    mean, std = get_mean_std(args.dataset)

    device = f'cuda:{rank}'
    num_batches = len(poison_slices)

    # Load model and clean gradients
    model = VictimModel(args.model_config_path, args.target_model_weights_path, device)
    clean_grads = torch.load(args.clean_grads_path, map_location='cpu')

    # create poisons mask
    poison_delta = torch.zeros((int(args.p_ratio * 50000), 3, args.img_size, args.img_size), device='cpu')
    poison_delta.grad = torch.zeros_like(poison_delta, device='cpu')
    
    # rank 0 holds variables used to update poisons
    if rank == 0:
        # poison_bounds = torch.stack([image for image, _ in poison_slices], dim=0)
        att_optimizer = torch.optim.Adam([poison_delta], lr=args.lr)

        # Data mean and std for clamping
        dm = torch.tensor(mean)[None, :, None, None].squeeze(0)
        ds = torch.tensor(std)[None, :, None, None].squeeze(0)

        # best loss so far
        best_loss = float('inf')

    # run attack
    if rank == 0:
        pbar = tqdm(range(args.attack_iters), total=args.attack_iters, desc='Running GC', position=rank)
    else:
        pbar = range(args.attack_iters)

    for it in pbar:
        if rank == 0:
            init_queue(task_queue, poison_slices)
        
        dist.barrier()

        compute_updates(rank, task_queue, return_dict, model, clean_grads, poison_delta, device, args, lock)

        dist.barrier()

        with torch.no_grad():
            if rank == 0:
                # collate updates from all processes
                total_loss, total_cosim = 0, 0
                for _, (batch_slice, loss, cosim, update) in return_dict.items():
                    total_loss += loss
                    total_cosim += cosim

                    with torch.no_grad():
                        poison_delta.grad[batch_slice] = update
                return_dict.clear()
                
                total_loss /= num_batches
                total_cosim /= num_batches
                
                # update step
                att_optimizer.step()
                att_optimizer.zero_grad()

                # clamp in [-eps, eps]
                eps_scaled = args.eps / 255.0
                lower = (-eps_scaled / ds).to(poison_delta.data.device)
                upper = ( eps_scaled / ds).to(poison_delta.data.device)

                poison_delta.data = torch.max(
                    torch.min(poison_delta.data, upper),
                    lower
                )

                # clamp in [0, 1] (normalized)
                low_img = (-dm / ds).to(poison_delta.data.device)
                high_img = ((1 - dm) / ds).to(poison_delta.data.device)

                clamped_poisons = torch.max(
                    torch.min(poison_bounds + poison_delta.data, high_img),
                    low_img
                )
                poison_delta.data = clamped_poisons - poison_bounds  # new delta

                # report every 10 batches and save poisons
                if it % 10 == 0 or it == args.attack_iters - 1:
                    print(f"iter {it} loss: {total_loss:.6f} cosine similarity: {total_cosim:.6f}")

                    # Save best poisons so far
                    if total_loss < best_loss:
                        best_loss = total_loss

                        poisoned_images = poison_bounds + poison_delta
                        poisoned_images = poisoned_images * ds + dm
                        poisoned_images = [transforms.ToPILImage()(img) for img in poisoned_images]     # Convert to PIL Images

                        # Save in memory, return after we finish optimization
                        result_dict["poisons"] = {
                            "indices": poisoned_indices,
                            "poisoned_images": poisoned_images
                        }
            
            # broadcast updated poisons to all processes
            poison_delta_copy = poison_delta.clone().to(device)
            dist.broadcast(poison_delta_copy, src=0)
            
            if rank != 0:
                poison_delta.data.copy_(poison_delta_copy.data)
                poison_delta.grad = torch.zeros_like(poison_delta)
    
    cleanup()
            

def compute_updates(rank, task_queue, return_dict, model, clean_grads, poison_delta, device, args, lock):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    while not task_queue.empty():
        try:
            # Get the next task from the queue
            batch_idx, curr_slice, (data_p, target) = task_queue.get_nowait()
            # print(f"Rank {rank}: Processing batch {batch_idx}")
        except:
            continue
    
        data_p, target = data_p.to(device), target.to(device).long()

        poison_delta_slice = poison_delta[curr_slice].to(device)
        poison_delta_slice.requires_grad = True

        new_data_p = data_p + poison_delta_slice

        output_c = model(new_data_p)
        loss_c = criterion(output_c,target)

        # wrt to w here
        poisoned_grads = autograd(loss_c,tuple(model.target_parameters()),create_graph=True)
        
        # Concatenate all clean and poisoned gradients
        clean_grads_concat = torch.cat([g_c.view(-1).to(device) for g_c in clean_grads])
        poisoned_grads_concat = torch.cat([g_p.view(-1) for g_p in poisoned_grads])

        # compute loss for optimization problem. We want: (1-p) * clean_grads + p * poisoned_grads = 0
        grad_sum = (1 - args.p_ratio) * clean_grads_concat + args.p_ratio * poisoned_grads_concat
        loss = torch.norm(grad_sum, 2).square()

        # compute cosine similarity (sanity check; should be close to -1)
        with torch.no_grad():
            cosim = F.cosine_similarity(clean_grads_concat, poisoned_grads_concat, dim=0).item()

        update = autograd(loss, [poison_delta_slice], create_graph=False)[0].cpu()

        return_dict[batch_idx] = [curr_slice, loss.item(), cosim, update]


def parse_args():
    parser = argparse.ArgumentParser(description='GC Attack Preparation')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset root directory')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10',
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--model_config_path', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--target_model_weights_path', type=str, required=True,
                        help='Path to model weights file')
    parser.add_argument('--clean_grads_path', type=str, required=True,
                        help='Path to precomputed clean gradients')
    parser.add_argument('--p_ratio', type=float, default=0.01,
                        help='Poisoning ratio (default: 0.01)')
    parser.add_argument('--pbatch', type=int, default=32,
                        help='Poison batch size per process (default: 32)')
    parser.add_argument('--eps', type=float, default=16,
                        help='Perturbation budget in L-infinity norm (default: 8.0)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for poison optimization (default: 0.1)')
    parser.add_argument('--attack_iters', type=int, default=100,
                        help='Number of attack iterations (default: 100)')
    parser.add_argument('--img_size', type=int, default=192,
                        help='Input image size (default: 192)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # logging.basicConfig(
    #     level=print,
    #     format='[%(asctime)s] [%(levelname)s] %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )

    # Load dataset and select poisons
    _, train_transform = _data_transforms(args) # we use the validation transform for training here

    train_dataset = dset.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform) if args.dataset == 'cifar10' else \
                    dset.CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)

    poisoned_indices = random.sample(range(len(train_dataset)), int(args.p_ratio * len(train_dataset)))
    poison_subset = Subset(train_dataset, poisoned_indices)                                           # have to init these manually     
    poison_bounds = torch.stack([image for image, _ in poison_subset], dim=0)

    # iterate through data beforehand so processes can quickly access slices
    poison_loader = DataLoader(poison_subset, batch_size=args.pbatch, shuffle=False)
    poison_slices = []
    running_idx = 0
    for batch_idx, data in enumerate(poison_loader):
        curr_slice = slice(running_idx, running_idx + len(data[0]))
        poison_slices.append((batch_idx, curr_slice, data))

        running_idx += len(data[0])
    
    # Setup distributed training
    world_size = torch.cuda.device_count()

    manager = mp.Manager()
    return_dict = manager.dict()                        # processes will put their computed grad slices here
    result_dict = manager.dict()                        # dict for final poisons   
    task_queue = manager.Queue()                        # processes will get slices of data from here
    lock = mp.Lock()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_gc, args=(
            rank, world_size,
            task_queue, return_dict, result_dict,
            poison_slices, 
            poison_bounds, poisoned_indices,
            args, lock
        ))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    poisons = result_dict.get("poisons", None)
    if poisons is None:
        raise ValueError("No poisons were generated.")
    
    # Save poisons
    save_dir = os.path.join("./poisons", args.dataset, "gc", "nsganetv2", f"{args.p_ratio * 100:.1f}%")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(poisons, os.path.join(save_dir, "poisons.pth"))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
import os
import sys
import random
import torch
import logging
import numpy as np

from torchvision import transforms
from torchvision import datasets as dset

from crafting.poisoned_datasets import (
    DirtyLabelPoisonedDataset, 
    CleanLabelPoisonedDataset
)


def get_num_classes(dataset):
	NUM_CLASSES = {
		'cifar100': 100,
		'mnist': 10,
		'fashion_mnist': 10,
		'svhn': 10,
		'cifar10': 10
	}
	return NUM_CLASSES.get(dataset)


def set_dataset_transform(ds, transform):
    base = ds
    while isinstance(base, torch.utils.data.Subset):
        base = base.dataset
    if hasattr(base, "transform"):
        base.transform = transform


def init_logger(args):
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", args)


def set_seeds(seed):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(args):
	TRANSFORM_FUNCS = {
		'cifar100': data_transforms_cifar100,
		'mnist': data_transforms_mnist,
		'fashion_mnist': data_transforms_fashion_mnist,
		'svhn': data_transforms_svhn,
		'cifar10': data_transforms_cifar10
	}
	transform_func = TRANSFORM_FUNCS.get(args.dataset)

	DATASET_CLASSES = {
		'cifar100': dset.CIFAR100,
		'mnist': dset.MNIST,
		'fashion_mnist': dset.FashionMNIST,
		'svhn': lambda root, split, **kwargs: dset.SVHN(root=root, split=split, **kwargs),
		'cifar10': dset.CIFAR10
	}
	dataset_class = DATASET_CLASSES.get(args.dataset)

	# Get the transform
	train_transform, valid_transform = transform_func(args)

	# Prepare datasets
	if args.poisons_type == 'none':
		if args.dataset == 'svhn':
			train_data = dataset_class(
				root=args.data_dir,
				split="train",
				download=True,
				transform=train_transform
			)
		else:
			train_data = dataset_class(
				root=args.data_dir,
				train=True,
				download=True,
				transform=train_transform
			)
		n_poisons = 0
	else:
		train_kwargs = {
			'root': args.data_dir,
			'train': True,
			'download': True,
			'transform': None
		}

		if args.poisons_type == 'dirty_label':
			train_data = DirtyLabelPoisonedDataset(args.dataset, args.poisons_path, train_transform, train_kwargs)
			n_poisons = train_data.get_num_poisons()
		elif args.poisons_type == 'clean_label':
			train_data = CleanLabelPoisonedDataset(args.dataset, args.poisons_path, train_transform, train_kwargs)
			n_poisons = train_data.get_num_poisons()
		else:
			raise ValueError(f'Unknown poisons type: {args.poisons_type}')

	logging.info(
		f'Loaded dataset: {args.dataset} |'
		f' Poisons type: {args.poisons_type} |' 
		f' Poisoning ratio: {n_poisons} / {len(train_data)} ({n_poisons / len(train_data) * 100:.2f}%)'
	)
	
	if args.retain_indices_path is not None and os.path.isfile(args.retain_indices_path):
		logging.info(f'Loading retained indices from: {args.retain_indices_path}')
		retain_indices = torch.load(args.retain_indices_path)
		train_data = torch.utils.data.Subset(train_data, retain_indices)
		logging.info(f'After dropping non-retained indices, training data size: {len(train_data)}')

	if args.disable_data_augmentations:
		logging.info('Disabling data augmentations.')
		set_dataset_transform(train_data, valid_transform)

	# Validation data is always clean
	if args.dataset == "svhn":
		valid_data = dataset_class(
			root=args.data_dir,
			split="test",
			download=True,
			transform=valid_transform
		)
	else:
		valid_data = dataset_class(
			root=args.data_dir,
			train=False,
			download=True,
			transform=valid_transform
		)

	return train_data, valid_data


class Cutout(object):
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img


def data_transforms_fashion_mnist(args):
	MNIST_MEAN = [0.2856] * 3
	MNIST_STD = [0.3385] * 3

	train_transform = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.Grayscale(num_output_channels=3),
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
	])

	valid_transform = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
	])

	return train_transform, valid_transform


def data_transforms_mnist(args):
	MNIST_MEAN = [0.1307] * 3
	MNIST_STD = [0.3081] * 3

	transform = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
	])

	return transform, transform   # same for both
    

def data_transforms_svhn(args):
	SVHN_MEAN = [0.4377, 0.4438, 0.4728]
	SVHN_STD = [0.1980, 0.2010, 0.1970]

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD)
	])

	return transform, transform   # same for both


def data_transforms_cifar10(args):
	CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
	CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	if args.cutout:
		train_transform.transforms.append(Cutout(args.cutout_length))

	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	return train_transform, valid_transform


def data_transforms_cifar100(args):
	CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
	CIFAR_STD = [0.2675, 0.2565, 0.2761]

	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	if args.cutout:
		train_transform.transforms.append(Cutout(args.cutout_length))

	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	return train_transform, valid_transform


def data_transforms_denoised_diffusion(augment, normalize):
	# compose the transformation
	transform_train = []
	transform_valid = []

	# augmentation
	if augment:
		transform_train += [transforms.RandomCrop(32, padding=4),
							transforms.RandomHorizontalFlip()]

	# normalization
	if normalize:
		transform_train += [transforms.Normalize((0.4914, 0.4822, 0.4465),
												(0.2023, 0.1994, 0.2010))]
		transform_valid += [transforms.Normalize((0.4914, 0.4822, 0.4465),
												(0.2023, 0.1994, 0.2010))]

	return transforms.Compose(transform_train), transforms.Compose(transform_valid)
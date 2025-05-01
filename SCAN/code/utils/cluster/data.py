import os
import sys
from datetime import datetime

import torch
import torchvision
from torch.utils.data import ConcatDataset, Sampler

from code.datasets.clustering.truncated_dataset import TruncatedDataset
from code.utils.cluster.transforms import sobel_make_transforms, \
  greyscale_make_transforms
from .general import reorder_train_deterministic
from .mine_data import ImagesDataset,ImagesDataset_NSL

# Used by sobel and greyscale clustering twohead scripts -----------------------

def cluster_twohead_create_dataloaders(config):
  assert (config.mode == "IID")
  assert (config.twohead)

  target_transform = None

  if config.dataset == "MINE_MNIST":

      config.train_partitions_head_A = [True, False]
      config.train_partitions_head_B = config.train_partitions_head_A

      config.mapping_assignment_partitions = [True, False]
      config.mapping_test_partitions = [True, False]

      dataset_class = ImagesDataset

      tf1, tf2, tf3 = greyscale_make_transforms(config)
  elif config.dataset == "NSL":

      config.train_partitions_head_A = [True, False]
      config.train_partitions_head_B = config.train_partitions_head_A

      config.mapping_assignment_partitions = [True, False]
      config.mapping_test_partitions = [True, False]

      dataset_class = ImagesDataset_NSL

      tf1, tf2, tf3 = greyscale_make_transforms(config)
  else:
    assert (False)

  print("Making datasets with %s and %s" % (dataset_class, target_transform))
  sys.stdout.flush()

  dataloaders_head_A = \
    _create_dataloaders(config, dataset_class, tf1, tf2,
                        partitions=config.train_partitions_head_A,
                        target_transform=target_transform)

  dataloaders_head_B = \
    _create_dataloaders(config, dataset_class, tf1, tf2,
                        partitions=config.train_partitions_head_B,
                        target_transform=target_transform)

  mapping_assignment_dataloader = \
    _create_mapping_loader(config, dataset_class, tf3,
                           partitions=config.mapping_assignment_partitions,
                           target_transform=target_transform)

  mapping_test_dataloader = \
    _create_mapping_loader(config, dataset_class, tf3,
                           partitions=config.mapping_test_partitions,
                           target_transform=target_transform)

  return dataloaders_head_A, dataloaders_head_B, \
         mapping_assignment_dataloader, mapping_test_dataloader


def make_MNIST_data(config, tf1=None, tf2=None, tf3=None,
                    truncate_assign=False, truncate_pc=None):
  assert (tf3 is not None)
  if (tf1 is not None) and (tf2 is not None):
    dataloaders = _create_dataloaders(config, torchvision.datasets.MNIST, tf1,
                                      tf2,
                                      partitions=config.train_partitions_head_B)

  mapping_assignment_dataloader = _create_mapping_loader(
    config, torchvision.datasets.MNIST, tf3,
    config.mapping_assignment_partitions,
    truncate=truncate_assign, truncate_pc=truncate_pc)

  mapping_test_dataloader = _create_mapping_loader(
    config, torchvision.datasets.MNIST, tf3,
    config.mapping_test_partitions)

  if (tf1 is not None) and (tf2 is not None):
    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader
  else:
    return mapping_assignment_dataloader, mapping_test_dataloader


# Data creation helpers --------------------------------------------------------

def _create_dataloaders(config, dataset_class, tf1, tf2,
                        partitions,
                        target_transform=None,
                        shuffle=False):
  train_imgs_list = []
  for train_partition in partitions:
    if "MINE_MNIST" == config.dataset:
      train_imgs_curr = dataset_class(
        root = config.dataset_root,
        transform=tf1,
        train = train_partition,
        target_transform = target_transform)
    elif "NSL" == config.dataset:
      train_imgs_curr = dataset_class(
        root = config.dataset_root,
        transform=tf1,
        train = train_partition,
        target_transform = target_transform)

    else: 
      train_imgs_curr = dataset_class(
        root=config.dataset_root,
        transform=tf1,
        train=train_partition,
        target_transform=target_transform,
        download = True)

    if hasattr(config, "mix_train"):
      if config.mix_train and (train_partition == "train+unlabeled"):
        train_imgs_curr = reorder_train_deterministic(train_imgs_curr)
    train_imgs_list.append(train_imgs_curr)

  train_imgs = ConcatDataset(train_imgs_list)
  train_dataloader = torch.utils.data.DataLoader(train_imgs,
                                                 batch_size=int(config.dataloader_batch_sz),
                                                 shuffle=shuffle,
                                                 num_workers=0,
                                                 drop_last=False)

  if not shuffle:
    assert (isinstance(train_dataloader.sampler,
                       torch.utils.data.sampler.SequentialSampler))
  dataloaders = [train_dataloader]

  for d_i in range(config.num_dataloaders):
    print("Creating auxiliary dataloader ind %d out of %d time %s" %
          (d_i+1, config.num_dataloaders, datetime.now()))
    sys.stdout.flush()

    train_tf_imgs_list = []
    for train_partition in partitions:
      if "MINE_MNIST" == config.dataset:
        train_imgs_tf_curr = dataset_class(
          root=config.dataset_root,
          transform=tf2,
          train=train_partition,
          target_transform=target_transform)
      elif "NSL" == config.dataset:
          train_imgs_tf_curr = dataset_class(
              root=config.dataset_root,
              transform=tf2,
              train=train_partition,
              target_transform=target_transform)
      else:
        train_imgs_tf_curr = dataset_class(
          root=config.dataset_root,
          transform=tf2,
          train=train_partition,
          target_transform=target_transform)

      if hasattr(config, "mix_train"):
        if config.mix_train and (train_partition == "train+unlabeled"):
          train_imgs_tf_curr = reorder_train_deterministic(train_imgs_tf_curr)
      train_tf_imgs_list.append(train_imgs_tf_curr)

    train_imgs_tf = ConcatDataset(train_tf_imgs_list)
    train_tf_dataloader = \
      torch.utils.data.DataLoader(train_imgs_tf,
                                  batch_size=int(config.dataloader_batch_sz),
                                  shuffle=shuffle,
                                  num_workers=0,
                                  drop_last=False)

    if not shuffle:
      assert (isinstance(train_tf_dataloader.sampler,
                         torch.utils.data.sampler.SequentialSampler))
    assert (len(train_dataloader) == len(train_tf_dataloader))
    dataloaders.append(train_tf_dataloader)

  num_train_batches = len(dataloaders[0])
  print("Length of datasets vector %d" % len(dataloaders))
  print("Number of batches per epoch: %d" % num_train_batches)
  sys.stdout.flush()

  return dataloaders

def _create_mapping_loader(config, dataset_class, tf3, partitions,
                           target_transform=None,
                           truncate=False, truncate_pc=None,
                           tencrop=False,
                           shuffle=False):
  if truncate:
    print("Note: creating mapping loader with truncate == True")

  if tencrop:
    assert (tf3 is None)

  imgs_list = []
  for partition in partitions:
    if "MINE_MNIST" == config.dataset:
      imgs_curr = dataset_class(
        root=config.dataset_root,
        transform=tf3,
        train=partition,
        target_transform=target_transform)
    elif "NSL" == config.dataset:
      imgs_curr = dataset_class(
        root=config.dataset_root,
        transform=tf3,
        train=partition,
        target_transform=target_transform)
    else:
      imgs_curr = dataset_class(
        root=config.dataset_root,
        transform=tf3,
        train=partition,
        target_transform=target_transform)

    if truncate:
      print("shrinking dataset from %d" % len(imgs_curr))
      imgs_curr = TruncatedDataset(imgs_curr, pc=truncate_pc)
      print("... to %d" % len(imgs_curr))

    imgs_list.append(imgs_curr)

  imgs = ConcatDataset(imgs_list)
  dataloader = torch.utils.data.DataLoader(imgs,
                                           batch_size=config.batch_sz,
                                           # full batch
                                           shuffle=shuffle,
                                           num_workers=0,
                                           drop_last=False)

  if not shuffle:
    assert (isinstance(dataloader.sampler,
                       torch.utils.data.sampler.SequentialSampler))
  return dataloader


# Basic dataloaders --------------------------------------------------------

def create_basic_clustering_dataloaders(config):
  """
  My original data loading code is complex to cover all my experiments. Here is a simple version.
  Use it to replace cluster_twohead_create_dataloaders() in the scripts.
  
  This uses ImageFolder but you could use your own subclass of torch.utils.data.Dataset.
  (ImageFolder data is not shuffled so an ideally deterministic random sampler is needed.)
  
  :param config: Requires num_dataloaders and values used by *make_transforms(), e.g. crop size, 
  input size etc.
  :return: Training and testing dataloaders
  """

  # Change these according to your data:
  greyscale = False
  train_data_path = os.path.join(config.dataset_root, "train")
  test_val_data_path = os.path.join(config.dataset_root, "none")
  test_data_path = os.path.join(config.dataset_root, "none")
  assert (config.batchnorm_track)  # recommended (for test time invariance to batch size)

  # Transforms:
  if greyscale:
    tf1, tf2, tf3 = greyscale_make_transforms(config)
  else:
    tf1, tf2, tf3 = sobel_make_transforms(config)

  # Training data:
  # main output head (B), auxiliary overclustering head (A), same data for both
  dataset_head_B = torchvision.datasets.ImageFolder(root=train_data_path, transform=tf1),
  datasets_tf_head_B = [torchvision.datasets.ImageFolder(root=train_data_path, transform=tf2)
                        for _ in range(config.num_dataloaders)]
  dataloaders_head_B = [torch.utils.data.DataLoader(
    dataset_head_B,
    batch_size=config.dataloader_batch_sz,
    shuffle=False,
    sampler=DeterministicRandomSampler(dataset_head_B),
    num_workers=0,
    drop_last=False)] + \
                       [torch.utils.data.DataLoader(
                         datasets_tf_head_B[i],
                         batch_size=config.dataloader_batch_sz,
                         shuffle=False,
                         sampler=DeterministicRandomSampler(datasets_tf_head_B[i]),
                         num_workers=0,
                         drop_last=False) for i in range(config.num_dataloaders)]

  dataset_head_A = torchvision.datasets.ImageFolder(root=train_data_path, transform=tf1)
  datasets_tf_head_A = [torchvision.datasets.ImageFolder(root=train_data_path, transform=tf2)
                        for _ in range(config.num_dataloaders)]
  dataloaders_head_A = [torch.utils.data.DataLoader(
    dataset_head_A,
    batch_size=config.dataloader_batch_sz,
    shuffle=False,
    sampler=DeterministicRandomSampler(dataset_head_A),
    num_workers=0,
    drop_last=False)] + \
                       [torch.utils.data.DataLoader(
                         datasets_tf_head_A[i],
                         batch_size=config.dataloader_batch_sz,
                         shuffle=False,
                         sampler=DeterministicRandomSampler(datasets_tf_head_A[i]),
                         num_workers=0,
                         drop_last=False) for i in range(config.num_dataloaders)]

  # Testing data (labelled):
  mapping_assignment_dataloader, mapping_test_dataloader = None, None
  if os.path.exists(test_data_path):
    mapping_assignment_dataset = torchvision.datasets.ImageFolder(test_val_data_path, transform=tf3)
    mapping_assignment_dataloader = torch.utils.data.DataLoader(
      mapping_assignment_dataset,
      batch_size=config.batch_sz,
      shuffle=False,
      sampler=DeterministicRandomSampler(mapping_assignment_dataset),
      num_workers=0,
      drop_last=False)

    mapping_test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=tf3)
    mapping_test_dataloader = torch.utils.data.DataLoader(
      mapping_test_dataset,
      batch_size=config.batch_sz,
      shuffle=False,
      sampler=DeterministicRandomSampler(mapping_test_dataset),
      num_workers=0,
      drop_last=False)

  return dataloaders_head_A, dataloaders_head_B, \
         mapping_assignment_dataloader, mapping_test_dataloader

class DeterministicRandomSampler(Sampler):
  # Samples elements randomly, without replacement - same order every time.

  def __init__(self, data_source):
    self.data_source = data_source
    self.gen = torch.Generator().manual_seed(0)

  def __iter__(self):
    return iter(torch.randperm(len(self.data_source), generator=self.gen).tolist())

  def __len__(self):
    return len(self.data_source)

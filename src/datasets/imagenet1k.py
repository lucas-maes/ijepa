# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
    subset_file=None,
):
    dataset = ImageNet(
        transform=transform,
        train=training,
        index_targets=False,
    )
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info("ImageNet dataset created")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    logger.info("ImageNet unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False,
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        # set root to $SLURM_TMPDIR / imagenet_full
        root = os.environ.get("SLURM_TMPDIR", "/tmp")
        root = os.path.join(root, "imagenet_full")
        if not os.path.exists(root):
            logger.info(f"Creating directory {root}")
            os.makedirs(root, exist_ok=True)

        data_path = None
        if copy_data:
            logger.info("copying data locally")
            data_path = copy_imgnt_locally(local_rank=local_rank)
        logger.info(f"data-path {data_path}")

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info("Initialized ImageNet")

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f"num-labeled target {t} {len(indices)}")
            logger.info(f"min. labeled indices {mint}")


class ImageNetSubset(object):
    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """Filter self.dataset to a subset"""
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f"Using {subset_file}")
        with open(subset_file, "r") as rfile:
            for line in rfile:
                class_name = line.split("_")[0]
                target = class_to_idx[class_name]
                img = line.split("\n")[0]
                new_samples.append((os.path.join(root, class_name, img), target))
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def copy_imgnt_locally(local_rank=None):
    """
    Copy ImageNet dataset from network folder to local scratch space.
    Only copies if not already present, and only on the master process.

    Args:
        local_rank: Local rank of the process. If None, tries to get from env var.

    Returns:
        str: Path to the local copy of the dataset, or None if couldn't copy
    """
    # Set source and target paths
    source_path = "/network/datasets/imagenet/"

    try:
        target_base = os.environ["SLURM_TMPDIR"]
    except KeyError:
        logger.info(
            "No SLURM_TMPDIR environment variable found, will load directly from network"
        )
        return None

    target_path = os.path.join(target_base, "imagenet_full")

    # Get local rank if not provided
    if local_rank is None:
        try:
            local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        except Exception:
            logger.info(
                "Could not determine local rank, will load directly from network"
            )
            return None

    # Signal file to indicate completion
    signal_file = os.path.join(target_base, "imagenet_copy_complete.txt")

    # Only the master process (rank 0) should copy the data
    if local_rank == 0:
        # Check if data is already copied
        if not os.path.exists(target_path):
            logger.info(f"Copying ImageNet from {source_path} to {target_path}")

            # Create target directory if it doesn't exist
            os.makedirs(target_path, exist_ok=True)

            # Use rsync for efficient copying
            start_time = time.time()
            try:
                subprocess.run(["rsync", "-a", source_path, target_path], check=True)
                duration = (time.time() - start_time) / 60.0
                logger.info(f"Copy completed in {duration:.2f} minutes")

                # Create signal file to indicate completion
                with open(signal_file, "w") as f:
                    f.write(f"Copy completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to copy ImageNet dataset: {e}")
                return None
        else:
            logger.info(f"ImageNet dataset already exists at {target_path}")

            # Ensure signal file exists even if the directory was already there
            if not os.path.exists(signal_file):
                with open(signal_file, "w") as f:
                    f.write(f"Copy verified at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Non-master processes wait for the signal file
    else:
        logger.info(f"Process {local_rank} waiting for master to copy data...")
        while not os.path.exists(signal_file):
            time.sleep(30)  # Check every 30 seconds instead of 60
            logger.info(f"Process {local_rank}: Still waiting for copy to complete...")

        logger.info(f"Process {local_rank}: Master finished copying data")

    return target_path

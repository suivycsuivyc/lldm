# Source: https://github.com/webdataset/webdataset-imagenet/blob/main/imagenet.py

import os
from functools import partial

import webdataset as wds
import torch
from torchvision import transforms

from datasets import register


def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


center_crop = True
random_flip = False
transform_224 = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(224) if center_crop else transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
transform_256 = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(256) if center_crop else transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
transform_512 = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(512) if center_crop else transforms.RandomCrop(512),
    transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def transform_224_256(sample):
    return [transform_224(sample['jpg']), transform_256(sample['jpg'])]


def transform_224_512(sample):
    return [transform_224(sample['jpg']), transform_512(sample['jpg'])]


def pack(sample):
    return {
        'encoder_pixel_values': sample[0],
        'dm_pixel_values': sample[1],
    }


@register('webdataset_images')
def make_webdataset_images(root_path, resolution_encoder=224, resolution_dm=512, center_crop=True, random_flip=False,
                           batch_size=1, training=True, cache_dir=None, shardshuffle=1000, resampled=True, shuffle=5000):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    assert batch_size % world_size == 0
    batch_size //= world_size

    l = sorted([_ for _ in os.listdir(root_path) if _.endswith('.tar')])
    sid = os.path.splitext(l[0])[0]
    eid = os.path.splitext(l[-1])[0]
    urls = os.path.join(root_path, f'{{{sid}..{eid}}}.tar')

    assert resolution_encoder == 224
    assert resolution_dm in [256, 512]
    assert center_crop == True
    assert random_flip == False

    transform = transform_224_256 if resolution_dm == 256 else transform_224_512

    # repeat dataset infinitely for training mode
    dataset = (
        wds.WebDataset(
            urls,
            repeat=training,
            cache_dir=cache_dir,
            shardshuffle=shardshuffle if training else False,
            resampled=resampled if training else False,
            handler=wds.ignore_and_continue,
            nodesplitter=None if (training and resampled) else nodesplitter,
        )
        .shuffle(shuffle if training else 0)
        .decode('pil')
        .map(transform)
        .batched(batch_size, partial=False)
        .map(pack)
        # .to_tuple("jpg;png;jpeg cls", handler=wds.ignore_and_continue)
        # .map_tuple(transform)
    )

    # Attach vis_samples for visualize
    _dataset = wds.WebDataset(
        urls,
        resampled=resampled if training else False,
        handler=wds.ignore_and_continue,
        nodesplitter=None if (training and resampled) else nodesplitter,
    ).decode('pil').map(transform).map(pack)
    vis_samples = []
    it = iter(_dataset)
    for _ in range(32):
        try:
            vis_samples.append(next(it))
        except StopIteration:
            break
    dataset.vis_samples = vis_samples

    return dataset

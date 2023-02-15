from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import spiga.data.loaders.alignments as zoo_alignments

zoos = [zoo_alignments]


def get_dataset(data_config, pretreat=None, debug=False):

    for zoo in zoos:
        dataset = zoo.get_dataset(data_config, pretreat=pretreat, debug=debug)
        if dataset is not None:
            return dataset
    raise NotImplementedError('Dataset not available')


def get_dataloader(batch_size, data_config, pretreat=None, sampler_cfg=None, debug=False):

    dataset = get_dataset(data_config, pretreat=pretreat, debug=debug)

    if (len(dataset) % batch_size) == 1 and data_config.shuffle == True:
        drop_last_batch = True
    else:
        drop_last_batch = False

    shuffle = data_config.shuffle
    sampler = None
    if sampler_cfg is not None:
        sampler = DistributedSampler(dataset, num_replicas=sampler_cfg.world_size, rank=sampler_cfg.rank)
        shuffle = False

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=data_config.num_workers,
                            pin_memory=True,
                            drop_last=drop_last_batch,
                            sampler=sampler)

    return dataloader, dataset

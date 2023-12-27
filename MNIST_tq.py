import torch
from torchquantum.dataset import MNIST


def MNISTDataLoaders(args):
    dataset = MNIST(
        root='data',
        train_valid_split_ratio=args.train_valid_split_ratio,
        center_crop=args.center_crop,
        resize=args.resize,
        resize_mode='bilinear',
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=args.digits_of_interest,
        n_test_samples=None,
        n_valid_samples=None,
        fashion=False,
        n_train_samples=None
        )
    dataflow = dict()
    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
            batch_size = args.batch_size
        else:
            # for valid and test, use SequentialSampler to make the train.py
            # and eval.py results consistent
            sampler = torch.utils.data.SequentialSampler(dataset[split])
            batch_size = len(dataset[split])

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True)

    return dataflow['train'], dataflow['valid'], dataflow['test']

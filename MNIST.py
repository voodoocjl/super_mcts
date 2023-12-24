import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, image, binary_threshold=0.1):
        self.image = image
        self.binary_threshold = binary_threshold
    def __len__(self):
        return len(self.image)
    def __getitem__(self, index):
        image, label = self.image[index]
        # binarization processing
        # binary_image = (image > self.binary_threshold).float()

        # transform tensor (1, 28, 28) to (28, 28)
        # binary_image = binary_image.squeeze(0)
        # return binary_image, torch.zeros_like(binary_image), torch.zeros_like(binary_image), label
        return image, label


def MNISTDataLoaders(args):
    # set seed
    torch.manual_seed(42)

    # define data transformation
    tran = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if not args.center_crop == 28:
        tran.append(transforms.CenterCrop(args.center_crop))
    if not args.resize == 28:
        tran.append(transforms.Resize(args.resize, interpolation=args.resize_mode))
    transform = transforms.Compose(tran)

    # load MNIST 'train' dataset and split into train and valid sets
    train_valid = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
    idx, _ = torch.stack([train_valid.targets == number for number in args.digits_of_interest]).max(dim=0)
    train_valid.targets = train_valid.targets[idx]
    train_valid.data = train_valid.data[idx]
    train_len = int(args.train_valid_split_ratio[0] * len(train_valid))
    split = [train_len, len(train_valid) - train_len]
    train_set, valid_set = torch.utils.data.random_split(train_valid, split, generator=torch.Generator().manual_seed(1))

    # load MNIST 'test' dataset
    test_set = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)
    idx, _ = torch.stack([test_set.targets == number for number in args.digits_of_interest]).max(dim=0)
    test_set.targets = test_set.targets[idx]
    test_set.data = test_set.data[idx]

    # customize dataset
    custom_train_dataset = CustomDataset(train_set)     #23516
    custom_val_dataset = CustomDataset(valid_set)       #1238
    custom_test_dataset = CustomDataset(test_set)       #4157

    # customize Dataloader
    custom_train_loader = DataLoader(custom_train_dataset, batch_size=args.batch_size, shuffle=True)
    custom_val_loader = DataLoader(custom_val_dataset, batch_size=len(custom_val_dataset), shuffle=False)
    custom_test_loader = DataLoader(custom_test_dataset, batch_size=len(custom_test_dataset), shuffle=False)
    return custom_train_loader, custom_val_loader, custom_test_loader


# if __name__ == '__main__':
#     train_loader, val_loader, test_loader = MNISTDataLoaders()
#     for data_a, data_v, data_t, target in train_loader:
#         print(data_a.shape, data_v.shape, data_t.shape, target.shape)
#         print(data_a.dtype, data_v.dtype, data_t.dtype, target.dtype)
#         break

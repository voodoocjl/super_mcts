import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score


AUDIO = 'COVAREP'
VISUAL = 'FACET_4.2'
TEXT = 'glove_vectors'
TARGET = 'Opinion Segment Labels'


class CustomDataset(Dataset):
    def __init__(self, audio, visual, text, target):
        self.audio = audio
        self.visual = visual
        self.text = text
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        audio_val = self.audio[index]
        visual_val = self.visual[index]
        text_val = self.text[index]
        target = self.target[index]
        return visual_val, audio_val, text_val, target


def MOSIDataLoaders():
    with open('data/mosi', 'rb') as file:
        tensors = pickle.load(file)

    train_data = tensors[0]
    train_audio = torch.from_numpy(train_data[AUDIO]).float()
    train_visual = torch.from_numpy(train_data[VISUAL]).float()
    train_text = torch.from_numpy(train_data[TEXT]).float()
    train_target = torch.from_numpy(train_data[TARGET]).squeeze(dim=2)

    val_data = tensors[1]
    val_audio = torch.from_numpy(val_data[AUDIO]).float()
    val_visual = torch.from_numpy(val_data[VISUAL]).float()
    val_text = torch.from_numpy(val_data[TEXT]).float()
    val_target = torch.from_numpy(val_data[TARGET]).squeeze(dim=2)

    test_data = tensors[2]
    test_audio = torch.from_numpy(test_data[AUDIO]).float()
    test_visual = torch.from_numpy(test_data[VISUAL]).float()
    test_text = torch.from_numpy(test_data[TEXT]).float()
    test_target = torch.from_numpy(test_data[TARGET]).squeeze(dim=2)

    train = CustomDataset(train_audio, train_visual, train_text, train_target)
    val = CustomDataset(val_audio, val_visual, val_text, val_target)
    test = CustomDataset(test_audio, test_visual, test_text, test_target)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=673, shuffle=True, pin_memory=True)
    return train_loader, val_loader, test_loader

# Import the associated dataloader for affect datasets, which MOSI is a part of.
# from datasets.affect.get_data import get_dataloader

# Create the training, validation, and test-set dataloaders.
traindata, validdata, testdata = MOSIDataLoaders()

# Here, we'll import several common modules should you want to mess with this more.
from unimodals.common_models import GRU, MLP, Sequential, Identity

# As this example is meant to be simple and easy to train, we'll pass in identity
# functions for each of the modalities in MOSI:
encoders = [Identity(), Identity(), Identity()]

# Import a fusion paradigm, in this case early concatenation.
from fusions.common_fusions import ConcatEarly  # noqa

# Initialize the fusion module
fusion = ConcatEarly()

head = Sequential(GRU(409, 512, dropout=True, has_padding=False,
                  batch_first=True, last_only=True), MLP(512, 512, 1))

# Standard supervised learning training loop
# from training_structures.Supervised_Learning import test

# For more information regarding parameters for any system, feel free to check out the documentation
# at multibench.readthedocs.io!
# train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
#       is_packed=False, lr=1e-3, save='mosi_ef_r0.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_ef_r0.pt', map_location=torch.device('cpu'))
# test(model, testdata, 'affect', is_packed=False,
#      criterion=torch.nn.L1Loss(), task="regression", no_robust=True)

def evaluate(model, data_loader):
    model.eval()
    metrics = {}
    with torch.no_grad():
        data_v, data_a, data_t, target = next(iter(data_loader))
        output = model([data_v, data_a, data_t])
    output = output.squeeze().cpu().numpy()
    target = target.squeeze().numpy()
    metrics['mae'] = np.mean(np.absolute(output - target)).item()
    metrics['corr'] = np.corrcoef(output, target)[0][1].item()
    metrics['multi_acc'] = round(sum(np.round(output) == np.round(target)) / float(len(target)), 5).item()
    true_label = (target >= 0)
    pred_label = (output >= 0)
    metrics['bi_acc'] = accuracy_score(true_label, pred_label).item()
    metrics['f1'] = f1_score(true_label, pred_label, average='weighted').item()
    return metrics

print(evaluate(model, testdata))

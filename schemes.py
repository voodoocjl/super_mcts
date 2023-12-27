import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from MNIST import MNISTDataLoaders
from FusionModel import QNet
from FusionModel import translator
from Arguments import Arguments
import random


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    print("\nTest Accuracy: {}".format(metrics))
    # print("Test correlation: {}".format(metrics['corr']))
    
def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for images, targets in data_loader:
        images = images.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)        
        loss.backward()
        optimizer.step()

def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor()
    output_all = torch.Tensor()
    with torch.no_grad():
        for data_image, target in data_loader:
            data_image = data_image.to(args.device)
            target = target.to(args.device)
            output = model(data_image)
            instant_loss = criterion(output, target).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, target), dim=0)
            output_all = torch.cat((output_all, output), dim=0) 
    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return total_loss, accuracy

def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        for data_image, target in data_loader:
            data_image, target = data_image.to(args.device), target.to(args.device)
            output = model(data_image)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(target.view(-1, 1).expand_as(indices))
    size = target.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy    
    return metrics

def Scheme(design, weight=None, epochs=None):
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    args = Arguments()
    if epochs == None:
        epochs = args.epochs
    if torch.cuda.is_available() and args.device == 'cuda':
        print("using cuda device")
    else:
        print("using cpu device")
    train_loader, val_loader, test_loader = MNISTDataLoaders(args)
    model = QNet(args, design).to(args.device)
    if weight != 'init':
        if weight != None:
            model.load_state_dict(weight, strict= False)
        else:
            model.load_state_dict(torch.load('base_weight'))
    criterion = nn.NLLLoss()
   
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 0

    start = time.time()
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, args)
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = evaluate(model, val_loader, args)
        val_loss_list.append(val_loss)
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            print(epoch, train_loss, val_loss, 'saving model')
            best_model = copy.deepcopy(model)           
        else:
            print(epoch, train_loss, val_loss)
        # metrics = evaluate(model, test_loader, args)
        # display(metrics)
        # print(epoch, train_loss, val_loss)
    end = time.time()
    print("Running time: %s seconds" % (end - start))
    # best_model = model
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'mae': metrics}
    
    # torch.save(best_model.state_dict(), 'base_weight')
    return best_model, report


if __name__ == '__main__':
    change_code = None
    change_code = [1, 1, 3, 0, 1]  #0.717825355
    change_code = [[2, 3, 3, 3, 2], [0, 1, 1, 1, 2], [1, 2, 1, 2, 2], [3, 1, 0, 1, 3]]

    # design = translator(change_code)    
    # best_model, report = Scheme(design)

    design = translator(change_code, 'full')    
    best_model, report = Scheme(design, 'init', 20)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from MNIST_tq import MNISTDataLoaders
from FusionModel import QNet
from FusionModel import translator, prune_single,gen_arch
from Arguments import Arguments
import random


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest Accuracy: {}".format(metrics) + RESET)

    
def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for feed_dict in data_loader:
        images = feed_dict['image'].to(args.device)
        targets = feed_dict['digit'].to(args.device)    
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
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images)
            instant_loss = criterion(output, targets).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, targets), dim=0)
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
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy    
    return metrics

def Scheme(design, weight='base', epochs=None):
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
        if weight != 'base':
            model.load_state_dict(weight, strict= False)
        else:            
            # model.load_state_dict(torch.load('weights/mnist_4_layers_reuploading'))
            model.load_state_dict(torch.load('weights/mnist_best_1'))
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
    
    # torch.save(best_model.state_dict(), 'base_weight_bad')
    return best_model, report


if __name__ == '__main__':
    change_code = None
    # change_code = [1, -1, -3, 2, 1]  #0.717825355
    # change_code = [[3, 0, 0, 0, 0, 0, 1, 0, 1], [4, 0, 0, 0, 0, 0, 0, 1, 0]]
    change_code = [[3, 0, 0, 0, 0, 0, 1, 0, 1], [4, 0, 0, 0, 0, 0, 0, 1, 0]]

    
    # import pickle
    # with open('search_space_mnist_single', 'rb') as file:
    #     search_space = pickle.load(file)
    
    # change_code = random.sample(search_space, 10)

    # for i in range(10):
    #     print(change_code[i])
    #     design = translator([change_code[i]])
    #     best_model, report = Scheme(design, 'base', 10)
    
    design = translator(change_code, 'full')
    best_model, report = Scheme(design, 'base', 1)


    # torch.save(best_model.state_dict(), 'weights/mnist_best_1')
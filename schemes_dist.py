import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MOSIDataLoaders
from FusionModel import QNet
from FusionModel import translator
from Arguments import Arguments
import random
import pickle
import csv, os
import torch.multiprocessing as mp

def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    print("\nTest mae: {}".format(metrics['mae']))    


def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for data_a, data_v, data_t, target in data_loader:
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()
        output = model(data_a, data_v, data_t)
        loss = criterion(output, target)
        # loss = output[1]
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_a, data_v, data_t, target in data_loader:
            data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
            target = target.to(args.device)
            output = model(data_a, data_v, data_t)
            instant_loss = criterion(output, target).item()
            total_loss += instant_loss
    total_loss /= len(data_loader.dataset)
    return total_loss


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    with torch.no_grad():
        data_a, data_v, data_t, target = next(iter(data_loader))
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        output = model(data_a, data_v, data_t)
    output = output.cpu().numpy()
    target = target.numpy()
    metrics['mae'] = np.mean(np.absolute(output - target)).item()
    metrics['corr'] = np.corrcoef(output, target)[0][1].item()
    metrics['multi_acc'] = round(sum(np.round(output) == np.round(target)) / float(len(target)), 5).item()
    true_label = (target >= 0)
    pred_label = (output >= 0)
    metrics['bi_acc'] = accuracy_score(true_label, pred_label).item()
    metrics['f1'] = f1_score(true_label, pred_label, average='weighted').item()    
    return metrics


def Scheme(design):
    
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    args = Arguments()

    # MOSI
    train_loader, val_loader, test_loader = MOSIDataLoaders(args)   
    model = QNet(args, design).to(args.device)
    # model.load_state_dict(torch.load('classical_weight'), strict= False)
    model.load_state_dict(torch.load('base_weight_tq'), strict= False)    

    # # MOSEI
    # train_loader, val_loader, test_loader = MOSEIDataLoaders(args)         
    # model = QNet_mosei(args, design).to(args.device)
    # model.load_state_dict(torch.load('classical_weight_MOSEI'), strict= False)
    
    criterion = nn.L1Loss(reduction='sum')
    # optimizer = optim.Adam([
    #     {'params': model.ClassicalLayer_a.parameters()},
    #     {'params': model.ClassicalLayer_v.parameters()},
    #     {'params': model.ClassicalLayer_t.parameters()},
    #     {'params': model.ProjLayer_a.parameters()},
    #     {'params': model.ProjLayer_v.parameters()},
    #     {'params': model.ProjLayer_t.parameters()},
    #     {'params': model.QuantumLayer.parameters(), 'lr': args.qlr},
    #     {'params': model.Regressor.parameters()}
    #     ], lr=args.clr)
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 10000

    start = time.time()
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, args)
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = test(model, val_loader, criterion, args)
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # print(epoch, train_loss, val_loss, 'saving model')
            best_model = copy.deepcopy(model)
        # else:
        #     print(epoch, train_loss, val_loss)
    end = time.time()
    print("Running time: %s seconds" % (end - start))
    
    # best_model = copy.deepcopy(model)
    # best_val_loss = val_loss
    metrics = evaluate(best_model, test_loader, args)
    # display(metrics)
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'metrics': metrics}
    return best_model, report

def search(train_space, index, size):
    filename = 'train_results_{}.csv'.format(index)
    if os.path.isfile(filename) == False:
        with open(filename, 'w+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow(['Num', 'sample_id', 'arch_code', 'val_loss', 'test_mae', 'test_corr',
                                'test_multi_acc', 'test_bi_acc', 'test_f1'])
   
    csv_reader = csv.reader(open(filename))
    i = len(list(csv_reader)) - 1
    j = index * size + i 
    base_code = [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0]
    
    while len(train_space) > 0:
        net = train_space[i]
        print('Net', j, ":", net)
        design = translator(net, base_code)
        best_model, report = Scheme(design)
        with open(filename, 'a+', newline='') as res:
            writer = csv.writer(res)
            best_val_loss = report['best_val_loss']
            metrics = report['metrics']
            writer.writerow([i, j, net, best_val_loss, metrics['mae'], metrics['corr'],
                                metrics['multi_acc'], metrics['bi_acc'], metrics['f1']])        
        j += 1
        i += 1


if __name__ == '__main__':
    
    # base_code = [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0]
    # # change_code = None
    # change_code = [1, 3, 2, 4, 1, 0]
    # design = translator(base_code, change_code)    
    # best_model, report = Scheme(design)

    train_space = []
    filename = 'data/train_space_1'

    with open(filename, 'rb') as file:
        train_space = pickle.load(file)
    
    num_processes = 2
    size = int(len(train_space) / num_processes)
    space = []
    for i in range(num_processes):
        space.append(train_space[i*size : (i+1)*size])
    args = Arguments()
    # if torch.cuda.is_available() and args.device == 'cuda':
    #     print("using cuda device")
    # else:
    #     print("using cpu device")
    with mp.Pool(processes = num_processes) as pool:        
        pool.starmap(search, [(space[i], i, size) for i in range(num_processes)])
    
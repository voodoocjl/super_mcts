import time
import random
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from FusionModel import gen_arch
from Network import FCN, ACN, RNN, Attention
from Classifier import get_label


# torch.cuda.is_available = lambda : False

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

with open('data/mosi_dataset', 'rb') as file:
    dataset = pickle.load(file)

def normalize(x):
    x = (x - torch.mean(x).unsqueeze(-1)) / torch.std(x).unsqueeze(-1)
    return x

def transform_2d(x, repeat):
    # x = x.reshape(-1, 5, 7)
    x = x.reshape(-1, repeat + 1, 8)
    return x
    

# base_code = [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0]
base_code = [5, 1, 2, 3, 4, 5, 6, 0]
arch_code, mae = [], []
repeat = 2
for key in dataset:
    # arch_code.append(gen_arch(eval(key), base_code))
    arch_code.append(base_code + ([0, 0] + eval(key)) * repeat)
    mae.append(dataset[key])
arch_code = torch.from_numpy(np.asarray(arch_code, dtype=np.float32))
arch_code = normalize(arch_code)
mae =  torch.from_numpy(np.asarray(mae, dtype=np.float32))

arch_code = transform_2d(arch_code, repeat)
# model = FCN(8, 2)
# model = ACN(32, pooling_size=(3, 3), output_size=2)
model = RNN(8, 16, 2)

Epoch = 3001
true_label = get_label(mae)
t_size = 5000
device = 'cpu'

def data(arch_code_t):
    arch_code_train = arch_code_t[:t_size].to(device)
    mae_train = mae[:t_size].to(device)
    label = get_label(mae_train).to(device)
    p_label = 2 * label - 1
    
    dataset = TensorDataset(arch_code_train, mae_train, p_label)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    arch_code_test = arch_code_t[t_size:].to(device)
    mae_test = mae[t_size:].to(device)
    test_label = true_label[t_size:].to(device)
    return dataloader, arch_code_train, arch_code_test, mae_test, label, test_label

# mean = mae_test.mean()
# good, bad, good_num, bad_num = 0, 0, 0, 0
# for i in mae_test:
#     if i < mean:
#         good += i
#         good_num += 1
#     else:
#         bad += i
#         bad_num += 1

# print("Ground truth:", good / good_num, bad / bad_num)

def train(model):
    acc_list =  []
    if torch.cuda.is_available():
        model.cuda()    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
    for epoch in range(1, Epoch):
        for x, y, z in dataloader:
            model.train()
            pred = model(x) 
            loss_s = loss_fn(pred[:, 0], y)  
            loss_e = loss_fn(pred[:, -1], z)             
            train_loss = loss_e + loss_s            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()        

        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(arch_code_train).cpu()
                # error = (pred[:,-1] - mae_train).abs().mean()
                # pred_label = get_label(pred[:, -1])
                pred_label = (pred[:, -1] > 0).float()                
                acc = accuracy_score(label.cpu().numpy(), pred_label.numpy())
                # acc = f1_score(label, pred_label)
                print(epoch, acc)
                acc_list.append(acc)
    return acc_list

def test(model):
    model.eval()
    with torch.no_grad():
        pred = model(arch_code_test).cpu()
        pred_label = (pred[:, -1] > 0).float()        
        acc = accuracy_score(test_label.cpu().numpy(), pred_label.numpy())
        # acc = f1_score(test_label, pred_label)
        good_num = pred_label.sum()
        bad_num = pred_label.shape[0] - good_num
        mae_good = (mae_test.cpu() * pred_label).sum() / good_num
        mae_bad =  (mae_test.cpu() * (1 - pred_label)).sum() / bad_num
        print("test acc:", acc, mae_good, mae_bad)
        acc_list.append(acc)
    print(acc_list)
    
    return pred_label

dataloader, arch_code_train, arch_code_test, mae_test, label, test_label = data(arch_code)
print("dataset size:", len(arch_code))
print("training size:", len(arch_code_train))
print("test size:", len(arch_code_test))

s = time.time()
acc_list = train(model)
e = time.time()
print('time: ', e-s)
pred_label = test(model)

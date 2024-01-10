import pickle
import os
import csv


with open('search_space_mnist_single', 'rb') as file:
    search_space_single = pickle.load(file)

with open('search_space_mnist_enta', 'rb') as file:
    search_space_enta = pickle.load(file)
    
arch_code_len = len(search_space_single[0])

# print("total architectures:", len(search_space))

with open('data/mnist_dataset_single', 'rb') as file:
    dataset = pickle.load(file)
    
dataset = {}

if os.path.isfile('results.csv') == False:
    with open('results.csv', 'w+', newline='') as res:
        writer = csv.writer(res)
        writer.writerow(['sample_id', 'arch_code', 'sample_node', 'ACC', 'p_ACC'])

if os.path.isfile('results_30_epoch.csv') == False:
    with open('results_30_epoch.csv', 'w+', newline='') as res:
        writer = csv.writer(res)
        writer.writerow(['iteration', 'arch_code', 'ACC'])

state_path = 'states'
if os.path.exists(state_path) == False:
    os.makedirs(state_path)
files = os.listdir(state_path)

def empty_arch(n_layers, n_qubits):            
    single = [[i] + [0]* (2*n_layers) for i in range(1,n_qubits+1)]
    enta = [[i] + [i]*n_layers for i in range(1,n_qubits+1)]
    return [single, enta]




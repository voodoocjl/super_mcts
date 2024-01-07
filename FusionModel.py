import copy
import pennylane as qml
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from torchquantum.encoding import encoder_op_list_name_dict
from Arguments import Arguments
import numpy as np

args = Arguments()

def gen_arch(change_code, base_code=args.base_code):
    arch_code = base_code[1:] * base_code[0]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        # for i in range(len(change_code)):
        #     q = change_code[i][0]  # the qubit changed
        #     for id, t in enumerate(change_code[i][1:]):
        #         arch_code[q - 1 + id * args.n_qubits] = t
    return arch_code

def prune_single(net):
    net_arr = np.array(net).reshape((args.n_layers, args.n_qubits))
    tmp = np.abs(net_arr) - [i for i in range(1, 1+args.n_qubits)]        # posiitons of all discarded CU3
    index = np.where(tmp == 0)
    net_arr[index] = 0
    index = np.where(net_arr < 0)
    net_arr[index] = index[1] + 1
    single = np.ones((args.n_layers, args.n_qubits)).astype(int)

    for i in range(args.n_layers):
        unique, counts = np.unique(net_arr[i], return_counts=True)
        redundant = unique[counts > 1]   # remove 0
        redundant[redundant != 0]
        single[i][redundant-1] = 0

    return single

def translator(change_code, trainable='partial', base_code=args.base_code):
    net = gen_arch(change_code)
    updated_design = {}
    updated_design['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:,0] - 1
        change_code = change_code.reshape(-1, length)    
        updated_design['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:            
            updated_design['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(2, -1)
            j += 1    

    if trainable == 'full' or change_code is None:
        updated_design['change_qubit'] = None
    else:
        if type(change_code[0]) != type([]): change_code = [change_code]
        updated_design['change_qubit'] = change_qbit[-1]

    # num of layers
    updated_design['n_layers'] = args.n_layers

    for layer in range(updated_design['n_layers']):
    # categories of single-qubit parametric gates
        for i in range(args.n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(args.n_qubits):
            if net[j + layer * args.n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * args.n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * args.n_qubits])-1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * args.n_qubits * 2
    return updated_design

class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict['4x4_ryzxy'])
        self.uploading = [tq.GeneralEncoder(encoder_op_list_name_dict['{}x4_ryzxy'.format(i)]) for i in range(4)]

        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        self.q_params_rot, self.q_params_enta = [], []
        for i in range(self.args.n_qubits):
            self.q_params_rot.append(pi * torch.rand(self.design['n_layers'], 3)) # each U3 gate needs 3 parameters
            self.q_params_enta.append(pi * torch.rand(self.design['n_layers'], 3)) # each CU3 gate needs 3 parameters

        for layer in range(self.design['n_layers']):
            for q in range(self.n_wires):
                # 'trainable' option
                if self.design['change_qubit'] is None:
                    rot_trainable = True
                    enta_trainable = True
                elif q == self.design['change_qubit']:
                    rot_trainable = True
                    enta_trainable = True
                else:
                    rot_trainable = False
                    enta_trainable = False
                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'U3':
                    self.rots.append(tq.U3(has_params=True, trainable=rot_trainable,
                                           init_params=self.q_params_rot[q][layer]))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                    self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                                             init_params=self.q_params_enta[q][layer]))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)  # 'down_sample_kernel_size' = 6
        # x = x.view(bsz, -1)
        x = x.view(bsz, 4, 4).transpose(1,2)

        # tmp = x[:, 4:8].clone()
        # x[:, 4:8] = x[:, 8:12]
        # x[:, 8:12] = tmp

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        # encode input image with '4x4_ryzxy' gates
        # for j in range(self.n_wires):
        #     self.uploading[j](qdev, x[:,j])

        for layer in range(self.design['n_layers']):            
            for j in range(self.n_wires):
                if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][0][layer] == 0):
                    self.uploading[j](qdev, x[:,j])
                if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][1][layer] == 0):
                    self.rots[j + layer * self.n_wires](qdev, wires=j)            
                # self.uploading[j](qdev, x[:,j])
                # self.rots[j + layer * self.n_wires](qdev, wires=j)
                
            for j in range(self.n_wires):
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        return self.measure(qdev)


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.QuantumLayer = TQLayer(self.args, self.design)

    def forward(self, x_image):
        exp_val = self.QuantumLayer(x_image)
        output = F.log_softmax(exp_val, dim=1)        
        return output

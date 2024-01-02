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
                
        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for i, t in enumerate(change_code[i][1:]):
                arch_code[q + i * args.n_qubits] = t
    return arch_code

def prune_single(net):
    net_arr = np.array(net).reshape((args.n_layers, args.n_qubits))
    tmp = net_arr - [i for i in range(args.n_qubits)]        # posiitons of all discarded CU3
    index = np.where(tmp == 0)
    net_arr[index] = -1
    single = np.ones((args.n_qubits, args.n_qubits)).astype(int)
    
    for i in range(args.n_layers):
        unique, counts = np.unique(net_arr[i], return_counts=True)
        redundant = unique[counts > 1]   # remove -1
        redundant[redundant != -1]        
        single[i][redundant] = 0

    return single

def translator(change_code, trainable='partial', base_code=args.base_code):    
    net = gen_arch(change_code, base_code)
    net_arr = np.array(net).reshape((args.n_layers, args.n_qubits))
    tmp = net_arr - [i for i in range(4)]        # posiitons of all discarded CU3
    index = np.where(tmp == 0)
    net_arr[index] = -1
    updated_design = {}
    
    for i in range(args.n_layers):
        unique, counts = np.unique(net_arr[i], return_counts=True)
        redundant = unique[counts > 1]   # remove -1
        redundant[redundant != -1]        
        updated_design['layer '+ str(i)] = list(redundant)

    if trainable == 'full' or change_code is None:
        updated_design['change_qubit'] = None
    else:
        if type(change_code[0]) != type([]): change_code = [change_code]
        updated_design['change_qubit'] = change_code[-1][0]

    # num of layers
    updated_design['n_layers'] = args.n_layers

    for layer in range(updated_design['n_layers']):
    # categories of single-qubit parametric gates        
        for i in range(args.n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(args.n_qubits):
            updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * args.n_qubits]])

        updated_design['total_gates'] = updated_design['n_layers'] * args.n_qubits
    return updated_design

class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict['4x4_ryzxy'])

        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        self.q_params_rot, self.q_params_enta = [], []
        for i in range(self.args.n_qubits):
            self.q_params_rot.append(pi * torch.rand(3 * self.design['n_layers'])) # each U3 gate needs 3 parameters
            self.q_params_enta.append(pi * torch.rand(3 * self.design['n_layers'])) # each CU3 gate needs 3 parameters

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
                                           init_params=self.q_params_enta[q][layer*3:(layer+1)*3].reshape((3,))))                
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                    self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                                             init_params=self.q_params_enta[q][layer*3:(layer+1)*3].reshape((3,))))                
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)  # 'down_sample_kernel_size' = 6
        x = x.view(bsz, -1)

        tmp = x[:, 4:8].clone()
        x[:, 4:8] = x[:, 8:12]
        x[:, 8:12] = tmp

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        # encode input image with '4x4_ryzxy' gates        
        self.encoder(qdev, x)

        for layer in range(self.design['n_layers']):
            for j in range(self.n_wires):
                if j not in self.design['layer '+str(layer)]:
                    self.rots[j + layer * self.n_wires](qdev, wires=j)
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

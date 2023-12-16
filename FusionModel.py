import copy
import pennylane as qml
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
from Arguments import Arguments
args = Arguments()


def gen_arch( change_code, base_code = args.base_code):    
    arch_code = base_code[1:] * base_code[0]
    if type(change_code[0]) != type([]):
        change_code = [change_code]
    change_qubit = change_code[-1][0]
    if change_code is not None:
        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for i, t in enumerate(change_code[i][1:]):
                arch_code[q + i * args.n_qubits] = t
    return arch_code


def translator(change_code, base_code = args.base_code):
    if type(change_code[0]) != type([]):
        change_code = [change_code]
    net = gen_arch(change_code, base_code)    
    updated_design = {}
    if change_code is None:
        updated_design['change_qubit'] = None
    else:
        updated_design['change_qubit'] = change_code[-1][0]

    # num of layers
    updated_design['n_layers'] = args.n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(args.n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'Rx'
    
        # categories and positions of entangled gates
        for j in range(args.n_qubits):
            updated_design['enta' + str(layer) + str(j)] = ('IsingZZ', [j, net[j + layer * args.n_qubits]])
    
        updated_design['total_gates'] = updated_design['n_layers'] * args.n_qubits
    return updated_design

dev = qml.device("lightning.qubit", wires=args.n_qubits)
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(q_input_features, q_weights_rot, q_weights_enta, **kwargs):
    current_design = kwargs['design']
    q_input_features = torch.transpose(q_input_features, 0, 1)  #(n_qubits, batches)
    q_input_features = q_input_features.reshape(args.n_qubits, 3, -1)  # (7, 3, 32)
     
    for layer in range(current_design['n_layers']):
        # data reuploading
        for i in range(args.n_qubits):
            qml.Rot(*q_input_features[i], wires=i)
        # single-qubit parametric gates and entangled gates
        for j in range(args.n_qubits):
            if current_design['rot' + str(layer) + str(j)] == 'Rx':
                qml.RX(q_weights_rot[j][layer], wires=j)
            else:
                qml.RY(q_weights_rot[j][layer], wires=j)

            if current_design['enta' + str(layer) + str(j)][1][0] != current_design['enta' + str(layer) + str(j)][1][1]:
                if current_design['enta' + str(layer) + str(j)][0] == 'IsingXX':
                    qml.IsingXX(q_weights_enta[j][layer], wires=current_design['enta' + str(layer) + str(j)][1])
                else:
                    qml.IsingZZ(q_weights_enta[j][layer], wires=current_design['enta' + str(layer) + str(j)][1])

    return [qml.expval(qml.PauliZ(i)) for i in range(args.n_qubits)]


class QuantumLayer(nn.Module):
    def __init__(self, arguments, design):
        super(QuantumLayer, self).__init__()
        self.args = arguments
        self.design = design
        self.q_params_rot, self.q_params_enta = nn.ParameterList(), nn.ParameterList()
        for i in range(self.args.n_qubits):
            if self.design['change_qubit'] is None:
                rot_trainable = True
                enta_trainable = True
            elif i == self.design['change_qubit']:
                rot_trainable = False
                enta_trainable = True
            else:
                rot_trainable = False
                enta_trainable = False
            self.q_params_rot.append(nn.Parameter(pi * torch.rand(self.design['n_layers']), requires_grad=rot_trainable))
            self.q_params_enta.append(nn.Parameter(pi * torch.rand(self.design['n_layers']), requires_grad=enta_trainable))

    def forward(self, input_features):        
        output = quantum_net(input_features, self.q_params_rot, self.q_params_enta, design=self.design)
        q_out = torch.stack([output[i] for i in range(len(output))]).float()        # (n_qubits, batch)
        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(1)
        q_out = torch.transpose(q_out, 0, 1)    #(batch, n_qubits)
        return q_out    


class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        for layer in range(self.design['n_layers']):
            for q in range(self.n_wires):
                # 'trainable' option
                if self.design['change_qubit'] is None:
                    rot_trainable = True
                    enta_trainable = True
                elif q == self.design['change_qubit']:
                    rot_trainable = False
                    enta_trainable = True
                else:
                    rot_trainable = False
                    enta_trainable = False
                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'Rx':
                    self.rots.append(tq.RX(has_params=True, trainable=rot_trainable))
                else:
                    self.rots.append(tq.RY(has_params=True, trainable=rot_trainable))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'IsingXX':
                    self.entas.append(tq.RXX(has_params=True, trainable=enta_trainable))
                else:
                    self.entas.append(tq.RZZ(has_params=True, trainable=enta_trainable))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.reshape(bsz, self.n_wires, 3)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        for layer in range(self.design['n_layers']):
            for i in range(self.n_wires):
                tqf.rot(qdev, wires=i, params=x[:, i])
            for j in range(self.n_wires):
                self.rots[j + layer * self.n_wires](qdev, wires=j)
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        return self.measure(qdev)


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.ClassicalLayer_a = nn.RNN(self.args.a_insize, self.args.a_hidsize)
        self.ClassicalLayer_v = nn.RNN(self.args.v_insize, self.args.v_hidsize)
        self.ClassicalLayer_t = nn.RNN(self.args.t_insize, self.args.t_hidsize)
        self.ProjLayer_a = nn.Linear(self.args.a_hidsize, self.args.a_hidsize)
        self.ProjLayer_v = nn.Linear(self.args.v_hidsize, self.args.v_hidsize)
        self.ProjLayer_t = nn.Linear(self.args.t_hidsize, self.args.t_hidsize)
        if args.backend == 'pennylane':
            self.QuantumLayer = QuantumLayer(self.args, self.design)
        else:
            self.QuantumLayer = TQLayer(self.args, self.design)
        self.Regressor = nn.Linear(self.args.n_qubits, 1)
        for name, param in self.named_parameters():
            if "QuantumLayer" not in name:
                param.requires_grad = False

    def forward(self, x_a, x_v, x_t):
        x_a = torch.permute(x_a, (1, 0, 2))
        x_v = torch.permute(x_v, (1, 0, 2))
        x_t = torch.permute(x_t, (1, 0, 2))
        a_h = self.ClassicalLayer_a(x_a)[0][-1]
        v_h = self.ClassicalLayer_v(x_v)[0][-1]
        t_h = self.ClassicalLayer_t(x_t)[0][-1]
        a_o = torch.relu(self.ProjLayer_a(a_h))
        v_o = torch.sigmoid(self.ProjLayer_v(v_h)) * pi
        t_o = torch.sigmoid(self.ProjLayer_t(t_h)) * pi
        x_p = torch.cat((a_o, v_o, t_o), 1)
        exp_val = self.QuantumLayer(x_p)
        output = torch.tanh(self.Regressor(exp_val).squeeze(dim=1)) * 3
        return output

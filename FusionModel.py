import copy
import pennylane as qml
import torch
import torch.nn as nn
from math import pi
from Arguments import Arguments
args = Arguments()


def gen_arch(base_code, change_code):
    arch_code = copy.copy(base_code)
    if change_code is not None:
        q = change_code[0]  # the qubit changed
        for i, t in enumerate(change_code[1:]):
            arch_code[q + i * args.n_qubits] = t
    return arch_code


def translator(base_code, change_code = None):
    assert type(base_code) == type([])
    net = gen_arch(base_code, change_code)
    
    updated_design = {}
    if change_code is None:
        updated_design['change_qubit'] = None
    else:
        updated_design['change_qubit'] = change_code[0]

    # num of layers
    updated_design['n_layers'] = 5

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(args.n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'Rx'
    
        # categories and positions of entangled gates
        for j in range(args.n_qubits):
            updated_design['enta' + str(layer) + str(j)] = ('IsingZZ', [j, net[j + layer * args.n_qubits]])
    
        updated_design['total_gates'] = updated_design['n_layers'] * args.n_qubits
    return updated_design


qml.disable_return()
dev = qml.device("lightning.qubit", wires=args.n_qubits)
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(q_input_features_flat, q_weights_rot, q_weights_enta, **kwargs):
    current_design = kwargs['design']
    q_input_features = q_input_features_flat.reshape(args.n_qubits, 3)
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
        self.q_params_rot = nn.Parameter(pi * torch.rand(self.args.n_qubits, self.design['n_layers']))
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.args.n_qubits, self.design['n_layers']))

    def forward(self, input_features):
        q_out = torch.Tensor(0, self.args.n_qubits)
        q_out = q_out.to(self.args.device)
        if self.design['change_qubit'] is None:
            q_params_rot = self.q_params_rot
            q_params_enta = self.q_params_enta
        else:
            q_params_rot = torch.zeros_like(self.q_params_rot)
            q_params_enta = torch.zeros_like(self.q_params_enta)            
            for i in range(self.args.n_qubits):
                if i != self.design['change_qubit']:
                    q_params_rot[i] = self.q_params_rot[i].detach()
                    q_params_enta[i] = self.q_params_enta[i].detach()
                else:
                    q_params_rot[i] = self.q_params_rot[i]
                    q_params_enta[i] = self.q_params_enta[i]        
            
        for elem in input_features:
            q_out_elem = quantum_net(elem, q_params_rot, q_params_enta, design=self.design).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return q_out
    
    # qml.drawer.use_style('black_white')
    # fig, ax = qml.draw_mpl(quantum_net)(elem, self.q_params_rot, self.q_params_enta, design=self.design)


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
        self.QuantumLayer = QuantumLayer(self.args, self.design)
        self.Regressor = nn.Linear(self.args.n_qubits, 1)
        # for name, param in self.named_parameters():
        #     if "QuantumLayer" not in name:
        #         param.requires_grad = False

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

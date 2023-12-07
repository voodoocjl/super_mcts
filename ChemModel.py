import pennylane as qml
from Arguments import Arguments
args = Arguments()


def translator(net):
    assert type(net) == type([])
    updated_design = {}

    # r = net[0]
    q = net[0:6]
    # c = net[8:15]
    p = net[6:12]

    # num of layer repetitions
    layer_repe = [5, 7]
    updated_design['layer_repe'] = layer_repe[0]

    # categories of single-qubit parametric gates
    for i in range(args.n_qubits):
        if q[i] == 0:
            category = 'Rx'
        else:
            category = 'Ry'
        updated_design['rot' + str(i)] = category

    # categories and positions of entangled gates
    for j in range(args.n_qubits):
        # if c[j] == 0:
        #     category = 'IsingXX'
        # else:
        #     category = 'IsingZZ'
        updated_design['enta' + str(j)] = ([j, p[j]])

    updated_design['total_gates'] = len(q) + len(p)
    return updated_design


def quantum_net(q_params, design):
    current_design = design
    q_weights = q_params.reshape(current_design['layer_repe'], args.n_qubits, 2)
    for layer in range(current_design['layer_repe']):
        for j in range(args.n_qubits):
            if current_design['rot' + str(j)] == 'Rx':
                qml.RX(q_weights[layer][j][0], wires=j)
            else:
                qml.RY(q_weights[layer][j][0], wires=j)

            qml.IsingZZ(q_weights[layer][j][1], wires=current_design['enta' + str(j)])

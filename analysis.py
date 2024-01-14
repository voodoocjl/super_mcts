from MCTS import MCTS, count_gates
import pickle
import numpy as np

saved_file = 'states/mcts_agent_735'
with open(saved_file, 'rb') as json_data:
    agent = pickle.load(json_data)

def analysis_result(samples, ranks):
    gate_stat = []    
    sorted_changes = [k for k, v in sorted(samples.items(), key=lambda x: x[1], reverse=True)]
    for i in range(ranks):
        _, gates = count_gates(eval(sorted_changes[i]))
        gate_stat.append(list(gates.values()))
    
    return np.array(gate_stat), sorted_changes

samples = agent.samples
rank = 50
gates, sorted = analysis_result(samples, rank)

def find_arch(condition, number):
    data = gates[:, 0]
    rot = gates[:, 1]
    enta = gates[:, 2]
    if condition == 'enta':
        index_enta = np.where(enta < number)[0]
    elif condition == 'single':
        index_enta = np.where(rot < number)[0]
    else:
        index_enta = np.where(data < number)[0]   
    for i in index_enta:
        print(samples[sorted[i]])

find_arch('enta', 10)
mean = np.mean(gates, axis=0)
print(gates)
from MCTS import MCTS, count_gates
import pickle
import numpy as np

# saved_file = 'saved_states/42_0.006'
# saved_file = 'saved_states/42_0.003'
# saved_file = 'saved_states/42_0'

# saved_file = 'saved_states/super_0.006'
# saved_file = 'saved_states/super_0.003'
saved_file = 'saved_states/super_0'

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
# samples_true = agent.samples_true
samples_true = agent.samples
rank = 10
gates, sorted = analysis_result(samples_true, rank)
# gates, sorted = analysis_result(samples, rank)


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
        print('acc:{} arch: {}'.format(samples_true[sorted[i]], gates[i]))
        # print('acc:{} arch: {}'.format(samples_true[sorted[i]], gates[i]))
        
find_arch('enta', 16)
# find_arch('single', 11)
mean = np.mean(gates, axis=0)
# print(gates)
print(mean)
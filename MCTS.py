import pickle
import os
import random
import json
import csv
import numpy as np
import torch
from Node import Node
from FusionModel import translator
from schemes import Scheme
import time
from sampling import sampling_node

class MCTS:
    def __init__(self, search_space, tree_height, arch_code_len):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        assert type(search_space[0]) == type([])

        self.search_space   = search_space        
        self.ARCH_CODE_LEN  = arch_code_len
        self.ROOT           = None
        self.Cp             = 0.2
        self.nodes          = []
        self.samples        = {}
        self.samples_latest = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.mae_list    = []
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_MAEINV     = 0
        self.MAX_SAMPNUM    = 0
        self.sample_nodes   = []
        self.stages         = 0
        self.sampling_num   = 0               

        self.tree_height    = tree_height

        # initialize a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 1:
                is_good_kid = True

            parent_id = i // 2 - 1
            if parent_id == -1:
                self.nodes.append(Node(None, is_good_kid, self.ARCH_CODE_LEN, True))
            else:
                self.nodes.append(Node(self.nodes[parent_id], is_good_kid, self.ARCH_CODE_LEN, False))

        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        self.weight = 'base'
        self.explorations = {'rate': 0.006, 'rate_decay': [0.006, 0.004, 0.002, 0]}        
        self.topology = [([i] + [(i+1)%4]* 5 ) for i in range(4)]   # 5 layers!  4 is number of qubits!
        self.init_train()


    def init_train(self):
        for i in range(0, 50):
            net = random.choice(self.search_space)
            self.search_space.remove(net)
            self.TASK_QUEUE.append(net)
            self.sample_nodes.append('random')

        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for initializing MCTS")

    def re_init_tree(self, mode=None):
        with open('search_space_4_layers_pm', 'rb') as file:
            self.search_space = pickle.load(file)   
        self.TASK_QUEUE = []
        self.sample_nodes = []
        # self.explorations['rate'] -= 0.005
        self.stages += 1
        for i in self.nodes:
            i.x_bar = float("inf")
        epochs = 30
        strategy = 'base'
        sorted_changes = [k for k, v in sorted(self.samples_latest.items(), key=lambda x: x[1], reverse=True)]
        sorted_changes = [change for change in sorted_changes if len(eval(change)) == self.stages]

        # pick best 3 and randomly choose one
        random.seed(self.ITERATION)
        
        best_changes = [eval(sorted_changes[i]) for i in range(3)]
        best_change = random.choice(best_changes)
        print('Current Change: ', best_change)
        design = translator(best_change, 'full')
        best_model, report = Scheme(design, strategy, epochs)
        self.samples_latest = {}

        with open('results_30_epoch.csv', 'a+', newline='') as res:
            writer = csv.writer(res)
            metrics = report['mae']
            writer.writerow([self.ITERATION, best_change, metrics])
        
        if mode is None:
            self.ROOT.base_code = best_change
        else:
            self.ROOT.base_code = best_change[1:]
            self.stages -= 1
            # self.reset_node_data()        # attention!
            # self.sampling_num += len(self.samples)
            # self.samples = {}
            design = translator(self.ROOT.base_code, 'full')
            best_model, report = Scheme(design, strategy, 5)
            with open('results_30_epoch.csv', 'a+', newline='') as res:
                writer = csv.writer(res)
                metrics = report['mae']
                writer.writerow([self.ITERATION, self.ROOT.base_code, metrics])

        qubits = [code[0] for code in self.ROOT.base_code]        
        self.weight = best_model.state_dict()

        random.seed(self.ITERATION)

        for i in range(0, 30):
            net = random.choice(self.search_space)
            while net[0] in qubits:
                net = random.choice(self.search_space)
            self.search_space.remove(net)       
            self.TASK_QUEUE.append(net)
            self.sample_nodes.append('random')

        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for re-initializing MCTS {}".format(self.ROOT.base_code))    

    def dump_all_states(self, num_samples):
        node_path = 'states/mcts_agent'
        with open(node_path+'_'+str(num_samples), 'wb') as outfile:
            pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()


    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag(json.loads(k), v)    


    def populate_prediction_data(self):
        # self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag(k, 0.0)


    def train_nodes(self):
        for i in self.nodes:
            i.train()


    def predict_nodes(self, method = None, dataset =None):
        for i in self.nodes:
            if dataset:
                i.predict_validation()
            else:
                i.predict(method)


    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len(i.bag)
        assert counter == len(self.search_space)


    def reset_to_root(self):
        self.CURT = self.ROOT


    def print_tree(self):
        print('\n'+'-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)


    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        self.ROOT.counter += 1
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            curt_node = curt_node.kids[np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]]
            self.nodes[curt_node.id].counter += 1
        return curt_node


    def evaluate_jobs(self):
        while len(self.TASK_QUEUE) > 0:
            job = self.TASK_QUEUE.pop()
            sample_node = self.sample_nodes.pop()

            try:
                # print("\nget job from QUEUE:", job)
                job_str = json.dumps(job)
                design = translator(job, 'full')
                # print("translated to:\n{}".format(design))
                # print("\nstart training:")
                if job_str in dataset:
                    report = {'mae': dataset.get(job_str)}
                    # print(report)
                else:
                    # _, report = Scheme(design, self.weight)
                    _, report = Scheme(design)
                acc = 1 * report['mae']
                self.DISPATCHED_JOB[job_str] = acc

                if type(job[0]) != type([]):
                    job = [job]

                job_str = json.dumps(job)
                zero_counts = [job[i].count(0) for i in range(len(job))]
                # exploration = ((abs(np.subtract(self.topology[job[0]], job))) % 2.4).round().sum()
                gate_reduced = np.sum(zero_counts)
                p_acc = acc + gate_reduced * self.explorations['rate']
                self.samples[job_str] = p_acc
                self.samples_latest[job_str] = p_acc
                # self.explorations[job_str]   = ((abs(np.subtract(self.topology[job[0]], job))) % 2.4).round().sum()
                self.mae_list.append(acc)

                with open('results.csv', 'a+', newline='') as res:
                    writer = csv.writer(res)
                    metrics = acc
                    # num_id = self.sampling_num + len(self.samples)
                    num_id = len(self.samples)
                    writer.writerow([num_id, job_str, sample_node, metrics, p_acc])

            except Exception as e:
                print(e)
                self.TASK_QUEUE.append(job)
                self.sample_nodes.append(sample_node)
                print("current queue length:", len(self.TASK_QUEUE))


    def search(self):
        while len(self.search_space) > 0 and self.ITERATION < 102:
            # save current state
            if self.ITERATION > 0:
                self.dump_all_states(self.sampling_num + len(self.samples))
            print("\niteration:", self.ITERATION)

            period = 5
            max_changes = 1
            
            if (self.ITERATION % period == 1): #and (self.ITERATION != 0):
                # if self.ITERATION % (max_changes*period) == 1:
                #     self.re_init_tree()
                # else:           
                #     self.re_init_tree('restart')
                if self.ITERATION >= (period+1):            
                    self.re_init_tree('restart')
                else:
                    self.re_init_tree()              
                    
                for i in range(len(self.TASK_QUEUE)):
                    net = self.ROOT.base_code.copy()
                    net.append(self.TASK_QUEUE[i])
                    self.TASK_QUEUE[i] = net
           
            # evaluate jobs:
            print("\nevaluate jobs...")
            self.evaluate_jobs()
            print("\nfinished all jobs in task queue")

            # assemble the training data:
            print("\npopulate training data...")
            self.populate_training_data()
            print("finished")

            # training the tree
            print("\ntrain classifiers in nodes...")
            if torch.cuda.is_available():
                print("using cuda device")
            else:
                print("using cpu device")
            start = time.time()
            self.train_nodes()
            print("finished")
            end = time.time()
            print("Running time: %s seconds" % (end - start))
            # self.print_tree()

            # clear the data in nodes
            self.reset_node_data() 

            print("\npopulate prediction data...")
            self.populate_prediction_data()
            print("finished")

            print("\npredict and partition nets in search space...")
            self.predict_nodes()
            self.check_leaf_bags()
            print("finished")
            self.print_tree()
            # # sampling nodes
            # # nodes = [0, 1, 2, 3, 8, 12, 13, 14, 15]
            # nodes = [0, 3, 12, 15]
            # sampling_node(self, nodes, dataset, self.ITERATION)

            random.seed(self.ITERATION)
            for i in range(0, 10):
                # select
                target_bin   = self.select()
                if self.ROOT.base_code == None:
                    qubits = None
                else:
                    qubits = [code[0] for code in self.ROOT.base_code]
                sampled_arch = target_bin.sample_arch(qubits)                
                # NOTED: the sampled arch can be None 
                if sampled_arch is not None:                    
                    # push the arch into task queue
                    if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                        self.TASK_QUEUE.append(sampled_arch)
                        # self.search_space.remove(sampled_arch)
                        self.sample_nodes.append(target_bin.id-7)
                else:
                    # trail 1: pick a network from the left leaf
                    for n in self.nodes:
                        if n.is_leaf == True:
                            sampled_arch = n.sample_arch(qubits)
                            if sampled_arch is not None:
                                print("\nselected node" + str(n.id-7) + " in leaf layer")                                
                                # print("sampled arch:", sampled_arch)
                                if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                                    self.TASK_QUEUE.append(sampled_arch)
                                    # self.search_space.remove(sampled_arch)
                                    self.sample_nodes.append(n.id-7)
                                    break
                            else:
                                continue
                if type(sampled_arch[0]) == type([]):
                    arch = sampled_arch[-1]
                else:
                    arch = sampled_arch
                self.search_space.remove(arch)                          

            self.ITERATION += 1


if __name__ == '__main__':
     # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)   

    with open('search_space_mnist_single', 'rb') as file:
        search_space = pickle.load(file)
    
    arch_code_len = len(search_space[0])
    print("\nthe length of base architecture codes:", arch_code_len)
    print("total architectures:", len(search_space))

    with open('data/mnist_dataset_single', 'rb') as file:
        dataset = pickle.load(file)
      
    # dataset = {}

    if os.path.isfile('results.csv') == False:
        with open('results.csv', 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['sample_id', 'arch_code', 'sample_node', 'ACC', 'p_ACC'])

    if os.path.isfile('results_30_epoch.csv') == False:
        with open('results_30_epoch.csv', 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['iteration', 'arch_code', 'ACC'])

    # agent = MCTS(search_space, 5, arch_code_len)
    # agent.search()

    state_path = 'states'
    if os.path.exists(state_path) == False:
        os.makedirs(state_path)
    files = os.listdir(state_path)
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        # node_path = 'states/mcts_agent_120'
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)
        print("\nresume searching,", agent.ITERATION, "iterations completed before")
        print("=====>loads:", len(agent.nodes), "nodes")
        print("=====>loads:", len(agent.samples), "samples")
        print("=====>loads:", len(agent.DISPATCHED_JOB), "dispatched jobs")
        print("=====>loads:", len(agent.TASK_QUEUE), "task_queue jobs from node:", agent.sample_nodes[0])        
        agent.search()
    else:
        agent = MCTS(search_space, 4, arch_code_len)
        agent.search()

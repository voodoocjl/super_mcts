import json
import math
from math import log2, ceil
import random
import numpy as np
from Classifier import Classifier


class Node:
    obj_counter   = 0

    def __init__(self, parent = None, is_good_kid = False, arch_code_len = 0, is_root = False):
        # Note: every node is initialized as a leaf,
        # only non-leaf nodes are equipped with classifiers to make decisions
        if not is_root:
            assert type(parent) == type(self)

        self.parent        = parent
        self.is_good_kid   = is_good_kid
        self.ARCH_CODE_LEN = arch_code_len
        self.is_root       = is_root
        self.x_bar         = float("inf")
        self.n             = 0
        self.uct           = 0
        self.counter       = 1

        self.kids          = []
        self.bag           = {}
        self.validation    = {}
        self.f1            = [0]

        # data for good and bad kids, respectively
        self.good_kid_data = {}
        self.bad_kid_data  = {}

        self.is_leaf       = True
        self.id            = Node.obj_counter
        self.layer        = ceil(log2(self.id + 2) - 1)
        self.classifier    = Classifier({}, self.ARCH_CODE_LEN, self.id)

        # insert current node into the kids of parent
        if parent is not None:
            self.parent.kids.append(self)
            if self.parent.is_leaf == True:
                self.parent.is_leaf = False
            assert len(self.parent.kids) <= 2

        Node.obj_counter += 1


    def clear_data(self):
        self.bag.clear()
        self.bad_kid_data.clear()
        self.good_kid_data.clear()


    def put_in_bag(self, net, maeinv):
        assert type(net) == type([])
        # assert type(maeinv) == type(float(0.1))
        net_k = json.dumps(net)
        self.bag[net_k] = (maeinv)


    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)


    def pad_str_to_8chars(self, ins):
        if len(ins) <= 14:
            ins += ' ' * (14 - len(ins))
            return ins
        else:
            return ins


    def __str__(self):
        name = self.get_name()
        name = self.pad_str_to_8chars(name)
        name += (self.pad_str_to_8chars('lf:' + str(self.is_leaf)))

        name += (self.pad_str_to_8chars(' val:{0:.4f}   '.format(round(self.get_xbar(), 4))))
        name += (self.pad_str_to_8chars(' uct:{0:.4f}   '.format(round(self.get_uct(Cp=0.2), 4))))
        name += self.pad_str_to_8chars('n:' + str(self.n))
        name += self.pad_str_to_8chars('visit:' + str(self.counter))
        if self.is_leaf == False:
            name += self.pad_str_to_8chars('acc:{0:.4f} '.format(round(self.classifier.training_accuracy[-1], 4)))
            name += self.pad_str_to_8chars('f1:{0:.4f}   '.format(round(self.f1[-1], 4)))

        else:
            name += self.pad_str_to_8chars('acc: ---- ')
            name += self.pad_str_to_8chars('f1:  ---- ')


        name += self.pad_str_to_8chars('sp:' + str(len(self.bag)))
        # name += (self.pad_str_to_8chars('g_k:' + str(len(self.good_kid_data))))
        # name += (self.pad_str_to_8chars('b_k:' + str(len(self.bad_kid_data))))

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent)

        name += (' parent:' + parent)
        # kids = ''
        # kid = ''
        # for k in self.kids:
        #     kid = self.pad_str_to_8chars(k.get_name())
        #     kids += kid
        # name += (' kids:' + kids)
        if self.is_leaf:
            name = Color.YELLOW + name +Color.RESET
        elif self.layer == 2:
            name = Color.GREEN + name +Color.RESET

        return name


    def get_uct(self, Cp):
        if self.is_root and self.parent == None:
            return float('inf')
        if self.n == 0 and len(self.bag) > 0:
            return float('inf')
        # coeff = 2 ** (5 - ceil(log2(self.id + 2)))
        if len(self.bag) == 0:
            return 0
        # return self.x_bar + Cp*math.sqrt(2*math.log(self.parent.n)/self.n)
        return self.x_bar + 2 * Cp*math.sqrt(2*math.log(self.parent.counter)/self.counter)


    def get_xbar(self):
        return self.x_bar


    def train(self):
        if self.parent == None and self.is_root == True:
        # training starts from the bag
            assert len(self.bag) > 0
            self.classifier.update_samples(self.bag, self.x_bar)
            self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
                self.classifier.update_samples(self.parent.good_kid_data, self.x_bar)
                self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
            else:
                self.bag = self.parent.bad_kid_data
                self.classifier.update_samples(self.parent.bad_kid_data, self.x_bar)
                self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
        self.x_bar = np.mean(np.array(list(self.bag.values())))
        self.n     = len(self.bag.values())


    def predict(self, method = None):
        if self.parent == None and self.is_root == True and self.is_leaf == False:
            self.good_kid_data, self.bad_kid_data, _ = self.classifier.split_predictions(self.bag, method)
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
                self.good_kid_data, self.bad_kid_data, xbar = self.classifier.split_predictions(self.parent.good_kid_data, method)
                # self.x_bar = xbar
            else:
                self.bag = self.parent.bad_kid_data
                self.good_kid_data, self.bad_kid_data, xbar = self.classifier.split_predictions(self.parent.bad_kid_data, method)
                # self.x_bar = xbar
        if method:
            self.validation = self.bag.copy()


    def predict_validation(self):
        if self.is_leaf == False:
            self.good_kid_data, self.bad_kid_data, _ = self.classifier.split_predictions(self.validation)
        if self.is_good_kid:
            self.bag = self.parent.good_kid_data


    def get_performance(self):
        i = 0
        for k in self.bag.keys():
            if k in self.validation:
                i += 1
        precision = i / (len(self.bag) + 1e-6)
        i = 0
        for k in self.validation.keys():
            if k in self.bag:
                i += 1
        recall = i / len(self.validation)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return f1


    def sample_arch(self):
        if len(self.bag) == 0:
            return None
        net_str = random.choice(list(self.bag.keys()))
        del self.bag[net_str]
        parent_node = self.parent
        for i in range(3):
            del parent_node.bag[net_str]
            parent_node = parent_node.parent
        return json.loads(net_str)


class Color:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

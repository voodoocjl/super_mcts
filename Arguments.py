class Arguments:
    def __init__(self):
        self.n_qubits   = 4
        
        self.device     = 'cpu'        
        self.clr        = 0.005
        self.qlr        = 0.01
        self.epochs     = 1
        self.batch_size = 256        
        self.sampling = 5

        self.n_layers = 4
        self.base_code = [self.n_layers, 2, 3, 4, 1]

        # self.backend    = 'pennylane'
        self.backend    = 'tq'
        self.digits_of_interest = [0, 1, 2, 3]
        self.train_valid_split_ratio = [0.95, 0.05]
        self.center_crop = 24
        self.resize = 28

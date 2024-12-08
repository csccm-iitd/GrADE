import logging
import argparse
import os
import sys
import torch as T
from itertools import count
import numpy as np
import os.path as osp
import torch
import time
from prettytable import PrettyTable
import json
import warnings
import sys
import psutil

sys.path.insert(1, '/src')

# from src.Training import init_model, training_epoch, testing_epoch
from Training import init_model, training_epoch, testing_epoch

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# writer.add_graph(model, input_to_model=[torch.tensor([0]), torch.tensor([0]), data.x.to(device)], verbose=True)
# writer.close()

# warnings.simplefilter('error')
warnings.filterwarnings("ignore")


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        # logger.info(f"name: {name} value{parameter}\n")
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='gnode_pde')
        
        folder = '/home/souvik/subhankar/Grade_benchmarking/grade'
        prjct_dir = folder
        # prjct_dir = '/content/drive/MyDrive/Colab Notebooks/gnode_pde'
        device = T.device('cuda:' + str(0) if T.cuda.is_available() else 'cpu')

        self.add_argument('--prjct_dir', type=str, default=prjct_dir)
        self.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
                          default='rk4')
        self.add_argument('--train_batch_size', type=int, default=20)
        self.add_argument('--test_batch_size', type=int, default=30)
        self.add_argument('--seed', type=int, default=25)
        self.add_argument('--device', type=str, default=device)
        self.add_argument('--os', type=str, default='linux', choices=['window', 'linux'])

        # will be changed down the code, just for demo
        self.add_argument('--ConvLayer', type=str, default='SpiderConv')
        self.add_argument('--data_type', type=str, default='pde_2d')
        self.add_argument('--op_type', type=str, default='burgers2d')
        self.add_argument('--Basis', type=str, default='Chebychev')
        self.add_argument('--N_Bases_ls', type=int, default=3)
        self.add_argument('--save_logs', action='store_true', default=True)

        # experimental purpose, need not be altered
        self.add_argument('--cont_in', type=str, choices=['t', 'dt'], default='t')  # continuous_in
        self.add_argument('--adaptive_graph', action='store_true', default=False)
        self.add_argument('--PINN', type=int, default=0)
        self.add_argument('--myepoch', type=list, default=[3, 4])
        self.add_argument('--recursive_integration', action='store_true', default=False)

    def parse(self):
        args = self.parse_args()
        args.Bases_ = ['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF',
                       'GaussianRBF', 'None', 'MultiquadRBF', 'PiecewiseConstant']
        args.Layers_ = ['PointNetLayer', 'GATConv', 'SpiderConv', 'GATConv2', 'SplineConv']
        args.ops = ['burgers1d', 'burgers2d']  # [r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}']
        args.data_types = ['pde_1d', 'pde_2d', 'TrajectoryExtrapolation']
        args.vary_ = ['lyr', 'bses', 'n_b', 'tsu', 'b_ts', 'rt', 'dt']
        args.rollout_ts = [0]
        return args

    def print_args(self, args):
        logger.info(f'continuous_in: {args.cont_in}')
        logger.info(f'Basis: {args.Basis}')
        logger.info(f'Number of Basis: {args.N_Basis}')
        logger.info(f'stencil: {args.stencil}')
        if args.train:
            logger.info(f'train_batch_size: {args.train_batch_size}')
        if not args.train:
            logger.info(f'test_batch_size: {args.test_batch_size}')


class TrainLoop():
    def __init__(self, **kwargs):
        self.run_loop(**kwargs)

    def set_value(self):
        if args.exp[:-3] == 'lit_train__lit_test':
            idx = args.last_intg_t_idx_train_ls.index(args.last_intg_t_idx_train)
            args.lr_epoch_ls = args.lr_epoch_ls_ls[idx]

        if args.exp[:-3] == 'lit_train__lit_test':
            if hasattr(args, 'niters_ls'):
                idx = args.last_intg_t_idx_train_ls.index(args.last_intg_t_idx_train)
                args.niters = args.niters_ls[idx]

        logger.info(f'learning rate: {args.lr}') if hasattr(args, 'lr') else 0
        logger.info(f'niters: {args.niters}') if hasattr(args, 'niters') else 0

    def lr_scheduler(self, optimizer, epoch):
        """use value of lr for specific epoch from 'lr_epoch_ls' list """

        if hasattr(args, 'lr_epoch_ls'):
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr_epoch_ls[epoch]

        for param_group in optimizer.param_groups:
            return param_group['lr']

    def run_loop(self, **kwargs):
        args.train = True
        for args.ConvLayer in args.Layer_ls:
            for args.Basis in args.Basis_ls:
                for args.N_Basis in args.N_Bases_ls:
                    for args.train_size_used in args.tsu:
                        for args.last_intg_t_idx_train in args.last_intg_t_idx_train_ls:
                            self.set_value()
                            args.add_str = args.add_str if hasattr(args, 'add_str') else ''
                            model, optimizer, train_loader, test_loader, PlotResults, target_list = init_model(args,
                                                                                                               Parser_,
                                                                                                               **kwargs)
                            count_parameters(model)
                            prev_loss = 1000  # any large no.
                            t1 = 0
                            for epoch in range(0, args.niters):
     
                                lr = self.lr_scheduler(optimizer, epoch)
                                start = time.time()
                                batch_time = args.lit_idx_train_epoch_ls[epoch] if hasattr(args,
                                                                                           'lit_idx_train_epoch_ls') and epoch < len(
                                    args.lit_idx_train_epoch_ls) else args.last_intg_t_idx_train

                                train_error = training_epoch(model, optimizer, train_loader, PlotResults, target_list,
                                                             epoch, batch_time, args)
                                t1 += time.time() - start
                                logger.info(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, '
                                            f'intgrtd_t: {batch_time}, lr: {lr:.4f}, time: {t1:.4f}')

                                path = args.get_path(args, W=1, exp=args.exp)
                                if epoch % args.sv_every == 0 and epoch > 0:
                                    # count_parameters(model) # todo remove
                                    start = time.time()
                                    test_error = testing_epoch(model, test_loader, PlotResults, target_list, epoch,
                                                               args.last_intg_t_idx_test, args)
                                    t2 = time.time() - start
                                    if test_error < prev_loss:
                                        model_wts = model.state_dict()
                                        prev_loss = test_error
                                        save_epoch = epoch
                                        logger.info(f'Epoch: {epoch:02d}, test_Loss: {test_error:.4f},  '
                                                    f'intgrtd_t: {args.last_intg_t_idx_test}, lr: {lr:.4f}, time: {t2:.4f} ***')
                                    else:
                                        logger.info(f'Epoch: {epoch:02d}, test_Loss: {test_error:.4f},  '
                                                    f'intgrtd_t: {args.last_intg_t_idx_test}, lr: {lr:.4f}, time: {t2:.4f}')

                                    if args.exp[:11] == 'time__epoch':
                                        if not "dr" in vars():
                                            dr = np.array([[t1], [test_error]])
                                        else:
                                            dr = np.concatenate((dr, np.array([[t1], [test_error]])), axis=1)

                            args.add_str = args.add_str if hasattr(args, 'add_str') else ''
                            if args.exp[:11] == 'time__epoch' and kwargs.get('save', None):
                                sv_dir = args.get_path(args, C3=1, rslt_dir=args.exp)
                                np.savetxt(osp.join(sv_dir, args.exp + args.add_str + '.csv'), dr, delimiter=",")

                            if kwargs.get('save', None):
                                torch.save(model_wts, osp.join(path, 'weights' + args.add_str))
                                args.logger.info(f'================saved weights at {save_epoch} epoch================')


class TestLoop():
    def __init__(self, csv_name, sv_dir, **kwargs):
        self.csv_name = csv_name
        self.sv_dir = sv_dir
        self.kwargs = kwargs
        self.run_loop(**kwargs)

    def rslt1(self, test_error):
        # if not "ar" in vars():
        if not hasattr(self, 'ar'):
            self.ar = np.array([[test_error]])
        else:
            self.ar = np.concatenate((self.ar, np.array([[test_error]])), axis=1)

        # print(self.ar)

    def rslt2(self):
        # if not "arr" in vars():
        if not hasattr(self, 'arr'):
            self.arr = self.ar
        else:
            self.arr = np.concatenate((self.arr, self.ar), axis=0)
        del self.ar

        # print(self.arr)

    def save_rslt(self):
        if self.kwargs.get('save', None):
            np.savetxt(osp.join(self.sv_dir, self.csv_name), self.arr, delimiter=",")
        del self.arr

    def run_loop(self, **kwargs):
        args.train = False
        args.dt_ls = args.dt_ls if hasattr(args, 'dt_ls') else [0]
        for args.ConvLayer in args.Layer_ls:
            for args.Basis in args.Basis_ls:
                for args.N_Basis in args.N_Bases_ls:
                    for args.train_size_used in args.tsu:
                        for args.dt in args.dt_ls:
                            for args.last_intg_t_idx_train in args.last_intg_t_idx_train_ls:
                                args.last_intg_t_idx_test_ls = args.last_intg_t_idx_test_ls if hasattr(args,
                                                                                                       'last_intg_t_idx_test_ls') else [
                                    0]
                                for args.last_intg_t_idx_test in args.last_intg_t_idx_test_ls:
                                    model, optimizer, train_loader, test_loader, PlotResults, target_list = init_model(
                                        args, Parser_, **kwargs)
                                    path = args.get_path(args, W=1, exp=args.exp)
                                    args.add_str = args.add_str if hasattr(args, 'add_str') else ''
                                    # pretrained_weights = osp.join(path, 'weights' + args.add_str)
                                    pretrained_weights = osp.join(path, 'weights')
                                    count_parameters(model)
                                    model.load_state_dict(torch.load(pretrained_weights, map_location=args.device))
                                    epoch = 0
                                    last_intg_t_idx_test = args.last_intg_t_idx_train if args.last_intg_t_idx_test_ls == [
                                        0] else args.last_intg_t_idx_test
                                    start = time.time()
                                    test_error = testing_epoch(model, test_loader, PlotResults, target_list, epoch,
                                                               last_intg_t_idx_test, args)
                                    logger.info(f'Epoch: {epoch:02d}, test_Loss: {test_error:.4f}, '
                                                f'intgrtd_t: {last_intg_t_idx_test}, time: {time.time() - start:.4f}')

                                    self.rslt1(test_error)
                                self.rslt2() if 'b_ts' == args.vary[1] else 0
                            self.rslt2() if 'dt' == args.vary[1] else 0
                        self.rslt2() if 'tsu' in args.vary[1] else 0
                    self.rslt2() if 'n_b' in args.vary[1] else 0
                self.rslt2() if 'bses' in args.vary[1] else 0
            self.rslt2() if 'lyr' in args.vary[1] else 0
        self.save_rslt()


def save_args(sv_args, sv_dir):
    del_args = ['logger', 'device', 'ts', 'small_del_t', 'n_split_locs', 'raw_data',
                'get_edge_index', 'pos_j', 'skip', 'len_data', 'get_path']
    [delattr(sv_args, g) for g in del_args if hasattr(sv_args, g)]
    with open(sv_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(sv_args), args_file, indent=4)


def save_logs(args, **kwargs):
    if args.data_type == 'pde_1d':
        # from src.pde_1d.Dataset import get_path
        from pde_1d.Dataset import get_path
    elif args.data_type == 'pde_2d':
        # from src.pde_2d.Dataset import get_path
        from pde_2d.Dataset import get_path
    # elif args.data_type == 'TrajectoryExtrapolation':
    #     from TrajectoryExtrapolation.Dataset import get_path

    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if args.save_logs:
        # Open errorLogs.log file to save printed output
        p_ = get_path(args, CL=1, exp=args.exp)
        path = osp.join(p_, 'errorLogs.log')

        fh = logging.FileHandler(path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    logger.info(f'\n*********************************************************************************************\n'
                f'                          experiment name: {args.exp}\n'
                f'*********************************************************************************************\n')

    args.logger = logger

    return get_path


# Different plot functions 1d===========================================================================================


def train_layer__lit_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:3]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [80]
    args.lit_idx_train_epoch_ls = [4] * 211 + [4] * 30 + [5] * 30 + [6] * 30 + [7] * 15 + [8] * 15
    args.last_intg_t_idx_train_ls = [4]
    args.last_intg_t_idx_test = 14
    args.sv_every = 30
    args.niters = 211
    args.exp = 'layer__lit_test_1d'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def layer__lit_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:3]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [80]
    args.last_intg_t_idx_train_ls = list(range(4, 24, 4))
    args.vary = args.vary_[4:5] + args.vary_[0:1]
    args.exp = 'layer__lit_test_1d'
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='layer__lit_test_1d')
    TestLoop('layer__lit_test_1d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_tr_size__lit_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = list(range(30, 151, 30))
    # args.lit_idx_train_epoch_ls = [4] * 120
    args.last_intg_t_idx_train_ls = [4]
    args.skip = 7
    args.lr = 0.07
    args.last_intg_t_idx_test = 5
    args.sv_every = 50
    args.niters = 211
    args.exp = 'tr_size__lit_test_1d'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def tr_size__lit_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [120]  # list(range(30, 121, 30))
    args.skip = 7
    args.last_intg_t_idx_train_ls = [30]  # list(range(3, 24, 3))
    args.vary = args.vary_[4:5] + args.vary_[3:4]
    args.exp = 'tr_size__lit_test_1d'
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='tr_size__lit_test_1d')
    TestLoop(args.exp + '.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_lit_train__lit_test_1d(**kwargs):
    
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [120]  # [140]

    
    # args.lit_idx_train_epoch_ls = [2] * 300 + [3] * 300 + [4]* 201 
    # args.lit_idx_train_epoch_ls = [2] * 200 + [3] * 200 + [4] * 401
    # args.lit_idx_train_epoch_ls = [2] * 200 + [3] * 100 + [4]* 501
    # args.lit_idx_train_epoch_ls = [2] * 50 + [3] * 50 + [4] * 701
    args.lit_idx_train_epoch_ls = [2] * 25 + [3] * 25 + [4] * 751
    # args.lit_idx_train_epoch_ls = [2] * 25 + [3] * 50 + [4] * 751
    # args.lit_idx_train_epoch_ls = [2] * 100 + [3] * 100 + [4] * 601


    args.last_intg_t_idx_train_ls = [4] 
    args.skip = 7
    # args.niters_ls = [351]  # [201, 401, 401, 401]
    args.niters_ls = [801]  # [201, 401, 401, 401]

    lr10 =    [0.07]*50 + [0.065]*50 + [0.060]*50 + [0.055]*50 \
            + [0.050]*50 + [0.045]*50 + [0.040]*50 + [0.035]*50 \
            + [0.030]*50 + [0.025]*50 + [0.020]*50 + [0.015]*50 \
            + [0.010]*50 + [0.005]*50 + [0.001]*50 + [0.0005]*51 
    
    lr11 =    [0.07]*51 + [0.065]*50 + [0.060]*25 + [0.055]*25 \
            + [0.050]*25 + [0.045]*25 + [0.040]*25 + [0.035]*25 \
            + [0.030]*25 + [0.025]*25 + [0.020]*25 + [0.015]*25 \
            + [0.010]*25 + [0.005]*25 + [0.001]*25 + [0.0005]*25

    args.lr_epoch_ls_ls = [lr10] 
    args.last_intg_t_idx_test = 4
    args.sv_every = 50
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.exp = 'lit_train__lit_test_1d'
    _ = save_logs(args)
    TrainLoop(**kwargs)

    # args.niters_ls = [301, 351]
    # lr1 = [0.05] * 801
    # lr5 = [0.06]*50 + [0.05]*100 + [0.045]*50 + [0.04]*50 + [0.035]*50 + [0.03]*50 + [0.025]*50 + [0.02]*50 + [0.015]*50 + [0.01]*50 
    # lr6 = [0.065]*50 + [0.060]*50 + [0.055]*50 + [0.05]*100 + [0.045]*100 + [0.04]*100 + [0.035]*100 + [0.03]*100 + [0.025]*100 + [0.02]*100 + [0.015]*100 + [0.01]*100
    # lr7 = [0.07]*50 + [0.065]*50 + [0.060]*50 + [0.055]*100 + [0.050]*151
    # lr8 = [0.07]*50 + [0.065]*100 + [0.060]*100 + [0.055]*100 + [0.050]*251
    # lr9 = [0.07]*100 + [0.065]*100 + [0.060]*100 + [0.055]*100 + [0.050]*251

    # # lr4 = [0.057] * 25 + [0.059] * 25 + [0.062] * 50 + [0.065] * 351
    # # lr2 = [0.05] * 25 + [0.052] * 25 + [0.054] * 50 + [0.056] * 351
    # # lr3 = [0.045] * 25 + [0.048] * 25 + [0.052] * 50 + [0.054] * 351
    # # lr4 = [0.07] * 401

    # lr2 = [0.05] * 25 + [0.052] * 25 + [0.054] * 50 + [0.056] * 301
    # lr3 = [0.045] * 25 + [0.048] * 25 + [0.052] * 50 + [0.054] * 301
    # lr4 = [0.040] * 25 + [0.045] * 25 + [0.050] * 50 + [0.025] * 301
    
    # lr4 = [0.04]*25 + [0.042]*25 + [0.044]*50 + [0.048]*251
    # args.lr_epoch_ls_ls = [lr1, lr1, lr2, lr3]


def lit_train__lit_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [120]
    args.skip = 7
    args.last_intg_t_idx_train_ls = list(range(2, 6, 1))
    args.last_intg_t_idx_test_ls = list(range(3, 24, 3))
    args.vary = args.vary_[4:5] + args.vary_[4:5]
    args.exp = 'lit_train__lit_test_1d'
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='lit_train__lit_test_1d')
    TestLoop('lit_train__lit_test_1d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_B__dt_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:2]
    args.N_Bases_ls = [3]
    args.tsu = [100]
    args.lit_idx_train_epoch_ls = [3] * 15 + [4] * 15 + [5] * 15 + [6] * 15 + [7] * 15 + [8] * 15
    args.last_intg_t_idx_train_ls = [12]
    args.last_intg_t_idx_test = 15
    TrainLoop(**kwargs)


def B__dt_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:2]
    args.N_Bases_ls = [3]
    args.tsu = [80]
    args.last_intg_t_idx_train_ls = list(range(4, 24, 4))
    args.train_batch_size = 1
    args.wbt = 8
    args.dt_ls = list(range(1, 5))
    args.vary = args.vary_[4:5] + args.vary_[0:1]
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='B__dt_test_1d')
    TestLoop('B__dt_test_1d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_B__lit_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[6:7]
    args.N_Bases_ls = [3]
    args.tsu = [80]
    args.lit_idx_train_epoch_ls = [4] * 15 + [5] * 15 + [6] * 15 + [7] * 15 + [8] * 15
    args.last_intg_t_idx_train_ls = [8]
    args.last_intg_t_idx_test = 15
    args.exp = 'B__lit_test_1d'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def B__lit_test_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:7]
    args.N_Bases_ls = [3]
    args.tsu = [80]
    args.last_intg_t_idx_train_ls = list(range(2, 16, 2))
    args.train_batch_size = 2
    args.wbt = 8
    args.vary = args.vary_[4:5] + args.vary_[1:2]
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='B__lit_test_1d')
    TestLoop('B__lit_test_1d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_B__n_B_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:7]
    args.N_Bases_ls = [1, 2, 3, 4, 5]
    args.tsu = [70]
    args.lit_idx_train_epoch_ls = [3] * 15 + [4] * 15 + [5] * 15 + [6] * 15 + [7] * 15
    args.last_intg_t_idx_train_ls = [7]
    args.last_intg_t_idx_test = 15
    TrainLoop(**kwargs)


def B__n_B_1d(**kwargs):
    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layer_ls[2:3]
    args.Basis_ls = args.Basis_ls[1:7]
    args.N_Bases_ls = [1, 2, 3, 4, 5]
    args.tsu = [70]
    args.last_intg_t_idx_train_ls = [8]
    args.train_batch_size = 2
    args.wbt = 7
    args.vary = args.vary_[4:5] + args.vary_[1:2]
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='B__n_B_1d')
    TestLoop('B__n_B_1d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_trial_1d(**kwargs):
    # args.data_type = args.data_types[0]
    # args.op_type = args.ops[0]
    # args.Layer_ls = args.Layers_[2:3]
    # args.Basis_ls = args.Bases_[5:6]
    # args.N_Bases_ls = [1]
    # args.tsu = [60]
    # args.lit_idx_train_epoch_ls = [3] * 251 + [4] * 30 + [5] * 30 + [6] * 30 + [7] * 15 + [8] * 15
    # args.last_intg_t_idx_train_ls = [4]
    # args.last_intg_t_idx_test = 14
    # args.sv_every = 50
    # args.niters = 251
    # args.train_batch_size = 20
    # args.test_batch_size = 20
    # args.lr = 0.006
    # # args.lr = 0.06  # gat no pinn pos_j - pos_i
    # # args.lr = 0.08  # spider no pinn pos_j - pos_i
    # # args.lr = 0.09  # spider no pinn pos_j
    # args.method = 'rk4'
    # args.exp = 'layer__lit_test_1d_PINN'
    # _ = save_logs(args)
    # TrainLoop(**kwargs)

    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [140]
    args.lit_idx_train_epoch_ls = [4] * 201 + [5] * 150 + [5] * 30 + [6] * 30 + [7] * 15 + [8] * 15
    args.last_intg_t_idx_train_ls = [4]
    args.skip = 7
    args.last_intg_t_idx_test = 3
    args.lr = 0.07
    args.sv_every = 1 #50
    args.niters = 2 #201
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.method = 'rk4'
    args.exp = 'lit_train__lit_test_1d_trial'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def trial_1d(**kwargs):
    # args.data_type = args.data_types[0]
    # args.op_type = args.ops[0]
    # args.Layer_ls = args.Layers_[2:3]
    # args.Basis_ls = args.Bases_[5:6]
    # args.N_Bases_ls = [1]
    # args.tsu = [60]
    # args.test_batch_size = 60
    # args.last_intg_t_idx_train_ls = [3]  # list(range(3, 24, 4))
    # args.vary = args.vary_[4:5] + args.vary_[0:1]
    # args.exp = 'layer__lit_test_1d_PINN'
    # args.save_logs = False
    # get_path = save_logs(args)
    # sv_dir = get_path(args, C3=1, rslt_dir=args.exp)
    # TestLoop('layer__lit_test_1d_PINN.csv', sv_dir, **kwargs)
    # sv_args = args
    # save_args(sv_args, sv_dir)

    args.data_type = args.data_types[0]
    args.op_type = args.ops[0]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [140]
    args.skip = 7
    # args.last_intg_t_idx_test_ls = [40]
    args.last_intg_t_idx_train_ls = [40]  # list(range(3, 24, 4))
    args.vary = args.vary_[4:5] + args.vary_[4:5]
    args.exp = 'lit_train__lit_test_1d_trial'
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir=args.exp)
    TestLoop('lit_train__lit_test_1d_trial.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


# Different plot functions 2d===========================================================================================


def train_layer__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:3]  # 0:3
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [120]
    args.lit_idx_train_epoch_ls = [3] * 501  # [2] * 15 + [3] * 15 + [4] * 15 + [5] * 15 + [6] * 15 + [7] * 15
    args.last_intg_t_idx_train_ls = [3]
    args.last_intg_t_idx_test = 2
    args.sv_every = 100
    args.niters = 501
    args.skip = 2
    args.lr_epoch_ls = [0.055] * 200 + [0.052] * 100 + [0.048] * 250
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.method = 'rk4'
    args.stencil = 'star'
    args.exp = 'layer__lit_test_2d'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def layer__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:3]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [120]
    args.skip = 2
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.stencil = 'star'
    args.last_intg_t_idx_train_ls = list(range(5, 35, 5))
    args.vary = args.vary_[4:5] + args.vary_[0:1]
    args.exp = 'layer__lit_test_2d'
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='layer__lit_test_2d')
    TestLoop(args.exp + '.csv', sv_dir, **kwargs)

    sv_args = args
    save_args(sv_args, sv_dir)


def train_tr_size__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    # args.tsu = list(range(20, 121, 10))
    args.tsu = list(range(80, 121, 20))
    args.sv_every = 100
    args.niters = 501
    args.lit_idx_train_epoch_ls = [3] * args.niters
    args.last_intg_t_idx_train_ls = [3]
    args.skip = 2
    args.last_intg_t_idx_test = 2
    args.lr_epoch_ls = [0.055] * 200 + [0.052] * 100 + [0.048] * 250
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.method = 'rk4'
    args.exp = 'tr_size__lit_test_2d'
    args.stencil = 'star'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def tr_size__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    # args.tsu = list(range(30, 121, 30))
    # args.tsu = list(range(20, 121, 10))
    args.tsu = [120]  # list(range(80, 121, 20))
    args.last_intg_t_idx_train_ls = [30]  # list(range(5, 35, 5))
    args.vary = args.vary_[4:5] + args.vary_[3:4]
    args.skip = 2
    args.exp = 'tr_size__lit_test_2d'
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.save_logs = False
    get_path = save_logs(args)
    args.stencil = 'star'
    sv_dir = get_path(args, C3=1, rslt_dir='tr_size__lit_test_2d')
    TestLoop(args.exp + '.csv', sv_dir, **kwargs)

    # args.add_str = '_9'
    # args.stencil = 9
    # TestLoop(args.exp + '_9' + '.csv', sv_dir, **kwargs)

    sv_args = args
    save_args(sv_args, sv_dir)


def train_B__dt_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:2] + args.Bases_[5:6]
    args.N_Bases_ls = [3]
    args.tsu = [30]
    args.lit_idx_train_epoch_ls = [2] * 15 + [3] * 15 + [4] * 15 + [5] * 15 + [6] * 15 + [7] * 15  #
    args.last_intg_t_idx_train_ls = [7]
    args.last_intg_t_idx_test = 12
    args.exp = 'B__dt_test_2d'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def B__dt_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:2] + args.Bases_[5:6]
    args.N_Bases_ls = [3]
    args.tsu = [30]
    args.last_intg_t_idx_train_ls = [10]
    args.dt_ls = list(range(1, 6))
    args.vary = args.vary_[6:7] + args.vary_[1:2]
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='B__dt_test_2d')
    TestLoop('B__dt_test_2d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_B__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:7]
    args.N_Bases_ls = [2]
    args.tsu = [80]
    args.lit_idx_train_epoch_ls = [2] * 15 + [3] * 15 + [4] * 15 + [5] * 15 + [6] * 15 + [7] * 15
    args.last_intg_t_idx_train_ls = [7]
    args.last_intg_t_idx_test = 12
    args.exp = 'B__lit_test_2d'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def B__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:7]
    args.N_Bases_ls = [2]
    args.tsu = [80]
    args.last_intg_t_idx_train_ls = list(range(2, 16, 2))
    args.vary = args.vary_[4:5] + args.vary_[1:2]
    args.exp = 'B__lit_test_2d'
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='B__lit_test_2d')
    TestLoop('B__lit_test_2d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_B__n_B_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:6]
    args.N_Bases_ls = [3, 5]
    args.tsu = [13]
    args.lit_idx_train_epoch_ls = [2] * 15 + [3] * 15 + [4] * 15 + [5] * 15 + [6] * 15 + [7] * 15
    args.last_intg_t_idx_train_ls = [7]
    args.last_intg_t_idx_test = 12
    TrainLoop(**kwargs)


def B__n_B_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[1:7]
    args.N_Bases_ls = [1, 2, 3, 4, 5]
    args.tsu = [70]
    args.last_intg_t_idx_train_ls = [8]
    args.train_batch_size = 2
    args.wbt = 7
    args.vary = args.vary_[4:5] + args.vary_[1:2]
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='B__n_B_2d')
    TestLoop('B__n_B_2d.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)


def train_lit_train__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [90]
    args.last_intg_t_idx_train_ls = [4]  # [2, 3, 4]
    args.skip = 2
    args.last_intg_t_idx_test = 3

    lr_ls1 = [0.055] * 200 + [0.053] * 100 + [0.05] * 250
    lr_ls2 = [0.055] * 200 + [0.054] * 100 + [0.03] * 250
    lr_ls3 = [0.045] * 400 + [0.044] * 300 + [0.043] * 250
    args.lr_epoch_ls_ls = [lr_ls1, lr_ls2, lr_ls3]
    args.lr_epoch_ls_ls = [lr_ls3]

    args.niters_ls = [701]  # [401, 401, 701]
    # args.niters_ls = [701]
    args.sv_every = 100
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.method = 'rk4'
    args.exp = 'lit_train__lit_test_2d'
    args.stencil = 'star'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def lit_train__lit_test_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [90]
    args.skip = 2
    args.last_intg_t_idx_train_ls = list(range(2, 5, 1))
    args.last_intg_t_idx_test_ls = list(range(5, 35, 5))
    args.vary = args.vary_[4:5] + args.vary_[4:5]
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.exp = 'lit_train__lit_test_2d'
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir='lit_train__lit_test_2d')
    TestLoop(args.exp + '.csv', sv_dir, **kwargs)

    # args.add_str = '_star'
    # TestLoop(args.exp + '_star' + '.csv', sv_dir, **kwargs)

    sv_args = args
    save_args(sv_args, sv_dir)


def time__epoch_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [90]
    args.last_intg_t_idx_test = 5
    args.skip = 2
    args.sv_every = 100
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.exp = 'time__epoch_2d'
    args.stencil = 'star'

    # args.last_intg_t_idx_train_ls = [4]
    # args.lr = 0.025
    # args.niters = 401

    args.last_intg_t_idx_train_ls = [4]
    args.lr = 0.045
    args.niters = 701
    _ = save_logs(args)
    TrainLoop(**kwargs)

    args.last_intg_t_idx_train_ls = [4]
    args.lit_idx_train_epoch_ls = [2] * 200 + [3] * 300 + [4] * 300

    lr1 = [0.06] * 200
    lr2 = [0.022] * 25 + [0.024] * 25 + [0.032] * 50 + [0.04] * 200  # + [0.045] * 200
    lr3 = [0.015] * 25 + [0.018] * 25 + [0.022] * 25 + [0.032] * 25 + [0.04] * 250  # + [0.045] * 200
    args.lr_epoch_ls = lr1 + lr2 + lr3
    # args.niters = 401
    # args.lit_idx_train_epoch_ls = [3] * 300 + [4] * 300 + [5] * 300
    # args.lr_epoch_ls = [0.055] * 300 + [0.025] * 50 + [0.025] * 150 + [0.01] * 100 + [0.013] * 200
    args.niters = 701
    args.add_str = '_vry'
    _ = save_logs(args)
    TrainLoop(**kwargs)


def train_trial_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[0:1]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [120]
    args.lit_idx_train_epoch_ls = [2] * 401 + [5] * 150 + [5] * 30 + [6] * 30 + [7] * 15 + [8] * 15
    args.last_intg_t_idx_train_ls = [2]
    args.skip = 2
    args.last_intg_t_idx_test = 3
    # args.lr = 0.055
    args.lr_epoch_ls = [0.05] * 400 + [0.05] * 200
    args.sv_every = 50
    args.niters = 401
    args.train_batch_size = 30
    args.test_batch_size = 30
    args.method = 'rk4'
    args.exp = 'lit_train__lit_test_2d_trial'
    _ = save_logs(args)
    # args.stencil = 'star'
    args.stencil = 'k_near'
    # args.stencil = 'k_nn'
    TrainLoop(**kwargs)


def trial_2d(**kwargs):
    args.data_type = args.data_types[1]
    args.op_type = args.ops[1]
    args.Layer_ls = args.Layers_[1:2]
    args.Basis_ls = args.Bases_[5:6]
    args.N_Bases_ls = [1]
    args.tsu = [90]
    args.skip = 2
    args.last_intg_t_idx_train_ls = [25]  # list(range(3, 24, 4))
    args.vary = args.vary_[4:5] + args.vary_[0:1]
    args.test_batch_size = 10
    args.exp = 'lit_train__lit_test_2d_trial'
    args.save_logs = False
    get_path = save_logs(args)
    sv_dir = get_path(args, C3=1, rslt_dir=args.exp)

    args.add_str = '_star'
    args.stencil = 'star'
    TestLoop(args.exp + '_star' + '.csv', sv_dir, **kwargs)
    sv_args = args
    save_args(sv_args, sv_dir)

    # args.add_str = '_9'
    # args.stencil = 9
    # TestLoop(args.exp + '_9' + '.csv', sv_dir, **kwargs)
    # sv_args = args
    # save_args(sv_args, sv_dir)


# ======================================================================================================================

if __name__ == '__main__':
    Parser_ = Parser()
    args = Parser_.parse()

    logger = logging.getLogger('my_module')
    logger.setLevel(logging.DEBUG)

    # train_trial_1d(save=1, show=0)
    #
    # train_layer__lit_test_1d(save=0, show=0)
    #
    # train_tr_size__lit_test_1d(save=1, show=0)
    #
    train_lit_train__lit_test_1d(save=1, show=0)
    #
    # train_B__dt_test_1d(save=1, show=0)
    #
    # train_B__lit_test_1d(save=1, show=0)
    #
    # train_adptive__none_1D(save=0, show=1)
    #
    # train_NoBasis__del(save=1, show=0)
    #
    # # *******************************************************
    #
    # train_trial_2d(save=1, show=0)
    #
    # train_layer__lit_test_2d(save=1, show=0)
    #
    # train_tr_size__lit_test_2d(save=1, show=0)
    #
    # train_B__dt_test_2d(save=1, show=0)
    #
    # train_B__lit_test_2d(save=1, show=0)
    #
    # train_B__n_B_2d(save=0, show=0)
    #
    # time__epoch_2d(save=1, show=0)
    #
    # train_lit_train__lit_test_2d(save=1, show=0)
    #
    # =============================================  test functions =====================================================
    #
    # trial_1d(save=1, show=0)
    #
    # layer__lit_test_1d(save=0, show=1)
    #
    # tr_size__lit_test_1d(save=1, show=0)
    #
    # lit_train__lit_test_1d(save=1, show=0)
    #
    # B__dt_test_1d(save=0, show=0)
    #
    # B__lit_test_1d(save=1, show=0)
    #
    # B__n_B_1d(save=0, show=0)
    #
    # # *******************************************************
    #
    # trial_2d(save=1, show=0)
    #
    # B__dt_test_2d(save=1, show=0)
    #
    # tr_size__lit_test_2d(save=1, show=0)
    #
    # layer__lit_test_2d(save=1, show=0)
    #
    # B__lit_test_2d(save=1, show=0)
    #
    # B__n_B_2d(save=0, show=0)
    #
    # lit_train__lit_test_2d(save=1, show=0)
    #
    # adptive__none(save=0, show=1)
    #
    # layer__lit_test(save=0, show=0)
    #
    # NoBasis__del(save=1, show=0)
    #
    # at_NoBasis__del(save=1, show=0)
    #
    # lit_train__lit_test_TE(save=1, show=0)
    #
    # # *******************************************************

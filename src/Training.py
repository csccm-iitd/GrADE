from torch_geometric.data import DataLoader
import numpy as np
# import torchdiffeq
# import multiprocessing as mp
import torch as T
import random
from itertools import count
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import os.path as osp
import math
import sys
import os
import psutil
sys.path.insert(0, '/src')
# prjct_dir = '/content/drive/MyDrive/Colab Notebooks/gnode_pde'
# sys.path.insert(0,os.path.join(prjct_dir,"src"))
# sys.path.insert(1,os.path.join(prjct_dir,"data"))

# from src.Model import Net
from Model import Net
from Utils import integrator


def init_model(args, Parser_, **kwargs):

    args.save, args.show = kwargs.get('save'), kwargs.get('show')
    get_value = lambda attr_name, default_value: getattr(args, attr_name) if hasattr(args, attr_name) else default_value
    if args.Basis == 'None':
        args.N_Basis = 1
    # from src.Utils import integrator
    from Utils import integrator

    if args.data_type == 'pde_1d':
        # from src.pde_1d.Dataset import Mydata, get_path, PlotResults
        from pde_1d.Dataset import Mydata, get_path, PlotResults, TargetList
        args.get_path = get_path

        args.n_Nodes = 511
        args.n_Nodes_d = 511
        args.lr = get_value('lr', 0.08)
        args.niters = get_value('niters', 70)  # 120

        if args.op_type == 'burgers1d':
            args.train_size = 170
            args.test_size = 30
            args.sv_every = get_value('sv_every', 5)
            # args.n_time_steps = 300300/
            args.max_batch_time = 2002
            if args.cont_in == 't':
                args.skip = get_value('skip', 4)
                args.ts = lambda skip: T.linspace(0, 2, args.max_batch_time)[0:-1:skip]
            elif args.cont_in == 'dt':
                if hasattr(args, 'dt'):
                    args.skip = lambda bs: T.tensor([args.dt])
                elif args.Basis != 'None':
                    args.skip = lambda bs: T.randint(1, 5, (bs, 1))
                elif args.Basis == 'None':
                    args.skip = lambda bs: T.randint(2, 3, (bs, 1))
                args.ts = lambda skip: T.linspace(0, 2, args.max_batch_time)[0:-1:skip]
            in_f, out_f = 1, 1
        else:
            raise Exception("Unkonwn operator type.")

        args.n_linear_layers = 1
        args.n_conv_layers = 3
        args.conv_hidden_dim = [1, 1]
        args.l_w = 1000
        args.stencil = args.stencil if hasattr(args, 'stencil') else 5
        model = Net(args, in_f, out_f).to(args.device)

    elif args.data_type == 'pde_2d':
        from pde_2d.Dataset import Mydata, get_path, PlotResults, TargetList
        # from src.pde_2d.Dataset import Mydata, get_path, PlotResults
        args.get_path = get_path

        args.n_Nodes = 63 * 63
        args.n_Nodes_d = 128 * 128
        args.lr = get_value('lr', 0.08)
        args.niters = get_value('niters', 150)

        if args.op_type == 'burgers2d':
            args.train_size = 121
            args.test_size = 20
            args.sv_every = get_value('sv_every', 20)
            args.ln_test_batch_t = get_value('ln_test_batch_t', 50)
            # args.n_time_steps = 2525
            if args.cont_in == 't':
                args.max_batch_time = 101
                args.skip = get_value('skip', 1)
                args.ts = lambda skip: T.linspace(0, 1, args.max_batch_time)[0:-1:skip]
            elif args.cont_in == 'dt':
                args.max_batch_time = 201
                if hasattr(args, 'dt'):
                    args.skip = lambda bs: T.tensor([args.dt])
                elif args.Basis != 'None':
                    args.skip = lambda bs: T.randint(1, 5, (bs, 1))
                elif args.Basis == 'None':
                    args.skip = lambda bs: T.randint(2, 3, (bs, 1))
                args.ts = lambda skip: T.linspace(0, 0.5, args.max_batch_time)[0:-1:skip]
            in_f, out_f = 2, 2
        else:
            raise Exception("Unkonwn operator type.")
        # args.small_del_t = args.ts[1] - args.ts[0]
        args.n_conv_layers = 3
        args.conv_hidden_dim = [4, 8]
        args.l_w = 10
        args.stencil = args.stencil if hasattr(args, 'stencil') else 'star'
        model = Net(args, in_f, out_f).to(args.device)

    # elif args.data_type == 'TrajectoryExtrapolation':
    #     from TrajectoryExtrapolation.Dataset import Mydata, get_path
    #     from TrajectoryExtrapolation.Utils import integrator, PlotResults
    #
    #     args.train_size = 30
    #     args.test_size = 10
    #     args.sv_every = 5
    #     args.n_time_steps = 1000  # 256
    #     args.n_Nodes = 10
    #     args.niters = 30
    #     args.max_batch_time = 30
    #     args.n_linear_layers = 1
    #     args.n_conv_layers = 1
    #     args.l_w = 1
    #     # args.conv_hidden_dim = 4
    #     # model.Layer(args, 8, 1)
    #     model = Net(args, 4, 1).to(args.device)

    Parser_.print_args(args)

    _ = get_path(args, P=1)
    path = get_path(args, F=1)

    mydata = Mydata(root=path, name='Pydata', args=args)

    dataset = mydata

    train_dataset = dataset[:args.train_size_used]
    test_dataset = dataset[args.train_size:]

    args.logger.info(f'Number of training graphs: {len(train_dataset)}')
    args.logger.info(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    target_list = TargetList()

    # print(model)
    args.logger.info(model)
    optimizer = T.optim.Adam(model.parameters(), lr=args.lr)  # lr=0.06)
    # criterion = T.nn.MSELoss()

    return model, optimizer, train_loader, test_loader, PlotResults, target_list


def append(i_ls, t_ls, full_i_ls, full_t_ls, i):
    if i == 0:
        full_i_ls, full_t_ls = i_ls, t_ls
    else:
        full_i_ls, full_t_ls = np.concatenate((full_i_ls, i_ls), axis=1), np.concatenate((full_t_ls, t_ls), axis=1)

    return full_i_ls, full_t_ls


def training_epoch(model, optimizer, train_loader, PlotResults, target_list, epoch, batch_time, args):
    model.train()

    total_loss = 0
    full_i_ls = full_t_ls = np.array(0)


    for data, i in zip(train_loader, count(0, 1)):
        optimizer.zero_grad()  # Clear gradients.

        args.bs = len(data.x[:, 0]) // args.n_Nodes
        args.bs_ = args.bs if args.cont_in == 't' else 1
        loss, update_total_loss, i_ls, t_ls = integrator(model, data, batch_time, True, epoch, i, target_list, args)
        loss.backward()  # Backward pass

        # for p in model.parameters():
        #     T.nn.utils.clip_grad_norm_(p, 0.05)
        # T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # T.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        # for name, params in model.named_parameters():
        #     args.logger.info(f'name:{name}, value:{params[0, 0, 0]}, grad:{params.grad[0, 0, 0]}')
        optimizer.step()  # Update model parameters

        # for name, params in model.named_parameters():
        #     args.logger.info(f'name:{name}, value:{params[0, 0, 0]}, grad:{params.grad[0, 0, 0]}')

        total_loss = total_loss + update_total_loss.sum()
        full_i_ls, full_t_ls = append(i_ls.cpu().numpy(), t_ls.cpu().numpy(), full_i_ls, full_t_ls, i)

    # if epoch % args.sv_every == 0 and epoch > 0:
    #     if args.save or args.show:
    #         c = 0#random.randint(0, args.bs-1)
    #         PlotResults(full_i_ls, full_t_ls, batch_time, args, model.pos_list, model.edge_index_list, c, epoch, 2)
    #         abcde = 0
    #     args.ignore = 1

    return total_loss / len(train_loader.dataset)


# def return_loss(data, i, model, test_loader, get_path, integrator, PlotResults, epoch, batch_time, args, kwargs):
#     args.bs = len(data.x[:, 0]) // args.n_Nodes
#     integrated_list, trgt_batch = integrator(model, data, batch_time, False, epoch, i, args.device, args)
#     # integrated_list = integrated_list[0::2]
#
#     # total_loss = total_loss + np.linalg.norm(
#     #     integrated_list.cpu().detach().numpy()[:, :] - trgt_batch.cpu().detach().numpy()) / np.linalg.norm(
#     #     integrated_list.cpu().detach().numpy()[:, :]) * data.num_graphs
#     loss = T.linalg.norm(
#         integrated_list.detach().clone()[:, :] - trgt_batch.detach().clone()) / T.linalg.norm(
#         integrated_list.detach().clone()[:, :]) * data.num_graphs
#
#     pos_list = model.pos_list
#     edge_index_list = model.edge_index_list
#     # if i == epoch and kwargs.get('show', None):
#     if i == 0 and kwargs.get('show', None):
#         if epoch % 1 == 0 and epoch >= 0:
#             PlotResults(trgt_batch.cpu().detach().numpy(), integrated_list[:, :].cpu().detach().numpy(),
#                         batch_time, args, pos_list, edge_index_list, epoch, 2)
#             args.ignore = 1
#
#     return loss
#
#
# @T.no_grad()
# def test(model, test_loader, get_path, integrator, PlotResults, epoch, batch_time, args, **kwargs):
#     model.eval()
#     pool = mp.Pool(2)
#     # losses = pool.map(return_loss, (data for data in test_loader))
#     # total_loss = sum(losses)
#     bb = [pool.apply_async(return_loss, args=(data, i, model, test_loader, get_path, integrator, PlotResults,
#                                               epoch, batch_time, args, kwargs)) for data, i in zip(test_loader, count(0, 1))]
#     cc = [p.get() for p in bb]
#     total_loss = sum(cc)
#
#     return total_loss / len(test_loader.dataset)

@T.no_grad()
def testing_epoch(model, test_loader, PlotResults, target_list, epoch, batch_time, args):

    # model.eval()
    total_loss = 0
    full_i_ls = full_t_ls = np.array(0)
    args.ignore = 1

    for data, i in zip(test_loader, count(0, 1)):

        args.bs = len(data.x[:, 0]) // args.n_Nodes
        args.bs_ = args.bs if args.cont_in == 't' else 1
        args.logger.info(f'RAM memory % used: {psutil.virtual_memory()[2]} i:{i}')
        # breakpoint()
        _, update_total_loss, i_ls, t_ls = integrator(model, data, batch_time, False, epoch, i, target_list, args)
        args.logger.info(f'RAM memory % used: {psutil.virtual_memory()[2]} i:{i}')

        total_loss = total_loss + update_total_loss.sum()
        full_i_ls, full_t_ls = append(i_ls.cpu().numpy(), t_ls.cpu().numpy(), full_i_ls, full_t_ls, i)

    # if i == epoch and kwargs.get('show', None):
    if args.save or args.show:
        if epoch % 1 == 0 and epoch >= 0:
            c_ls = [9, 10, 16 , 26, 15, 21]#  [0, 15]  # random.randint(0, args.bs-1)
            for c in c_ls:
                PlotResults(full_i_ls, full_t_ls, batch_time, args, model.pos_list, model.edge_index_list, epoch, c, 2)
            args.ignore = 1

    return total_loss / len(test_loader.dataset)

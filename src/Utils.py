import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F
from torch import linalg as LA
import pdb
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.interpolate import interp1d
import sys
import psutil

sys.path.insert(1, '/src')

# from src.torchdiffeq import odeint
from torchdiffeq import odeint


class RecursiveIntegration:

    def __init__(self, args, model, size, data, time_step, method, integrated_ls):
        super(RecursiveIntegration, self).__init__()
        self.args = args
        self.model = model
        self.device = self.args.device
        self.method = method
        self.array = np.ndarray
        self.list = []
        self.size = batch_time = size
        self.solution_ls = T.zeros((self.size, 21 * 21, 1), device=self.device)
        self.data = data
        self.time_step = time_step
        self.calculate(integrated_ls, batch_time)

    def dev(aaa):
        if aaa < 2:
            return 1
        else:
            return 2 ** (aaa - 1)

    def calculate(self, tensort, xaxis):  # u2,u3 ,u4,   3

        if xaxis == self.size:
            self.solution_ls[0] = tensort[0]

        for i in range(xaxis - 1):
            aaa = self.size - xaxis + i + 1
            deva = self.dev(aaa)
            bbb = (tensort[i + 1] / self.dev(aaa))
            self.solution_ls[aaa] = self.solution_ls[aaa] + bbb

        for i in range(xaxis - 1):
            t_start = (self.size - xaxis + i + 1) * self.time_step
            N_steps = self.size - (self.size - xaxis + i + 1)
            t_end = self.time_step * (self.size - 1)

            if self.method == 'from_zero':
                batch_t = T.linspace(0, t_end - t_start, N_steps)
            elif self.method == 'from_start':
                batch_t = T.linspace(t_start, t_end, N_steps)
            else:
                raise ValueError('method is not recognized')

            # assert batch_t.s
            # temp = odeint(TrueOperator, tensort[i + 1], batch_t, method='euler')
            temp = odeint(self.model, tensort[i + 1], batch_t, method=self.args.method).to(self.device)
            # temp = odeint(self.model, tensort[i + 1].to(self.device), self.data.batch.to(self.device), self.data.edge_index.to(self.device),
            #               self.data.pos.to(self.device), self.data.edge_attr.to(self.device), batch_t.to(self.device),
            #               method=self.args.method).to(self.device)
            self.calculate(temp, xaxis - i - 1)


def integrator(model, data, batch_time, train_bool, epoch, i, target_list, args):
    # if args.data_type == 'pde_1d':
    #     if args.os == 'window':
    #         from src.pde_1d.Dataset import target_batch
    #     else:
    #         from pde_1d.Dataset import target_batch
    # if args.data_type == 'pde_2d':
    #     if args.os == 'window':
    #         from src.pde_2d.Dataset import TargetList
    #     else:
    #         from pde_2d.Dataset import TargetList

    device = args.device
    if args.cont_in == 't':
        """ network take uses 't' for determining weights at different depth(layer number) of network """

        ts = args.ts(args.skip)
        batch_t = ts[0:batch_time]
        # batch_t = T.linspace(0, batch_t[-1], args.ln_test_batch_t, device=batch_t.device)
        del_t = batch_t[1].item() - batch_t[0].item()
        idx_ls = T.tensor(range(0, batch_time)) # [0,1,2,4]


        # Calculate edge index for defining connection in graph
        data.edge_index = model.get_edge_index(data.x, data.pos.to(device), data.batch.to(device)).cpu()
        data.edge_attr = T.tensor([1])
        data.pos = data.pos if data.pos is not None else T.ones((len(data.x[:, 0]), 2))

        # Setting values required by model
        model.del_t, model.epoch, model.batch, model.edge_attr = del_t, epoch, data.batch.to(device), data.edge_attr
        model.pos_list, model.edge_index_list, model.prev_idx = [data.pos.to(device)], [data.edge_index.to(device)], -1

        if args.PINN and 1 and args.data_type == 'pde_1d':
            u_t_autograd = []
            model.u_t_pred_ls = []
            mu = 0.0025
            xc = data.pos.to(device).clone()
            xc.requires_grad = True
            model.pos_list = xc
            integrated_ls = odeint(model, data.x.to(device).float(), batch_t.to(device), method=args.method)

            # debugging =====================================
            # integrated_ls[1].sum().backward()
            # model.u_t_pred_ls[1].sum().backward()
            # for name, params in model.named_parameters():
            #     args.logger.info(f'name:{name}, value:{params[0, 0, 0]}, grad:{params.grad[0, 0, 0]}')
            # stay_cool = 0
            # debugging =====================================

            # time_ls = T.cat((batch_t, T.tensor([batch_t[-1].item()+del_t])), 0)
            for t_idx in range(1, len(batch_t)):
                upred = integrated_ls[t_idx]
                upred_x = T.autograd.grad(upred.sum(), xc, create_graph=True)[0]
                upred_xx = T.autograd.grad(upred_x.sum(), xc, create_graph=True)[0]
                u_t = -upred * upred_x + mu * upred_xx
                # u_t = - upred_x
                # u_t = - upred_x * upred
                # u_t = upred_xx
                u_t_autograd.append(u_t)
            u_t_autograd_ls = T.stack(u_t_autograd, dim=0).float()
            u_t_model_ls = T.stack(model.u_t_pred_ls[1:], dim=0)

            # debugging =====================================
            # crit = T.nn.MSELoss()
            # integrated_ls[1].sum().backward()
            # model.u_t_pred_ls[1].sum().backward()
            # u_t.sum().backward()
            # loss = crit(model.u_t_pred_ls[1], u_t)
            # loss = crit(u_t_autograd_ls, u_t_model_ls)
            # loss.backward()
            # for name, params in model.named_parameters():
            #     args.logger.info(f'name:{name}, value:{params[0, 0, 0]}, grad:{params.grad[0, 0, 0]}')
            # stay_cool = 0

            # u_t = model(1, 0, data.x.to(device))
            # # u = data.x.to(device) + u_t
            # upred_x = T.autograd.grad(u_t.sum(), xc, create_graph=True)[0]
            # upred_xx = T.autograd.grad(upred_x.sum(), xc, create_graph=True)[0]
            # jkjljk = 0

            # if epoch == 130:
            def plot_for_debug():
                fig = plt.figure()  # figsize=(5, 6))
                # widths = [3, 3, 3]
                # heights = [1, 1]
                spec5 = fig.add_gridspec(ncols=2, nrows=2)  # , width_ratios=widths, height_ratios=heights)
                a1 = fig.add_subplot(spec5[0, 0])
                a2 = fig.add_subplot(spec5[0, 1])
                b1 = fig.add_subplot(spec5[1, 0])
                b2 = fig.add_subplot(spec5[1, 1])

                start = 511 * 0
                end = start + 511

                _u_ = data.x.detach().cpu()[start:end, 0]

                # a1.plot(_u_)
                # a1.set_title('u')
                # abc = np.gradient(_u_)
                # a2.plot(-abc)
                # a2.set_title('u_x')
                # b1.plot(u_t_autograd_ls[0, :, 0].detach().cpu())
                # b1.set_title('u_x autograd')
                # b2.plot(u_t_model_ls[0, :, 0].detach().cpu())
                # b2.set_title('u_x model')

                # a1.plot(_u_)
                # a1.set_title('u')
                # _u_x = T.tensor(np.gradient(_u_))
                # a2.plot(- _u_x * _u_)
                # a2.set_title('-u*u_x')
                # b1.plot(u_t_autograd_ls[0, start:end, 0].detach().cpu())
                # b1.set_title('-u*u_x autograd')
                # b2.plot(u_t_model_ls[0, start:end, 0].detach().cpu())
                # b2.set_title('-u*u_x model')

                a1.plot(_u_)
                a1.set_title('u')
                _u_x = T.tensor(np.gradient(_u_))
                _u_xx = T.tensor(np.gradient(_u_x))
                a2.plot(-_u_ * _u_x + mu * _u_x)
                a2.set_title('u_t')
                b1.plot(u_t_autograd_ls[0, start:end, 0].detach().cpu())
                b1.set_title('u_t autograd')
                b2.plot(u_t_model_ls[0, start:end, 0].detach().cpu())
                b2.set_title('u_t model')

                plt.show()

            plot_for_debug()
            # debugging =====================================

            stan = lambda x: (x - x.mean()) / x.std()
            l1 = F.mse_loss(stan(u_t_autograd_ls), stan(u_t_model_ls))
            l2 = (u_t_autograd_ls ** -2).sum() * 1e-18
            # l3 = (u_t_model_ls**-2 * 1e+1).sum()*1e-4
            l3 = T.tensor([0.000])

            target_ls = (target_list(data.x, i, 0, train_bool, idx_ls, args)).float().to(args.device)
            l4 = F.mse_loss(target_ls, integrated_ls)
            loss = l1  # + l4  # + l2

            if i % 5 == 0:
                args.logger.info(
                    f'l1:{round(l1.item(), 4)}, l2:{round(l2.item(), 4)}, l3:{round(l4.item(), 4)}')

            ilc, tlc = integrated_ls.detach().clone(), target_ls.detach().clone()
            update_total_loss = loss.detach().clone()

        elif args.adaptive_graph and not model.training and args.data_type == 'pde_1d':
            """ Not ready """

            # list for storing integrated values
            integrated_ls_new = data.x.to(device)[None]

            # list for storing values at initial node (for calculating loss we want values at initial node loc.)
            integrated_ls = data.x.to(device)[None]
            p = model.pos_list[0]

            def cp(x):
                return x.detach().cpu()

            def new_pos(args, diff):
                pos = T.arange(args.n_dense_Nodes, dtype=T.float32, device=args.device)[:, None]
                x1 = T.tensor(range(0, args.n_dense_Nodes - 1), device=diff.device)
                x2 = T.tensor(range(1, args.n_dense_Nodes), device=diff.device)

                y1 = diff[0:args.n_dense_Nodes - 1]
                y2 = diff[1:args.n_dense_Nodes]

                del_z = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                del_z_stack = T.zeros((len(del_z) + 1))
                for i in range(len(del_z)):
                    del_z_stack[i + 1] = del_z[0:i].sum()

                dis_stack = T.linspace(0, del_z_stack[-1].item(), args.n_Nodes)
                ohno = T.cat([del_z_stack.unsqueeze(0)] * args.n_Nodes, dim=0)
                ohyes = T.cat([dis_stack.unsqueeze(1)] * len(del_z_stack), dim=1)
                positions = T.argmin(T.abs(ohno - ohyes), dim=1)

                # del_z_stack = T.cat([del_z.unsqueeze(0)]*len(del_z))
                # del_z_sum = del_z_stack.sum(dim=0)
                pos = pos[positions]
                return pos

            def intp(old_co_ordinates, old_values, new_co_ordinates):
                f = interp1d(old_co_ordinates, old_values)
                return f(new_co_ordinates)

            for time in range(batch_time - 1):
                h_t = integrated_ls_new[time]
                pos = model.pos_list[time]
                batch_1 = batch_t[time: time + 2]
                model.pos_list.append(pos.detach().clone())

                # evaluate next time value(h_t1)
                h_t1 = odeint(model, h_t, batch_1, method=args.method)[-1]

                # evaluate differance between h_t and next time value(h_t1) on dense nodes(n_dense_Nodes)
                locs = T.linspace(0, args.n_Nodes - 1, args.n_Nodes).to(args.device)
                locs_dense = T.linspace(0, args.n_Nodes - 1, args.n_dense_Nodes).to(args.device)
                diff = T.abs((intp(locs, h_t1.T, locs_dense) - intp(locs, h_t.T, locs_dense) * 1.3) * 20).long() + 20
                # --------------------------------------

                # evaluate new node location based on calculated differance(diff)
                pos_n0 = new_pos(args, diff.reshape(1000)) * (args.n_Nodes - 1) / (args.n_dense_Nodes - 1)
                sorted_pos_n, indices = T.sort(T.transpose(pos_n0, 0, 1))
                pos_n = T.transpose(sorted_pos_n, 0, 1)
                # --------------------------------------

                # remove old pos from pos_list used for evaluating h_t1
                model.pos_list.pop(-1)
                model.pos_list.append(pos_n.detach().clone())
                # --------------------------------------

                # evaluate current and next time value at pos_n
                h_new = (intp(pos[:, 0], h_t.T, pos_n[:, 0])).T
                h_t1_new = odeint(model, h_new, batch_1, method=args.method)[-1]
                integrated_ls_new = T.cat((integrated_ls_new, h_t1_new.unsqueeze(0)), dim=0)
                model.edge_index_list.append(model.edge_index)
                # --------------------------------------

                # next time value at initial node locations(p) for calculating loss
                # value will not be used again in loop(only stored)
                h_t1_new_old_pos = intp(cp(pos_n[:, 0]), cp(h_t1_new), cp(p[:, 0]))[:, None].to(args.device)
                integrated_ls = T.cat((integrated_ls, h_t1_new_old_pos.unsqueeze(0)), dim=0)
                # --------------------------------------

        else:

            integrated_ls = odeint(model, data.x.to(device), batch_t.to(device), method=args.method)
            target_ls = (target_list[data.x, i, 0, train_bool, idx_ls, batch_time, args]).to(args.device)

            if args.recursive_integration and epoch in args.myepoch:
                """ Not used """
                recursive_intg = RecursiveIntegration(args, model, batch_time, data, del_t, args.method, integrated_ls)
                integrated_ls = recursive_intg.solution_ls

            # calculate loss on which loss.backward will be called
            ilc, tlc = integrated_ls.detach().clone(), target_ls.detach().clone()
            weightage = T.tensor(range(len(ilc[:, 0])), device=args.device)[:, None, None] * args.l_w
            loss = F.mse_loss(integrated_ls * weightage, target_ls * weightage) if train_bool else 0
            # --------------------------------------

            # calculate update_total_loss for analysing performance using only final time step
            # value of update_total_loss do not vary by changing batch size
            split_graphs = lambda x: T.stack(T.split(x, args.n_Nodes, dim=1), dim=0)
            cat_t_ls = lambda y: T.cat(T.split(y, 1, dim=1), dim=2)[:, 0]
            reshape_ls = lambda z: cat_t_ls(split_graphs(z))

            # pdb.set_trace()

            def total_loss(ilc, tlc):
                """
                ARGS:
                    ilc (torch.tensor): [lit_idx, n_Nodes*batch_size, 1]
                VARS:
                    reshape_ls(ilc) (torch.tensor): [batch_size, n_Nodes*lit_idx, 1]
                RETURNS: (torch.tensor): [batch_size, 1]
                """
                # breakpoint()
                return LA.norm(reshape_ls(ilc - tlc), dim=1) / LA.norm(reshape_ls(ilc), dim=1)

            update_total_loss = total_loss(ilc[-1][None], tlc[-1][None]) # Total loss is being calculated as Rmse at last time step
            # args.logger.info(f'{update_total_loss}')
            # --------------------------------------

    elif args.cont_in == 'dt':
        """ network take uses 'dt' for determining weights at different depth(layer number) of network """

        trgt_batch_ls = []
        integrated_list = []
        T.random.seed()
        skip = args.skip(args.bs)

        for z in range(args.bs):
            ts = args.ts(skip[z])
            batch_t = ts[0:batch_time]
            idx_ls = T.tensor(range(0, batch_time))

            len_1_batch = len(data.x) // args.bs
            u0 = data.x[len_1_batch * z:len_1_batch * (z + 1)]
            batch = data.batch[0:len_1_batch]
            pos = data.pos[len_1_batch * z:len_1_batch * (z + 1)]

            data.edge_index = model.get_edge_index(u0, pos.to(args.device), batch.to(args.device)).cpu()
            # data.x = data.x.to(T.float32)
            pos = pos if pos is not None else T.ones((len(data.x[:, 0]), 2))
            data.edge_attr = T.tensor([1])
            del_t = batch_t[1].item() - batch_t[0].item()

            model.del_t = del_t
            model.epoch = epoch

            model.del_t, model.epoch, model.batch, model.edge_attr = del_t, epoch, batch.to(device), data.edge_attr
            model.pos_list, model.edge_index_list, model.prev_idx = [pos.to(device)], [data.edge_index.to(device)], -1

            integrated_list.append(odeint(model, u0.to(device), batch_t.to(device), method=args.method))
            trgt_batch_ls.append(target_list(0, i, z, train_bool, idx_ls, args))

        assert T.equal(data.x[:len_1_batch],
                       trgt_batch_ls[0][0]), 'starting target different from starting training data'
        trgt_batch = T.cat(trgt_batch_ls, dim=1)
        integrated_list = T.cat(integrated_list, dim=1)

    return loss, update_total_loss, ilc, tlc

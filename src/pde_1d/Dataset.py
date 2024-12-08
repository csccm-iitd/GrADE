import sys
import os.path as osp
import os
import logging

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
from scipy.interpolate import interp1d
import argparse
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import shutil


def get_path(args, **kwargs) -> str:
    """ Make folders of various path based on string in kwargs
    Return:
         path of folder created
    """

    if hasattr(args, 'ConvLayer'):
        if args.ConvLayer == 'PointNetLayer':
            dir1 = 'PointNetLayer'
        elif args.ConvLayer == 'GATConv':
            dir1 = 'GATConv'
        elif args.ConvLayer == 'SpiderConv':
            dir1 = 'SpiderConv'
        elif args.ConvLayer == 'SplineConv':
            dir1 = 'SplineConv'

    if hasattr(args, 'Basis'):
        if args.Basis == 'Polynomial':
            dir2 = 'Polynomial'
        elif args.Basis == 'Chebychev':
            dir2 = 'Chebychev'
        elif args.Basis == 'Fourier':
            dir2 = 'Fourier'
        elif args.Basis == 'VanillaRBF':
            dir2 = 'VanillaRBF'
        elif args.Basis == 'GaussianRBF':
            dir2 = 'GaussianRBF'
        elif args.Basis == 'MultiquadRBF':
            dir2 = 'MultiquadRBF'
        elif args.Basis == 'PiecewiseConstant':
            dir2 = 'PiecewiseConstant'
        elif args.Basis == 'None':
            dir2 = 'No_Basis'

    if args.op_type == 'burgers1d':
        dir3 = 'burgers1d'

    if kwargs.get('W', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_1d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir, 'src/pde_1d/weights_results')
        if args.exp == 'lit_train__lit_test_1d':
            path = osp.join(p_, dir3, kwargs.get('exp', None), dir1, dir2, 'N_Basis{}'.format(args.N_Basis),
                            'train_size{}'.format(args.train_size_used), 'lit_idx_train{}'.format(args.last_intg_t_idx_train))
        else:
            path = osp.join(p_, dir3, kwargs.get('exp', None), dir1, dir2, 'N_Basis{}'.format(args.N_Basis),
                            'train_size{}'.format(args.train_size_used))

    elif kwargs.get('C1', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_1d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir, 'src/pde_1d/weights_results')
        path = osp.join(p_, dir3, dir1, dir2)

    elif kwargs.get('C2', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_1d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir, 'src/pde_1d/weights_results')
        if kwargs.get('rslt_dir', None):
            path = osp.join(p_, dir3, dir1, kwargs.get('rslt_dir', None))
        else:
            path = osp.join(p_, dir3, dir1)

    elif kwargs.get('C3', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_1d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir, 'src/pde_1d/weights_results')
        if kwargs.get('rslt_dir', None):
            path = osp.join(p_, dir3, kwargs.get('rslt_dir', None))
        else:
            path = osp.join(p_, dir3)

    elif kwargs.get('CL', None):
        path = osp.join(args.prjct_dir, 'src/pde_1d\weights_results', dir3, kwargs.get('exp', None)) if args.os == 'window' else osp.join(args.prjct_dir, 'src/pde_1d/weights_results', dir3, kwargs.get('exp', None))

    elif kwargs.get('F', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_1d')
        path = osp.join(p_, 'torch_data')

    elif kwargs.get('R', None):
        p_ = osp.join(args.prjct_dir, 'data/pde_1d/fenics_data') if args.os == 'window' else osp.join(args.prjct_dir, 'data/pde_1d/fenics_data')
        path = osp.join(p_)

    elif kwargs.get('P', None):
        path = osp.join(args.prjct_dir, 'src/pde_1d/torch_data')
        if os.path.exists(path):
            shutil.rmtree(path)
        mkm = 0

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def raw_sequential_data(args, **kwargs):

    if hasattr(args, 'raw_data') and hasattr(args, 'last_data_type') and args.data_type == args.last_data_type:
        # args.logger.info(f'using previously loaded data')
        pass
    else:
        mbt = args.max_batch_time
        t_end = round(T.linspace(0, 2, args.max_batch_time)[-1].item(), 5)
        path = get_path(args, R=1)


        bool_files = [
            (osp.exists(osp.join(path, 'raw{:d}_mbt{:d}_tend{}_nx{}.npy'.format(run, mbt, t_end, args.n_Nodes+1))),
             osp.join(path, 'raw{:d}_mbt{:d}_tend{}_nx{}.npy'.format(run, mbt, t_end, args.n_Nodes+1))) for run in range(200)] # range(300)]

        assert any([bool_files[i][0] for i in
                    range(len(bool_files))]), 'data not present, run fenics_burger1d.py in /data folder to get data'
        data_ = np.stack([(np.load(nm)) for bl, nm in bool_files if bl], 0)
        data_ = T.tensor(data_[:, :, 0:args.n_Nodes])
        args.len_data = np.arange(0, len(data_[:, 0, 0]))
        # args.n_time_steps = len(data_[:, 0])
        args.raw_data = data_.to(T.float32)

        args.logger.info(f'data loaded ')

    if kwargs.get('plot_data', False):
        wv = data_[0]
        fps = 5  # frame per sec
        frn = len(data_[:, 0, 0])

        def update_plot(frame_number, wv, plot1):
            plot1[0].set_ydata(wv[frame_number])
            n = 0

        fig = plt.figure(figsize=(4, 4))
        widths = [4]
        heights = [4]
        aa = np.linspace(0, 1, args.n_Nodes)
        spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths,
                                 height_ratios=heights)
        bx = fig.add_subplot(spec5[0, 0])
        # bx = fig.add_subplot(spec5[1, 0])
        plot1 = bx.plot(aa, wv[0])

        ani = animation.FuncAnimation(fig, update_plot, frn,
                                      fargs=(wv, plot1),
                                      interval=1000 / fps)
        # ====================================== saving data gif ======================================
        gif = 'anim{:d}_mbt{:d}_tend{}_nx{}'.format(len(data_[:, 0, 0]), mbt, t_end, args.n_Nodes)
        gif_path = osp.join(path, gif)
        # ani.save(gif_path + '.mp4', writer='ffmpeg', fps=fps)
        # ani.save(gif_path + '.gif', writer='imagemagick', fps=fps)
        plt.show()

    args.last_data_type = args.data_type
    return args.raw_data


def traindata_position_in_raw_data(args):
    # assert args.train_size % args.max_batch_time == 0, "train_size should be multiple of batch_time"
    # assert args.train_size < (
    #         args.n_time_steps + 1 - args.max_batch_time) / args.max_batch_time, "train_size should be smaller that n_time_steps/batch_time"
    # assert args.test_size < (
    #         args.n_time_steps + 1 - args.max_batch_time) / args.max_batch_time, "test_size should be smaller that n_time_steps/batch_time"

    np.random.seed(10)
    s = T.from_numpy(np.random.choice(args.len_data, args.train_size, replace=False))
    np.random.seed(5)
    q = T.from_numpy(np.random.choice(args.len_data, args.test_size, replace=False))
    return s.long(), q.long()


class TargetList:
    def __init__(self):
        self.target_tr_ls = {}
        self.target_ts_ls = {}

    def __getitem__(self, item):
        batched_u, position, z, train_bool, idx_ls, lit_idx, args = item

        if train_bool:
            if lit_idx in list(self.target_tr_ls.keys()):
                if position in list(self.target_tr_ls[lit_idx].keys()):
                    # args.logger.info(f'using previously processed train target data')
                    pass
                else:
                    self.compute(batched_u, position, z, train_bool, idx_ls, lit_idx, args)

            else:
                self.target_tr_ls[lit_idx] = {}
                self.compute(batched_u, position, z, train_bool, idx_ls, lit_idx, args)
        else:

            if lit_idx in list(self.target_ts_ls.keys()):
                if position in list(self.target_ts_ls[lit_idx].keys()):
                    args.logger.info(f'using previously processed test target data')
                    pass
                else:
                    self.compute(batched_u, position, z, train_bool, idx_ls, lit_idx, args)
            else:
                self.target_ts_ls[lit_idx] = {}
                self.compute(batched_u, position, z, train_bool, idx_ls, lit_idx, args)

        return self.target_tr_ls[lit_idx][position] if train_bool else self.target_ts_ls[lit_idx][position]

    def compute(self, batched_u, position, z, train_bool, idx_ls, lit_idx, args):

        if args.cont_in == 't':
            s, q = traindata_position_in_raw_data(args)
            assert idx_ls[-1] <= args.max_batch_time, 'batch_time should be smaller than args.max_batch_time'

            x_array = raw_sequential_data(args, plot_data=False).unsqueeze(3)

            nx = args.n_Nodes

            if train_bool:
                bs = args.bs # batch_size
                trbs = args.train_batch_size
                # breakpoint()
                #args.skip is the time skip
                batch_y = T.cat([x_array[:, 0:-1:args.skip][s[position * trbs + z], idx_ls, :] for z in range(bs)], dim=1)
                # batch_y = train_x_array
                assert T.equal(batched_u[:len(x_array[0, :, 0])], batch_y[0, :len(x_array[0, :, 0])]), 'starting target different from starting training data'
                self.target_tr_ls[lit_idx][position] = batch_y.to(args.device)
            else:
                bs = args.bs
                tbs = args.test_batch_size
                batch_y = T.cat([x_array[:, 0:-1:args.skip][q[position * tbs + z], idx_ls, :] for z in range(bs)], dim=1)
                assert T.equal(batched_u[:len(x_array[0, 0])],
                               batch_y[0, :len(x_array[0, 0])]), 'starting target different from starting training data'
                self.target_ts_ls[lit_idx][position] = batch_y.to(args.device)


        elif args.cont_in == 'dt':

            s, q = traindata_position_in_raw_data(args)
            skip = args.skip(args.bs)
            # assert idx_ls[-1] <= args.max_batch_time/args.skip, 'batch_time should be smaller than args.max_batch_time'

            data_ = raw_sequential_data(args, plot_data=False).unsqueeze(2)

            if train_bool:
                trbs = args.train_batch_size
                batch_y = (data_[:, 0:-1:skip[z]][s[position * trbs + z], idx_ls, :]).transpose(2, 1)
                self.target_tr_ls[lit_idx][position] = batch_y.to(args.device)

            else:
                tbs = args.test_batch_size
                batch_y = (data_[:, 0:-1:skip[z]][q[position * tbs + z], idx_ls, :]).transpose(2, 1)
                self.target_ts_ls[lit_idx][position] = batch_y.to(args.device)


class Mydata(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, args, split="public", num_train_per_class=20,
                 num_val=500, num_test=1000, transform=None, pre_transform=None):
        self.name = name
        self.args = args

        super(Mydata, self).__init__(root, transform, pre_transform)
        self.data, self.slices = T.load(self.processed_paths[0])

        self.split = split
        assert self.split in ['public', 'full', 'random']

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = ['train_x_array', 'test_x_array', 'train_y_array', 'test_y_array']
        return ['{}.npy'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            nothing = 0

    def process(self):

        self.data, self.slices = self.read_Mydata_data(self.raw_dir, self.name, self.args)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        T.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def read_Mydata_data(self, folder, prefix, args):

        x_array = raw_sequential_data(args, plot_data=False)
        x_array = x_array.unsqueeze(2)

        s, q = traindata_position_in_raw_data(args)
        len_train_and_test = len(s) + len(q)

        test_x_array = x_array[q, :][:, 0]
        train_x_array = x_array[s, :][:, 0]

        x_list = T.cat(list(train_x_array), dim=1).transpose(0, 1)
        xt_list = T.cat(list(test_x_array), dim=1).transpose(0, 1)
        x = T.cat((x_list, xt_list), dim=0)


        batch = np.zeros((len_train_and_test * args.n_Nodes), dtype=int)
        for i in range(len(s) + len(q)):
            batch[i * args.n_Nodes:i * args.n_Nodes + args.n_Nodes] = i
        batch = T.tensor(batch).long()

        # pos = T.zeros((args.n_Nodes, 2))
        # pos[:, 0] = T.linspace(0, args.n_Nodes-1, args.n_Nodes)
        pos = T.zeros((args.n_Nodes, 1))
        pos[:, 0] = T.linspace(0, args.n_Nodes - 1, args.n_Nodes)
        pos = T.cat([pos] * (len(s) + len(q)))

        data = Data(x=x, pos=pos)  # , y=y)
        data, slices = self.split(data, batch)
        return data, slices

    def split(self, data, batch):
        node_slice = T.cumsum(T.from_numpy(np.bincount(batch)), 0)
        # print('node_slice = ', node_slice, node_slice.shape)
        node_slice = T.cat([T.tensor([0]), node_slice])
        # print('node_slice = ', node_slice, node_slice.shape)

        # row, _ = data.edge_index
        # edge_slice = T.cumsum(T.from_numpy(np.bincount(batch[row])), 0)
        # edge_slice = T.cat([T.tensor([0]), edge_slice])
        # print('edge_slice = ', edge_slice, edge_slice.shape)

        # Edge indices should start at zero for every graph.
        # data.edge_index -= node_slice[batch[row]].unsqueeze(0)
        # print(data.edge_index)
        data.__num_nodes__ = T.bincount(batch).tolist()
        # print(data.__num_nodes__)

        # slices = {'edge_index': edge_slice}

        if data.x is not None:
            # slices['x'] = node_slice
            slices = {'x': node_slice}
        # if data.edge_attr is not None:
        #     slices['edge_attr'] = edge_slice
        if data.pos is not None:
            slices['pos'] = node_slice
        # if data.y is not None:
        #     # if data.y.size(0) == batch.size(0):
        #     slices['y'] = node_slice

        return data, slices


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
        mask = T.zeros(pos.size(0), dtype=T.bool)
        mask[index] = True
        plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
        plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.pause(1)
    plt.show()


def PlotResults(i_ls, t_ls, batch_time, args, pos_ls, edge_index_list, epoch, c, view_mode):
    """plot results

    Args:
        i_ls (tensor): [batch_time, n_Nodes * train_size_used, 1] or [batch_time, n_Nodes * test_size, 1]
        t_ls (tensor): [batch_time, n_Nodes * train_size_used, 1] or [batch_time, n_Nodes * test_size, 1]
        view_mode (int): 1. animation 2. image
    """

    srt, end = args.n_Nodes*c, args.n_Nodes*c + args.n_Nodes
    t_ls = t_ls[:, srt:end, 0]
    i_ls = i_ls[:, srt:end, 0]

    if view_mode == 1:

        if isinstance(pos_ls, list):
            if len(pos_ls[0][0, :]) == 1:
                pos_ls = [T.cat((pos_ls[i], T.zeros(pos_ls[i].shape).to(pos_ls[i].device)), dim=1) for i in
                            range(len(pos_ls))]
        else:
            pos_ls = [pos_ls]*100

        def visualize_points(pos, edge_index=None, index=None):
            # fig = plt.figure(figsize=(4, 4))
            plot3 = []
            i = 0
            if edge_index is not None:
                # for (src, dst) in T.tensor(edge_index).t().tolist():
                for (src, dst) in edge_index.t().tolist():
                    src = pos[src].tolist()
                    dst = pos[dst].tolist()
                    plot3.append(bx.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black'))
                    i = i + 1
                    # print(i)
            plot4 = bx.scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), s=50, zorder=1000)
            plt.axis('off')
            # plt.show()
            return plot3, plot4

        def update_plot(frame_number, t_ls, i_ls, pos_ls, edge_index_list, plot1, plot2):#, plot3, plot4):
            # plot.set_offsets(chng_evolution[frame_number])
            plot1[0].set_ydata(t_ls[frame_number])  # [np.nan] * len(x))
            plot2[0].set_ydata(i_ls[frame_number])
            i = 0
            # for (src, dst) in T.tensor(edge_index_list[frame_number]).t().tolist():

            # for (src, dst) in edge_index_list[frame_number].t().tolist():
            #     src = pos_ls[frame_number][src].tolist()
            #     dst = pos_ls[frame_number][dst].tolist()
            #     plot3[i][0].set_ydata([src[1], dst[1]])
            #     i = i + 1
            #     # print(i)
            #     # plot3 = bx.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
            # plot4.set_offsets(pos_ls[frame_number].cpu())

        fps = 2  # frame per sec
        frn = len(i_ls[:, 0])
        aa = np.linspace(0, args.n_Nodes-1, args.n_Nodes)
        fig = plt.figure(figsize=(4, 4))
        widths = [4]
        heights = [4]  # , 4]
        spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths,
                                 height_ratios=heights)
        # ax = fig.add_subplot(spec5[0, 0])
        bx = fig.add_subplot(spec5[0])  # 1, 0])

        # visualize_points(pos[0:21 * 21], edge_index=edge_index[:, 0:int(
        #     len(edge_index[0, :]) / (len(s) + len(q)))])

        # plot = ax.scatter(chng_evolution[0, :, 0], chng_evolution[0, :, 1], s=50, zorder=1000)
        plot1 = bx.plot(aa, t_ls[0], color='r')
        plot2 = bx.plot(aa, i_ls[0], color='b')
        # plot3, plot4 = visualize_points(pos_ls[0], edge_index_list[0])

        ani = animation.FuncAnimation(fig, update_plot, frn,
                                      fargs=(t_ls, i_ls, pos_ls, edge_index_list, plot1, plot2),#, plot3, plot4),
                                      interval=1000 / fps)

    if view_mode == 2:

        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        rc('text', usetex=False)
        # a = np.random.randn(512, 50)
        final_time = args.ts(args.skip)[0:batch_time][-1]
        tt = [0, final_time]

        # fig = plt.figure(figsize=(15, 8), dpi=150)
        fig = plt.figure()
        ax = []
        ax.append(plt.subplot2grid((3, 16), (0, 0), colspan=14))
        ax.append(plt.subplot2grid((3, 16), (1, 0), colspan=14))
        ax.append(plt.subplot2grid((3, 16), (2, 0), colspan=14))

        cmap = matplotlib.cm.get_cmap('inferno')
        c0 = ax[1].imshow(i_ls, interpolation='nearest', cmap=cmap, aspect='auto',
                          extent=[0, 1, tt[-1], tt[0]])
        c_max = np.max(i_ls)
        c_min = np.min(i_ls)
        c0.set_clim(vmin=c_min, vmax=c_max)

        c0 = ax[0].imshow(t_ls, interpolation='nearest', cmap=cmap, aspect='auto',
                          extent=[0, 1, tt[-1], tt[-0]])
        c0.set_clim(vmin=c_min, vmax=c_max)

        p0 = ax[0].get_position().get_points().flatten()
        p1 = ax[1].get_position().get_points().flatten()
        # {left, bottom, right, top}
        ax_cbar = fig.add_axes([p1[2] + 0.015, p1[1], 0.020, p0[3] - p1[1]])  # {‘top’, ‘bottom’,’left’,’right’}.
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

        c0 = ax[2].imshow(np.abs(i_ls - t_ls), interpolation='nearest', cmap=cmap, aspect='auto',
                          extent=[0, 1, tt[-1], tt[0]])
        p0 = ax[2].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[2] + 0.015, p0[1], 0.020, p0[3] - p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
        # tickLabels = ["{:.2e}".format(t0) for t0 in tickLabels]
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

        ax[0].set_ylabel('t (true)', fontsize=14)
        ax[1].set_ylabel('t (prediction)', fontsize=14)
        ax[2].set_ylabel('t (L1 error)', fontsize=14)
        ax[1].set_xlabel('x', fontsize=14)

    path = args.get_path(args, W=1, exp=args.exp)
    file_loc = osp.join(path, 'fig_c{}'.format(c) + args.add_str + '.pdf')
    plt.savefig(file_loc, format='pdf') if args.save else 0
    plt.show() if args.show else 0
    plt.close(fig)
    plt.close('all')
    # sv_f = lambda sv: ani.save('animation.gif', writer='imagemagick', fps=fps) if sv else 0
    # plt.show()
    # return sv_f


if __name__ == '__main__':

    # ---------------------------- Argument Parser -----------------------------
    parser = argparse.ArgumentParser('glance at data')
    prjct_dir = os.getcwd()
    parser.add_argument('--prjct_dir', type=str, default=prjct_dir)
    parser.add_argument('--data_type', type=str, default='pde_1d')
    parser.add_argument('--op_type', type=str, default='burgers1d')
    args = parser.parse_args()

    # -------------------------------- logger ----------------------------------
    logger = logging.getLogger('my_module')
    logger.setLevel(logging.DEBUG)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    args.logger = logger

    args.n_Nodes = 511
    args.n_Nodes_d = 511
    args.os = 'window'
    if args.op_type == 'burgers1d':
        args.max_batch_time = 2002

    # -------------------------- load and plot data ----------------------------
    raw_sequential_data(args, plot_data=1)

    # self.visualize_points(pos[0:nx * nx], edge_index=edge_index[:, 0:int(
    #             len(edge_index[0, :]) / (len(s) + len(q)))])  # 3087-7 2646-6 1764-4 3528-8

    # TargetList(0, 1, True, 5, args)
    # s,q =traindata_position_in_raw_data(args)

import os.path as osp
import os
import logging
from numpy import random
from torch._C import dtype
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
import numpy as np
import torch as T
# from scipy.interpolate import interp2d
from scipy.interpolate import Rbf
from scipy.interpolate import RegularGridInterpolator
# from scipy import interpolate
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib
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
        elif args.ConvLayer == 'GATConv2':
            dir1 = 'GATConv2'

    if hasattr(args, 'Basis'):
        if args.Basis == 'Polynomial':
            dir2 = 'Polynomial'
        elif args.Basis == 'Chebychev':
            dir2 = 'Chebychev'
        elif args.Basis == 'Fourier':
            dir2 = 'Fourier'
        elif args.Basis == 'MultiquadRBF':
            dir2 = 'MultiquadRBF'
        elif args.Basis == 'GaussianRBF':
            dir2 = 'GaussianRBF'
        elif args.Basis == 'VanillaRBF':
            dir2 = 'VanillaRBF'
        elif args.Basis == 'PiecewiseConstant':
            dir2 = 'PiecewiseConstant'
        elif args.Basis == 'None':
            dir2 = 'No_Basis'

    if args.op_type == 'burgers2d':
        dir3 = 'burgers2d'

    if kwargs.get('W', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_2d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir,
                                                                                                         'src/pde_2d/weights_results')
        if args.exp == 'lit_train__lit_test_2d':
            path = osp.join(p_, dir3, kwargs.get('exp', None), dir1, dir2, 'N_Basis{}'.format(args.N_Basis),
                            'train_size{}'.format(args.train_size_used),
                            'lit_idx_train{}'.format(args.last_intg_t_idx_train))
        else:
            path = osp.join(p_, dir3, kwargs.get('exp', None), dir1, dir2, 'N_Basis{}'.format(args.N_Basis),
                            'train_size{}'.format(args.train_size_used))

    elif kwargs.get('C1', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_2d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir,
                                                                                                         'src/pde_2d/weights_results')
        path = osp.join(p_, dir3, dir1, dir2)

    elif kwargs.get('C2', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_2d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir,
                                                                                                         'src/pde_2d/weights_results')
        if kwargs.get('rslt_dir', None):
            path = osp.join(p_, dir3, dir1, kwargs.get('rslt_dir', None))
        else:
            path = osp.join(p_, dir3, dir1)

    elif kwargs.get('C3', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_2d\weights_results') if args.os == 'window' else osp.join(args.prjct_dir,
                                                                                                         'src/pde_2d/weights_results')
        if kwargs.get('rslt_dir', None):
            path = osp.join(p_, dir3, kwargs.get('rslt_dir', None))
        else:
            path = osp.join(p_, dir3)

    elif kwargs.get('CL', None):
        path = osp.join(args.prjct_dir, 'src/pde_2d\weights_results', dir3,
                        args.exp) if args.os == 'window' else osp.join(args.prjct_dir, 'src/pde_2d/weights_results',
                                                                       dir3, args.exp)

    elif kwargs.get('F', None):
        p_ = osp.join(args.prjct_dir, 'src/pde_2d')
        path = osp.join(p_, 'torch_data')

    elif kwargs.get('R', None):
        p_ = osp.join(args.prjct_dir, 'data\pde_2d/fenics_data') if args.os == 'window' else osp.join(args.prjct_dir,
                                                                                                      'data/pde_2d/fenics_data')
        path = osp.join(p_)

    elif kwargs.get('P', None):
        path = osp.join(args.prjct_dir, 'src/pde_2d/torch_data')
        if os.path.exists(path):
            shutil.rmtree(path)
        mkm = 0

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def raw_sequential_data(args, **kwargs):
    """ Loads data present in /data folder and saves in args.raw_data. Returns data stored in args.raw_data
    after first time till args.data_type is unchanged.

    :var bool_files : list of 300 tuples (bool, file_path)

    Return:
          args.raw_data (tensor): [n_initial conditions, n_sequence generated with an initial
                                    conditions, n_components(x, y), n_Nodes(64*64)]
    """

    if hasattr(args, 'raw_data'):
        # args.logger.info(f'using previously loaded data')
        pass
    else:
        mbt = args.max_batch_time
        t_end = 1.0  # round(T.linspace(0, 1, args.max_batch_time)[-1].item(), 5)
        n_x = int(args.n_Nodes ** 0.5)
        n_y = n_x
        n__x = int(args.n_Nodes_d ** 0.5)
        n__y = n__x

        p__x = np.linspace(0, 1, n__x)
        p__y = np.linspace(0, 1, n__y)
        p_x = np.linspace(0, 1, n_x)
        p_y = np.linspace(0, 1, n_y)

        path = get_path(args, R=1)

        def file_path(path, mbt, t_end, n__x):
            return lambda run: osp.join(path, 'run{:d}'.format(run),
                                        'raw{:d}_mbt{:d}_tend{}_nx{}.npy'.format(run, mbt, t_end, n__x))

        fp = file_path(path, mbt, t_end, n__x)

        bool_files = [(osp.exists(fp(run)), fp(run)) for run in range(300)]

        assert any([bool_files[i][0] for i in range(len(bool_files))]), 'data not present, run fenics_burger2d.py ' \
                                                                        'in /data folder to get data'
        data_ = np.stack([(np.load(nm)) for bl, nm in bool_files if bl], 0)
        data = data_[0]
        args.len_data = np.arange(0, len(data_[:, 0, 0]))
        args.logger.info(f'data loaded ')
        data_ = data_.reshape((-1, 101, 2, 64, 64))[:, :, :, :63, :63].reshape((-1, 101, 2, 63 * 63))
        args.raw_data = T.tensor(data_, dtype=T.float32)

    if kwargs.get('plot_data', False):
        t_ls_x = data[:, 0, :]
        t_ls_y = data[:, 1, :]
        nx = int(args.n_Nodes ** 0.5)
        fps = 5  # frame per sec
        frn = len(data[:, 0])  # frame number of the animation

        def update_plot(frame_number, t_ls_x, t_ls_y, plot1, plot2):
            plot1.set_array(t_ls_x[frame_number].reshape((nx, nx)))
            plot2.set_array(t_ls_y[frame_number].reshape((nx, nx)))

        fig = plt.figure(figsize=(5, 6))
        widths = [2]
        heights = [2, 2]
        spec5 = fig.add_gridspec(ncols=1, nrows=2, width_ratios=widths,
                                 height_ratios=heights)
        ax = fig.add_subplot(spec5[0, 0])
        ax.set_title('data_x')
        bx = fig.add_subplot(spec5[1, 0])
        bx.set_title('data_y')

        nx = int(args.n_Nodes ** 0.5)
        plot1 = ax.imshow(t_ls_x[0].reshape((nx, nx)))
        plot2 = bx.imshow(t_ls_y[0].reshape((nx, nx)))

        ani = animation.FuncAnimation(fig, update_plot, frn,
                                      fargs=(t_ls_x, t_ls_y, plot1, plot2),
                                      interval=1000 / fps)

        # ====================================== saving data gif ======================================
        gif = 'anim{:d}_mbt{:d}_tend{}_nx{}'.format(len(data_[:, 0, 0]), mbt, t_end, n__x)
        gif_path = osp.join(path, gif)
        # ani.save(gif_path + '.gif', writer='imagemagick', fps=fps)

        plt.show()

    return args.raw_data


def uniform_pos(nx):
    a = T.ones(nx)
    b = T.arange(nx)
    c = T.stack([a] * nx, dim=0)
    d = (c * b).transpose(0, 1).reshape(-1)
    e = T.cat(([b] * nx), dim=0)
    f = T.stack((d, e), dim=1)
    # pos = T.cat([f]*len_train_and_test, dim=0)
    pos = T.cat([f] * 1, dim=0)
    return pos.numpy()


def random_pos(n_rand_nodes):
    """
    Args:
        n_rand_nodes: num of nodes randomly distributed in between domain

    Returns:
        pos: position of all nodes in domain
    """
    np.random.seed(10)
    x = np.random.random(n_rand_nodes) * 58 + 2
    y = np.random.random(n_rand_nodes) * 58 + 2
    pos_random = np.concatenate((x[:, None], y[:, None]), axis=1)

    u_pos = uniform_pos(63)
    r_ls1 = np.concatenate([np.arange(128, 187) + 63 * i for i in range(59)])
    r_ls2 = np.arange(u_pos.shape[0], u_pos.shape[0] + (x.shape[0] - r_ls1.shape[0]))
    map = np.concatenate((r_ls1, r_ls2))

    ls1 = np.arange(r_ls1.shape[0])
    ls2 = np.arange(r_ls1.shape[0], x.shape[0])
    u_pos[r_ls1] = pos_random[ls1]
    pos = np.concatenate((u_pos, pos_random[ls2]), axis=0)

    # u_pos[r_ls] = pos_random
    # pos = u_pos
    return pos.astype('float32'), pos_random.astype('float32'), map

    # pos_boundary = np.zeros((62 * 4 + 60 * 4, 2))

    # x1s = [np.arange(62), 62, np.arange(62, 0, -1), 0]
    # x2s = [np.arange(1, 61), 61, np.arange(61, 1, -1), 1]
    # y1s = [0, np.arange(62), 62, np.arange(62, 0, -1)]
    # y2s = [1, np.arange(1, 61), 61, np.arange(61, 1, -1)]
    # count = [0, 1, 2, 3]

    # for x1, y1, x2, y2, i in zip(x1s, y1s, x2s, y2s, count):
    #     srt, end = i * 122, i * 122 + 62
    #     pos_boundary[srt:end, 0] = x1
    #     pos_boundary[srt:end, 1] = y1

    #     srt, end = i * 122 + 62, i * 122 + 124 - 2
    #     pos_boundary[srt:end, 0] = x2
    #     pos_boundary[srt:end, 1] = y2
    #     # mknkl=0

    # pos = np.concatenate((pos_boundary, pos_random), axis=0)
    # pos = pos.astype('float32')
    # return pos


def intp(old_co_ordinates, old_values, new_co_ordinates):
    """
    Args:
        old_co_ordinates:
        old_values:
        new_co_ordinates:

    Returns:

    """
    # f = Rbf(old_co_ordinates[:, 0], old_co_ordinates[:, 1], old_values)
    # return f(new_co_ordinates[:, 0], new_co_ordinates[:, 1])
    f = RegularGridInterpolator((old_co_ordinates[:63, 1], old_co_ordinates[:63, 1]),
                                old_values.reshape([63, 63]).numpy())
    new_points = np.concatenate((new_co_ordinates[:, 0][:, None], new_co_ordinates[:, 1][:, None]),
                                axis=1)  # [n_points, 2]
    return f(new_points)


def scatter_3d(u_pos, u_data, r_pos, r_data):
    fig = plt.figure()
    spec5 = fig.add_gridspec(ncols=1, nrows=1)
    ax = fig.add_subplot(spec5[0, 0], projection='3d')
    ax.scatter(u_pos[:, 0], u_pos[:, 1], u_data, color='black')
    ax.scatter(r_pos[:, 0], r_pos[:, 1], r_data + 5, color='red')
    plt.show()


def scatter_data(data, nx, n_rand_nodes):
    u_pos, r_pos = uniform_pos(nx), random_pos(n_rand_nodes)
    r_pos, _, _ = r_pos
    data_r = np.zeros((data.shape[0], data.shape[1], r_pos.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_r[i, j] = intp(u_pos, data[i, j], r_pos)

    # scatter_3d(u_pos, data[0, 0], r_pos, data_r[0, 0])
    return T.from_numpy(data_r).to(dtype=T.float32)


def traindata_position_in_raw_data(args):
    # assert args.train_size % args.batch_time == 0, "train_size should be multiple of batch_time"
    # assert args.train_size < (
    #         args.n_time_steps + 1) / args.max_batch_time, "train_size should be smaller that n_time_steps/batch_time"
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
                    # args.logger.info(f'using previously processed test target data')
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
            assert idx_ls[
                       -1] <= args.max_batch_time / args.skip, 'batch_time should be smaller than args.max_batch_time'

            nx = int(args.n_Nodes ** 0.5)
            x_array = raw_sequential_data(args, plot_data=False)

            if train_bool:
                bs = args.bs
                trbs = args.train_batch_size

                if args.stencil == 'k_near' or args.stencil == 'k_nn':
                    cat_ls = []
                    for z in range(bs):
                        seq = scatter_data(x_array[:, 0:-1:args.skip][s[position * trbs + z], idx_ls, :], nx,
                                           args.n_rand_nodes)
                        cat_ls.append(seq.transpose(2, 1))
                    batch_y = T.cat(cat_ls, dim=1)
                else:
                    batch_y = T.cat(
                        [(x_array[:, 0:-1:args.skip][s[position * trbs + z], idx_ls, :]).transpose(2, 1) for z in
                         range(bs)],
                        dim=1)

                assert T.equal(batched_u[:len(x_array[0, 0])],
                               batch_y[0, :len(x_array[0, 0])]), 'starting target different from starting training data'
                self.target_tr_ls[lit_idx][position] = batch_y.to(args.device)
            else:
                bs = args.bs
                tbs = args.test_batch_size

                if args.stencil == 'k_near' or args.stencil == 'k_nn':
                    cat_ls = []
                    for z in range(bs):
                        seq = scatter_data(x_array[:, 0:-1:args.skip][q[position * tbs + z], idx_ls, :], nx,
                                           args.n_rand_nodes)
                        cat_ls.append(seq.transpose(2, 1))
                    batch_y = T.cat(cat_ls, dim=1)
                else:
                    batch_y = T.cat(
                        [(x_array[:, 0:-1:args.skip][q[position * tbs + z], idx_ls, :]).transpose(2, 1) for z in
                         range(bs)],
                        dim=1)

                assert T.equal(batched_u[:len(x_array[0, 0])],
                               batch_y[0, :len(x_array[0, 0])]), 'starting target different from starting training data'
                self.target_ts_ls[lit_idx][position] = batch_y.to(args.device)

        elif args.cont_in == 'dt':

            s, q = traindata_position_in_raw_data(args)
            skip = args.skip(args.bs)
            # assert idx_ls[-1] <= args.max_batch_time/args.skip, 'batch_time should be smaller than args.max_batch_time'

            data_ = raw_sequential_data(args, plot_data=False)

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
        # del self.data['train_mask']
        # self.data.pop('train_mask', None)

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
        s, q = traindata_position_in_raw_data(args)
        len_train_and_test = len(s) + len(q)
        nx = int(args.n_Nodes ** 0.5)

        test_x_array = x_array[q][:, 0]
        train_x_array = x_array[s][:, 0]

        if args.stencil == 'k_near':
            args.n_rand_nodes = n_rand_nodes = (63 - 4) ** 2
            test_x_array = scatter_data(test_x_array, nx, n_rand_nodes)
            train_x_array = scatter_data(train_x_array, nx, n_rand_nodes)
            pos, args.r_pos, args.map = random_pos(n_rand_nodes)
        elif args.stencil == 'k_nn':
            args.n_rand_nodes = n_rand_nodes = (80) ** 2
            test_x_array = scatter_data(test_x_array, nx, n_rand_nodes)
            train_x_array = scatter_data(train_x_array, nx, n_rand_nodes)
            pos, args.r_pos, args.map = random_pos(n_rand_nodes)
        else:
            pos = uniform_pos(nx)

        args.n_Nodes = n_Nodes = pos.shape[0]
        pos = T.cat([T.from_numpy(pos)] * len_train_and_test, dim=0)

        x_list = T.cat(list(train_x_array), dim=1).transpose(0, 1)
        xt_list = T.cat(list(test_x_array), dim=1).transpose(0, 1)

        x = T.cat((x_list, xt_list), dim=0)

        batch = np.zeros((len_train_and_test * n_Nodes), dtype=int)
        for i in range(len(s) + len(q)):
            batch[i * n_Nodes:i * n_Nodes + n_Nodes] = i
        batch = T.tensor(batch).long()

        data = Data(x=x, pos=pos)
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
        # slices['edge_attr'] = edge_slice
        if data.pos is not None:
            slices['pos'] = node_slice
        # if data.y is not None:
        #     # if data.y.size(0) == batch.size(0):
        #     slices['y'] = node_slice

        return data, slices

    def visualize_points(self, pos, edge_index=None, index=None):
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


def PlotResults(i_ls, t_ls, batch_time, args, pos_list, edge_index_list, epoch, idx, view_mode):
    nx = int(args.n_Nodes ** 0.5)

    def get_sample(idx, component):
        srt = args.n_Nodes * idx
        end = srt + args.n_Nodes
        if component == 'x':
            target, pred = t_ls[:, srt:end, 0], i_ls[:, srt:end, 0]
        elif component == 'y':
            target, pred = t_ls[:, srt:end, 1], i_ls[:, srt:end, 1]
        return pred, target

    i_ls_x, t_ls_x = get_sample(idx, 'x')
    i_ls_y, t_ls_y = get_sample(idx, 'y')

    if view_mode == 1:

        x = np.arange(-1, 1, 0.097)
        y = np.arange(-1, 1, 0.097)
        X, Y = np.meshgrid(x, y)

        fps = 5  # frame per sec
        frn = len(i_ls[:, 0])  # frame number of the animation

        def update_plot(frame_number, t_ls, i_ls, plot, plot2, plot3):
            plot[0].remove()
            plot[1].remove()
            plot2.axes.lines.remove(plot2.axes.lines[0])
            plot3.axes.lines.remove(plot3.axes.lines[0])
            plot[0] = ax.plot_surface(X, Y, t_ls[frame_number].reshape((nx, nx)),
                                      cmap="magma")
            plot[1] = ax.plot_surface(X, Y, i_ls[frame_number].reshape((nx, nx)),
                                      cmap="viridis", alpha=0.3)
            plot2 = bx.plot(x, t_ls[frame_number].reshape((nx, nx))[10, :],
                            color='0.3')  # cmap(ys1[10]))
            plot3 = bx.plot(x, i_ls[frame_number].reshape((nx, nx))[10, :],
                            color='0.75')  # cmap(ys2[10]))

        fig = plt.figure(figsize=(5, 6))
        widths = [2]
        heights = [2, 1]
        spec5 = fig.add_gridspec(ncols=1, nrows=2, width_ratios=widths,
                                 height_ratios=heights)
        ax = fig.add_subplot(spec5[0, 0], projection='3d')
        bx = fig.add_subplot(spec5[1, 0])
        bx.set_title('black-true, grey-estimated')

        Ys = Y * 0
        Ys[:, :] = .3
        Ys1 = Ys
        Ys2 = Ys
        Ys1[10] = 1
        Ys2[10] = .4
        ys1 = Ys1[:, 0]
        ys2 = Ys2[:, 0]
        cmap = cm.magma

        nx = args.n_Nodes
        plot = [
            ax.plot_surface(X, Y, t_ls[0].reshape((nx, nx)), color='0.75', rstride=1,
                            cstride=1, facecolors=cmap(Ys1)),
            ax.plot_surface(X, Y, i_ls[0].reshape((nx, nx)), color='0.75', rstride=1,
                            cstride=1, facecolors=cmap(Ys2))]
        # ax.set_zlim(0, 1.1)
        plot2, = bx.plot(x, t_ls[0].reshape((nx, nx))[10, :], color=cmap(ys1[10]))
        plot3, = bx.plot(x, i_ls[0].reshape((nx, nx))[10, :], color=cmap(ys2[10]))
        ani = animation.FuncAnimation(fig, update_plot, frn,
                                      fargs=(t_ls, i_ls, plot, plot2, plot3),
                                      interval=1000 / fps)
        fn = 'plot_anim'
        ani.save(fn + '.gif', writer='imagemagick', fps=fps)
        print('saved', epoch)

    elif view_mode == 2:
        def get_slice(nx, c):
            """ Get slice """
            return lambda ls: ls.reshape(-1, nx, nx)[:, :, c]

        gs = get_slice(nx, 32)

        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        rc('text', usetex=False)
        rc('font', size=8)
        # a = np.random.randn(512, 50)
        final_time = args.ts(args.skip)[0:batch_time][-1]
        tt = [0, final_time]
        tick_font_size = 5

        # fig = plt.figure(figsize=(15, 8), dpi=150)
        fig = plt.figure()
        ax = []
        ax.append(plt.subplot2grid((3, 36), (0, 0), colspan=13))
        ax.append(plt.subplot2grid((3, 36), (1, 0), colspan=13))
        ax.append(plt.subplot2grid((3, 36), (2, 0), colspan=13))
        ax.append(plt.subplot2grid((3, 36), (0, 20), colspan=13))
        ax.append(plt.subplot2grid((3, 36), (1, 20), colspan=13))
        ax.append(plt.subplot2grid((3, 36), (2, 20), colspan=13))

        c_max = np.max(i_ls_x)
        c_min = np.min(i_ls_x)
        cmap = matplotlib.cm.get_cmap('inferno')

        c0 = ax[1].imshow(gs(i_ls_x), interpolation='nearest', cmap=cmap, aspect='auto', extent=[0, 1, tt[-1], tt[0]])
        c0.set_clim(vmin=c_min, vmax=c_max)
        c0 = ax[0].imshow(gs(t_ls_x), interpolation='nearest', cmap=cmap, aspect='auto', extent=[0, 1, tt[-1], tt[0]])
        c0.set_clim(vmin=c_min, vmax=c_max)

        p0 = ax[0].get_position().get_points().flatten()
        p1 = ax[1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2] + 0.015, p1[1], 0.020, p0[3] - p1[1]])  # {left, bottom, right, top}
        ticks = np.linspace(0, 1, 5)
        tick_labels = np.linspace(c_min, c_max, 5)
        tick_labels = ["{:02.2f}".format(t0) for t0 in tick_labels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tick_labels)
        # cbar.tick_params(right=False)

        # cmap = "viridis"
        c0 = ax[2].imshow(np.abs(gs(i_ls_x) - gs(t_ls_x)), interpolation='nearest', cmap=cmap, aspect='auto',
                          extent=[0, 1, tt[-1], tt[0]])
        p0 = ax[2].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[2] + 0.015, p0[1], 0.020, p0[3] - p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
        # tickLabels = ["{:.2e}".format(t0) for t0 in tickLabels]
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)
        # cbar.ax[2].tick_params(labelsize=tick_font_size)

        ax[0].set_ylabel('t (true)', fontsize=14)
        ax[1].set_ylabel('t (prediction)', fontsize=14)
        ax[2].set_ylabel('t (L1 error)', fontsize=14)
        ax[1].set_xlabel('x', fontsize=14)

        # ---------------------------------------------------------

        c_max = np.max(i_ls_y)
        c_min = np.min(i_ls_y)
        cmap = matplotlib.cm.get_cmap('inferno')

        c0 = ax[4].imshow(gs(i_ls_y), interpolation='nearest', cmap=cmap, aspect='auto', extent=[0, 1, tt[-1], tt[0]])
        c0.set_clim(vmin=c_min, vmax=c_max)
        c0 = ax[3].imshow(gs(t_ls_y), interpolation='nearest', cmap=cmap, aspect='auto', extent=[0, 1, tt[-1], tt[0]])
        c0.set_clim(vmin=c_min, vmax=c_max)

        p0 = ax[3].get_position().get_points().flatten()
        p1 = ax[4].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2] + 0.015, p1[1], 0.020, p0[3] - p1[1]])  # {left, bottom, right, top}
        ticks = np.linspace(0, 1, 5)
        tick_labels = np.linspace(c_min, c_max, 5)
        tick_labels = ["{:02.2f}".format(t0) for t0 in tick_labels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tick_labels)

        c0 = ax[5].imshow(np.abs(gs(i_ls_y) - gs(t_ls_y)), interpolation='nearest', cmap=cmap, aspect='auto',
                          extent=[0, 1, tt[-1], tt[0]])
        p0 = ax[5].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[2] + 0.015, p0[1], 0.020, p0[3] - p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
        # tickLabels = ["{:.2e}".format(t0) for t0 in tickLabels]
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)
        # cbar.ax[2].tick_params(labelsize=tick_font_size)

        ax[4].set_xlabel('x', fontsize=14)
        ax[3].tick_params(labelleft=False)
        ax[4].tick_params(labelleft=False)
        ax[5].tick_params(labelleft=False)

        name = 'fig_im{}'.format(idx)
        path = args.get_path(args, W=1, exp=args.exp)
        file_loc = osp.join(path, name + args.add_str + '.pdf')
        plt.savefig(file_loc, format='pdf') if args.save else 0
        plt.show() if args.show else 0
        plt.close(fig)
        plt.close('all')

    elif view_mode == 3:
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        rc('text', usetex=False)
        rc('font', size=8)
        cmap = matplotlib.cm.get_cmap('inferno')  # "plasma"
        cmap_error = matplotlib.cm.get_cmap('inferno')  # "virdis"
        target_steps = pred_steps = np.array(range(0, 30, 5))

        nx = 63
        uTarget, uPred = np.stack((t_ls_x, t_ls_y), axis=1).reshape((-1, 2, nx, nx)), np.stack((i_ls_x, i_ls_y),
                                                                                               axis=1).reshape(
            (-1, 2, nx, nx))
        target, prediction = uTarget[target_steps], uPred[pred_steps]
        error = np.abs(prediction - target)

        fig, ax = plt.subplots(6, len(pred_steps), figsize=(len(pred_steps) * 3, 15))
        fig.subplots_adjust(wspace=0.5)

        for i in range(len(pred_steps)):
            for j in range(2):

                c_max = np.max(np.array([target[i, j], prediction[i, j]]))
                c_min = np.min(np.array([target[i, j], prediction[i, j]]))
                ax[3 * j, i].imshow(target[i, j], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto',
                                    extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)
                ax[3 * j + 1, i].imshow(prediction[i, j], interpolation='nearest', cmap=cmap, origin='lower',
                                        aspect='auto', extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)

                ax[3 * j + 2, i].imshow(error[i, j], interpolation='nearest', cmap=cmap_error, origin='lower',
                                        aspect='auto', extent=[0, 1, 0, 1])
                c_max_error = np.max(error[i, j])
                c_min_error = np.min(error[i, j])

                p0 = ax[3 * j, i].get_position().get_points().flatten()
                p1 = ax[3 * j + 1, i].get_position().get_points().flatten()
                ax_cbar = fig.add_axes([p1[2] + 0.0075, p1[1], 0.005, p0[3] - p1[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min, c_max, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                cmap
                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
                cbar.set_ticklabels(tickLabels)

                p0 = ax[3 * j + 2, i].get_position().get_points().flatten()
                ax_cbar = fig.add_axes([p0[2] + 0.0075, p0[1], 0.005, p0[3] - p0[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min_error, c_max_error, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap_error), orientation='vertical',
                                                 ticks=ticks)
                cbar.set_ticklabels(tickLabels)

                for ax0 in ax[:-1, i]:
                    ax0.set_xticklabels([])

                for ax0 in ax[:, i]:
                    ax0.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                    ax0.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                    if (i > 0):
                        ax0.set_yticklabels([])

                if i == 0:
                    ax[3 * j, i].set_ylabel('y (true)', fontsize=14)
                    ax[3 * j + 1, i].set_ylabel('y (prediction)', fontsize=14)
                    ax[3 * j + 2, i].set_ylabel('y (L1 error)', fontsize=14)

            ax[0, i].set_title('t={:.02f}'.format(args.ts(args.skip)[pred_steps[i]]), fontsize=14)
            ax[-1, i].set_xlabel('x', fontsize=14)

        name = name = 'plot2D'
        path = args.get_path(args, W=1, exp=args.exp)
        file_loc = osp.join(path, name + args.add_str + '.pdf')
        plt.savefig(file_loc, format='pdf') if args.save else 0
        plt.show() if args.show else 0
        plt.close(fig)
        plt.close('all')

    elif view_mode == 4:

        nx = int(args.n_Nodes ** 0.5)
        fps = 5  # frame per sec
        frn = len(i_ls[:, 0])  # frame number of the animation

        def update_plot(frame_number, t_ls, i_ls, plot1, plot2, plot3, plot4):
            plot1.set_array(t_ls_x[frame_number].reshape((nx, nx)))
            plot2.set_array(t_ls_y[frame_number].reshape((nx, nx)))
            plot3.set_array(t_ls_x[frame_number].reshape((nx, nx)))
            plot4.set_array(i_ls_y[frame_number].reshape((nx, nx)))

        fig = plt.figure(figsize=(5, 6))
        widths = [2, 2]
        heights = [2, 2]
        spec5 = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                                 height_ratios=heights)
        ax = fig.add_subplot(spec5[0, 0])
        ax.set_title('target_x')
        bx = fig.add_subplot(spec5[1, 0])
        bx.set_title('target_y')
        cx = fig.add_subplot(spec5[0, 1])
        cx.set_title('pred_x')
        dx = fig.add_subplot(spec5[1, 1])
        dx.set_title('pred_y')

        nx = int(args.n_Nodes ** 0.5)
        plot1 = ax.imshow(t_ls_x[0].reshape((nx, nx)))
        plot2 = bx.imshow(t_ls_y[0].reshape((nx, nx)))
        plot3 = cx.imshow(i_ls_x[0].reshape((nx, nx)))
        plot4 = dx.imshow(i_ls_y[0].reshape((nx, nx)))

        ani = animation.FuncAnimation(fig, update_plot, frn,
                                      fargs=(t_ls, i_ls, plot1, plot2, plot3, plot4),
                                      interval=1000 / fps)
        fn = 'result_animation_epoch{:d}'.format(epoch)

    # sv_f = lambda sv: ani.save('animation.gif', writer='imagemagick', fps=fps) if sv else 0


if __name__ == '__main__':

    # ---------------------------- Argument Parser -----------------------------
    parser = argparse.ArgumentParser('glance at data')
    prjct_dir = os.getcwd()
    parser.add_argument('--prjct_dir', type=str, default=prjct_dir)
    parser.add_argument('--data_type', type=str, default='pde_2d')
    parser.add_argument('--op_type', type=str, default='burgers2d')
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

    args.n_Nodes = 64 * 64
    args.n_Nodes_d = 128 * 128
    args.os = 'window'
    if args.op_type == 'burgers2d':
        args.max_batch_time = 101

    # -------------------------- load and plot data ----------------------------
    raw_sequential_data(args, plot_data=1)

    # TargetList(0, 1, True, 5, args)
    # s,q =traindata_position_in_raw_data(args)
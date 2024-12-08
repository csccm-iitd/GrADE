import numpy as np
from torch_cluster import knn_graph, knn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import random
import math
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from torch import linalg as LA
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import matplotlib.pyplot as plt
from torch_geometric.nn.inits import glorot, zeros, uniform
from Basis import *

import warnings
from typing import Union, Tuple, List
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import torch as T
from torch_geometric.utils.repeat import repeat
from torch_spline_conv import spline_basis, spline_weighting
import os.path as osp
import itertools
from scipy import ndimage


# from src.pde_2d.Dataset import random_pos

def mm(input, w, b):
    w.T
    result = [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*w)] for A_row in input]
    return result + b
    # return T.tensor(input.detach().cpu().numpy() @ w.T.detach().cpu().numpy() + b.detach().cpu().numpy())


class PointNetLayer(MessagePassing):
    # def __init__(self, bias=True, basisfunc=Fourier(5), dilation=True, shift=True):
    #     super().__init__()
    #     self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
    #     self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
    #     self.basisfunc = basisfunc
    #     self.n_eig = n_eig = self.basisfunc.n_eig
    #     self.deg = deg = self.basisfunc.deg

    def __init__(self, args, in_features, out_features, layer_num, agmnt_featur,
                 bias=True, basisfunc=Fourier(5), dilation=True, shift=True):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('mean')

        # if
        self.args = args
        self.device = args.device
        self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
        self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
        self.basisfunc = basisfunc
        self.n_eig = n_eig = self.basisfunc.n_eig
        self.deg = deg = self.basisfunc.deg
        self.in_features, self.out_features = in_features, out_features
        self.agmnt_featur = agmnt_featur
        self.layer_num = layer_num

        if hasattr(args, 'hidden'):
            self.hidden = args.hidden
        else:
            if self.args.n_linear_layers > 1:
                assert 1 == 0, 'number of Neurones in hidden layer is required'
            else:
                self.hidden = out_features
        # if hidden is None:
        #     self.hidden = self.out_features  # out_features
        # else:
        #     assert 1 == 0, 'number of neuron in hidden layers is required'
        self.weight = T.Tensor(out_features, in_features)
        if bias:
            self.bias = T.Tensor(out_features)
        else:
            self.register_parameter('bias', None)

        self.activation = nn.Tanh() if args.PINN else nn.LeakyReLU(0.2)
        self.coeffs = nn.ParameterDict(
            {'1': T.nn.Parameter(T.Tensor((in_features + 1) * self.hidden, self.deg, self.n_eig))})
        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                self.coeffs['{:d}'.format(c + 2)] = T.nn.Parameter(
                    T.Tensor((self.hidden + 1) * self.hidden, self.deg, self.n_eig))
            else:
                self.coeffs['{:d}'.format(c + 2)] = T.nn.Parameter(
                    T.Tensor((self.hidden + 1) * out_features, self.deg, self.n_eig))

        # self.coeffs1 = T.nn.Parameter(T.Tensor((in_features + 1) * self.hidden, self.deg, self.n_eig))
        # self.coeffs2 = T.nn.Parameter(T.Tensor((self.hidden + 1) * out_features, self.deg, self.n_eig))
        self.reset_parameters()

    # def __init__(self, time_d, in_features, out_features, N_Basis):
    #     # Message passing with "max" aggregation.
    #     super(PointNetLayer, self).__init__('add')
    #     self.time_d = time_d
    #     self.in_features = in_features
    #     self.out_features = out_features
    #
    #     # Initialization of the MLP:
    #     # Here, the number of input features correspond to the hidden node
    #     # dimensionality plus point dimensionality (=3).
    #     # self.mlp = Sequential(Linear(in_features * 2, out_features),
    #     #                       ReLU(),
    #     #                       Linear(out_features, out_features))
    #
    #     # self.weight = T.nn.Parameter(T.randn(3, out_features, in_features, width, width))
    #     # self.bias = T.nn.Parameter(T.zeros(3, out_features))
    #     self.N_Basis = N_Basis
    #     self.weight = T.nn.Parameter(T.randn(N_Basis, out_features, in_features * 2))
    #     self.bias = T.nn.Parameter(T.zeros(N_Basis, out_features))
    #     kl = 0

    def reset_parameters(self):
        if self.args.PINN:
            T.nn.init.uniform_(self.coeffs['{:d}'.format(1)], a=1.0, b=2.0)
            for c in range(self.args.n_linear_layers - 1):
                T.nn.init.uniform_(self.coeffs['{:d}'.format(c + 2)], a=1.0, b=2.0)
        else:
            T.nn.init.uniform_(self.coeffs['{:d}'.format(1)], a=0.0, b=0.05)
            for c in range(self.args.n_linear_layers - 1):
                T.nn.init.uniform_(self.coeffs['{:d}'.format(c + 2)], a=0.0, b=0.05)
        # T.nn.init.normal_(self.coeffs['{:d}'.format(1)])
        # for c in range(self.args.n_linear_layers - 1):
        #     T.nn.init.normal_(self.coeffs['{:d}'.format(c + 2)])

    def calculate_weights(self, s, coeffs):
        "Expands `s` following the chosen eigenbasis"
        n_range = T.linspace(0, self.deg, self.deg).to(self.args.device)
        basis = self.basisfunc(n_range, s * self.dilation.to(self.args.device) + self.shift.to(self.args.device))
        B = []
        for i in range(self.n_eig):
            Bin = T.eye(self.deg).to(self.args.device)
            Bin[range(self.deg), range(self.deg)] = basis[i]
            B.append(Bin)
        B = T.cat(B, 1).to(self.args.device)
        coeffss = T.cat([coeffs[:, :, i] for i in range(self.n_eig)], 1).transpose(0, 1).to(self.args.device)
        X = T.matmul(B, coeffss)
        return X.sum(0)

    def forward(self, h, edge_index, pos, edge_attr, t):
        # Start propagating messages.
        # print('edge_index ==',edge_index, 'h==',h)
        assert not T.isnan(h).any()
        return self.propagate(edge_index, pos=pos, edge_attr=edge_attr, h=h,
                              t=t)  # pos=pos, edge_attr=edge_attr, h=h, t=t)

    # def forward(self, input):
    #     # For the moment, GalLayers rely on DepthCat to access the `s` variable. A better design would free the user
    #     # of having to introduce DepthCat(1) every time a GalLayer is used
    #     s = input[-1, -1]
    #     input = input[:, :-1]
    #     w = self.calculate_weights(s)
    #     self.weight = w[0:self.in_features * self.out_features].reshape(self.out_features, self.in_features)
    #     self.bias = w[self.in_features * self.out_features:(self.in_features + 1) * self.out_features].reshape(
    #         self.out_features)
    #     return F.linear(input, self.weight, self.bias)

    def message(self, h_j, h_i, pos_j, pos_i, t):

        s = 1 if self.args.Basis == 'None' else t
        input = self.agmnt_featur(self.layer_num, h_j, h_i, pos_j, pos_i)

        w1 = self.calculate_weights(s, self.coeffs['{:d}'.format(1)])
        self.weight1 = w1[0:self.in_features * self.hidden].reshape(self.hidden, self.in_features)
        self.bias1 = w1[self.in_features * self.hidden:(self.in_features + 1) * self.hidden].reshape(
            self.hidden)
        valu1 = F.linear(input, self.weight1, self.bias1)

        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                valu1 = self.activation(valu1)
                w2 = self.calculate_weights(s, self.coeffs['{:d}'.format(c + 2)])
                self.weight2 = w2[0:self.hidden * self.hidden].reshape(self.hidden, self.hidden)
                self.bias2 = w2[self.hidden * self.hidden:(self.hidden + 1) * self.hidden].reshape(
                    self.hidden)
                valu1 = F.linear(valu1, self.weight2, self.bias2)
            else:
                valu1 = self.activation(valu1)
                w2 = self.calculate_weights(s, self.coeffs['{:d}'.format(c + 2)])
                self.weight2 = w2[0:self.hidden * self.out_features].reshape(self.out_features, self.hidden)
                self.bias2 = w2[self.hidden * self.out_features:(self.hidden + 1) * self.out_features].reshape(
                    self.out_features)
                valu1 = F.linear(valu1, self.weight2, self.bias2)
        # val1 = valu1.activation()
        # w2 = self.calculate_weights(s, self.coeffs2)
        # self.weight2 = w2[0:self.hidden * self.out_features].reshape(self.out_features, self.hidden)
        # self.bias2 = w2[self.hidden * self.out_features:(self.hidden + 1) * self.out_features].reshape(
        #     self.out_features)
        # valu2 = F.linear(valu1, self.weight2, self.bias2)

        return valu1
        # return input


class GATConv2(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_features (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_features (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    # def __init__(self, in_features, out_features, bias=True, basisfunc=Fourier(5), dilation=True, shift=True):
    #     # Message passing with "max" aggregation.
    #     super(PointNetLayer, self).__init__('add')
    #
    #     self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
    #     self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
    #     self.basisfunc = basisfunc
    #     self.n_eig = n_eig = self.basisfunc.n_eig
    #     self.deg = deg = self.basisfunc.deg
    #
    #     in_features = in_features*2
    #     self.in_features, self.out_features = in_features, out_features
    #     self.weight = T.Tensor(out_features, in_features)
    #     if bias:
    #         self.bias = T.Tensor(out_features)
    #     else:
    #         self.register_parameter('bias', None)
    #     self.coeffs = T.nn.Parameter(T.Tensor((in_features + 1) * out_features, self.deg, self.n_eig))
    #     self.reset_parameters()

    def __init__(self, args, in_features: Union[int, Tuple[int, int]],
                 out_features: int, layer_num, agmnt_featur, heads: int = 1, concat: bool = True,
                 basisfunc=Fourier(5), dilation=True, shift=True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = False, bias: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv2, self).__init__(node_dim=0, **kwargs)

        self.args = args
        if isinstance(in_features, tuple):
            self.in_features1, self.in_features2 = in_features[0], in_features[1]
            self.in_features = in_features[1]
            self.sep = True
        else:
            self.in_features = self.in_features1 = self.in_features2 = in_features
            self.sep = False

        self.hidden = args.hidden
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.agmnt_featur = agmnt_featur
        self.layer_num = layer_num
        self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
        self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
        self.basisfunc = basisfunc
        self.n_eig = n_eig = self.basisfunc.n_eig
        self.deg = deg = self.basisfunc.deg

        # self.weight = T.Tensor(out_features, in_features)
        # if bias:
        #     self.bias = T.Tensor(out_features)
        # else:
        #     self.register_parameter('bias', None)

        I1, I2, H, C, D = self.in_features1, self.in_features2, self.heads, self.out_features, self.hidden
        # self.coeffs1 = Parameter(T.Tensor((I1 + 1) * H * C, self.deg, self.n_eig))
        # self.coeffs2 = Parameter(T.Tensor((I2 + 1) * H * C, self.deg, self.n_eig)) if self.sep else 0
        self.activation = nn.Tanh() if args.PINN else nn.LeakyReLU(0.2)
        self.coeffs1 = nn.ParameterDict(
            {'1': T.nn.Parameter(T.Tensor((I1 + 1) * D, self.deg, self.n_eig))})
        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                self.coeffs1['{:d}'.format(c + 2)] = T.nn.Parameter(
                    T.Tensor((D + 1) * D, self.deg, self.n_eig))
            else:
                self.coeffs1['{:d}'.format(c + 2)] = T.nn.Parameter(
                    T.Tensor((D + 1) * H * C, self.deg, self.n_eig))

        if self.sep:
            self.coeffs2 = nn.ParameterDict(
                {'1': T.nn.Parameter(T.Tensor((I2 + 1) * D, self.deg, self.n_eig))})
            for c in range(self.args.n_linear_layers - 1):
                if c + 2 < self.args.n_linear_layers:
                    self.coeffs2['{:d}'.format(c + 2)] = T.nn.Parameter(
                        T.Tensor((D + 1) * D, self.deg, self.n_eig))
                else:
                    self.coeffs2['{:d}'.format(c + 2)] = T.nn.Parameter(
                        T.Tensor((D + 1) * H * C, self.deg, self.n_eig))

        # self.coeffs = Parameter(T.Tensor((in_features) * heads * out_features, self.deg, self.n_eig))

        # if isinstance(in_features, int):
        #     self.lin_l = Linear(in_features, heads * out_features, bias=False)
        #     self.lin_r = self.lin_l
        # else:
        #     self.lin_l = Linear(in_features[0], heads * out_features, False)
        #     self.lin_r = Linear(in_features[1], heads * out_features, False)

        # self.att_l = Parameter(T.Tensor(1, heads, out_features))
        # self.att_r = Parameter(T.Tensor(1, heads, out_features))

        # self.att_l_weight = T.Tensor(1, heads, out_features)
        self.att_l = Parameter(T.Tensor(1 * H * C, self.deg, self.n_eig))
        self.att_r = Parameter(T.Tensor(1 * H * C, self.deg, self.n_eig))

        if bias and concat:
            self.bias = Parameter(T.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = Parameter(T.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        T.nn.init.uniform_(self.coeffs1['{:d}'.format(1)], a=0.0, b=0.05)
        for c in range(self.args.n_linear_layers - 1):
            T.nn.init.uniform_(self.coeffs1['{:d}'.format(c + 2)], a=0.0, b=0.05)
        if self.sep:
            T.nn.init.uniform_(self.coeffs2['{:d}'.format(1)], a=0.0, b=0.05)
            for c in range(self.args.n_linear_layers - 1):
                T.nn.init.uniform_(self.coeffs2['{:d}'.format(c + 2)], a=0.0, b=0.05)

        # T.nn.init.normal_(self.coeffs1['{:d}'.format(1)])
        # for c in range(self.args.n_linear_layers - 1):
        #     T.nn.init.normal_(self.coeffs1['{:d}'.format(c + 2)])
        # if self.sep:
        #     T.nn.init.normal_(self.coeffs2['{:d}'.format(1)])
        #     for c in range(self.args.n_linear_layers - 1):
        #         T.nn.init.normal_(self.coeffs2['{:d}'.format(c + 2)])

        # T.nn.init.normal_(self.att_l_weight)
        # T.nn.init.normal_(self.att_r_weight)

        # glorot(self.lin_l.weight)
        # glorot(self.lin_r.weight)  # TODO: change
        # glorot(self.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def calculate_weights(self, s, coeffs):
        "Expands `s` following the chosen eigenbasis"

        n_range = T.linspace(0, self.deg, self.deg).to(self.args.device)
        basis = self.basisfunc(n_range, s * self.dilation.to(self.args.device) + self.shift.to(self.args.device))
        B = []
        for i in range(self.n_eig):
            Bin = T.eye(self.deg).to(self.args.device)
            Bin[range(self.deg), range(self.deg)] = basis[i]
            B.append(Bin)
        B = T.cat(B, 1).to(self.args.device)
        coeffs = T.cat([coeffs[:, :, i] for i in range(self.n_eig)], 1).transpose(0, 1).to(self.args.device)
        X = T.matmul(B, coeffs)
        return X.sum(0)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, pos, edge_attr, t,
                size: Size = None, return_attention_weights=None):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights_results for each edge. (default: :obj:`None`)
        """
        assert not T.isnan(x).any()
        s = t
        I1, I2, H, C = self.in_features1, self.in_features2, self.heads, self.out_features

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, s=t, size=size)
        # out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i, pos_j, pos_i, s,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        I1, I2, H, C, D = self.in_features1, self.in_features2, self.heads, self.out_features, self.hidden
        x1, x2 = self.agmnt_featur(self.layer_num, x_j, x_i, pos_j, pos_i)

        w = self.calculate_weights(s, self.coeffs1['{:d}'.format(1)])
        weight = w[0:I1 * D].reshape(D, I1)
        biass = w[I1 * D:(I1 + 1) * D].reshape(D)
        x_l = F.linear(x1, weight, biass)

        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                x_l = self.activation(x_l)
                w = self.calculate_weights(s, self.coeffs1['{:d}'.format(c + 2)])
                weight = w[0:D * D].reshape(D, D)
                bias = w[D * D:(D + 1) * D].reshape(D)
                x_l = F.linear(x_l, weight, bias)
            else:
                x_l = self.activation(x_l)
                w = self.calculate_weights(s, self.coeffs1['{:d}'.format(c + 2)])
                weight = w[0:D * H * C].reshape(H * C, D)
                bias = w[D * H * C:(D + 1) * H * C].reshape(H * C)
                x_l = F.linear(x_l, weight, bias).view(-1, H, C)

        att_l = self.calculate_weights(s, self.att_l)
        att_l_weight = att_l.reshape(1, H, C)
        alpha_l = (x_l * att_l_weight).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        alpha = alpha_l  # if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha1 = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha1
        alpha2 = F.dropout(alpha1, p=self.dropout, training=self.training)

        # w = self.calculate_weights(s, self.coeffs2) if self.sep else self.calculate_weights(s, self.coeffs1)
        # weight = w[0:I2 * H * C].reshape(H * C, I2)
        # biass = w[I2 * H * C:(I2 + 1) * H * C].reshape(H * C)
        # x_x = F.linear(x2, weight, biass).view(-1, H, C)

        coeffs2 = self.coeffs2 if self.sep else self.coeffs1
        w = self.calculate_weights(s, coeffs2['{:d}'.format(1)])
        weight = w[0:I2 * D].reshape(D, I2)
        biass = w[I2 * D:(I2 + 1) * D].reshape(D)
        x_x = F.linear(x2, weight, biass)

        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                x_x = self.activation(x_x)
                w = self.calculate_weights(s, self.coeffs1['{:d}'.format(c + 2)])
                weight = w[0:D * D].reshape(D, D)
                bias = w[D * D:(D + 1) * D].reshape(D)
                x_x = F.linear(x_x, weight, bias)
            else:
                x_x = self.activation(x_x)
                w = self.calculate_weights(s, self.coeffs1['{:d}'.format(c + 2)])
                weight = w[0:D * H * C].reshape(H * C, D)
                bias = w[D * H * C:(D + 1) * H * C].reshape(H * C)
                x_x = F.linear(x_x, weight, bias).view(-1, H, C)

        bb = x_x * alpha2.unsqueeze(-1)
        return bb

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_features,
                                             self.out_features, self.heads)


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_features (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_features (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, args, in_features: Union[int, Tuple[int, int]], out_features: int,
                 layer_num, agmnt_featur, heads: int = 1, concat: bool = True,
                 basisfunc=Fourier(5), dilation=True, shift=True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = False, bias: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
        self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
        self.basisfunc = basisfunc
        self.n_eig = n_eig = self.basisfunc.n_eig
        self.deg = deg = self.basisfunc.deg
        self.agmnt_featur = agmnt_featur
        self.layer_num = layer_num
        if hasattr(args, 'hidden'):
            self.hidden = args.hidden
        else:
            if self.args.n_linear_layers > 1:
                assert 1 == 0, 'number of Neurones in hidden layer is required'
            else:
                self.hidden = out_features

        I, C, D = self.in_features, self.out_features, self.hidden

        if args.PINN:
            # self.activation = lambda x, b=-0.1: F.softplus(x, beta=1) - (F.softplus(x, beta=b)) * b
            self.activation = lambda x: T.pow(x * ((x > 0).int() * 2 - 1), 5 / 7) * ((x > 0).int() * 2 - 1)
            # self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(0.2)

        # self.coeffs = Parameter(T.Tensor((in_features + 1) * heads * out_features, self.deg, self.n_eig))
        self.coeffs = nn.ParameterDict(
            {'1': T.nn.Parameter(T.Tensor((I + 1) * D, self.deg, self.n_eig))})
        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                self.coeffs['{:d}'.format(c + 2)] = T.nn.Parameter(T.Tensor((D + 1) * D, self.deg, self.n_eig))
            else:
                self.coeffs['{:d}'.format(c + 2)] = T.nn.Parameter(T.Tensor((D + 1) * C, self.deg, self.n_eig))

        if bias and concat:
            self.bias = Parameter(T.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = Parameter(T.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        # pass
        T.manual_seed(self.args.seed)
        if self.args.data_type == 'pde_1d':
            if self.args.PINN:
                glorot(self.coeffs['{:d}'.format(1)])
                for c in range(self.args.n_linear_layers - 1):
                    glorot(self.coeffs['{:d}'.format(c + 2)])

                # T.nn.init.uniform_(self.coeffs['{:d}'.format(1)], a=0.1, b=0.3)
                # for c in range(self.args.n_linear_layers - 1):
                #     T.nn.init.uniform_(self.coeffs['{:d}'.format(c + 2)], a=0.1, b=0.3)
            else:
                T.nn.init.uniform_(self.coeffs['{:d}'.format(1)], a=0.0, b=0.005)
                for c in range(self.args.n_linear_layers - 1):
                    T.nn.init.uniform_(self.coeffs['{:d}'.format(c + 2)], a=0.0, b=0.005)
            # T.nn.init.normal_(self.coeffs['{:d}'.format(1)])
            # for c in range(self.args.n_linear_layers - 1):
            #     T.nn.init.normal_(self.coeffs['{:d}'.format(c + 2)])
        elif self.args.data_type == 'pde_2d':
            T.nn.init.uniform_(self.coeffs['{:d}'.format(1)], a=0.0, b=0.05)
            for c in range(self.args.n_linear_layers - 1):
                T.nn.init.uniform_(self.coeffs['{:d}'.format(c + 2)], a=0.0, b=0.05)

    def calculate_weights(self, s, coeffs):
        "Expands `s` following the chosen eigenbasis"

        n_range = T.linspace(0, self.deg, self.deg).to(self.args.device)
        basis = self.basisfunc(n_range, s * self.dilation.to(self.args.device) + self.shift.to(self.args.device))
        B = []
        for i in range(self.n_eig):
            Bin = T.eye(self.deg).to(self.args.device)
            Bin[range(self.deg), range(self.deg)] = basis[i]
            B.append(Bin)
        B = T.cat(B, 1).to(self.args.device)
        coeffs = T.cat([coeffs[:, :, i] for i in range(self.n_eig)], 1).transpose(0, 1).to(self.args.device)
        X = T.matmul(B, coeffs)
        return X.sum(0)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, pos, edge_attr, t,
                size: Size = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights_results for each edge. (default: :obj:`None`)
        """

        assert not T.isnan(x).any()
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, t=t, size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out1 = out.view(-1, self.heads * self.out_features)
        else:
            out1 = out.mean(dim=1)

        if self.bias is not None:
            out1 += self.bias

        return out1

    def message(self, x_j: Tensor, x_i, pos_j, pos_i, t,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        s = 1 if self.args.Basis == 'None' else t
        I, C, D = self.in_features, self.out_features, self.hidden
        inpt, xx = self.agmnt_featur(self.layer_num, x_j, x_i, pos_j, pos_i)

        w1 = self.calculate_weights(s, self.coeffs['{:d}'.format(1)])
        weight = w1[0:I * D].reshape(D, I)
        bias = w1[I * D:(I + 1) * D].reshape(D)

        valu = F.linear(inpt, weight, bias)
        # valu = mm(inpt, weight, bias)

        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                valu = self.activation(valu)
                w = self.calculate_weights(s, self.coeffs['{:d}'.format(c + 2)])
                weight = w[0:D * D].reshape(D, D)
                bias = w[D * D:(D + 1) * D].reshape(D)
                valu = F.linear(valu, weight, bias)
            else:
                valu = self.activation(valu)
                w = self.calculate_weights(s, self.coeffs['{:d}'.format(c + 2)])
                weight = w[0:D * C].reshape(C, D)
                bias = w[D * C:(D + 1) * C].reshape(C)
                valu = F.linear(valu, weight, bias)

        bb = xx * valu
        assert not T.isnan(bb).any()
        return bb

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_features,
                                             self.out_features, self.heads)


class SpiderConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_features (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_features (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    # def __init__(self, in_features, out_features, bias=True, basisfunc=Fourier(5), dilation=True, shift=True):
    #     # Message passing with "max" aggregation.
    #     super(PointNetLayer, self).__init__('add')
    #
    #     self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
    #     self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
    #     self.basisfunc = basisfunc
    #     self.n_eig = n_eig = self.basisfunc.n_eig
    #     self.deg = deg = self.basisfunc.deg
    #
    #     in_features = in_features*2
    #     self.in_features, self.out_features = in_features, out_features
    #     self.weight = T.Tensor(out_features, in_features)
    #     if bias:
    #         self.bias = T.Tensor(out_features)
    #     else:
    #         self.register_parameter('bias', None)
    #     self.coeffs = T.nn.Parameter(T.Tensor((in_features + 1) * out_features, self.deg, self.n_eig))
    #     self.reset_parameters()

    def __init__(self, args, in_features: Union[int, Tuple[int, int]], out_features: int,
                 layer_num, agmnt_featur, heads: int = 1, concat: bool = True,
                 basisfunc=Fourier(5), dilation=True, shift=True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SpiderConv, self).__init__(node_dim=0, **kwargs)

        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
        self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
        self.basisfunc = basisfunc
        self.n_eig = n_eig = self.basisfunc.n_eig
        self.deg = deg = self.basisfunc.deg
        self.agmnt_featur = agmnt_featur
        self.layer_num = layer_num

        if hasattr(args, 'hidden'):
            self.hidden = args.hidden
        else:
            if self.args.n_linear_layers > 1:
                assert 1 == 0, 'number of Neurones in hidden layer is required'
            else:
                self.hidden = out_features

        # weight for spider conv
        # self.coeffs = T.nn.Parameter(T.Tensor(in_features * out_features+2, self.deg, self.n_eig))
        self.coeffs = nn.ParameterDict(
            {'1': T.nn.Parameter(T.Tensor(in_features * self.hidden + 2, self.deg, self.n_eig))})

        assert self.args.n_linear_layers == 1, 'multiple layer not supported in Spider conv'
        for c in range(self.args.n_linear_layers - 1):
            if c + 2 < self.args.n_linear_layers:
                self.coeffs['{:d}'.format(c + 2)] = T.nn.Parameter(
                    T.Tensor(self.hidden * self.hidden + 2, self.deg, self.n_eig))
            else:
                self.coeffs['{:d}'.format(c + 2)] = T.nn.Parameter(
                    T.Tensor(self.hidden * out_features + 2, self.deg, self.n_eig))

        if bias and concat:
            self.bias = Parameter(T.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = Parameter(T.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        # T.nn.init.uniform_(self.coeffs, a=0.0, b=0.05)
        T.manual_seed(self.args.seed)
        if self.args.data_type == 'pde_1d':
            T.nn.init.uniform_(self.coeffs['{:d}'.format(1)], a=0.0, b=0.005)
            for c in range(self.args.n_linear_layers - 1):
                T.nn.init.uniform_(self.coeffs['{:d}'.format(c + 2)], a=0.0, b=0.005)
            # T.nn.init.normal_(self.coeffs['{:d}'.format(1)])
            # for c in range(self.args.n_linear_layers - 1):
            #     T.nn.init.normal_(self.coeffs['{:d}'.format(c + 2)])
        elif self.args.data_type == 'pde_2d':
            T.nn.init.uniform_(self.coeffs['{:d}'.format(1)], a=0.0, b=0.05)
            for c in range(self.args.n_linear_layers - 1):
                T.nn.init.uniform_(self.coeffs['{:d}'.format(c + 2)], a=0.0, b=0.05)

        # T.nn.init.normal_(self.coeffs)  # TODO: change
        # T.nn.init.normal_(self.att_l_weight)
        # T.nn.init.normal_(self.att_r_weight)

        # glorot(self.lin_l.weight)
        # glorot(self.lin_r.weight)  # TODO: change
        # glorot(self.weight)
        zeros(self.bias)

    def calculate_weights(self, s, coeffs):
        "Expands `s` following the chosen eigenbasis"

        n_range = T.linspace(0, self.deg, self.deg).to(self.args.device)
        basis = self.basisfunc(n_range, s * self.dilation.to(self.args.device) + self.shift.to(self.args.device))
        B = []
        for i in range(self.n_eig):
            Bin = T.eye(self.deg).to(self.args.device)
            Bin[range(self.deg), range(self.deg)] = basis[i]
            B.append(Bin)
        B = T.cat(B, 1).to(self.args.device)
        coeffs = T.cat([coeffs[:, :, i] for i in range(self.n_eig)], 1).transpose(0, 1).to(self.args.device)
        X = T.matmul(B, coeffs)
        return X.sum(0)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, pos, edge_attr, t,
                size: Size = None, return_attention_weights=None):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights_results for each edge. (default: :obj:`None`)
        """

        assert not T.isnan(x).any()
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, t=t, size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out1 = out.view(-1, self.heads * self.out_features)
        else:
            out1 = out.mean(dim=1)

        if self.bias is not None:
            out1 += self.bias

        return out1

    def message(self, x_j: Tensor, x_i, pos_j, pos_i, t,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        s = 1 if self.args.Basis == 'None' else t
        H, Hd, I, C = self.heads, self.hidden, self.in_features, self.out_features
        inpt, xx = self.agmnt_featur(self.layer_num, x_j, x_i, pos_j, pos_i)

        # w = self.calculate_weights(s, self.coeffs)
        # self.weight = w[0:self.in_features * H * C].reshape(H * C, self.in_features)
        # self.biass = w[self.in_features * H * C:(self.in_features + 1) * H * C].reshape(H * C)
        # x_l = x_r = F.linear(inpt, self.weight, self.biass).view(-1, H, C)

        nmjk = 0
        # weight = (self.calculate_weights(s, self.coeffs))
        # w = weight[0:I * C].reshape(I, C)
        # ww = weight[I * C]
        # bb = weight[I * C + 1]

        f = lambda a: math.factorial(a)

        def bases(x, y=None):
            if y is None:
                # one = T.ones(x.shape).to(self.args.device)
                # asd = T.stack((one, x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7), dim=1)
                rr = 10
                stkd = T.stack([x ** r / f(r) for r in range(rr)], dim=1)
                return stkd
            else:
                lst = []
                r = 3
                for i in range(r):
                    for j in range(r - 1):
                        lst.append(x ** i * y ** j / f(i) / f(j))
                return T.stack(lst, dim=1)

        if len(inpt[0]) == 2:
            w = self.calculate_weights(s, self.coeffs['{:d}'.format(1)])
            weight = w[0:I * Hd].reshape(I, Hd)
            ww = w[I * Hd]
            bb = w[I * Hd + 1]
            x, y = inpt[:, 0] * ww + bb, inpt[:, 1] * ww + bb
            # x, y = inpt[:, 0], inpt[:, 1]
            one = T.ones(x.shape).to(self.args.device)
            bases_ = bases(x, y)
            # bases_ = T.stack((one, x, y, x * y, x ** 2, y ** 2, x * y ** 2, y * x ** 2, x ** 3, y ** 3), dim=1)
            valu1 = T.mm(bases_, weight)

        elif len(inpt[0]) == 1:
            w = self.calculate_weights(s, self.coeffs['{:d}'.format(1)])
            weight = w[0:I * Hd].reshape(I, Hd)
            ww = w[I * Hd]
            bb = w[I * Hd + 1]
            bases_ = bases(x=inpt[:, 0] * ww + bb)
            valu1 = T.mm(bases_, weight)

            for c in range(self.args.n_linear_layers - 1):
                if c + 2 < self.args.n_linear_layers:
                    valu1 = self.activation(valu1)
                    w = self.calculate_weights(s, self.coeffs['{:d}'.format(c + 2)])
                    weight = w[0:Hd * Hd].reshape(Hd, Hd)
                    ww = w[Hd * Hd]
                    bb = w[Hd * Hd + 1]
                    bases_ = bases(x=valu1 * ww + bb)
                    valu1 = T.mm(bases_, weight)
                else:
                    valu1 = self.activation(valu1)
                    w = self.calculate_weights(s, self.coeffs['{:d}'.format(c + 2)])
                    weight = w[0:Hd * C].reshape(Hd, C)
                    ww = w[Hd * C]
                    bb = w[Hd * C + 1]
                    bases_ = bases(x=valu1 * ww + bb)
                    valu1 = T.mm(bases_, weight)

        x_l = valu1

        assert x_l is not None
        # assert alpha_l is not None

        # alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        # alpha = F.leaky_relu(alpha_l, self.negative_slope)
        # alpha1 = softmax(alpha, index, ptr, size_i)
        # self._alpha = alpha1
        # alpha2 = F.dropout(alpha1, p=self.dropout, training=self.training)
        # bb = x_j * alpha2.unsqueeze(-1)
        # bb = xx * alpha_l.sum(dim=1)
        bb = xx * x_l
        return bb

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_features,
                                             self.out_features, self.heads)


class SplineConv(MessagePassing):
    r"""The spline-based convolutional operator from the `"SplineCNN: Fast
    Geometric Deep Learning with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in
        \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a kernel function defined
    over the weighted B-Spline tensor product basis.

    .. note::

        Pseudo-coordinates must lay in the fixed interval :math:`[0, 1]` for
        this method to work as intended.

    Args:
        in_features (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_features (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Size of the convolving kernel.
        is_open_spline (bool or [bool], optional): If set to :obj:`False`, the
            operator will use a closed B-spline basis in this dimension.
            (default :obj:`True`)
        degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, args,
                 in_features: Union[int, Tuple[int, int]],
                 out_features: int,
                 dim: int,
                 kernel_size: Union[int, List[int]],
                 agmnt_featur,
                 basisfunc=Fourier(5),
                 is_open_spline: bool = True,
                 degree: int = 1,
                 aggr: str = 'mean',
                 root_weight: bool = False,
                 bias: bool = False,
                 dilation=True, shift=True,
                 **kwargs):  # yapf: disable
        super(SplineConv, self).__init__(aggr=aggr, **kwargs)

        if spline_basis is None:
            raise ImportError('`SplineConv` requires `T-spline-conv`.')

        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.degree = degree
        self.agmnt_featur = agmnt_featur
        self.args = args
        self.basisfunc = basisfunc
        self.n_eig = n_eig = self.basisfunc.n_eig
        self.deg = deg = self.basisfunc.deg
        self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
        self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)

        kernel_size = T.tensor(repeat(kernel_size, dim), dtype=T.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = T.tensor(is_open_spline, dtype=T.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        K = kernel_size.prod().item()
        if isinstance(in_features, int):
            in_features = (in_features, in_features)
            self.I = in_features
            self.O = out_features
            self.K = K

        self.weight = Parameter(T.Tensor(K * in_features[0] * out_features, self.deg, self.n_eig))

        if root_weight:
            self.root = Parameter(T.Tensor(in_features[1] * out_features, self.deg, self.n_eig))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(T.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # size = self.weight.size(0) * self.weight.size(1)
        # uniform(size, self.weight)
        # uniform(size, self.root)
        T.nn.init.normal_(self.weight)
        if self.root is not None:
            T.nn.init.normal_(self.root)
        # zeros(self.weight)
        # zeros(self.root))
        zeros(self.bias)

    def calculate_weights(self, s, coeffs):
        "Expands `s` following the chosen eigenbasis"
        n_range = T.linspace(0, self.deg, self.deg).to(self.args.device)
        basis = self.basisfunc(n_range, s * self.dilation.to(self.args.device) + self.shift.to(self.args.device))
        B = []
        for i in range(self.n_eig):
            Bin = T.eye(self.deg).to(self.args.device)
            Bin[range(self.deg), range(self.deg)] = basis[i]
            B.append(Bin)
        B = T.cat(B, 1).to(self.args.device)
        coeffss = T.cat([coeffs[:, :, i] for i in range(self.n_eig)], 1).transpose(0, 1).to(self.args.device)
        X = T.matmul(B, coeffss)
        return X.sum(0)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, pos, t,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        assert not T.isnan(x).any()
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if not x[0].is_cuda:
            warnings.warn(
                'We do not recommend using the non-optimized CPU version of '
                '`SplineConv`. If possible, please move your data to GPU.')

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, pos=pos, x=x, edge_attr=edge_attr, t=t, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            self.r = (self.calculate_weights(t, self.root)).reshape(self.I[1], self.O)
            out += T.matmul(x_r, self.r)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, x_i, pos_j, pos_i, t, edge_attr: Tensor) -> Tensor:
        s = 1 if self.args.Basis == 'None' else t
        x, edge_attr = self.agmnt_featur(layer_num=1, pos_j=pos_j, pos_i=pos_j, h_j=x_j, h_i=x_i)
        # print(edge_attr.size(1))
        # print('kernel_size', self.kernel_size.numel())
        # pseudo.size(1) == kernel_size.numel()
        data = spline_basis(edge_attr, self.kernel_size, self.is_open_spline,
                            self.degree)

        self.w = (self.calculate_weights(s, self.weight)).reshape(self.K, self.I[0], self.O)
        rtrn = spline_weighting(x, self.w, *data)
        return rtrn

    def __repr__(self):
        return '{}({}, {}, dim={})'.format(self.__class__.__name__,
                                           self.in_features, self.out_features,
                                           self.dim)


class linear(T.nn.Module):
    def __init__(self, args, in_features, out_features,
                 basisfunc=Fourier(5), dilation=True, shift=True):
        super(linear, self).__init__()
        self.args = args
        self.in_features, self.out_features = in_features, out_features
        self.t_ = 1
        self.dilation = T.ones(1) if not dilation else nn.Parameter(data=T.ones(1), requires_grad=True)
        self.shift = T.zeros(1) if not shift else nn.Parameter(data=T.zeros(1), requires_grad=True)
        self.basisfunc = basisfunc
        self.n_eig = n_eig = self.basisfunc.n_eig
        self.deg = deg = self.basisfunc.deg
        I, C = self.in_features, self.out_features = in_features, out_features
        if args.PINN:
            # self.activation = lambda x, b=-0.1: F.softplus(x, beta=1) - (F.softplus(x, beta=b)) * b
            self.activation = lambda x: T.pow(x * ((x > 0).int() * 2 - 1), 5 / 7) * ((x > 0).int() * 2 - 1)
            # self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(0.2)
        self.coeffs = Parameter(T.Tensor((I + 1) * C, self.deg, self.n_eig))
        self.reset_parameters()

    def reset_parameters(self):
        # pass
        if self.args.data_type == 'pde_1d':
            if self.args.PINN:
                T.nn.init.uniform_(self.coeffs, a=0.1, b=0.8)
            else:
                # T.nn.init.normal_(self.coeffs)
                T.nn.init.uniform_(self.coeffs, a=0, b=0.05)
        elif self.args.data_type == 'pde_2d':
            T.nn.init.uniform_(self.coeffs, a=0.0, b=0.05)

    def calculate_weights(self, s, coeffs):
        "Expands `s` following the chosen eigenbasis"

        n_range = T.linspace(0, self.deg, self.deg).to(self.args.device)
        basis = self.basisfunc(n_range, s * self.dilation.to(self.args.device) + self.shift.to(self.args.device))
        B = []
        for i in range(self.n_eig):
            Bin = T.eye(self.deg).to(self.args.device)
            Bin[range(self.deg), range(self.deg)] = basis[i]
            B.append(Bin)
        B = T.cat(B, 1).to(self.args.device)
        coeffs = T.cat([coeffs[:, :, i] for i in range(self.n_eig)], 1).transpose(0, 1).to(self.args.device)
        X = T.matmul(B, coeffs)
        return X.sum(0)

    def forward(self, h_):
        assert not T.isnan(h_).any()
        h = h_
        t = self.t_
        # h = h_[0]
        # t = h_[1]
        # t = self.t_
        # t = kwargs.get('t', -1)
        # assert not t == -1
        I, C = self.in_features, self.out_features

        w = self.calculate_weights(t, self.coeffs)
        weight = w[0:I * C].reshape(C, I)
        bias = w[I * C:(I + 1) * C].reshape(C)
        out = F.linear(h, weight, bias)
        assert not T.isnan(out).any()
        return out


class Net(T.nn.Module):

    def __init__(self, args, in_features, out_features):
        super(Net, self).__init__()

        self.r = 3
        self.args = args
        self.del_t = 1
        self.epoch = 0
        self.batch_prev = T.tensor([0]).to(args.device)
        self.bs_prev = -1
        self.device = self.args.device
        T.manual_seed(12345)
        if args.PINN:
            # self.activation = lambda x, b=-0.1: F.softplus(x, beta=1) - (F.softplus(x, beta=b)) * b
            self.activation = lambda x: T.pow(x * ((x > 0).int() * 2 - 1), 5 / 7) * ((x > 0).int() * 2 - 1)
            # self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(0.2)
        self.Layer(args, in_features, out_features)

        self.u_t_pred_ls = []
        self.edge_index_list = []

    def polar(self, x_y):
        r = ((x_y[:, 0]) ** 2 + (x_y[:, 1]) ** 2) ** 0.5
        theta = T.atan(x_y[:, 1] / x_y[:, 0])
        r_theta = T.cat((r[:, None], theta[:, None]), dim=1)
        return r_theta

    def c_pos(self, pos_j):
        """ Change value of positions of nodes connected to nodes on boundary of domain
        Args:
            pos_j (tensor): stacked positions of nodes connected to every node
                            i.e. :math:`[[p_{j_1}^{i_1}x,y], [p_{j_2}^{i_1}x,y],
                            [p_{j_1}^{i_2}x,y], [p_{j_2}^{i_2}x,y],...
                            [p_{j_1}^{i_N}x,y, p_{j_2}^{i_N}x,y]]`
                            shape [n_edges * batch_size, co-ordinates(2 i.e.(x, y))]
        """

        with T.no_grad():
            if self.args.data_type == 'pde_1d':
                ln = len(pos_j[:, 0]) // self.args.bs_
                for d in range(0, len(pos_j[:, 0]), ln):
                    for n0 in [0, 1, self.args.n_Nodes - 2, self.args.n_Nodes - 1]:
                        # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
                        ok = T.tensor([n0 - 1, n0 + 1, n0 - 2, n0 + 2]).unsqueeze(1)
                        pos_j[4 * n0 + d:4 * n0 + d + 4] = ok

                    # for n0 in [0]:
                    #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
                    #     ok = T.tensor([n0 + 1, n0 - 1, n0 + 2, n0 - 2]).unsqueeze(1)
                    #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
                    #
                    # for n0 in [1]:
                    #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
                    #     ok = T.tensor([n0 - 1, n0 + 1, n0 + 2, n0 - 2]).unsqueeze(1)
                    #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
                    #
                    # for n0 in [self.args.n_Nodes - 1]:
                    #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
                    #     ok = T.tensor([n0 - 1, n0 + 1, n0 - 2, n0 + 2]).unsqueeze(1)
                    #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
                    #
                    # for n0 in [self.args.n_Nodes - 2]:
                    #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
                    #     ok = T.tensor([n0 + 1, n0 - 1, n0 - 2, n0 + 2]).unsqueeze(1)
                    #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
                    #
                    #     # ok = T.tensor([n0 + 1, n0 - 1, n0 + 2, n0 - 2, n0 + 3, n0 - 3]).unsqueeze(1)
                    #     # pos_j[6 * n0:6 * n0 + 6] = ok

            elif self.args.data_type == 'pde_2d':

                #  compute pos_j once and use same until batch size changes
                # print('cpos_called', ' bs:', self.args.bs_, ' bs_prev:', self.bs_prev, ' len:', len(pos_j[:, 0]))
                if hasattr(self.args, 'pos_j') and self.args.bs_ == self.bs_prev:
                    pos_j = self.args.pos_j
                else:
                    if self.args.stencil == 5:
                        ln = len(pos_j[:, 0]) // self.args.bs_
                        for d in range(0, len(pos_j[:, 0]), ln):
                            loop = [range(0, 4032, 64), range(4032, 4095),
                                    range(4095, 63, -64), range(63, 0, -1)]
                            for i in loop:
                                for n in i:
                                    pos = T.ones(4, 2)
                                    n0 = (n - n % 64) // 64
                                    n1 = n % 64
                                    pos[:, 0] = T.tensor([n0 - 1, n0, n0, n0 + 1])
                                    pos[:, 1] = T.tensor([n1, n1 - 1, n1 + 1, n1])
                                    pos_j[4 * n + d:4 * n + d + 4, :] = pos

                    elif self.args.stencil == 9:
                        ln = len(pos_j[:, 0]) // self.args.bs_
                        for d in range(0, len(pos_j[:, 0]), ln):
                            loop = [range(0, 3906, 63), range(3906, 3968, 1), range(3968, 62, -63), range(62, 0, -1),
                                    range(64, 3844, 63), range(3844, 3904, 1), range(3904, 124, -63),
                                    range(124, 64, -1)]
                            for i in loop:
                                for n in i:
                                    # l, u, d, r, ll, uu, dd, rr
                                    pos = T.ones(8, 2)
                                    n0 = (n - n % 63) // 63
                                    n1 = n % 63
                                    pos[:, 0] = T.tensor([n0 - 1, n0, n0, n0 + 1, n0 - 2, n0, n0, n0 + 2])
                                    pos[:, 1] = T.tensor([n1, n1 - 1, n1 + 1, n1, n1, n1 - 2, n1 + 2, n1])
                                    pos_j[8 * n + d:8 * n + d + 8, :] = pos

                    elif self.args.stencil == 'star':
                        ln = len(pos_j[:, 0]) // self.args.bs_
                        for d in range(0, len(pos_j[:, 0]), ln):
                            loop = [range(0, 3906, 63), range(3906, 3968),
                                    range(3968, 62, -63), range(62, 0, -1)]
                            for i in loop:
                                for n in i:
                                    # l, d, u, r, ld, lu, rd, ru
                                    pos = T.ones(8, 2)
                                    n0 = (n - n % 63) // 63
                                    n1 = n % 63
                                    pos[:, 0] = T.tensor([n0 - 1, n0, n0, n0 + 1, n0 - 1, n0 - 1, n0 + 1, n0 + 1])
                                    pos[:, 1] = T.tensor([n1, n1 - 1, n1 + 1, n1, n1 - 1, n1 + 1, n1 - 1, n1 + 1])
                                    pos_j[8 * n + d:8 * n + d + 8, :] = pos

                    elif self.args.stencil == 'k_near':
                        ln = len(pos_j[:, 0]) // self.args.bs_
                        for d in range(0, len(pos_j[:, 0]), ln):
                            loop = [range(0, 3906, 63), range(3906, 3968),
                                    range(3968, 62, -63), range(62, 0, -1)]
                            for i in loop:
                                for n in i:
                                    # l, d, u, r, ld, lu, rd, ru
                                    pos = T.ones(8, 2)
                                    n0 = (n - n % 63) // 63
                                    n1 = n % 63
                                    pos[:, 0] = T.tensor([n0 - 1, n0, n0, n0 + 1, n0 - 1, n0 - 1, n0 + 1, n0 + 1])
                                    pos[:, 1] = T.tensor([n1, n1 - 1, n1 + 1, n1, n1 - 1, n1 + 1, n1 - 1, n1 + 1])
                                    pos_j[8 * n + d:8 * n + d + 8, :] = pos

                    elif self.args.stencil == 'k_nn':
                        ln = len(pos_j[:, 0]) // self.args.bs_
                        for d in range(0, len(pos_j[:, 0]), ln):
                            loop = [range(0, 3906, 63), range(3906, 3968),
                                    range(3968, 62, -63), range(62, 0, -1)]
                            count = 0
                            for i in loop:
                                for n in i:
                                    # l, d, u, r, ld, lu, rd, ru
                                    pos = T.ones(8, 2)
                                    n0 = (n - n % 63) // 63
                                    n1 = n % 63
                                    pos[:, 0] = T.tensor([n0 - 1, n0, n0, n0 + 1, n0 - 1, n0 - 1, n0 + 1, n0 + 1])
                                    pos[:, 1] = T.tensor([n1, n1 - 1, n1 + 1, n1, n1 - 1, n1 + 1, n1 - 1, n1 + 1])
                                    pos_j[8 * count + d:8 * count + d + 8, :] = pos
                                    count += 1

                    self.args.pos_j = pos_j
                    self.bs_prev = self.args.bs_

            return pos_j

    def agmnt_featur(self, layer_num, h_j, h_i, pos_j, pos_i):
        # diff = h_j - h_i

        if self.args.data_type == 'pde_1d':

            if self.args.ConvLayer == 'PointNetLayer':
                if layer_num == 1 and h_j is not None:
                    pos_jj = self.c_pos(pos_j)
                    Rpos = pos_jj - pos_i
                    input = T.cat([diff / Rpos], dim=-1)
                    assert not T.isnan(input).any()

                if layer_num == 2 and h_j is not None:
                    pos_jj = self.c_pos(pos_j)
                    Rpos = pos_jj - pos_i
                    input = T.cat([diff / Rpos], dim=-1)

                input = input.float()
                return input

            elif self.args.ConvLayer == 'GATConv':

                if layer_num == 1 and h_j is not None:
                    input = self.c_pos(pos_j)
                    xx = h_j - h_i

                if layer_num >= 2 and h_j is not None:
                    input = self.c_pos(pos_j)
                    xx = h_j - h_i

                if self.args.PINN:
                    # pass
                    # input = input - pos_i.detach()
                    # input = input - pos_i
                    input = input / 512
                else:
                    # pass
                    # input = input / 512
                    input = input - pos_i

                return input, xx

            elif self.args.ConvLayer == 'SpiderConv':

                if layer_num == 1 and h_j is not None:
                    input = self.c_pos(pos_j)
                    xx = h_j

                if layer_num >= 2 and h_j is not None:
                    input = self.c_pos(pos_j)
                    xx = h_j

                if self.args.PINN:
                    # pass
                    # input = input - pos_i.detach()
                    # input = input - pos_i
                    input = input / 512
                else:
                    # pass
                    # input = input / 512 * 30
                    input = input - pos_i

                return input, xx

            elif self.args.ConvLayer == 'SplineConv':

                if layer_num == 1 and h_j is not None:
                    pos_jj = self.c_pos(pos_j)
                    Rpos = pos_jj - pos_i
                    input = Rpos
                    xx = h_j

                if layer_num == 2 and h_j is not None:
                    pos_jj = self.c_pos(pos_j)
                    Rpos = pos_jj - pos_i
                    input = Rpos
                    xx = h_j

                input = input.float()
                return input, xx

        elif self.args.data_type == 'pde_2d':

            if self.args.ConvLayer == 'PointNetLayer':
                if layer_num == 1 and h_j is not None:
                    Rpos = self.c_pos(pos_j) - pos_i
                    # d = T.stack([(Rpos[:, 0] ** 2 + Rpos[:, 1] ** 2) ** 1] * len(h_j[0]), dim=1)
                    # Lpos = T.fliplr(Rpos)
                    # input = T.cat([h_j * Rpos / d, h_j * Lpos / d], dim=-1)
                    input = T.cat([h_j, Rpos], dim=-1)

                if layer_num == 2 and h_j is not None:
                    Rpos = self.c_pos(pos_j) - pos_i
                    # d = T.stack([(Rpos[:, 0] ** 2 + Rpos[:, 1] ** 2) ** 1] * 4, dim=1)
                    # Rpos = T.cat([Rpos] * 2, dim=1)
                    # Lpos = T.fliplr(Rpos)
                    # input = T.cat([h_j * Rpos / d, h_j * Lpos / d], dim=-1)
                    input = T.cat([h_j, Rpos, Rpos], dim=-1)

                assert not T.isnan(input).any()
                input = input.float()
                return input

            elif self.args.ConvLayer == 'GATConv':
                if layer_num == 1 and h_j is not None:
                    pos_in = self.c_pos(pos_j) - pos_i
                    xx = T.cat([h_j - h_i, h_j - h_i], dim=-1)
                    # xx = T.cat([h_j, h_j], dim=-1)

                elif layer_num == 2 and h_j is not None:
                    pos_in = self.c_pos(pos_j) - pos_i
                    xx = T.cat([h_j - h_i, h_j - h_i], dim=-1)
                    # xx = T.cat([h_j, h_j], dim=-1)

                # pos_in = self.polar(pos_in)
                return pos_in, xx

            elif self.args.ConvLayer == 'GATConv2':
                # if layer_num == 1 and h_j is not None:
                #     pos_jj = self.c_pos(pos_j)
                #     Rpos = pos_jj - pos_i
                #     input = Rpos
                #     xx = h_j
                #
                # elif layer_num == 2 and h_j is not None:
                #     pos_jj = self.c_pos(pos_j)
                #     Rpos = pos_jj - pos_i
                #     input = Rpos
                #     xx = h_j
                #
                # input = input.float()
                return pos_j, h_j

            elif self.args.ConvLayer == 'SpiderConv':
                if layer_num == 1 and h_j is not None:
                    input = self.c_pos(pos_j) - pos_i
                    xx = T.cat([h_j - h_i, h_j - h_i], dim=-1)
                    # xx = T.cat([h_j, h_j], dim=-1)

                if layer_num == 2 and h_j is not None:
                    input = self.c_pos(pos_j) - pos_i
                    xx = T.cat([h_j - h_i, h_j - h_i], dim=-1)
                    # xx = T.cat([h_j, h_j], dim=-1)

                return input, xx

            elif self.args.ConvLayer == 'SplineConv':

                if layer_num == 1 and h_j is not None:
                    pos_jj = self.c_pos(pos_j)
                    Rpos = pos_jj - pos_i
                    input = Rpos

                if layer_num == 2 and h_j is not None:
                    pos_jj = self.c_pos(pos_j)
                    Rpos = pos_jj - pos_i
                    input = Rpos

                input = input.float()
                return h_j, input

        elif self.args.data_type == 'TrajectoryExtrapolation':

            if self.args.ConvLayer == 'PointNetLayer':
                if h_j is not None:
                    input = T.cat([h_j, diff], dim=-1)
                    input = input.float()
                    return input

            elif self.args.ConvLayer == 'GATConv2':
                if h_j is not None:
                    input = T.cat([h_j, diff], dim=-1)
                    input = input.float()
                    return input, input

    def Layer(self, args, in_features, out_features):
        """
        Initialize layers of net. Make dictionary(self.conv) of these layers.
        Args:
            in_features (int): in features of net
            out_features (int): out feature of net
        """

        assert hasattr(args, 'conv_hidden_dim'), 'number of Neurones in hidden layer is required'
        assert isinstance(args.conv_hidden_dim, list)
        if len(args.conv_hidden_dim) > 1:
            self.conv_hidden_dim1 = args.conv_hidden_dim[0]
            self.conv_hidden_dim2 = args.conv_hidden_dim[1]
        else:
            self.conv_hidden_dim1 = args.conv_hidden_dim[0]

        dl = sf = False if args.Basis == 'Polynomial' or args.Basis == 'None' else True
        self.basisfunc = self.Basis(args)
        if hasattr(args, 'hidden'):
            del args.hidden

        lin = lambda in_, out_: linear(args, in_, out_, basisfunc=self.basisfunc, dilation=dl, shift=sf)
        PN = lambda in_, out_, layer_idx: PointNetLayer(args, in_, out_, layer_idx,
                                                        self.agmnt_featur, basisfunc=self.basisfunc,
                                                        dilation=dl, shift=sf)
        GT = lambda in_, out_, layer_idx, heads=1: GATConv(args, in_, out_, layer_idx,
                                                           self.agmnt_featur, heads=heads, basisfunc=self.basisfunc,
                                                           dilation=dl, shift=sf)
        GT2 = lambda in_, out_, layer_idx, heads=1: GATConv2(args, in_, out_, layer_idx,
                                                             self.agmnt_featur, heads=heads, basisfunc=self.basisfunc,
                                                             dilation=dl, shift=sf)
        SC = lambda in_, out_, layer_idx: SpiderConv(args, in_, out_, layer_idx,
                                                     self.agmnt_featur, heads=1, basisfunc=self.basisfunc,
                                                     dilation=dl, shift=sf)
        SP = lambda in_, out_, dim, kernel_size, layer_idx: SplineConv(args, in_, out_, dim, kernel_size,
                                                                       self.agmnt_featur, basisfunc=self.basisfunc,
                                                                       dilation=dl, shift=sf)

        def layers_dict(layers):
            self.conv = nn.ModuleDict({'1': layers[0]})
            for j in range(len(layers) - 1):
                self.conv['{:d}'.format(j + 2)] = layers[j + 1]

        if self.args.data_type == 'pde_1d':

            if args.ConvLayer == 'PointNetLayer':
                args.n_linear_layers = 1

                l1 = PN(in_features, self.conv_hidden_dim1, 1)
                l2 = PN(self.conv_hidden_dim1, self.conv_hidden_dim2, 2)
                l3 = lin(3, 32)
                l4 = lin(32, out_features)
                layers_dict((l1, l2, l3, l4))

                # debugging =====================================
                # l1 = PN(in_features, self.conv_hidden_dim1, 1)
                # l2 = lin(2, 32)
                # l3 = lin(32, out_features)
                # layers_dict((l1, l2, l3))
                # debugging =====================================

            elif args.ConvLayer == 'GATConv':
                args.n_linear_layers = 2
                args.hidden = 32

                l1 = GT(1, self.conv_hidden_dim1, 1)
                l2 = GT(1, self.conv_hidden_dim2, 2)
                l3 = lin(4, 32)
                l4 = lin(32, out_features)
                layers_dict((l1, l2, l3, l4))

                # debugging =====================================
                # l1 = GT(1, self.conv_hidden_dim1, 1)
                # l2 = lin(2, 32)
                # l3 = lin(32, out_features)
                # layers_dict((l1, l2, l3))

                # l1 = GT(1, self.conv_hidden_dim1, 1)
                # l2 = lin(2, 1)
                # layers_dict((l1, l2))

                # args.n_linear_layers = 2
                # args.hidden = 32
                # l1 = GT(1, 1, 1)
                # layers_dict([l1])

                # l1 = GT(1, self.conv_hidden_dim1, 1)
                # l2 = GT(1, self.conv_hidden_dim2, 2)
                # l3 = GT(1, self.conv_hidden_dim2, 2)
                # l4 = GT(1, self.conv_hidden_dim2, 2)
                # l5 = lin(3, 32)
                # l6 = lin(32, out_features)
                # layers_dict((l1, l2, l3, l4, l5, l6))
                # debugging =====================================

            elif args.ConvLayer == 'SpiderConv':
                args.n_linear_layers = 1

                l1 = SC(10, self.conv_hidden_dim1, 1)
                l2 = SC(10, self.conv_hidden_dim2, 2)
                l3 = lin(4, 32)
                l4 = lin(32, out_features)
                layers_dict((l1, l2, l3, l4))

            elif args.ConvLayer == 'SplineConv':
                l1 = SP(in_features, self.conv_hidden_dim1, 1, [5], 1)
                l2 = SP(self.conv_hidden_dim1, self.conv_hidden_dim2, 1, [5], 2)
                l3 = lin(self.conv_hidden_dim2 * 2 + 1, 32)
                l4 = lin(32, out_features)
                layers_dict((l1, l2, l3, l4))

        elif self.args.data_type == 'pde_2d':

            if args.ConvLayer == 'PointNetLayer':
                args.n_linear_layers = 2
                args.hidden = 128

                # l1 = PN(in_features*2, self.conv_hidden_dim1, 1)
                # l2 = PN(self.conv_hidden_dim1*2, self.conv_hidden_dim2, 2)
                # l3 = lin(self.conv_hidden_dim2+6, 32)
                # l4 = lin(32, out_features)
                # layers_dict((l1, l2, l3, l4))

                l1 = PN(in_features + 2, self.conv_hidden_dim1, 1)
                l2 = PN(self.conv_hidden_dim1 + 4, self.conv_hidden_dim2, 2)
                l3 = lin(2 * 3 + self.conv_hidden_dim1 * 3 + self.conv_hidden_dim2, 64)
                l4 = lin(64, out_features)
                layers_dict((l1, l2, l3, l4))

            elif args.ConvLayer == 'SpiderConv':
                args.n_linear_layers = 1

                l1 = SC(6, self.conv_hidden_dim1, 1)
                l2 = SC(6, self.conv_hidden_dim2, 2)
                # l3 = lin(self.conv_hidden_dim2+6, 32)
                l3 = lin(2 * 3 + self.conv_hidden_dim1 * 3 + self.conv_hidden_dim2, 64)
                l4 = lin(64, out_features)
                layers_dict((l1, l2, l3, l4))

            elif args.ConvLayer == 'GATConv':
                args.n_linear_layers = 2
                args.hidden = 32

                # l1 = GT(2, self.conv_hidden_dim1, 1)
                # l2 = GT(2, self.conv_hidden_dim2, 2)
                # l3 = lin(2 * 1 + self.conv_hidden_dim1 * 1 + self.conv_hidden_dim2, 32)
                # l4 = lin(32, out_features)
                # layers_dict((l1, l2, l3, l4))

                l1 = GT(2, self.conv_hidden_dim1, 1)
                l2 = GT(2, self.conv_hidden_dim2, 2)
                l3 = lin(2 * 3 + self.conv_hidden_dim1 * 3 + self.conv_hidden_dim2, 64)
                l4 = lin(64, out_features)
                layers_dict((l1, l2, l3, l4))

            elif args.ConvLayer == 'GATConv2':
                args.n_linear_layers = 2
                args.hidden = 4

                l1 = GT2((2, 2), self.conv_hidden_dim1, 1, heads=1)
                l2 = GT2((2, 4), 1, 2, heads=self.conv_hidden_dim2)
                l3 = lin(self.conv_hidden_dim2 + 6, 32)
                l4 = lin(32, out_features)
                layers_dict((l1, l2, l3, l4))

            elif args.ConvLayer == 'SplineConv':
                args.n_linear_layers = 1
                l1 = SP(2, self.conv_hidden_dim1, 2, [5, 5], 1)
                l2 = SP(4, self.conv_hidden_dim2, 2, [2, 2], 2)
                l3 = lin(self.conv_hidden_dim2 + 6, 32)
                l4 = lin(32, out_features)
                layers_dict((l1, l2, l3, l4))

            elif args.ConvLayer == 'GMLS':

                args.n_linear_layers = 1

                # args.hidden = 16

                # ====================================================
                device = self.args.device
                num_dim = 2
                nchannels_u = 1
                Nc = nchannels_u

                Nx = 21
                Ny = 21
                NNx = 3 * Nx
                NNy = 3 * Ny  # simple periodic by tiling for now
                aspect_ratio = NNx / float(NNy)
                xx = np.linspace(-1.5, aspect_ratio * 1.5, NNx)
                xx = xx.astype(float)
                yy = np.linspace(-1.5, 1.5, NNy)
                yy = yy.astype(float)

                aa = np.meshgrid(xx, yy)
                np_xj = np.array([aa[0].flatten(), aa[1].flatten()]).T

                aa = np.meshgrid(xx, yy)
                np_xi = np.array([aa[0].flatten(), aa[1].flatten()]).T

                # make T tensors
                xj = T.from_numpy(np_xj).float().to(device)  # convert to T tensors
                xj.requires_grad = False

                xi = T.from_numpy(np_xi).float().to(device)  # convert to T tensors
                xi.requires_grad = False

    def Basis(self, args):
        """ Initialize Basis class in Basis.py for making model continuous in depth.
        i.e. parameters of networks(weights) at any depth(t) will be combination of basis
        """
        if args.Basis == 'Chebychev':
            basis = Chebychev(args.N_Basis)
        elif args.Basis == 'Fourier':
            basis = Fourier(args.N_Basis)
        elif args.Basis == 'VanillaRBF':
            basis = VanillaRBF(args.N_Basis)
        elif args.Basis == 'GaussianRBF':
            basis = GaussianRBF(args.N_Basis)
        elif args.Basis == 'MultiquadRBF':
            basis = MultiquadRBF(args.N_Basis)
        elif args.Basis == 'PiecewiseConstant':
            basis = PiecewiseConstant(args.N_Basis)
        elif args.Basis == 'Polynomial' or args.Basis == 'None':
            assert args.N_Basis == 1 if args.Basis == 'None' else 1
            basis = Polynomial(args.N_Basis)
        return basis

    def get_edge_index(self, hh, pos, batch):
        """Connect nodes with its neighbours. Compute edge_index once and use same until batch size changes

        Args:
            hh (tensor): net input [n_Nodes * batch_size, in_features]
            pos (tensor): position of nodes [n_Nodes * batch_size, co-ordinates(2 i.e.(x, y))]
            batch (tensor): indexing for pos [n_Nodes * batch_size] e.g {0, 0, 1, 1, 2, 2...19, 19}
                            for batch_size=20, n_Nodes=2
        Return:
            edge_index (tensor): pairs of nodes connected by an edge [2, n_edges]
        """

        if self.args.data_type == 'pde_1d':

            #  compute edge_index once and use same until batch size changes
            if hasattr(self.args, 'get_edge_index') and T.equal(batch, self.batch_prev):
                edge_index = self.args.get_edge_index
            else:
                if self.args.stencil == 5:
                    edge_index = knn_graph(pos, k=4, batch=batch, loop=False)
                    ed = edge_index

                    l1 = len(edge_index[0]) // self.args.bs_
                    tn = self.args.n_Nodes * self.args.bs_
                    for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):

                        for j in [0, 1, 509, 510]:
                            edge_corners = T.ones(2, 4)
                            edge_corners[1] = j + n_
                            if j == 0:
                                edge_corners[0] = T.tensor([510 + n_, 1 + n_, 509 + n_, 2 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                            if j == 1:
                                edge_corners[0] = T.tensor([0 + n_, 2 + n_, 510 + n_, 3 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                            if j == 509:
                                edge_corners[0] = T.tensor([508 + n_, 510 + n_, 507 + n_, 0 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                            if j == 510:
                                edge_corners[0] = T.tensor([509 + n_, 0 + n_, 508 + n_, 1 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                self.args.get_edge_index = edge_index
                self.batch_prev = batch

            # # edge index for periodic bc
            # cor = np.linspace(0, 2 * np.pi, self.args.n_Nodes + 1)
            # cor = cor[:-1]
            # c_pos = np.zeros((self.args.n_Nodes, 2))
            # c_pos[:, 0] = np.cos(cor)
            # c_pos[:, 1] = np.sin(cor)
            # c_pos = T.tensor(c_pos).to(self.args.device)
            # c_pos = T.cat([c_pos] * self.args.bs_)
            # edge_index = knn_graph(c_pos, k=self.k, batch=batch, loop=False)

        elif self.args.data_type == 'pde_2d':

            #  compute edge_index once and use same until batch size changes
            if hasattr(self.args, 'get_edge_index') and T.equal(batch, self.batch_prev):
                edge_index = self.args.get_edge_index
            else:
                if self.args.stencil == 5:
                    # edge_index = T.zeros((2, 64 * 64 * 4), dtype=int)
                    # zy = T.arange(64 * 64)
                    # zy = zy.unsqueeze(1)
                    # z1 = zy
                    # z2 = zy - 1
                    # z3 = zy + 1
                    # z4 = zy + 64
                    # z5 = zy - 64
                    # zx = T.cat([z2, z3, z4, z5], dim=1)
                    # edge_index[0] = zx.reshape(-1)
                    # edge_index[1] = T.cat([z1] * 4, dim=1).reshape(-1)
                    # edge_index = edge_index.to(self.args.device)

                    edge_index = knn_graph(pos, k=4, batch=batch, loop=False)
                    ed = edge_index

                    l1 = len(edge_index[0]) // self.args.bs_
                    tn = self.args.n_Nodes * self.args.bs_
                    for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):
                        # for d in range(0, l1*7, l1):
                        for i in range(1, 63):
                            edge_l = T.ones(2, 4)
                            l = i
                            ll = i + n_
                            edge_l[1] = ll
                            edge_l[0] = T.tensor([ll + 64 * 63, ll - 1, ll + 1, ll + 64])
                            edge_index[:, 4 * l + p_:4 * l + p_ + 4] = edge_l

                            edge_r = T.ones(2, 4)
                            r = i + 4032
                            rr = i + 4032 + n_
                            edge_r[1] = rr
                            edge_r[0] = T.tensor([rr - 64, rr - 1, rr + 1, rr - 64 * 63])
                            edge_index[:, 4 * r + p_:4 * r + p_ + 4] = edge_r

                            edge_b = T.ones(2, 4)
                            b = i * 64
                            bb = i * 64 + n_
                            edge_b[1] = bb
                            edge_b[0] = T.tensor([bb - 64, bb + 64 - 1, bb + 1, bb + 64])
                            edge_index[:, 4 * b + p_:4 * b + p_ + 4] = edge_b

                            edge_u = T.ones(2, 4)
                            u = i * 64 + 63
                            uu = i * 64 + 63 + n_
                            edge_u[1] = uu
                            edge_u[0] = T.tensor([uu - 64, uu - 1, uu - 64 + 1, uu + 64])
                            edge_index[:, 4 * u + p_:4 * u + p_ + 4] = edge_u

                        for j in [0, 63, 4032, 4095]:
                            edge_corners = T.ones(2, 4)
                            edge_corners[1] = j + n_
                            if j == 0:
                                edge_corners[0] = T.tensor([4032 + n_, 63 + n_, 1 + n_, 64 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                            if j == 63:
                                edge_corners[0] = T.tensor([4095 + n_, 62 + n_, 0 + n_, 127 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                            if j == 4032:
                                edge_corners[0] = T.tensor([3968 + n_, 4095 + n_, 4033 + n_, 0 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                            if j == 4095:
                                edge_corners[0] = T.tensor([4031 + n_, 4094 + n_, 4032 + n_, 63 + n_])
                                edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners

                elif self.args.stencil == 9:

                    nx = int(self.args.n_Nodes ** 0.5)
                    edge_index = T.zeros((2, nx * nx * 8 * self.args.bs_), device=self.args.device, dtype=T.int64)
                    l1 = len(edge_index[0]) // self.args.bs_
                    tn = self.args.n_Nodes * self.args.bs_
                    e = [0] * 8

                    zy = T.arange(nx * nx)[:, None]
                    z_ = [zy - 63, zy + 1, zy - 1, zy + 63, zy - 63 * 2, zy + 2, zy - 2, zy + 63 * 2]
                    zx = T.cat(z_, dim=1).reshape(-1)
                    edge_index[0] = T.cat([zx + self.args.n_Nodes * i for i in range(self.args.bs_)], dim=0)
                    edge_index[1] = T.cat(
                        [T.cat([zy] * 8, dim=1).reshape(-1) + self.args.n_Nodes * i for i in range(self.args.bs_)],
                        dim=0)

                    def edge_(e, edge_index, s, ss, p_):
                        """
                        ARGS:
                            s (int): node index in domain
                            ss (int): node index for graph in mini-batch
                            p_ (int): graph index in mini-batch i.e. graph 1, graph 2
                            e (list): change to be made in node index of nodes connected to node s
                        """
                        edge_ = T.ones(2, 8)
                        se = np.add(e, ss).tolist()
                        edge_[1] = ss
                        edge_[0] = T.tensor([se[0] - 63, se[1] - 1, se[2] + 1, se[3] + 63,
                                             se[4] - 63 * 2, se[5] - 2, se[6] + 2, se[7] + 63 * 2])
                        edge_index[:, 8 * s + p_:8 * s + p_ + 8] = edge_
                        return edge_index, [0] * 8

                    l, d, u, r = 63 * 63, 63, -63, -63 * 63
                    for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):
                        for i in range(2, 61):  # l, d, u, r, ll, dd, uu, rr

                            # =======================================
                            e[0] = e[4] = 63 * 63  # left boundary of image
                            edge_index, e = edge_(e, edge_index, i, i + n_, p_)

                            e[3] = e[7] = -63 * 63  # right boundary of image
                            edge_index, e = edge_(e, edge_index, i + 3906, i + 3906 + n_, p_)

                            e[2] = e[6] = -63  # up boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63 + 62, i * 63 + 62 + n_, p_)

                            e[1] = e[5] = 63  # down boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63, i * 63 + n_, p_)
                            # =======================================

                            e[4] = 63 * 63  # 2nd left boundary of image
                            edge_index, e = edge_(e, edge_index, i + 63, i + 63 + n_, p_)

                            e[7] = -63 * 63  # 2nd right boundary of image
                            edge_index, e = edge_(e, edge_index, i + 3906 - 63, i + 3906 - 63 + n_, p_)

                            e[6] = -63  # 2nd up boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63 + 62 - 1, i * 63 + 62 - 1 + n_, p_)

                            e[5] = 63  # 2nd down boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63 + 1, i * 63 + 1 + n_, p_)
                            # =======================================

                        # ======================================= l, d, u, r, ll, dd, uu, rr
                        e[0], e[1], e[4], e[5] = l, d, l, d
                        edge_index, e = edge_(e, edge_index, 0, 0 + n_, p_)

                        e[0], e[4], e[5] = l, l, d
                        edge_index, e = edge_(e, edge_index, 1, 1 + n_, p_)

                        e[1], e[4], e[5] = d, l, d
                        edge_index, e = edge_(e, edge_index, 63, 63 + n_, p_)

                        e[4], e[5] = l, d
                        edge_index, e = edge_(e, edge_index, 64, 64 + n_, p_)
                        # =======================================

                        # =======================================
                        e[0], e[2], e[4], e[6] = l, u, l, u
                        edge_index, e = edge_(e, edge_index, 62, 62 + n_, p_)

                        e[2], e[4], e[6] = u, l, u
                        edge_index, e = edge_(e, edge_index, 125, 125 + n_, p_)

                        e[0], e[4], e[6] = l, l, u
                        edge_index, e = edge_(e, edge_index, 61, 61 + n_, p_)

                        e[4], e[6] = l, u
                        edge_index, e = edge_(e, edge_index, 124, 124 + n_, p_)
                        # =======================================

                        # =======================================
                        e[1], e[3], e[5], e[7] = d, r, d, r
                        edge_index, e = edge_(e, edge_index, 3906, 3906 + n_, p_)

                        e[1], e[5], e[7] = d, d, r
                        edge_index, e = edge_(e, edge_index, 3843, 3843 + n_, p_)

                        e[3], e[5], e[7] = r, d, r
                        edge_index, e = edge_(e, edge_index, 3907, 3907 + n_, p_)

                        e[5], e[7] = d, r
                        edge_index, e = edge_(e, edge_index, 3844, 3844 + n_, p_)
                        # =======================================

                        # =======================================
                        e[2], e[3], e[6], e[7] = u, r, u, r
                        edge_index, e = edge_(e, edge_index, 3968, 3968 + n_, p_)

                        e[3], e[6], e[7] = r, u, r
                        edge_index, e = edge_(e, edge_index, 3967, 3967 + n_, p_)

                        e[2], e[6], e[7] = u, u, r
                        edge_index, e = edge_(e, edge_index, 3905, 3905 + n_, p_)

                        e[6], e[7] = u, r
                        edge_index, e = edge_(e, edge_index, 3904, 3904 + n_, p_)
                        # =======================================

                elif self.args.stencil == 'star':
                    edge_index = knn_graph(pos, k=8, batch=batch, loop=False)

                    def edge_(e, edge_index, s, ss, p_):
                        """
                        ARGS:
                            s (int): node index in domain
                            ss (int): node index for graph in mini-batch
                            p_ (int): graph index in mini-batch i.e. graph 1, graph 2
                            e (list): change to be made in node index of nodes connected to node s
                        """
                        edge_ = T.ones(2, 8, device=self.args.device, dtype=T.int64)
                        se = np.add(e, ss).tolist()
                        edge_[1] = ss
                        edge_[0] = T.tensor([se[0] - 63, se[1] - 1, se[2] + 1, se[3] + 63,
                                             se[4] - 63 - 1, se[5] - 63 + 1, se[6] + 63 - 1, se[7] + 63 + 1])
                        edge_index[:, 8 * s + p_:8 * s + p_ + 8] = edge_
                        return edge_index, [0] * 8

                    l1 = len(edge_index[0]) // self.args.bs_  # len of edge_index of 1 graph
                    tn = self.args.n_Nodes * self.args.bs_  # total n_Nodes in mini-batch
                    e = [0] * 8
                    l, d, u, r = 63 * 63, 63, -63, -63 * 63

                    for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):
                        for i in range(1, 62):  # l, d, u, r, ld, lu, rd, ru

                            e[0] = e[4] = e[5] = l  # left boundary of image
                            edge_index, e = edge_(e, edge_index, i, i + n_, p_)

                            e[3] = e[6] = e[7] = r  # right boundary of image
                            edge_index, e = edge_(e, edge_index, i + 3906, i + 3906 + n_, p_)

                            e[2] = e[5] = e[7] = u  # up boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63 + 62, i * 63 + 62 + n_, p_)

                            e[1] = e[4] = e[6] = d  # down boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63, i * 63 + n_, p_)

                        # ======================================= l, d, u, r, ld, lu, rd, ru
                        e[0], e[1], e[4], e[5], e[6] = l, d, l + d, l, d
                        edge_index, e = edge_(e, edge_index, 0, 0 + n_, p_)

                        e[1], e[3], e[4], e[6], e[7] = d, r, d, r + d, r
                        edge_index, e = edge_(e, edge_index, 3906, 3906 + n_, p_)

                        e[2], e[3], e[5], e[6], e[7] = u, r, u, r, u + r
                        edge_index, e = edge_(e, edge_index, 3968, 3968 + n_, p_)

                        e[0], e[2], e[4], e[5], e[7] = l, u, l, l + u, u
                        edge_index, e = edge_(e, edge_index, 62, 62 + n_, p_)
                        # =======================================

                elif self.args.stencil == 'k_near':
                    edge_index = knn_graph(pos, k=8, batch=batch, loop=False)

                    def edge_(e, edge_index, s, ss, p_):
                        """
                        ARGS:
                            s (int): node index in domain
                            ss (int): node index for graph in mini-batch
                            p_ (int): graph index in mini-batch times len i.e. graph 1*len, graph 2*len
                                        where len is number of edges in one graph
                            e (list): change to be made in node index of nodes connected to node s
                        """
                        edge_ = T.ones(2, 8, device=self.args.device, dtype=T.int64)
                        se = np.add(e, ss).tolist()
                        edge_[1] = ss
                        edge_[0] = T.tensor([se[0] - 63, se[1] - 1, se[2] + 1, se[3] + 63,
                                             se[4] - 63 - 1, se[5] - 63 + 1, se[6] + 63 - 1, se[7] + 63 + 1])
                        edge_index[:, 8 * s + p_:8 * s + p_ + 8] = edge_
                        return edge_index, [0] * 8

                    l1 = len(edge_index[0]) // self.args.bs_  # len of edge_index of 1 graph
                    tn = self.args.n_Nodes * self.args.bs_  # total n_Nodes in mini-batch
                    e = [0] * 8
                    l, d, u, r = 63 * 63, 63, -63, -63 * 63

                    for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):
                        for i in range(1, 62):  # l, d, u, r, ld, lu, rd, ru

                            e[0] = e[4] = e[5] = l  # left boundary of image
                            edge_index, e = edge_(e, edge_index, i, i + n_, p_)

                            e[3] = e[6] = e[7] = r  # right boundary of image
                            edge_index, e = edge_(e, edge_index, i + 3906, i + 3906 + n_, p_)

                            e[2] = e[5] = e[7] = u  # up boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63 + 62, i * 63 + 62 + n_, p_)

                            e[1] = e[4] = e[6] = d  # down boundary of image
                            edge_index, e = edge_(e, edge_index, i * 63, i * 63 + n_, p_)

                        # ======================================= l, d, u, r, ld, lu, rd, ru
                        e[0], e[1], e[4], e[5], e[6] = l, d, l + d, l, d
                        edge_index, e = edge_(e, edge_index, 0, 0 + n_, p_)

                        e[1], e[3], e[4], e[6], e[7] = d, r, d, r + d, r
                        edge_index, e = edge_(e, edge_index, 3906, 3906 + n_, p_)

                        e[2], e[3], e[5], e[6], e[7] = u, r, u, r, u + r
                        edge_index, e = edge_(e, edge_index, 3968, 3968 + n_, p_)

                        e[0], e[2], e[4], e[5], e[7] = l, u, l, l + u, u
                        edge_index, e = edge_(e, edge_index, 62, 62 + n_, p_)
                        # =======================================

                elif self.args.stencil == 'k_nn':
                    k = 32
                    r_pos = T.from_numpy(self.args.r_pos).to(self.args.device)
                    n_Nodes = pos.shape[0] // self.args.bs_
                    batch_pos = T.zeros((pos[:n_Nodes].shape[0]), dtype=T.int64, device=self.args.device)
                    batch_r_pos = T.zeros((r_pos.shape[0]), dtype=T.int64, device=self.args.device)
                    edge_index_r = knn(pos[:n_Nodes], r_pos, k + 1, batch_pos, batch_r_pos)

                    # del_col_self_edge = T.arange(0, edge_index_r.shape[1], k)
                    del_col_self_edge = T.cat(
                        [T.arange(i * (k + 1) + 1, i * (k + 1) + k + 1) for i in range(r_pos.shape[0])])
                    edge_index_r = edge_index_r[:, del_col_self_edge]

                    map = self.args.map
                    edge_index_r[0] = T.cat([T.tensor([map[i]] * k) for i in range(map.shape[0])], dim=0)
                    edge_index_r = T.flip(edge_index_r, [0])

                    def edge_(e, edge_index, s, ss, p_):
                        """
                        ARGS:
                            s (int): node index in domain
                            ss (int): node index for graph in mini-batch
                            p_ (int): graph index in mini-batch times len i.e. graph 1*len, graph 2*len
                                        where len is number of edges in one graph
                            e (list): change to be made in node index of nodes connected to node s
                        """
                        edge_ = T.ones(2, 8, device=self.args.device, dtype=T.int64)
                        se = np.add(e, ss).tolist()
                        edge_[1] = ss
                        edge_[0] = T.tensor([se[0] - 63, se[1] - 1, se[2] + 1, se[3] + 63,
                                             se[4] - 63 - 1, se[5] - 63 + 1, se[6] + 63 - 1, se[7] + 63 + 1])
                        edge_index = T.cat((edge_index, edge_), dim=1)
                        return edge_index, [0] * 8

                    # l1 = len(edge_index[0]) // self.args.bs_  # len of edge_index of 1 graph
                    # tn = n_Nodes * self.args.bs_  # total n_Nodes in mini-batch
                    e = [0] * 8
                    l, d, u, r = 63 * 63, 63, -63, -63 * 63

                    edge_index = T.tensor([], dtype=T.int64, device=self.args.device)
                    # for n_, p_ in zip(range(0, tn, n_Nodes), range(0, len(edge_index[0]), l1)):
                    n_, p_ = 0, 0
                    # l, d, u, r, ld, lu, rd, ru
                    # =======================================

                    e[0], e[1], e[4], e[5], e[6] = l, d, l + d, l, d
                    edge_index, e = edge_(e, edge_index, 0, 0 + n_, p_)
                    for i in range(1, 62):
                        e[1] = e[4] = e[6] = d  # down boundary of image
                        edge_index, e = edge_(e, edge_index, i * 63, i * 63 + n_, p_)
                    # =======================================

                    e[1], e[3], e[4], e[6], e[7] = d, r, d, r + d, r
                    edge_index, e = edge_(e, edge_index, 3906, 3906 + n_, p_)
                    for i in range(1, 62):
                        e[3] = e[6] = e[7] = r  # right boundary of image
                        edge_index, e = edge_(e, edge_index, i + 3906, i + 3906 + n_, p_)
                    # =======================================

                    e[2], e[3], e[5], e[6], e[7] = u, r, u, r, u + r
                    edge_index, e = edge_(e, edge_index, 3968, 3968 + n_, p_)
                    for i in range(1, 62):
                        e[2] = e[5] = e[7] = u  # up boundary of image
                        edge_index, e = edge_(e, edge_index, i * 63 + 62, i * 63 + 62 + n_, p_)
                    # =======================================

                    e[0], e[2], e[4], e[5], e[7] = l, u, l, l + u, u
                    edge_index, e = edge_(e, edge_index, 62, 62 + n_, p_)
                    for i in range(1, 62):
                        e[0] = e[4] = e[5] = l  # left boundary of image
                        edge_index, e = edge_(e, edge_index, i, i + n_, p_)
                    # =======================================

                    edge_index = T.cat([edge_index, edge_index_r], dim=1)
                    edge_index = T.cat([edge_index + n_Nodes * i for i in range(self.args.bs_)], dim=1)

                # self.visualize_points(pos[:n_Nodes], edge_index[:, :edge_index.shape[1]//self.args.bs])
                self.args.get_edge_index = edge_index
                self.batch_prev = batch

        elif self.args.data_type == 'TrajectoryExtrapolation':

            for i in range(len(hh)):
                for j in range(len(hh)):
                    if 2 * LA.norm(hh[i, :2] - hh[j, :2]) <= self.r:  # and i != j:
                        # data.edge_index[] =
                        edge_ = T.empty(2, 1, dtype=int).to(self.device)
                        edge_[0] = j
                        edge_[1] = i
                        try:
                            T.is_tensor(edge_index)
                            edge_index = T.cat((edge_index, edge_), dim=1)
                        except:
                            edge_index = edge_

        return edge_index

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
        # plt.pause(1)
        plt.show()

    def plot_graph_layer_output(self, hh, g_out, pos, **kwargs):

        if self.args.data_type == 'pde_1d':
            h1, h2 = g_out

            fig = plt.figure()  # figsize=(5, 6))
            widths = [3, 3, 3]
            heights = [1, 1]
            spec5 = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths, height_ratios=heights)
            a = [fig.add_subplot(spec5[0, 0])]
            b = [fig.add_subplot(spec5[0, 1]), fig.add_subplot(spec5[1, 1])]
            c = [fig.add_subplot(spec5[0, 2]), fig.add_subplot(spec5[1, 2])]

            start = 511 * 2
            end = start + 511

            a[0].plot(hh[start:end, 0].detach().cpu())
            a[0].set_title('u')

            b[0].plot(np.gradient(hh[start:end, 0].detach().cpu()))
            b[0].set_title('u-dx')
            b[1].set_ylabel('graph pred', fontsize=15)
            b[1].plot(h1[start:end, 0].detach().cpu())

            h2_00 = np.gradient(hh[start:end, 0].detach().cpu())
            c[0].plot(np.gradient(h2_00))
            c[0].set_title('u-dx-dx')
            c[1].plot(h2[start:end, 0].detach().cpu())

            for ii in [a, b, c]:
                for jj in ii:
                    jj.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            path = self.args.get_path(self.args, W=1, exp=self.args.exp)
            file_loc = osp.join(path, 'fig{}.pdf'.format(self.epoch))
            plt.savefig(file_loc, format='pdf') if self.args.save else 0
            plt.show() if self.args.show else 0
            plt.close(fig)
            plt.close('all')
            self.args.ignore = 0

        if self.args.data_type == 'pde_2d':

            h1, h2 = g_out
            slx, sly = kwargs.get('slx', 32), kwargs.get('sly', range(0, 63))
            view_mode = 2
            fig = plt.figure()  # figsize=(5, 6))
            spec5 = fig.add_gridspec(ncols=8, nrows=4)  # , width_ratios=w, height_ratios=h)
            spec5.update(wspace=0.1, hspace=0.4)

            def asp(fig, spec5):
                """ add sub plot """
                return lambda r, c: fig.add_subplot(spec5[r, c])

            ap = asp(fig, spec5)

            a, b, c = [], [], []
            a = [ap(r, c) for r, c in itertools.product(range(1), range(2))]
            b = [ap(r, c) for c, r in itertools.product(range(4, 8), range(2))]
            c = [ap(r, c) for c, r in itertools.product(range(8), range(2, 4))]

            def reshape(grid_x, grid_y, points, stencil, n_Nodes):
                """ Reshape """
                if stencil == 'k_near' or stencil == 'k_nn':
                    return lambda x, c: griddata(points, x[:n_Nodes, c].detach().cpu(), (grid_x, grid_y),
                                                 method='cubic')
                else:
                    return lambda x, c: x[:n_Nodes, c].reshape(63, 63).detach().cpu()

            grid_x, grid_y = np.mgrid[0:62:63j, 0:62:63j]
            points = pos.detach().cpu().numpy()[:self.args.n_Nodes]
            n_Nodes = self.args.n_Nodes
            rs = reshape(grid_x, grid_y, points, self.args.stencil, n_Nodes)

            def gd(x, ax):
                """ get derivative """
                x0, x1 = np.gradient(x)
                assert ax in [0, 1]
                return x0 if ax == 0 else x1
                # return ndimage.sobel(x, axis=ax, mode='constant')

            u0, u1 = rs(hh, 0), rs(hh, 1)
            u0_0, u0_1, u1_0, u1_1 = gd(u0, 0), gd(u0, 1), gd(u1, 0), gd(u1, 1)
            u0_00, u0_01, u0_10, u0_11 = gd(u0_0, 0), gd(u0_0, 1), gd(u0_1, 0), gd(u0_1, 1)
            u1_00, u1_01, u1_10, u1_11 = gd(u1_0, 0), gd(u1_0, 1), gd(u1_1, 0), gd(u1_1, 1)

            U = (u0, u1)
            U_ = (u0_0, u0_1, u1_0, u1_1)
            U__ = (u0_00, u0_01, u0_10, u0_11, u1_00, u1_01, u1_10, u1_11)

            if view_mode == 1:
                # a1.imshow(hh[:4096, 0].reshape(64, 64).detach().cpu())
                # a1.set_title('hh0')
                # a2.imshow(hh[:4096, 1].reshape(64, 64).detach().cpu())
                # a2.set_title('uy')
                #
                # b1.imshow(ndimage.sobel(hh[:4096, 0].reshape(64, 64).detach().cpu(), axis=0, mode='constant'))
                # b1.set_title('hh0-dx')
                # b2.imshow(h1[:4096, 0].reshape(64, 64).detach().cpu())
                # b3.imshow(ndimage.sobel(hh[:4096, 1].reshape(64, 64).detach().cpu(), axis=1, mode='constant'))
                # b3.set_title('uy-dy')
                # b4.imshow(h1[:4096, 1].reshape(64, 64).detach().cpu())
                # b5.imshow(ndimage.sobel(hh[:4096, 0].reshape(64, 64).detach().cpu(), axis=1, mode='constant'))
                # b5.set_title('hh0-dy')
                # b6.imshow(h1[:4096, 2].reshape(64, 64).detach().cpu())
                # b7.imshow(ndimage.sobel(hh[:4096, 1].reshape(64, 64).detach().cpu(), axis=0, mode='constant'))
                # b7.set_title('uy-dx')
                # b8.imshow(h1[:4096, 3].reshape(64, 64).detach().cpu())
                #
                # h2_00 = ndimage.sobel(hh[:4096, 0].reshape(64, 64).detach().cpu(), axis=0, mode='constant')
                # c1.imshow(ndimage.sobel(h2_00, axis=0, mode='constant'))
                # c1.set_title('hh0-dx-dx')
                # c2.imshow(h2[:4096, 0].reshape(64, 64).detach().cpu())
                # c3.imshow(ndimage.sobel(h2_00, axis=1, mode='constant'))
                # c3.set_title('hh0-dx-dy')
                # c4.imshow(h2[:4096, 4].reshape(64, 64).detach().cpu())
                #
                # h2_01 = ndimage.sobel(hh[:4096, 0].reshape(64, 64).detach().cpu(), axis=1, mode='constant')
                # c5.imshow(ndimage.sobel(h2_01, axis=0, mode='constant'))
                # c5.set_title('hh0-dy-dx')
                # c6.imshow(h2[:4096, 2].reshape(64, 64).detach().cpu())
                # c7.imshow(ndimage.sobel(h2_01, axis=1, mode='constant'))
                # c7.set_title('hh0-dy-dy')
                # c8.imshow(h2[:4096, 6].reshape(64, 64).detach().cpu())
                #
                # h2_10 = ndimage.sobel(hh[:4096, 1].reshape(64, 64).detach().cpu(), axis=0, mode='constant')
                # c9.imshow(ndimage.sobel(h2_10, axis=0, mode='constant'))
                # c9.set_title('uy-dx-dx')
                # c10.imshow(h2[:4096, 3].reshape(64, 64).detach().cpu())
                # c11.imshow(ndimage.sobel(h2_10, axis=1, mode='constant'))
                # c11.set_title('uy-dx-dy')
                # c12.imshow(h2[:4096, 7].reshape(64, 64).detach().cpu())
                #
                # h2_11 = ndimage.sobel(hh[:4096, 1].reshape(64, 64).detach().cpu(), axis=1, mode='constant')
                # c13.imshow(ndimage.sobel(h2_11, axis=0, mode='constant'))
                # c13.set_title('uy-dy-dx')
                # c14.imshow(h2[:4096, 1].reshape(64, 64).detach().cpu())
                # c15.imshow(ndimage.sobel(h2_11, axis=1, mode='constant'))
                # c15.set_title('uy-dy-dy')
                # c16.imshow(h2[:4096, 5].reshape(64, 64).detach().cpu())

                a[0].imshow(u0)
                a[0].set_title('ux')
                a[1].imshow(u1)
                a[1].set_title('uy')

                b[0].imshow(u0_0)
                b[0].set_title('ux-dx')
                b[1].imshow(rs(h1, 0))
                b[2].imshow(u1_1)
                b[2].set_title('uy-dy')
                b[3].imshow(rs(h1, 1))
                b[4].imshow(u0_1)
                b[4].set_title('ux-dy')
                b[5].imshow(rs(h1, 2))
                b[6].imshow(u1_0)
                b[6].set_title('uy-dx')
                b[7].imshow(rs(h1, 3))

                c[0].imshow(u0_00)
                c[0].set_title('ux-dx-dx')
                c[1].imshow(rs(h2, 0))
                c[2].imshow(u0_01)
                c[2].set_title('ux-dx-dy')
                c[3].imshow(rs(h2, 4))

                c[4].imshow(u0_10)
                c[4].set_title('ux-dy-dx')
                c[5].imshow(rs(h2, 2))
                c[6].imshow(u0_11)
                c[6].set_title('ux-dy-dy')
                c[7].imshow(rs(h2, 6))

                c[8].imshow(u1_00)
                c[8].set_title('uy-dx-dx')
                c[9].imshow(rs(h2, 3))
                c[10].imshow(u1_01)
                c[10].set_title('uy-dx-dy')
                c[11].imshow(rs(h2, 7))

                c[12].imshow(u1_10)
                c[12].set_title('uy-dy-dx')
                c[13].imshow(rs(h2, 1))
                c[14].imshow(u1_11)
                c[14].set_title('uy-dy-dy')
                c[15].imshow(rs(h2, 5))

            if view_mode == 2:

                def plot_slice(a, b, c, slx, sly):
                    fs1 = 10
                    fs2 = 10
                    u0, u1 = U
                    u0_0, u0_1, u1_0, u1_1 = U_
                    u0_00, u0_01, u0_10, u0_11, u1_00, u1_01, u1_10, u1_11 = U__

                    a[0].plot(u0[slx, sly])
                    a[0].set_title('ux', fontsize=fs1)
                    # a[0].tick_params(axis ='both', which ='both', length = 0, labelsize=5)
                    a[1].plot(u1[slx, sly])
                    a[1].set_title('uy', fontsize=fs1)

                    b[0].plot(u0_0[slx, sly])
                    b[0].set_title('ux-dx', fontsize=fs1)
                    b[0].set_ylabel('true', fontsize=fs2)
                    b[1].set_ylabel('graph pred', fontsize=fs2)
                    b[1].plot(-rs(h1, 2)[slx, sly])
                    b[2].plot(u1_1[slx, sly])
                    b[2].set_title('uy-dy', fontsize=fs1)
                    b[3].plot(-rs(h1, 3)[slx, sly])
                    b[4].plot(u0_1[slx, sly])
                    b[4].set_title('ux-dy', fontsize=fs1)
                    b[5].plot(-rs(h1, 0)[slx, sly])
                    b[6].plot(u1_0[slx, sly])
                    b[6].set_title('uy-dx', fontsize=fs1)
                    b[7].plot(-rs(h1, 1)[slx, sly])

                    c[0].plot(u0_00[slx, sly])
                    c[0].set_title('ux-dx-dx', fontsize=fs1)
                    c[0].set_ylabel('true', fontsize=fs2)
                    c[1].set_ylabel('graph pred', fontsize=fs2)
                    c[1].plot(-rs(h2, 2)[slx, sly])
                    c[2].plot(u0_01[slx, sly])
                    c[2].set_title('ux-dx-dy', fontsize=fs1)
                    c[3].plot(rs(h2, 6)[slx, sly])

                    c[4].plot(u0_10[slx, sly])
                    c[4].set_title('ux-dy-dx', fontsize=fs1)
                    c[5].plot(rs(h2, 4)[slx, sly])
                    c[6].plot(u0_11[slx, sly])
                    c[6].set_title('ux-dy-dy', fontsize=fs1)
                    c[7].plot(-rs(h2, 0)[slx, sly])

                    c[8].plot(u1_00[slx, sly])
                    c[8].set_title('uy-dx-dx', fontsize=fs1)
                    c[9].plot(-rs(h2, 5)[slx, sly])
                    c[10].plot(u1_01[slx, sly])
                    c[10].set_title('uy-dx-dy', fontsize=fs1)
                    c[11].plot(-rs(h2, 1)[slx, sly])

                    c[12].plot(u1_10[slx, sly])
                    c[12].set_title('uy-dy-dx', fontsize=fs1)
                    c[13].plot(-rs(h2, 3)[slx, sly])
                    c[14].plot(u1_11[slx, sly])
                    c[14].set_title('uy-dy-dy', fontsize=fs1)
                    c[15].plot(-rs(h2, 7)[slx, sly])

                    for ii in [a, b, c]:
                        for jj in ii:
                            jj.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

                plot_slice(a, b, c, slx, sly)

            path = self.args.get_path(self.args, W=1, exp=self.args.exp)
            name = kwargs.get('name') if kwargs.get('name') else 'fig'
            file_loc = osp.join(path, '{}{}.pdf'.format(name, self.epoch))
            plt.savefig(file_loc, format='pdf') if self.args.save else 0
            plt.show() if self.args.show else 0
            plt.close(fig)
            plt.close('all')
            self.args.ignore = 0

    def calc_output(self, hh, edge_index, pos, edge_attr, t):

        assert hh.dtype == T.float32 and pos.dtype == T.float32

        if self.args.data_type == 'pde_1d':

            h1 = self.conv['{:d}'.format(1)](hh, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            h2 = self.conv['{:d}'.format(2)](h1, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            h3 = T.cat([hh.float(), h1, h1, h2], dim=-1)
            self.conv['{:d}'.format(3)].t = self.conv['{:d}'.format(4)].t = t
            h4 = self.conv['{:d}'.format(3)](h3)
            h4 = self.activation(h4)
            h = self.conv['{:d}'.format(4)](h4)

            # debugging =====================================
            # h1 = self.conv['{:d}'.format(1)](hh.float(), edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            # h3 = T.cat([hh.float(), h1], dim=-1)
            # self.conv['{:d}'.format(2)].t = self.conv['{:d}'.format(3)].t = t
            # h4 = self.conv['{:d}'.format(2)](h3)
            # h4 = self.activation(h4)
            # h = self.conv['{:d}'.format(3)](h4)

            # h1 = self.conv['{:d}'.format(1)](hh.float(), edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            # h3 = T.cat([hh.float(), h1], dim=-1)
            # self.conv['{:d}'.format(2)].t = t
            # h = self.conv['{:d}'.format(2)](h3)

            # h = self.conv['{:d}'.format(1)](hh.float(), edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)

            # h1 = self.conv['{:d}'.format(1)](hh.float(), edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            # h2 = self.conv['{:d}'.format(2)](h1, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            # h3 = self.conv['{:d}'.format(3)](h2, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            # h4 = self.conv['{:d}'.format(4)](h3, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            # h4 = T.cat([hh.float(), h1, h4], dim=-1)
            # self.conv['{:d}'.format(5)].t = self.conv['{:d}'.format(6)].t = t
            # h5 = self.conv['{:d}'.format(5)](h4)
            # h5 = self.activation(h5)
            # h = self.conv['{:d}'.format(6)](h5)
            # debugging =====================================

            if hasattr(self.args, 'ignore') and self.args.ignore == 1 and (self.args.save or self.args.show):
                self.plot_graph_layer_output(hh, (h1, h2), pos)

        elif self.args.data_type == 'pde_2d':

            h1 = self.conv['{:d}'.format(1)](hh, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            h2 = self.conv['{:d}'.format(2)](h1, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
            # h3 = T.cat([hh, h1, h2], dim=-1)
            h3 = T.cat([hh, hh, hh, h1, h1, h1, h2], dim=-1)
            self.conv['{:d}'.format(3)].t = self.conv['{:d}'.format(4)].t = t
            h4 = self.conv['{:d}'.format(3)](h3)
            h4 = self.activation(h4)
            h = self.conv['{:d}'.format(4)](h4)

            # h4 = self.conv['{:d}'.format(3)]((h3, t))
            # h4 = h4.tanh()
            # h = self.conv['{:d}'.format(4)]((h4, t))

            if hasattr(self.args, 'ignore') and self.args.ignore == 1 and (self.args.save or self.args.show):
                self.plot_graph_layer_output(hh, (h1, h2), pos, slx=31, sly=range(63), name='fig_h')
                self.plot_graph_layer_output(hh, (h1, h2), pos, slx=range(63), sly=31, name='fig_v')

        assert not T.isnan(h).any()
        return h

    def forward(self, t, dt, data):  # hh, batch, edge_index, pos, edge_attr):

        _t_ = (t + dt) / self.del_t
        self.idx = int((t + 0.00001) / self.del_t)
        # print(self.prev_idx, self.idx, t)
        # dt_ = t - self.del_t * self.idx
        # dt = self.del_t  # if dt_ == 0 else dt_
        # if self.args.Basis != 'None':
        #     self.basisfunc.idx = self.idx
        hh = data
        n_Nodes = len(hh[:, 0])
        batch = self.batch.long()
        pos = self.pos_list[self.idx] if isinstance(self.pos_list, list) else self.pos_list
        edge_attr = self.edge_attr

        self.edge_index = edge_index = self.get_edge_index(hh, pos, batch)

        if self.args.cont_in == 't':
            h = self.calc_output(hh, edge_index, pos, edge_attr, _t_)
        elif self.args.cont_in == 'dt':
            h = self.calc_output(hh, edge_index, pos, edge_attr, self.del_t)

        if self.idx != self.prev_idx:
            if self.args.adaptive_graph and not self.training:
                pass
            else:
                self.pos_list.append(pos.detach().clone()) if isinstance(self.pos_list, list) else 0
                self.edge_index_list.append(edge_index.detach().clone())
            self.prev_idx = self.idx

        self.u_t_pred_ls.append(h) if dt == 0.0 else 0
        if 1 == 0:
            plt.figure()
            plt.plot(hh[:, 0].detach().cpu())
            plt.plot(h[:, 0].detach().cpu())
            plt.show()

        return h


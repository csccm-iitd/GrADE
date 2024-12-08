
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ adaptive none 1d  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def train_adptive__none_1d(**kwargs):  #
#     for B in ['Chebychev']:  # , 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#         for n_B in [3]:  # [1, 2, 3, 5, 10]:  # range(10):
#             for train_size in [10]:  # [5, 15, 25, 35, 45, 50]:
#                 for batch_time in [3]:  # [2, 6, 10, 18, 26, 38]:
#                     parser = argparse.ArgumentParser('ODE demo')
#                     parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
#                                         default='rk4')  # dopri5') rk4
#                     # parser.add_argument('--n_time_steps', type=int, default=500)  # i.e. 0th 1th ..999th = 1000 time value
#                     # parser.add_argument('--train_size', type=int, default=0)
#                     # parser.add_argument('--test_size', type=int, default=0)
#                     # parser.add_argument('--batch_time', type=int, default=5)
#                     parser.add_argument('--batch_size', type=int, default=1)
#                     # parser.add_argument('--niters', type=int, default=65)
#                     parser.add_argument('--test_freq', type=int, default=20)
#                     # parser.add_argument('--viz', action='store_true')
#                     parser.add_argument('--viz', default=True)
#                     parser.add_argument('--gpu', type=int, default=0)
#                     parser.add_argument('--adjoint', action='store_true')
#                     parser.add_argument('--tr_style', type=str, choices=['MyclassV2', 'None'], default=None)
#                     parser.add_argument('--myepoch', type=list, default=[3, 4])
#                     parser.add_argument('--N_Basis', type=int, default=n_B)
#                     # parser.add_argument('--N_particles', type=int, default='101')
#                     parser.add_argument('--n_dense_Nodes', type=int, default='1000')
#                     parser.add_argument('--ConvLayer', type=str, choices=['PointNetLayer', 'GATConv'],
#                                         default='PointNetLayer')
#                     parser.add_argument('--data_type', type=str,
#                                         choices=['pde_1d', 'pde_2d', 'TrajectoryExtrapolation'],
#                                         default='pde_1d')
#                     parser.add_argument('--Basis', type=str,
#                                         choices=['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF',
#                                                  'MultiquadRBF'],
#                                         default='{}'.format(B))
#                     parser.add_argument('--op_type', type=str,
#                                         choices=[r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}', 'ux', 'u*ux',
#                                                  'ux*ux', 'u*uxx', 'uxx*uxx'],
#                                         default='u*ux')
#                     parser.add_argument('--adaptive_graph', action='store_true', default=False)
#                     args = parser.parse_args()
#                     args.train_size_used = train_size
#                     args.last_intg_t_idx_train = batch_time
#                     print('Number of Basis:', n_B)
#                     print('steps for integration:', batch_time)
#                     print('Basis:', '{}'.format(B))
#                     model, optimizer, criterion, train_loader, test_loader, get_path, integrator, PlotResults = init_model(
#                         args, Parser_, **kwargs)
#                     for epoch in range(0, args.niters):
#                         # litt = [2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10,
#                         #        10]
#                         # litt = [2,2,2,2,4,4,6,6,8,8,10,10,10, 10, 10, 10,11,11,11,11 ,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13]
#                         # batch_time = litt[epoch]
#                         batch_time = 3
#                         train_error = training_epoch(model, optimizer, criterion, train_loader, get_path, integrator,
#                                             PlotResults,
#                                             epoch, batch_time, args, **kwargs)
# 
#                         args.rollout_t = 20
#                         args.adaptive_graph = True
#                         test_errorA1 = testing_epoch(model, test_loader, get_path, integrator, PlotResults, epoch, 3,
#                                             args, **kwargs)
#                         args.adaptive_graph = False
#                         test_error1 = testing_epoch(model, test_loader, get_path, integrator, PlotResults, epoch, 3,
#                                            args, **kwargs)
#                         del args.rollout_t
#                         args.rollout_t = 35
#                         args.adaptive_graph = True
#                         test_errorA2 = testing_epoch(model, test_loader, get_path, integrator, PlotResults, epoch, 3,
#                                             args, **kwargs)
#                         args.adaptive_graph = False
#                         test_error2 = testing_epoch(model, test_loader, get_path, integrator, PlotResults, epoch, 3,
#                                            args, **kwargs)
#                         del args.rollout_t
#                         print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, '
#                               f'Test Accuracy: {test_errorA1:.4f}, Test Accuracy: {test_error1:.4f}, '
#                               f'Test Accuracy: {test_errorA2:.4f}, Test Accuracy: {test_error2:.4f}')
#                         # print(f'Epoch: {epoch:02d}, Loss: {test_error:.4f}')
#                         # tt = time.time() - start
#                         # print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}')
# 
# 
# def adptive__none_1d(**kwargs):
#     # ========================================= adptive__none ==========================================
#     for ad in [True, True]:
#         for B in ['Chebychev']:  # , 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#             for n_B in [2]:  # [1, 2, 3, 5, 10]:  # range(10):
#                 for train_size in [10]:  # [5, 15, 25, 35, 45, 50]:
#                     for batch_time in [3]:  # [2, 6, 10, 18, 26, 38]:
#                         for last_intg_t_idx_test in [5, 17, 20, 26, 35]:
#                             args.train_size_used = train_size
#                             args.last_intg_t_idx_train = batch_time
#                             print('Number of Basis:', n_B)
#                             print('Time of integration:', batch_time)
#                             print('Basis:', '{}'.format(B))
#                             model, optimizer, criterion, train_loader, test_loader, integrator, PlotResults = init_model(
#                                 args, Parser_, **kwargs)
#                             path = args.get_path(args, W=1)
#                             pretrained_weights = osp.join(path, 'weights_lit{}'.format(batch_time))
#                             model.load_state_dict(torch.load(pretrained_weights))
#                             for epoch in [0]:
#                                 # batch_time = 10
#                                 test_error = testing_epoch(model, test_loader, integrator, PlotResults, epoch, last_intg_t_idx_test,
#                                                   args)
#                                 # print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, Test Accuracy: {test_error:.4f}')
#                                 print(f'Epoch: {epoch:02d}, Loss: {test_error:.4f}')
# 
#                             # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#                             if not "er" in vars():
#                                 er = np.array([[test_error]])
#                             else:
#                                 er = np.concatenate((er, np.array([[test_error]])), axis=1)
#         if not "err" in vars():
#             err = er
#         else:
#             err = np.concatenate((err, er), axis=0)
#         del er
# 
#     if kwargs.get('save', None):
#         path2 = args.get_path(args, C1=1)
#         np.savetxt(osp.join(path2, 'adptive__none_1D.csv'), err, delimiter=",")
#     del err
#     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ my data pos  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# a = T.ones(nx)
# b = T.arange(nx)
# c = T.stack([a] * nx, dim=0)
# d = (c*b).transpose(0, 1).reshape(-1)
# e = T.cat(([b] * nx), dim=0)
# f = T.stack((d, e), dim=1)
# pos = T.cat([f]*len_train_and_test, dim=0)

# x = train xt = test
# x_list = T.zeros((train_x_array.shape[0] * train_x_array.shape[2], 2))
# for i in range(len(s)):
#     for j in range(args.n_Nodes):
#         train_x = T.reshape(train_x_array[i, :, j], (1, -1))
#         try:
#             T.is_tensor(x_list)
#             x_list = T.cat((x_list, train_x), dim=0)
#         except:
#             x_list = train_x
#         if i < len(q):
#             test_x = T.reshape(test_x_array[i, :, j], (1, -1))
#             try:
#                 T.is_tensor(xt_list)
#                 xt_list = T.cat((xt_list, test_x), dim=0)
#             except:
#                 xt_list = test_x

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ target_batch 1d  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def target_batch(batched_u, position, z, train, idx_ls, args):
#
#     if args.cont_in == 't':
#         s, q = traindata_position_in_raw_data(args)
#         assert idx_ls[-1] <= args.max_batch_time, 'batch_time should be smaller than args.max_batch_time'
#
#         x_array = raw_sequential_data(args, plot_data=False).unsqueeze(3)
#         nx = args.n_Nodes
#
#         if train:
#             bs = args.bs
#             trbs = args.train_batch_size
#             batch_y = T.cat([x_array[:, 0:-1:args.skip][s[position * trbs + z], idx_ls, :] for z in range(bs)], dim=1)
#             # batch_y = train_x_array
#             assert T.equal(batched_u[:len(x_array[0, :, 0])],
#                                batch_y[0, :len(x_array[0, :, 0])]), 'starting target different from starting training data'
#         else:
#             bs = args.bs
#             tbs = args.test_batch_size
#             batch_y = T.cat([x_array[:, 0:-1:args.skip][q[position * tbs + z], idx_ls, :] for z in range(bs)], dim=1)
#             assert T.equal(batched_u[:len(x_array[0, 0])],
#                                batch_y[0, :len(x_array[0, 0])]), 'starting target different from starting training data'
#
#         return batch_y.to(args.device)
#
#     elif args.cont_in == 'dt':
#
#         s, q = traindata_position_in_raw_data(args)
#         skip = args.skip(args.bs)
#         # assert idx_ls[-1] <= args.max_batch_time/args.skip, 'batch_time should be smaller than args.max_batch_time'
#
#         data_ = raw_sequential_data(args, plot_data=False).unsqueeze(2)
#
#         if train:
#             trbs = args.train_batch_size
#             batch_y = (data_[:, 0:-1:skip[z]][s[position * trbs + z], idx_ls, :]).transpose(2, 1)
#
#         else:
#             tbs = args.test_batch_size
#             batch_y = (data_[:, 0:-1:skip[z]][q[position * tbs + z], idx_ls, :]).transpose(2, 1)
#
#         return batch_y.to(args.device)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ at_NoBasis__del_2d  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def at_NoBasis__del_2d(**kwargs):
#     for B in ['GaussianRBF', 'None']:
#         # 'PiecewiseConstant']:  # ['Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#         for n_B in [4]:  # [1, 2, 3, 5, 10]:  # range(10):
#             for train_size in [15]:  # [5, 15, 25, 35, 45, 50]:
#                 for batch_time in [2]:  # [2, 6, 10, 18, 26, 38]:
#                     for rollout_t in range(5, 35, 4):  # range(4, 44, 5):  # 13 elements 3-39
#                         if B == 'GaussianRBF':
#                             batch_time = 6
#                         parser = argparse.ArgumentParser('ODE demo')
#                         parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
#                                             default='dopri5')  # dopri5') rk4
#                         # parser.add_argument('--n_time_steps', type=int, default=500)  # i.e. 0th 1th ..999th = 1000 time value
#                         # parser.add_argument('--train_size', type=int, default=0)
#                         # parser.add_argument('--test_size', type=int, default=0)
#                         # parser.add_argument('--batch_time', type=int, default=5)
#                         parser.add_argument('--batch_size', type=int, default=1)
#                         # parser.add_argument('--niters', type=int, default=65)
#                         parser.add_argument('--test_freq', type=int, default=20)
#                         # parser.add_argument('--viz', action='store_true')
#                         parser.add_argument('--viz', default=True)
#                         parser.add_argument('--gpu', type=int, default=0)
#                         parser.add_argument('--adjoint', action='store_true')
#                         parser.add_argument('--tr_style', type=str, choices=['MyclassV2', 'None'], default=None)
#                         parser.add_argument('--myepoch', type=list, default=[3, 4])
#                         parser.add_argument('--N_Basis', type=int, default=n_B)
#                         # parser.add_argument('--N_particles', type=int, default='101')
#                         parser.add_argument('--n_dense_Nodes', type=int, default='1000')
#                         parser.add_argument('--ConvLayer', type=str, choices=['PointNetLayer', 'GATConv'],
#                                             default='PointNetLayer')
#                         parser.add_argument('--data_type', type=str,
#                                             choices=['pde_1d', 'pde_2d', 'TrajectoryExtrapolation'],
#                                             default='pde_2d')
#                         parser.add_argument('--Basis', type=str,
#                                             choices=['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF',
#                                                      'MultiquadRBF'],
#                                             default='{}'.format(B))
#                         parser.add_argument('--op_type', type=str,
#                                             choices=[r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}'],
#                                             default=r'\Delta{u}*\Delta{u}')
#                         parser.add_argument('--adaptive_graph', action='store_true', default=False)
#                         args = parser.parse_args()
#                         args.train_size_used = train_size
#                         args.last_intg_t_idx_train = batch_time
#                         args.rollout_t = rollout_t
#                         print('Number of Basis:', n_B)
#                         print('Time of integration:', batch_time)
#                         print('Basis:', '{}'.format(B))
#                         model, optimizer, criterion, train_loader, test_loader, get_path, integrator, PlotResults = init_model(
#                             args, Parser_, **kwargs)
#                         path = get_path(args, W=1)
#                         pretrained_weights = osp.join(path, 'weights_lit{}'.format(batch_time))
#                         model.load_state_dict(torch.load(pretrained_weights))
#                         for epoch in [0]:
#                             # train_error = train(model, optimizer, criterion, train_loader, get_path, integrator,
#                             #                     PlotResults, epoch, batch_time, args, **kwargs)
#                             if B == 'GaussianRBF':
#                                 batch_time = rollout_t + 1
#                             test_error = test(model, test_loader, get_path, integrator, PlotResults, epoch, batch_time,
#                                               args, **kwargs)
#                             # tt = time.time() - start
#                             print(f'Epoch: {epoch:02d}, Loss: {test_error:.4f}')

#                         # ((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
#                         if not "fr" in vars():
#                             fr = np.array([[test_error]])
#                         else:
#                             fr = np.concatenate((fr, np.array([[test_error]])), axis=1)
#         if not "frr" in vars():
#             frr = fr
#         else:
#             frr = np.concatenate((frr, fr), axis=0)
#         del fr

#     if kwargs.get('save', None):
#         path2 = get_path(args, C1=1)
#         np.savetxt(osp.join(path2, 'at_NoBasis__del.csv'), frr, delimiter=",")
#     del frr
#     # ((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ all  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def all(**kwargs):
#     for B in ['Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#         for n_B in [3]:  # [1, 2, 3, 5, 10]:  # range(10):
#             for train_size in [15]:  # [5, 15, 25, 35, 45, 50]:
#                 for batch_time in [10]:  # [2, 6, 10, 18, 26, 38]:
#                     for last_intg_t_idx_test in [5, 10, 15, 20, 25, 30]:
#                         parser = argparse.ArgumentParser('ODE demo')
#                         parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
#                                             default='rk4')  # dopri5') rk4
#                         # parser.add_argument('--n_time_steps', type=int, default=500)  # i.e. 0th 1th ..999th = 1000 time value
#                         # parser.add_argument('--train_size', type=int, default=0)
#                         # parser.add_argument('--test_size', type=int, default=0)
#                         # parser.add_argument('--batch_time', type=int, default=5)
#                         parser.add_argument('--batch_size', type=int, default=1)
#                         # parser.add_argument('--niters', type=int, default=65)
#                         parser.add_argument('--test_freq', type=int, default=20)
#                         # parser.add_argument('--viz', action='store_true')
#                         parser.add_argument('--viz', default=True)
#                         parser.add_argument('--gpu', type=int, default=0)
#                         parser.add_argument('--adjoint', action='store_true')
#                         parser.add_argument('--tr_style', type=str, choices=['MyclassV2', 'None'], default=None)
#                         parser.add_argument('--myepoch', type=list, default=[3, 4])
#                         parser.add_argument('--N_Basis', type=int, default=n_B)
#                         # parser.add_argument('--N_particles', type=int, default='101')
#                         parser.add_argument('--n_dense_Nodes', type=int, default='1000')
#                         parser.add_argument('--ConvLayer', type=str, choices=['PointNetLayer', 'GATConv'],
#                                             default='PointNetLayer')
#                         parser.add_argument('--data_type', type=str,
#                                             choices=['pde_1d', 'pde_2d', 'TrajectoryExtrapolation'],
#                                             default='pde_2d')
#                         parser.add_argument('--Basis', type=str,
#                                             choices=['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF',
#                                                      'MultiquadRBF'],
#                                             default='{}'.format(B))
#                         parser.add_argument('--op_type', type=str,
#                                             choices=[r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}'],
#                                             default=r'\Delta{u}*\Delta{u}')
#                         parser.add_argument('--adaptive_graph', action='store_true', default=False)
#                         args = parser.parse_args()
#                         args.train_size_used = train_size
#                         args.last_intg_t_idx_train = batch_time
#                         print('Number of Basis:', n_B)
#                         print('Time of integration:', batch_time)
#                         print('Basis:', '{}'.format(B))
#                         model, optimizer, criterion, train_loader, test_loader, integrator, PlotResults = init_model(
#                             args, Parser_, **kwargs)
#                         path = args.get_path(args, W=1)
#                         pretrained_weights = osp.join(path, 'weights_lit{}'.format(batch_time))
#                         model.load_state_dict(torch.load(pretrained_weights))
#                         for epoch in [0]:  # range(0, args.niters):
#                             # if epoch % 2 == 0:
#                             #     plot = True
#                             # batch_time = args.last_intg_t_idx_train
#                             # batch_time = 2 + 1 * int(epoch / 3)
#                             # train_error = train(model, optimizer, criterion, train_loader, get_path, integrator, PlotResults,
#                             #                     epoch, batch_time, args, **kwargs)
#                             # batch_time = 10
#                             test_error = testing_epoch(model, test_loader, integrator, PlotResults, epoch, last_intg_t_idx_test,
#                                               args)
#                             # print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, Test Accuracy: {test_acc:.4f}')
#                             print(f'Epoch: {epoch:02d}, Loss: {test_error:.4f}')

#                         # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                 #         if not "cr" in vars():
#                 #             cr = np.array([[test_error]])
#                 #         else:
#                 #             cr = np.concatenate((cr, np.array([[test_error]])), axis=1)
#                 #     if not "crr" in vars():
#                 #         crr = cr
#                 #     else:
#                 #         crr = np.concatenate((crr, cr), axis=0)
#                 #     del cr
#                 # path2 = get_path(args, C2=1)
#                 # np.savetxt(osp.join(path2, 'lit_train__lit_test.csv'), crr, delimiter=",")
#                 # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#     #                     if not "br" in vars():
#     #                         br = np.array([[test_error]])
#     #                     else:
#     #                         br = np.concatenate((br, np.array([[test_error]])), axis=1)
#     #     if not "brr" in vars():
#     #         brr = br
#     #     else:
#     #         brr = np.concatenate((brr, br), axis=0)
#     #     del br
#     #
#     # path2 = get_path(args, C2=1)
#     # np.savetxt(osp.join(path2, 'B__lit_test.csv'), brr, delimiter=",")

#     # **********************************************************************
#     #         if not "ar" in vars():
#     #             ar = np.array([[test_error]])
#     #         else:
#     #             ar = np.concatenate((ar, np.array([[test_error]])), axis=1)
#     #     if not "arr" in vars():
#     #         arr = ar
#     #     else:
#     #         arr = np.concatenate((arr, ar), axis=0)
#     #     del ar
#     # # dictionary = {'n_B{}'.format(n_B): arr}
#     # path = get_path(args, C1=1)
#     # # scipy.io.savemat('test.mat', dict(x=x, y=y))
#     # # np.save(osp.join(path, 'n_B{}.npy'.format(n_B)), arr)
#     # np.savetxt(osp.join(path, 'n_B{}.csv'.format(n_B)), arr, delimiter=",")
#     # del arr
#     # **********************************************************************

#     # =============================================================================
#     #         if not "arrr" in vars():
#     #             arrr = np.array([[test_error]])
#     #         else:
#     #             arrr = np.concatenate((arrr, np.array([[test_error]])), axis=1)  # arrr = [1,5]
#     #
#     #     if not "arrrr" in vars():
#     #         arrrr = arrr
#     #     else:
#     #         arrrr = np.concatenate((arrrr, arrr), axis=0)
#     #     del arrr
#     #
#     # path2 = get_path(args, C2=1)
#     # np.savetxt(osp.join(path2, 'B__n_B.csv'), arrrr, delimiter=",")
#     # =============================================================================



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ c_pos  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def c_pos(self, pos_j):
#     """ Change value of positions of nodes connected to nodes on boundary of domain
#     Args:
#         pos_j (tensor): stacked positions of nodes connected to every node
#                         i.e. :math:`[[p_{j_1}^{i_1}x,y], [p_{j_2}^{i_1}x,y],
#                         [p_{j_1}^{i_2}x,y], [p_{j_2}^{i_2}x,y],...
#                         [p_{j_1}^{i_N}x,y, p_{j_2}^{i_N}x,y]]`
#                         shape [n_edges * batch_size, co-ordinates(2 i.e.(x, y))]
#     """
#
#     with T.no_grad():
#         if self.args.data_type == 'pde_1d':
#             ln = len(pos_j[:, 0]) // self.args.bs_
#             for d in range(0, len(pos_j[:, 0]), ln):
#                 for n0 in [0, 1, self.args.n_Nodes - 2, self.args.n_Nodes - 1]:
#                     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
#                     ok = T.tensor([n0 - 1, n0 + 1, n0 - 2, n0 + 2]).unsqueeze(1)
#                     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
#
#                 # for n0 in [0]:
#                 #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
#                 #     ok = T.tensor([n0 + 1, n0 - 1, n0 + 2, n0 - 2]).unsqueeze(1)
#                 #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
#                 #
#                 # for n0 in [1]:
#                 #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
#                 #     ok = T.tensor([n0 - 1, n0 + 1, n0 + 2, n0 - 2]).unsqueeze(1)
#                 #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
#                 #
#                 # for n0 in [self.args.n_Nodes - 1]:
#                 #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
#                 #     ok = T.tensor([n0 - 1, n0 + 1, n0 - 2, n0 + 2]).unsqueeze(1)
#                 #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
#                 #
#                 # for n0 in [self.args.n_Nodes - 2]:
#                 #     # ok = T.tensor([n0 + d + 1, n0 + d - 1, n0 + d + 2, n0 + d - 2]).unsqueeze(1)
#                 #     ok = T.tensor([n0 + 1, n0 - 1, n0 - 2, n0 + 2]).unsqueeze(1)
#                 #     pos_j[4 * n0 + d:4 * n0 + d + 4] = ok
#                 #
#                 #     # ok = T.tensor([n0 + 1, n0 - 1, n0 + 2, n0 - 2, n0 + 3, n0 - 3]).unsqueeze(1)
#                 #     # pos_j[6 * n0:6 * n0 + 6] = ok
#
#         elif self.args.data_type == 'pde_2d':
#
#             #  compute pos_j once and use same until batch size changes
#             # print('cpos_called', ' bs:', self.args.bs_, ' bs_prev:', self.bs_prev, ' len:', len(pos_j[:, 0]))
#             if hasattr(self.args, 'pos_j') and self.args.bs_ == self.bs_prev:
#                 pos_j = self.args.pos_j
#             else:
#                 if self.args.stencil == 5:
#                     ln = len(pos_j[:, 0]) // self.args.bs_
#                     for d in range(0, len(pos_j[:, 0]), ln):
#                         loop = [range(0, 4032, 64), range(4032, 4095),
#                                 range(4095, 63, -64), range(63, 0, -1)]
#                         for i in loop:
#                             for n in i:
#                                 pos = T.ones(4, 2)
#                                 n0 = (n - n % 64) // 64
#                                 n1 = n % 64
#                                 pos[:, 0] = T.tensor([n0 - 1, n0, n0, n0 + 1])
#                                 pos[:, 1] = T.tensor([n1, n1 - 1, n1 + 1, n1])
#                                 pos_j[4 * n + d:4 * n + d + 4, :] = pos
#
#                 elif self.args.stencil == 9:
#                     loop = [range(0, 4032, 64), range(4032, 4095), range(4095, 63, -64), range(63, 0, -1)]
#                     for i in loop:
#                         ignore = 0
#                         for n in i:
#                             pos = T.ones(9, 2)
#                             n0 = (n - n % 64) // 64
#                             n1 = n % 64
#                             pos[:, 0] = T.tensor([n0, n0, n0 + 1, n0 - 1, n0, n0 + 1, n0 - 1, n0 + 1, n0 - 1])
#                             pos[:, 1] = T.tensor([n1, n1 + 1, n1, n1, n1 - 1, n1 + 1, n1 - 1, n1 - 1, n1 + 1])
#                             pos_j[8 * n:9 * n + 9, :] = pos
#
#                 else:
#                     ln = len(pos_j[:, 0]) // self.args.bs_
#                     for d in range(0, len(pos_j[:, 0]), ln):
#                         loop = [range(0, 3906, 63), range(3906, 3968),
#                                 range(3968, 62, -63), range(62, 0, -1)]
#                         for i in loop:
#                             for n in i:
#                                 pos = T.ones(8, 2)
#                                 n0 = (n - n % 63) // 63
#                                 n1 = n % 63
#                                 pos[:, 0] = T.tensor([n0 - 1, n0, n0, n0 + 1, n0 - 1, n0 - 1, n0 + 1, n0 + 1])
#                                 pos[:, 1] = T.tensor([n1, n1 - 1, n1 + 1, n1, n1 - 1, n1 + 1, n1 - 1, n1 + 1])
#                                 pos_j[8 * n + d:8 * n + d + 8, :] = pos
#
#                 self.args.pos_j = pos_j
#                 self.bs_prev = self.args.bs_
#
#         return pos_j

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ get_edge_index  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def get_edge_index(self, hh, pos, batch):
#     """Connect nodes with its neighbours. Compute edge_index once and use same until batch size changes
#
#     Args:
#         hh (tensor): net input [n_Nodes * batch_size, in_features]
#         pos (tensor): position of nodes [n_Nodes * batch_size, co-ordinates(2 i.e.(x, y))]
#         batch (tensor): indexing for pos [n_Nodes * batch_size] e.g {0, 0, 1, 1, 2, 2...19, 19}
#                         for batch_size=20, n_Nodes=2
#     Return:
#         edge_index (tensor): pairs of nodes connected by an edge [2, n_edges]
#     """
#
#     if self.args.data_type == 'pde_1d':
#
#         #  compute edge_index once and use same until batch size changes
#         if hasattr(self.args, 'get_edge_index') and T.equal(batch, self.batch_prev):
#             edge_index = self.args.get_edge_index
#         else:
#             if self.args.stencil == 5:
#                 edge_index = knn_graph(pos, k=4, batch=batch, loop=False)
#                 ed = edge_index
#
#                 l1 = len(edge_index[0]) // self.args.bs_
#                 tn = self.args.n_Nodes * self.args.bs_
#                 for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):
#
#                     for j in [0, 1, 509, 510]:
#                         edge_corners = T.ones(2, 4)
#                         edge_corners[1] = j + n_
#                         if j == 0:
#                             edge_corners[0] = T.tensor([510 + n_, 1 + n_, 509 + n_, 2 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#                         if j == 1:
#                             edge_corners[0] = T.tensor([0 + n_, 2 + n_, 510 + n_, 3 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#                         if j == 509:
#                             edge_corners[0] = T.tensor([508 + n_, 510 + n_, 507 + n_, 0 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#                         if j == 510:
#                             edge_corners[0] = T.tensor([509 + n_, 0 + n_, 508 + n_, 1 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#             self.args.get_edge_index = edge_index
#             self.batch_prev = batch
#
#         # # edge index for periodic bc
#         # cor = np.linspace(0, 2 * np.pi, self.args.n_Nodes + 1)
#         # cor = cor[:-1]
#         # c_pos = np.zeros((self.args.n_Nodes, 2))
#         # c_pos[:, 0] = np.cos(cor)
#         # c_pos[:, 1] = np.sin(cor)
#         # c_pos = T.tensor(c_pos).to(self.args.device)
#         # c_pos = T.cat([c_pos] * self.args.bs_)
#         # edge_index = knn_graph(c_pos, k=self.k, batch=batch, loop=False)
#
#     elif self.args.data_type == 'pde_2d':
#
#         #  compute edge_index once and use same until batch size changes
#         if hasattr(self.args, 'get_edge_index') and T.equal(batch, self.batch_prev):
#             edge_index = self.args.get_edge_index
#         else:
#             if self.args.stencil == 5:
#                 # edge_index = T.zeros((2, 64 * 64 * 4), dtype=int)
#                 # zy = T.arange(64 * 64)
#                 # zy = zy.unsqueeze(1)
#                 # z1 = zy
#                 # z2 = zy - 1
#                 # z3 = zy + 1
#                 # z4 = zy + 64
#                 # z5 = zy - 64
#                 # zx = T.cat([z2, z3, z4, z5], dim=1)
#                 # edge_index[0] = zx.reshape(-1)
#                 # edge_index[1] = T.cat([z1] * 4, dim=1).reshape(-1)
#                 # edge_index = edge_index.to(self.args.device)
#
#                 edge_index = knn_graph(pos, k=4, batch=batch, loop=False)
#                 ed = edge_index
#
#                 l1 = len(edge_index[0]) // self.args.bs_
#                 tn = self.args.n_Nodes * self.args.bs_
#                 for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):
#                     # for d in range(0, l1*7, l1):
#                     for i in range(1, 63):
#                         edge_l = T.ones(2, 4)
#                         l = i
#                         ll = i + n_
#                         edge_l[1] = ll
#                         edge_l[0] = T.tensor([ll + 64 * 63, ll - 1, ll + 1, ll + 64])
#                         edge_index[:, 4 * l + p_:4 * l + p_ + 4] = edge_l
#
#                         edge_r = T.ones(2, 4)
#                         r = i + 4032
#                         rr = i + 4032 + n_
#                         edge_r[1] = rr
#                         edge_r[0] = T.tensor([rr - 64, rr - 1, rr + 1, rr - 64 * 63])
#                         edge_index[:, 4 * r + p_:4 * r + p_ + 4] = edge_r
#
#                         edge_b = T.ones(2, 4)
#                         b = i * 64
#                         bb = i * 64 + n_
#                         edge_b[1] = bb
#                         edge_b[0] = T.tensor([bb - 64, bb + 64 - 1, bb + 1, bb + 64])
#                         edge_index[:, 4 * b + p_:4 * b + p_ + 4] = edge_b
#
#                         edge_u = T.ones(2, 4)
#                         u = i * 64 + 63
#                         uu = i * 64 + 63 + n_
#                         edge_u[1] = uu
#                         edge_u[0] = T.tensor([uu - 64, uu - 1, uu - 64 + 1, uu + 64])
#                         edge_index[:, 4 * u + p_:4 * u + p_ + 4] = edge_u
#
#                     for j in [0, 63, 4032, 4095]:
#                         edge_corners = T.ones(2, 4)
#                         edge_corners[1] = j + n_
#                         if j == 0:
#                             edge_corners[0] = T.tensor([4032 + n_, 63 + n_, 1 + n_, 64 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#                         if j == 63:
#                             edge_corners[0] = T.tensor([4095 + n_, 62 + n_, 0 + n_, 127 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#                         if j == 4032:
#                             edge_corners[0] = T.tensor([3968 + n_, 4095 + n_, 4033 + n_, 0 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#                         if j == 4095:
#                             edge_corners[0] = T.tensor([4031 + n_, 4094 + n_, 4032 + n_, 63 + n_])
#                             edge_index[:, 4 * j + p_:4 * j + p_ + 4] = edge_corners
#
#             elif self.args.stencil == 9:
#                 edge_index = T.zeros((2, 64 * 64 * 9))
#                 zy = T.arange(64 * 64)
#                 zy = zy.unsqueeze(1)
#                 z1 = zy
#                 z2 = zy - 1
#                 z3 = zy + 1
#                 z4 = zy + 64
#                 z5 = zy - 64
#                 z6 = zy + 2
#                 z7 = zy - 2
#                 z8 = zy + 64 * 2
#                 z9 = zy - 64 * 2
#                 zx = T.cat([z1, z2, z3, z4, z5, z6, z7, z8, z9], dim=1)
#                 edge_index[0] = zx.reshape(-1)
#                 edge_index[1] = T.cat([z1] * 9, dim=1).reshape(-1)
#
#                 for i in range(2, 62):
#                     edge_l = T.ones(2, 9)
#                     edge_l[1] = i
#                     edge_l[0] = T.tensor([i, i - 1, i + 1, i + 64, i + 64 * 63, i + 64 * 2,
#                                           i + 2, i - 2, i + 64 * 63 - 64])
#                     edge_index[:, 9 * i:9 * i + 9] = edge_l
#
#                     edge_r = T.ones(2, 9)
#                     rr = i + 4032
#                     edge_r[1] = rr
#                     edge_r[0] = T.tensor([rr, rr - 1, rr + 1, rr - 64, rr - 64 * 63, rr - 64 * 2,
#                                           rr + 2, rr - 2, rr - 64 * 63 + 64])
#                     edge_index[:, 9 * rr:9 * rr + 9] = edge_r
#
#                     edge_b = T.ones(2, 9)
#                     bb = i * 64
#                     edge_b[1] = bb
#                     edge_b[0] = T.tensor([bb, bb - 64, bb + 1, bb + 64, bb + 64 - 1,
#                                           bb - 64 * 2, bb + 64 * 2, bb + 2, bb + 64 - 2])
#                     edge_index[:, 9 * bb:9 * bb + 9] = edge_b
#
#                     edge_u = T.ones(2, 9)
#                     uu = i * 64 + 63
#                     edge_u[1] = uu
#                     edge_u[0] = T.tensor([uu, uu - 64, uu - 1, uu - 64 + 1, uu + 64,
#                                           uu - 64 * 2, uu + 64 * 2, uu - 2, uu - 64 + 2])
#                     edge_index[:, 9 * uu:9 * uu + 9] = edge_u
#                     # =======================================
#
#                     edge_li = T.ones(2, 9)
#                     edge_li[1] = i + 64
#                     edge_li[0] = T.tensor([i, i - 1, i + 1, i + 64, i - 64, i + 64 * 2,
#                                            i + 2, i - 2, i + 64 * 63 - 64])
#                     edge_index[:, 9 * i:9 * i + 9] = edge_li
#
#                     edge_ri = T.ones(2, 9)
#                     rr = i + 4032 - 64
#                     edge_ri[1] = rr
#                     edge_ri[0] = T.tensor([rr, rr - 1, rr + 1, rr - 64, rr + 64, rr - 64 * 2,
#                                            rr + 2, rr - 2, rr - 64 * 63 + 64])
#                     edge_index[:, 9 * rr:9 * rr + 9] = edge_ri
#
#                     edge_bi = T.ones(2, 9)
#                     bb = i * 64 + 1
#                     edge_bi[1] = bb
#                     edge_bi[0] = T.tensor([bb, bb - 64, bb + 1, bb + 64, bb - 1,
#                                            bb - 64 * 2, bb + 64 * 2, bb + 2, bb + 64 - 2])
#                     edge_index[:, 9 * bb:9 * bb + 9] = edge_bi
#
#                     edge_ui = T.ones(2, 9)
#                     uu = i * 64 + 63 - 1
#                     edge_ui[1] = uu
#                     edge_ui[0] = T.tensor([uu, uu - 64, uu - 1, uu + 1, uu + 64,
#                                            uu - 64 * 2, uu + 64 * 2, uu - 2, uu - 64 + 2])
#                     edge_index[:, 9 * uu:9 * uu + 9] = edge_ui
#
#                 for j in [0, 63, 4032, 4095, 62, 1, 64, 3968, 4033, 4095, 4031, 127, 65, 3969, 4030, 126]:
#                     edge_corners = T.ones(2, 9)
#                     edge_corners[1] = j
#                     if j == 0:
#                         edge_corners[0] = T.tensor([j, 1, 64, 63, 4032, 2, 3968, 128, 62])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 63:
#                         edge_corners[0] = T.tensor([j, 62, 127, 0, 4095, 61, 191, 4031, 1])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 4032:
#                         edge_corners[0] = T.tensor([j, 4033, 3968, 4095, 0, 3904, 4034, 4094, 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 4095:
#                         edge_corners[0] = T.tensor([j, 4094, 4031, 4032, 63, 127, 3967, 4093, 4033])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 62:
#                         edge_corners[0] = T.tensor(
#                             [j, j - 1, j + 1, j + 64, j + 64 * 63, j + 64 * 2, j + 2 - 64,
#                              j - 2, j + 64 * 63 - 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 1:
#                         edge_corners[0] = T.tensor([j, j - 1, j + 1, j + 64, j + 64 * 63, j + 64 * 2,
#                                                     j + 2, j - 2 + 64, j + 64 * 63 - 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 64:
#                         edge_corners[0] = T.tensor([j, 4033, 3968, 4095, 0, 3904, 4034, 4094, 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 3968:
#                         edge_corners[0] = T.tensor([j, 4094, 4031, 4032, 63, 127, 3967, 4093, 4033])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 4033:
#                         edge_corners[0] = T.tensor([rr, rr - 1, rr + 1, rr - 64, rr - 64 * 63, rr - 64 * 2,
#                                                     rr + 2, rr - 2, rr - 64 * 63 + 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 4094:
#                         edge_corners[0] = T.tensor([rr, rr - 1, rr + 1, rr - 64, rr - 64 * 63, rr - 64 * 2,
#                                                     rr + 2, rr - 2, rr - 64 * 63 + 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 4031:
#                         edge_corners[0] = T.tensor([j, 4033, 3968, 4095, 0, 3904, 4034, 4094, 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 127:
#                         edge_corners[0] = T.tensor([j, 4094, 4031, 4032, 63, 127, 3967, 4093, 4033])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 65:
#                         edge_corners[0] = T.tensor([j, 1, 64, 63, 4032, 2, 3968, 128, 62])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 3969:
#                         edge_corners[0] = T.tensor([j, 62, 127, 0, 4095, 61, 191, 4031, 1])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 4030:
#                         edge_corners[0] = T.tensor([j, 4033, 3968, 4095, 0, 3904, 4034, 4094, 64])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#                     if j == 126:
#                         edge_corners[0] = T.tensor([j, 4094, 4031, 4032, 63, 127, 3967, 4093, 4033])
#                         edge_index[:, 9 * j:9 * j + 9] = edge_corners
#
#             else:
#                 edge_index = knn_graph(pos, k=8, batch=batch, loop=False)
#
#                 l1 = len(edge_index[0]) // self.args.bs_
#                 tn = self.args.n_Nodes * self.args.bs_
#                 for n_, p_ in zip(range(0, tn, self.args.n_Nodes), range(0, len(edge_index[0]), l1)):
#                     for i in range(1, 63):
#                         edge_l = T.ones(2, 8)
#                         l = i
#                         ll = i + n_
#                         edge_l[1] = ll
#                         # edge_[0] = T.tensor([1, 0, 2, 65, 4033, 66, 64, 4034, 4032])
#                         edge_l[0] = T.tensor([ll + 64 * 63, ll + 1, ll - 1, ll + 64,
#                                               ll + 64 * 63 - 1, ll + 64 * 63 + 1, ll + 64 - 1, ll + 64 + 1])
#                         edge_index[:, 8 * l + p_:8 * l + p_ + 8] = edge_l
#
#                         edge_r = T.ones(2, 8)
#                         r = i + 4032
#                         rr = i + 4032 + n_
#                         edge_r[1] = rr
#                         # edge_[0] = T.tensor([1, 0, 2, 65, 4033, 66, 64, 4034, 4032])
#                         edge_r[0] = T.tensor([rr - 64, rr + 1, rr - 1, rr - 64 * 63,
#                                               rr - 64 - 1, rr - 64 + 1, rr - 64 * 63 - 1, rr - 64 * 63 + 1])
#                         edge_index[:, 8 * r + p_:8 * r + p_ + 8] = edge_r
#
#                         edge_b = T.ones(2, 8)
#                         b = i * 64
#                         bb = i * 64 + n_
#                         edge_b[1] = bb
#                         # edge__[0] = T.tensor([64, 0, 65, 128, 127, 1, 129, 63, 191])
#                         edge_b[0] = T.tensor([bb - 64, bb + 1, bb + 64 - 1, bb + 64,
#                                               bb - 1, bb - 64 + 1, bb + 64 + 63, bb + 64 + 1])
#                         edge_index[:, 8 * b + p_:8 * b + p_ + 8] = edge_b
#
#                         edge_u = T.ones(2, 8)
#                         u = i * 64 + 63
#                         uu = i * 64 + 63 + n_
#                         edge_u[1] = uu
#                         # edge__[0] = T.tensor([64, 0, 65, 128, 127, 1, 129, 63, 191])
#                         edge_u[0] = T.tensor([uu - 64, uu - 64 + 1, uu - 1, uu + 64,
#                                               uu - 64 - 1, uu - 64 + 1 - 64, uu + -1 + 64, uu - 64 + 1 + 64])
#                         edge_index[:, 8 * u + p_:8 * u + p_ + 8] = edge_u
#
#                     for j in [0, 63, 4032, 4095]:
#                         edge_corners = T.ones(2, 8)
#                         edge_corners[1] = j + n_
#                         if j == 0:
#                             edge_corners[0] = T.tensor([4032, 1, 63, 64, 4095, 4033, 127, 65])
#                             edge_index[:, 8 * j + p_:8 * j + p_ + 8] = edge_corners
#
#                         if j == 63:
#                             edge_corners[0] = T.tensor([4095, 0, 62, 127, 4094, 4032, 126, 64])
#                             edge_index[:, 8 * j + p_:8 * j + p_ + 8] = edge_corners
#
#                         if j == 4032:
#                             edge_corners[0] = T.tensor([3968, 4033, 4095, 0, 4031, 3969, 63, 1])
#                             edge_index[:, 8 * j + p_:8 * j + p_ + 8] = edge_corners
#
#                         if j == 4095:
#                             edge_corners[0] = T.tensor([4094, 4031, 4032, 63, 62, 3968, 4030, 0]) # might be wrong
#                             edge_index[:, 8 * j + p_:8 * j + p_ + 8] = edge_corners
#
#                 # self.visualize_points(pos, edge_index)
#
#             self.args.get_edge_index = edge_index
#             self.batch_prev = batch
#
#     elif self.args.data_type == 'TrajectoryExtrapolation':
#
#         for i in range(len(hh)):
#             for j in range(len(hh)):
#                 if 2 * LA.norm(hh[i, :2] - hh[j, :2]) <= self.r:  # and i != j:
#                     # data.edge_index[] =
#                     edge_ = T.empty(2, 1, dtype=int).to(self.device)
#                     edge_[0] = j
#                     edge_[1] = i
#                     try:
#                         T.is_tensor(edge_index)
#                         edge_index = T.cat((edge_index, edge_), dim=1)
#                     except:
#                         edge_index = edge_
#
#     return edge_index

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ plot graph output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# a[0].plot(u0[:, sl])
# a[0].set_title('ux')
# a[1].plot(u1[:, sl])
# a[1].set_title('uy')

# b[0].plot(u0_0[:, sl])
# b[0].set_title('ux-dx')
# b[1].plot(rs(h1, 0)[:, sl])
# b[2].plot(u1_1[:, sl])
# b[2].set_title('uy-dy')
# b[3].plot(rs(h1, 1)[:, sl])
# b[4].plot(u0_1[:, sl])
# b[4].set_title('ux-dy')
# b[5].plot(rs(h1, 2)[:, sl])
# b[6].plot(u1_0[:, sl])
# b[6].set_title('uy-dx')
# b[7].plot(rs(h1, 3)[:, sl])

# c[0].plot(u0_00[:, sl])
# c[0].set_title('ux-dx-dx')
# c[1].plot(rs(h2, 0)[:, sl])
# c[2].plot(u0_01[:, sl])
# c[2].set_title('ux-dx-dy')
# c[3].plot(rs(h2, 4)[:, sl])

# c[4].plot(u0_10[:, sl])
# c[4].set_title('ux-dy-dx')
# c[5].plot(rs(h2, 2)[:, sl])
# c[6].plot(u0_11[:, sl])
# c[6].set_title('ux-dy-dy')
# c[7].plot(rs(h2, 6)[:, sl])

# c[8].plot(u1_00[:, sl])
# c[8].set_title('uy-dx-dx')
# c[9].plot(rs(h2, 3)[:, sl])
# c[10].plot(u1_01[:, sl])
# c[10].set_title('uy-dx-dy')
# c[11].plot(rs(h2, 7)[:, sl])

# c[12].plot(u1_10[:, sl])
# c[12].set_title('uy-dy-dx')
# c[13].plot(rs(h2, 1)[:, sl])
# c[14].plot(u1_11[:, sl])
# c[14].set_title('uy-dy-dy')
# c[15].plot(rs(h2, 5)[:, sl])

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ TE Layers @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# if args.ConvLayer == 'PointNetLayer':
#
#     args.n_linear_layers = 2
#
#     args.hidden = 16
#     self.conv = nn.ModuleDict({'1': PointNetLayer(args, in_features * 2, self.conv_hidden_dim1, 1,
#                                                   self.agmnt_featur, basisfunc=self.basisfunc,
#                                                   dilation=self.dilation, shift=self.shift)})
#     self.conv['{:d}'.format(2)] = linear(args, self.conv_hidden_dim1 + 4, out_features,
#                                          basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)
#
# elif args.ConvLayer == 'GATConv2':
#
#     args.n_linear_layers = 2
#
#     args.hidden = 16
#     self.conv = nn.ModuleDict({'1': GATConv2(args,
#                                              in_features * 2, self.conv_hidden_dim1, 1,
#                                              self.agmnt_featur, heads=2, concat=False,
#                                              basisfunc=self.basisfunc, dilation=self.dilation,
#                                              shift=self.shift)})
#     self.conv['{:d}'.format(2)] = linear(args, self.conv_hidden_dim1 + 4, out_features,
#                                          basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ Different plot functions TE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def train_layer__bt_test_TE(**kwargs):
#     args.data_type = args.data_types[2]
#     args.Layers = args.Layers_[0:1] + args.Layers_[3:4]  #
#     args.Bases = args.Bases_[1:2]
#     args.N_Bases = [3]
#     args.tsu = [30]
#     args.bts = [2] * 5 + [3] * 6 + [4] * 7 + [5] * 7 + [6] * 7 + [7] * 5
#     args.batch_times = [5]
#     args.test_bt = 10
#     args.exp = 'layer__bt_test_TE'
#     _ = save_logs(args)
#     train_loop(**kwargs)
#
#
# def layer__bt_test_TE(**kwargs):
#     args.data_type = args.data_types[2]
#     args.Layers = args.Layers_[0:1] + args.Layers_[3:4]  #
#     args.Bases = args.Bases_[1:2]
#     args.N_Bases = [3]
#     args.tsu = [30]
#     args.batch_times = list(range(4, 28, 4))
#     args.vary = args.vary_[4:5] + args.vary_[0:1]
#     args.exp = 'layer__bt_test_TE'
#     args.save_logs = False
#     get_path = save_logs(args)
#     sv_dir = get_path(args, C3=1, rslt_dir='layer__bt_test_TE')
#     test_loop('layer__bt_test_TE.csv', sv_dir, **kwargs)
#     sv_args = args
#     save_args(sv_args, sv_dir)
#
#
# def train_B__bt_test_TE(**kwargs):
#     args.data_type = args.data_types[2]
#     args.Layers = args.Layers_[3:4]
#     args.Bases = args.Bases_[0:7]
#     args.N_Bases = [3]
#     args.tsu = [30]
#     args.bts = [2] * 5 + [3] * 6 + [4] * 7 + [5] * 7 + [6] * 7 + [7] * 5
#     args.batch_times = [5]
#     args.test_bt = 10
#     args.exp = 'B__bt_test_TE'
#     _ = save_logs(args)
#     train_loop(**kwargs)
#
#
# def B__bt_test_TE(**kwargs):
#     args.data_type = args.data_types[2]
#     args.Layers = args.Layers_[3:4]
#     args.Bases = args.Bases_[0:7]
#     args.N_Bases = [3]
#     args.tsu = [30]
#     args.batch_times = list(range(4, 28, 4))
#     args.vary = args.vary_[4:5] + args.vary_[1:2]
#     args.exp = 'B__bt_test_TE'
#     args.save_logs = False
#     get_path = save_logs(args)
#     sv_dir = get_path(args, C3=1, rslt_dir='B__bt_test_TE')
#     test_loop('B__bt_test_TE.csv', sv_dir, **kwargs)
#     sv_args = args
#     save_args(sv_args, sv_dir)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ Calculate TE output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# else:  # self.args.data_type == 'TrajectoryExtrapolation':
# h1 = self.conv['{:d}'.format(1)](hh, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
# h1 = self.activation(h1)
# h2 = torch.cat([hh.float(), h1], dim=-1)
# self.conv['{:d}'.format(2)].t = t
# h3 = self.conv['{:d}'.format(2)](h2)
#
# retrn = torch.empty(len(hh[:, 0]), 4).to(self.args.device)
# retrn[:, :2] = hh[:, 2:4]
# retrn[:, 2:] = h3  # -hh[:, :2] * 2 - h3
# h = retrn

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ PointNetLayer_No_B @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class PointNetLayer_No_B(MessagePassing):
#     # def __init__(self, bias=True, basisfunc=Fourier(5), dilation=True, shift=True):
#     #     super().__init__()
#     #     self.dilation = torch.ones(1) if not dilation else nn.Parameter(data=torch.ones(1), requires_grad=True)
#     #     self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
#     #     self.basisfunc = basisfunc
#     #     self.n_eig = n_eig = self.basisfunc.n_eig
#     #     self.deg = deg = self.basisfunc.deg
#
#     def __init__(self, args, in_features, out_features, layer_num, agmnt_featur, bias=True):
#         # Message passing with "max" aggregation.
#         super(PointNetLayer_No_B, self).__init__('mean')
#
#         # if
#         self.args = args
#         self.device = args.device
#         self.agmnt_featur = agmnt_featur
#         self.layer_num = layer_num
#         self.in_features, self.out_features = in_features, out_features
#
#         if hasattr(args, 'hidden'):
#             self.hidden = args.hidden
#         else:
#             if self.args.n_linear_layers > 1:
#                 assert 1 == 0, 'number of Neurones in hidden layer is required'
#             else:
#                 self.hidden = out_features
#         # if hidden is None:
#         #     self.hidden = self.out_features  # out_features
#         # else:
#         #     assert 1 == 0, 'number of neuron in hidden layers is required'
#         self.weight = torch.Tensor(out_features, in_features)
#         if bias:
#             self.bias = torch.Tensor(out_features)
#         else:
#             self.register_parameter('bias', None)
#
#         # aaa = torch.nn.Parameter(torch.Tensor((in_features + 1) * self.hidden, self.deg, self.n_eig))#.to(self.args.device)
#         self.ModuleDict = torch.nn.ModuleDict(
#             {'1': Linear(in_features, self.hidden)})
#         for c in range(self.args.n_linear_layers - 1):
#             if c + 2 < self.args.n_linear_layers:
#                 self.ModuleDict['{:d}'.format(c + 2)] = Linear(self.hidden, self.hidden)
#             else:
#                 self.ModuleDict['{:d}'.format(c + 2)] = Linear(self.hidden, out_features)
#
#         # self.coeffs1 = torch.nn.Parameter(torch.Tensor((in_features + 1) * self.hidden, self.deg, self.n_eig))
#         # self.coeffs2 = torch.nn.Parameter(torch.Tensor((self.hidden + 1) * out_features, self.deg, self.n_eig))
#         self.reset_parameters()
#
#     # def __init__(self, time_d, in_features, out_features, N_Basis):
#     #     # Message passing with "max" aggregation.
#     #     super(PointNetLayer, self).__init__('add')
#     #     self.time_d = time_d
#     #     self.in_features = in_features
#     #     self.out_features = out_features
#     #
#     #     # Initialization of the MLP:
#     #     # Here, the number of input features correspond to the hidden node
#     #     # dimensionality plus point dimensionality (=3).
#     #     # self.mlp = Sequential(Linear(in_features * 2, out_features),
#     #     #                       ReLU(),
#     #     #                       Linear(out_features, out_features))
#     #
#     #     # self.weight = torch.nn.Parameter(torch.randn(3, out_features, in_features, width, width))
#     #     # self.bias = torch.nn.Parameter(torch.zeros(3, out_features))
#     #     self.N_Basis = N_Basis
#     #     self.weight = torch.nn.Parameter(torch.randn(N_Basis, out_features, in_features * 2))
#     #     self.bias = torch.nn.Parameter(torch.zeros(N_Basis, out_features))
#     #     kl = 0
#
#     # def xpnd_in(self, in_dim):
#     #     if self.args.data_type == '1Dwave':
#     #         # expanded_in = in_dim * 2 + 2
#     #         expanded_in = in_dim * 2 + 1  # TODO:
#     #
#     #     elif self.args.data_type == 'pde_2d':
#     #         expanded_in = in_dim * 2 + 2
#     #
#     #     elif self.args.data_type == 'TrajectoryExtrapolation':
#     #         expanded_in = in_dim * 2
#     #
#     #     return expanded_in
#
#     def reset_parameters(self):
#         pass
#         # torch.nn.init.normal_(self.coeffs['{:d}'.format(1)])
#         # for c in range(self.args.n_linear_layers - 1):
#         #     torch.nn.init.normal_(self.coeffs['{:d}'.format(c + 2)])
#
#     def forward(self, h, edge_index, pos, edge_attr, t):
#         # Start propagating messages.
#         # print('edge_index ==',edge_index, 'h==',h)
#         return self.propagate(edge_index, pos=pos, edge_attr=edge_attr, h=h,
#                               t=t)  # pos=pos, edge_attr=edge_attr, h=h, t=t)
#
#     # def forward(self, input):
#     #     # For the moment, GalLayers rely on DepthCat to access the `s` variable. A better design would free the user
#     #     # of having to introduce DepthCat(1) every time a GalLayer is used
#     #     s = input[-1, -1]
#     #     input = input[:, :-1]
#     #     w = self.calculate_weights(s)
#     #     self.weight = w[0:self.in_features * self.out_features].reshape(self.out_features, self.in_features)
#     #     self.bias = w[self.in_features * self.out_features:(self.in_features + 1) * self.out_features].reshape(
#     #         self.out_features)
#     #     return torch.nn.functional.linear(input, self.weight, self.bias)
#
#     def message(self, h_j, h_i, pos_j, pos_i, t):
#         s = t
#         input = self.agmnt_featur(self.layer_num, h_j, h_i, pos_j, pos_i)
#         # diff = h_j - h_i  # Compute spatial relation.
#         # Rpos = pos_j - pos_i
#         #
#         # if self.args.data_type == 'pde_1d':
#         #     if h_j is not None:
#         #         input = torch.cat([h_j / 5, diff, Rpos], dim=-1)
#         #         # input = torch.cat([diff, Rpos], dim=-1)
#         #         input = input.float()
#         #
#         # elif self.args.data_type == 'pde_2d':
#         #     if h_j is not None:
#         #         input = torch.cat([h_j, diff, Rpos], dim=-1)
#         #         input = input.float()
#         #
#         # elif self.args.data_type == 'TrajectoryExtrapolation':
#         #     if h_j is not None:
#         #         input = torch.cat([h_j, diff], dim=-1)
#         #         input = input.float()
#
#         valu1 = self.ModuleDict['{:d}'.format(1)](input)
#
#         for c in range(self.args.n_linear_layers - 1):
#             valu1 = valu1.relu()
#             valu1 = self.ModuleDict['{:d}'.format(c + 2)](valu1)
#
#         return valu1

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ ORIGINAL GATConv @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class GATConv(MessagePassing):
#     r"""The graph attentional operator from the `"Graph Attention Networks"
#     <https://arxiv.org/abs/1710.10903>`_ paper
#
#     .. math::
#         \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
#         \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
#
#     where the attention coefficients :math:`\alpha_{i,j}` are computed as
#
#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
#         \right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
#         \right)\right)}.
#
#     Args:
#         in_features (int or tuple): Size of each input sample. A tuple
#             corresponds to the sizes of source and target dimensionalities.
#         out_features (int): Size of each output sample.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         concat (bool, optional): If set to :obj:`False`, the multi-head
#             attentions are averaged instead of concatenated.
#             (default: :obj:`True`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         add_self_loops (bool, optional): If set to :obj:`False`, will not add
#             self-loops to the input graph. (default: :obj:`True`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     _alpha: OptTensor
#
#     def __init__(self, in_features: Union[int, Tuple[int, int]],
#                  out_features: int, heads: int = 1, concat: bool = True,
#                  negative_slope: float = 0.2, dropout: float = 0.,
#                  add_self_loops: bool = True, bias: bool = True, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super(GATConv, self).__init__(node_dim=0, **kwargs)
#
#         self.in_features = in_features
#         self.out_features = out_features
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#
#         if isinstance(in_features, int):
#             self.lin_l = Linear(in_features, heads * out_features, bias=False)
#             self.lin_r = self.lin_l
#         else:
#             self.lin_l = Linear(in_features[0], heads * out_features, False)
#             self.lin_r = Linear(in_features[1], heads * out_features, False)
#
#         self.att_l = Parameter(torch.Tensor(1, heads, out_features))
#         self.att_r = Parameter(torch.Tensor(1, heads, out_features))
#
#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_features))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#         self._alpha = None
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         glorot(self.lin_l.weight)
#         glorot(self.lin_r.weight)
#         glorot(self.att_l)
#         glorot(self.att_r)
#         zeros(self.bias)
#
#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 size: Size = None, return_attention_weights=None):
#         # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
#         r"""
#
#         Args:
#             return_attention_weights (bool, optional): If set to :obj:`True`,
#                 will additionally return the tuple
#                 :obj:`(edge_index, attention_weights)`, holding the computed
#                 attention weights_results for each edge. (default: :obj:`None`)
#         """
#         H, C = self.heads, self.out_features
#
#         x_l: OptTensor = None
#         x_r: OptTensor = None
#         alpha_l: OptTensor = None
#         alpha_r: OptTensor = None
#         if isinstance(x, Tensor):
#             assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
#             x_l = x_r = self.lin_l(x).view(-1, H, C)
#             alpha_l = (x_l * self.att_l).sum(dim=-1)
#             alpha_r = (x_r * self.att_r).sum(dim=-1)
#         else:
#             x_l, x_r = x[0], x[1]
#             assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
#             x_l = self.lin_l(x_l).view(-1, H, C)
#             alpha_l = (x_l * self.att_l).sum(dim=-1)
#             if x_r is not None:
#                 x_r = self.lin_r(x_r).view(-1, H, C)
#                 alpha_r = (x_r * self.att_r).sum(dim=-1)
#
#         assert x_l is not None
#         assert alpha_l is not None
#
#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 num_nodes = x_l.size(0)
#                 if x_r is not None:
#                     num_nodes = min(num_nodes, x_r.size(0))
#                 if size is not None:
#                     num_nodes = min(size[0], size[1])
#                 edge_index, _ = remove_self_loops(edge_index)
#                 edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 edge_index = set_diag(edge_index)
#
#         # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
#         out = self.propagate(edge_index, x=(x_l, x_r),
#                              alpha=(alpha_l, alpha_r), size=size)
#
#         alpha = self._alpha
#         self._alpha = None
#
#         if self.concat:
#             out = out.view(-1, self.heads * self.out_features)
#         else:
#             out = out.mean(dim=1)
#
#         if self.bias is not None:
#             out += self.bias
#
#         if isinstance(return_attention_weights, bool):
#             assert alpha is not None
#             if isinstance(edge_index, Tensor):
#                 return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out
#
#     def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
#                 index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:
#         alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, index, ptr, size_i)
#         self._alpha = alpha
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return x_j * alpha.unsqueeze(-1)
#
#     def __repr__(self):
#         return '{}({}, {}, heads={})'.format(self.__class__.__name__,
#                                              self.in_features,
#                                              self.out_features, self.heads)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ adaptive_graph in integrator @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# elif args.adaptive_graph and not model.training and args.data_type == 'pde_1d':
#
# integrated_list = data.x.to(device).unsqueeze(0)
# # il = data.x.to(device).unsqueeze(0)  # todo
# integrated_list_n = data.x.to(device).unsqueeze(0)
# p = model.pos_list[0]
#
# for jk in range(batch_time - 1):
#     # hhh = il[jk]
#     # p = model.pos_list[0]
#     # model.pos_list.append(p.detach().clone())
#     # batch_1 = batch_t[jk: jk + 2]
#     # hhh_t1 = odeint(model, hhh, batch_1, method=args.method)[-1].unsqueeze(0)
#     # model.pos_list.pop(-1)
#     # il = T.cat((il, hhh_t1), dim=0)
#
#     h_t = integrated_list_n[jk]
#     pos = model.pos_list[jk]
#     batch_1 = batch_t[jk: jk + 2]
#     model.pos_list.append(pos.detach().clone())
#     h_t1 = odeint(model, h_t, batch_1, method=args.method)[-1]
#
#     aa = T.linspace(0, args.n_Nodes - 1, args.n_Nodes).to(args.device)
#     aa_new = T.linspace(0, args.n_Nodes - 1, args.n_dense_Nodes).to(args.device)
#     f1 = Interp1d()
#     diff = T.abs((f1(aa, T.transpose(h_t1, 0, 1), aa_new) - f1(aa, T.transpose(h_t, 0, 1),
#                                                                aa_new) * 1.3) * 20).long() + 20
#     chng = change(args)
#     pos_n0 = chng.new_pos(diff.reshape((1000))) * (args.n_Nodes - 1) / (args.n_dense_Nodes - 1)
#     sorted_pos_n, indices = T.sort(T.transpose(pos_n0, 0, 1))
#     pos_n = T.transpose(sorted_pos_n, 0, 1)
#     # edge_index_n = knn_graph(pos_n, k=model.k, batch=model.batch, loop=False)
#
#     model.pos_list.pop(-1)
#     model.pos_list.append(pos_n.detach().clone())
#     h_new = T.transpose(f1(pos[:, 0], T.transpose(h_t, 0, 1), pos_n[:, 0]), 0, 1)
#     # h_neww = interp(pos[:, 0].detach().cpu(), h_t[:,0].detach().cpu(), pos_n[:, 0].detach().cpu()).unsqueeze(1).to(args.device)
#     # - f1(aa, T.transpose(h_t, 0, 1), aa_new)
#     h_t1_new = odeint(model, h_new, batch_1, method=args.method)[-1]
#     # h_newww = interpolate(pos[:, 0], h_t, pos_n[:, 0])
#     h_t1_new_old_pos = interp(pos_n[:, 0].detach().cpu(), h_t1_new[:, 0].detach().cpu(),
#                               p[:, 0].detach().cpu()).unsqueeze(1).to(args.device)
#
#     # gg = interp1d(pos[:, 0].detach().cpu().numpy(), T.transpose(h_t, 0, 1).detach().cpu().numpy(), kind='cubic')
#     # gk = T.transpose(T.tensor(gg(pos_n[:, 0].detach().cpu().numpy())), 0, 1)
#     # pp(gk.unsqueeze(0), [pos_n])
#     # pp(h_t.unsqueeze(0), [pos])
#     # pp(h_t1.unsqueeze(0), [pos])
#     # pp(h_new.unsqueeze(0), [pos_n])
#
#     ###############
#     # tt = interp1d(pos[:, 0].detach().clone().cpu().numpy(), T.transpose(h_t, 0, 1).detach().clone().cpu().numpy(), kind='cubic')
#     # kk = T.tensor(tt(pos_n[:, 0].detach().clone().cpu().numpy())).to(args.device)
#     # if epoch==1:
#     #     fig = plt.figure(figsize=(4, 4))
#     #     def pp(logits, pos_list, color='b'):
#     #         if len(pos_list[0][0, :]) == 1:
#     #             pos_list = [
#     #                 T.cat((pos_list[i], T.zeros(pos_list[i].shape).to(pos_list[i].device)), dim=1) for i
#     #                 in
#     #                 range(len(pos_list))]
#     #
#     #         def visualize_points(pos, edge_index=None, index=None):
#     #             # fig = plt.figure(figsize=(4, 4))
#     #             plot3 = []
#     #             i = 0
#     #             if edge_index is not None:
#     #                 # for (src, dst) in T.tensor(edge_index).t().tolist():
#     #                 for (src, dst) in edge_index.t().tolist():
#     #                     src = pos[src].tolist()
#     #                     dst = pos[dst].tolist()
#     #                     plot3.append(bx.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black'))
#     #                     i = i + 1
#     #                     # print(i)
#     #             plot4 = bx.scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), color=color, s=50, zorder=1000)
#     #             plt.axis('off')
#     #             # plt.show()
#     #             return plot3, plot4
#     #
#     #         fps = 2  # frame per sec
#     #         frn = len(logits[:, 0])
#     #         aa = np.linspace(0, len(logits[0, :]) - 1, len(logits[0, :]))
#     #         # fig = plt.figure(figsize=(4, 4))
#     #         widths = [4]
#     #         heights = [4]  # , 4]
#     #         spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths,
#     #                                  height_ratios=heights)
#     #         # ax = fig.add_subplot(spec5[0, 0])
#     #         bx = fig.add_subplot(spec5[0])  # 1, 0])
#     #
#     #         plot2 = bx.plot(aa, logits[0].cpu(), color=color)
#     #         plot3, plot4 = visualize_points(pos_list[0])
#     #         # plt.show()
#     #     pp(h_t1_new.unsqueeze(0), [pos_n], color='b')
#     #     pp(hhh_t1, [p], color='r')
#     #     pp(h_t1.unsqueeze(0), [p], color='g')
#     #
#     #     pp(h_t.unsqueeze(0), [p], color='c')
#     #     pp(kk, [p], color='y')
#     #     pp(h_new.unsqueeze(0), [p], color='m')
#     #     pp(h_neww.unsqueeze(0), [p], color='k')
#     #     pp(h_t1_new_old_pos.unsqueeze(0), [p], color='tab:orange')
#     #     plt.show()
#     #     ll=0
#     ##################
#     # pp(T.transpose(diff, 0, 1).unsqueeze(0), [pos])
#
#     integrated_list = T.cat((integrated_list, h_t1_new_old_pos.unsqueeze(0)), dim=0)
#     integrated_list_n = T.cat((integrated_list_n, h_t1_new.unsqueeze(0)), dim=0)
#     model.edge_index_list.append(model.edge_index)
#     # integrated_list.append(h_t1_new)
#     # integrated_list = il # Todo

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ batch_t & idx_ls in integrator when args.rollout_t @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# if hasattr(args, 'rollout_t'):
#     if not args.rollout_t == 0:
#         assert (args.rollout_t + 1) % batch_time == 0, 'rollout_t is expected to of form of x*batch_time-1 ' \
#                                                        'otherwise interpolation between time steps will ' \
#                                                        'be required, which is not implemented yet'
#         batch_t = T.linspace(0, args.small_del_t * args.rollout_t, batch_time)
#         idx_ls = T.linspace(0, args.rollout_t, batch_time).long()
# else:
#     batch_t = ts[0:batch_time]
#     idx_ls = T.tensor(range(0, batch_time))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ pp plot 1d graph after interpolation @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def pp(logits, pos_list, color='b'):
#     if len(pos_list[0][0, :]) == 1:
#         pos_list = [torch.cat((pos_list[i], torch.zeros(pos_list[i].shape).to(pos_list[i].device)), dim=1) for i in
#                     range(len(pos_list))]
#
#     def visualize_points(pos, edge_index=None, index=None):
#         # fig = plt.figure(figsize=(4, 4))
#         plot3 = []
#         i = 0
#         if edge_index is not None:
#             # for (src, dst) in torch.tensor(edge_index).t().tolist():
#             for (src, dst) in edge_index.t().tolist():
#                 src = pos[src].tolist()
#                 dst = pos[dst].tolist()
#                 plot3.append(bx.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black'))
#                 i = i + 1
#                 # print(i)
#         plot4 = bx.scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), color=color, s=50, zorder=1000)
#         plt.axis('off')
#         # plt.show()
#         return plot3, plot4
#
#     fps = 2  # frame per sec
#     frn = len(logits[:, 0])
#     aa = np.linspace(0, len(logits[0, :])-1, len(logits[0, :]))
#     # fig = plt.figure(figsize=(4, 4))
#     widths = [4]
#     heights = [4]  # , 4]
#     spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths,
#                              height_ratios=heights)
#     # ax = fig.add_subplot(spec5[0, 0])
#     bx = fig.add_subplot(spec5[0])  # 1, 0])
#
#     plot2 = bx.plot(aa, logits[0].cpu(), color=color)
#     plot3, plot4 = visualize_points(pos_list[0])
#     # plt.show()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ interpolate function @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def interpolate(pos, h_tt, pos_n):
#     h_t = h_tt[:,0]
#     ohno = torch.cat([pos.unsqueeze(0)] * len(pos), dim=0)
#     ohyes = torch.cat([pos_n.unsqueeze(1)] * len(pos), dim=1)
#
#     sorted, indices = torch.sort(torch.abs(ohno - ohyes), dim=1)
#     positions = torch.argmin(torch.abs(ohno - ohyes), dim=1)
#     assert torch.equal(positions, indices[:, 0]), 'not ok'
#
#     v1 = h_t[positions]
#     v2 = h_t[indices[:, 1]]
#
#     v = v1+(v1-v2)/(pos[positions]-pos[indices[1]])*(pos_n-pos[positions])
#     return v.unsqueeze(1)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ SplineConv without basis @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class SplineConv(MessagePassing):
#     r"""The spline-based convolutional operator from the `"SplineCNN: Fast
#     Geometric Deep Learning with Continuous B-Spline Kernels"
#     <https://arxiv.org/abs/1711.08920>`_ paper
#
#     .. math::
#         \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in
#         \mathcal{N}(i)} \mathbf{x}_j \cdot
#         h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),
#
#     where :math:`h_{\mathbf{\Theta}}` denotes a kernel function defined
#     over the weighted B-Spline tensor product basis.
#
#     .. note::
#
#         Pseudo-coordinates must lay in the fixed interval :math:`[0, 1]` for
#         this method to work as intended.
#
#     Args:
#         in_features (int or tuple): Size of each input sample. A tuple
#             corresponds to the sizes of source and target dimensionalities.
#         out_features (int): Size of each output sample.
#         dim (int): Pseudo-coordinate dimensionality.
#         kernel_size (int or [int]): Size of the convolving kernel.
#         is_open_spline (bool or [bool], optional): If set to :obj:`False`, the
#             operator will use a closed B-spline basis in this dimension.
#             (default :obj:`True`)
#         degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
#         aggr (string, optional): The aggregation operator to use
#             (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
#             (default: :obj:`"mean"`)
#         root_weight (bool, optional): If set to :obj:`False`, the layer will
#             not add transformed root node features to the output.
#             (default: :obj:`True`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#
#     def __init__(self, args,
#                  agmnt_featur,
#                  in_features: Union[int, Tuple[int, int]],
#                  out_features: int,
#                  dim: int,
#                  kernel_size: Union[int, List[int]],
#                  basisfunc=Fourier(5),
#                  is_open_spline: bool = True,
#                  degree: int = 1,
#                  aggr: str = 'mean',
#                  root_weight: bool = True,
#                  bias: bool = False,
#                  **kwargs):  # yapf: disable
#         super(SplineConv, self).__init__(aggr=aggr, **kwargs)
#
#         if spline_basis is None:
#             raise ImportError('`SplineConv` requires `torch-spline-conv`.')
#
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dim = dim
#         self.degree = degree
#         self.agmnt_featur = agmnt_featur
#         self.args = args
#
#         kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
#         self.register_buffer('kernel_size', kernel_size)
#
#         is_open_spline = repeat(is_open_spline, dim)
#         is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
#         self.register_buffer('is_open_spline', is_open_spline)
#
#         if isinstance(in_features, int):
#             in_features = (in_features, in_features)
#
#         K = kernel_size.prod().item()
#         self.weight = Parameter(torch.Tensor(K, in_features[0], out_features))
#
#         if root_weight:
#             self.root = Parameter(torch.Tensor(in_features[1], out_features))
#         else:
#             self.register_parameter('root', None)
#
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # size = self.weight.size(0) * self.weight.size(1)
#         # uniform(size, self.weight)
#         # uniform(size, self.root)
#         zeros(self.weight)
#         zeros(self.root)
#         zeros(self.bias)
#
#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, pos, t,
#                 edge_attr: OptTensor = None, size: Size = None) -> Tensor:
#         """"""
#         if isinstance(x, Tensor):
#             x: OptPairTensor = (x, x)
#
#         if not x[0].is_cuda:
#             warnings.warn(
#                 'We do not recommend using the non-optimized CPU version of '
#                 '`SplineConv`. If possible, please move your data to GPU.')
#
#         # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, pos=pos, x=x, edge_attr=edge_attr, t=t, size=size)
#
#         x_r = x[1]
#         if x_r is not None and self.root is not None:
#             out += torch.matmul(x_r, self.root)
#
#         if self.bias is not None:
#             out += self.bias
#
#         return out
#
#     def message(self, x_j: Tensor, x_i, pos_j, pos_i, t, edge_attr: Tensor) -> Tensor:
#         s = 1 if self.args.Basis == 'None' else t
#         x, edge_attr = self.agmnt_featur(layer_num=1, pos_j=pos_j, pos_i=pos_j, h_j=x_j, h_i=x_i)
#         # print(edge_attr.size(1))
#         # print('kernel_size', self.kernel_size.numel())
#         # pseudo.size(1) == kernel_size.numel()
#         data = spline_basis(edge_attr, self.kernel_size, self.is_open_spline,
#                             self.degree)
#
#         rtrn = spline_weighting(x, self.weight, *data)
#         return rtrn
#
#     def __repr__(self):
#         return '{}({}, {}, dim={})'.format(self.__class__.__name__,
#                                            self.in_features, self.out_features,
#                                            self.dim)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ raw data plot (mountain) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# X, Y = np.meshgrid(p_x, p_y)
# x = p_x  # np.arange(-1, 1, 0.097)
# y = p_y  # np.arange(-1, 1, 0.097)
# Z = data[0]
# l_z = len(data[:, 0])
# dat = data
# # data[:, 10, :] = 0
#
# fps = 15  # frame per sec
# frn = l_z  # frame number of the animation
#
# def update_plot(frame_number, dat, data, plot1, plot2):
#     plot1[0].remove()
#     plot2.axes.lines.remove(plot2.axes.lines[0])
#     # plot1[0] = ax.plot_surface(X, Y, data[frame_number],
#     #                            cmap="magma", facecolors=cmap(Ys))
#     # plot2 = bx.plot(x, data[frame_number][10, :],
#     #                 color=cmap(ys[10]))
#     plot1[0] = ax.plot_surface(X, Y, data[frame_number],
#                                cmap="magma")
#     plot2 = bx.plot(x, dat[frame_number][10, :],
#                     color=cmap(ys[10]))
#
# fig = plt.figure(figsize=(5, 6))
# widths = [2]
# heights = [2, 1]
# spec5 = fig.add_gridspec(ncols=1, nrows=2, width_ratios=widths,
#                          height_ratios=heights)
# ax = fig.add_subplot(spec5[0, 0], projection='3d')
# bx = fig.add_subplot(spec5[1, 0])
#
# Ys = Y * 0
# Ys[:, :] = .3
# Ys[10] = .6
# # Ys[20] = .7
# ys = Ys[:, 0]
# cmap = cm.viridis
# # plot1 = [ax.plot_surface(X, Y, data[0], color='0.75',
# #                          rstride=1, cstride=1, facecolors=cmap(Ys))]
# # plot2, = bx.plot(x, data[0][10, :], color=cmap(ys[10]))
# plot1 = [ax.plot_surface(X, Y, data[0], color='0.75')]
# plot2, = bx.plot(x, dat[0][10, :], color=cmap(ys[10]))
#
# ani = animation.FuncAnimation(fig, update_plot, frn,
#                               fargs=(dat, data, plot1, plot2),
#                               interval=1000 / fps)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ edge_index with donuts @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # edge index for periodic bc
# nx = int(self.args.n_Nodes ** 0.5)
# cor1 = np.linspace(0, 2 * np.pi, nx+1)
# cor2 = np.linspace(0, 2 * np.pi, nx+1)
# phi, theta = np.meshgrid(cor1[0:-1], cor2[0:-1])
# phi, theta = phi.reshape(-1), theta.reshape(-1)
# c_pos = np.zeros((self.args.n_Nodes, 3))
# c_pos[:, 0] = np.cos(phi) * 10 + np.cos(phi) * np.cos(theta)* 10
# c_pos[:, 1] = np.sin(theta)
# c_pos[:, 2] = - np.sin(phi) * 10 - np.sin(phi) * np.cos(theta)*10
# c_pos_ = torch.tensor(c_pos).to(self.args.device)
# edge_index = knn_graph(c_pos_, k=self.k, batch=batch, loop=True)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(c_pos[:, 0], c_pos[:, 1], c_pos[:, 2])
# plt.show()

# c_pos = np.zeros(((nx + 1) * 4, 2))
# lin = np.arange(65)
# c_pos[(nx + 1) * 0:(nx + 1) * 1, 0] = lin
# c_pos[(nx + 1) * 0:(nx + 1) * 1, 1] = -1
# c_pos[(nx + 1) * 1:(nx + 1) * 2, 1] = lin
# c_pos[(nx + 1) * 1:(nx + 1) * 2, 0] = nx
# c_pos[(nx + 1) * 2:(nx + 1) * 3, 0] = np.flip(lin - 1, 0)
# c_pos[(nx + 1) * 2:(nx + 1) * 3, 1] = nx
# c_pos[(nx + 1) * 3:(nx + 1) * 4, 1] = np.flip(lin - 1, 0)
# c_pos[(nx + 1) * 3:(nx + 1) * 4, 0] = -1
# c_pos = torch.tensor(c_pos).to(self.args.device)
# pos = torch.tensor(pos).to(self.args.device)
# pos_n = torch.cat((pos, c_pos), dim=0)
# edge_index = knn_graph(pos_n, k=self.k, loop=True)  # k = 8
#
# for j in range(len(edge_index[0])):
#     for k in range(0, 260):
#         if edge_index[0, j].item() in range(4096 + 65 * 0, 4096 + 65 * 1):
#             edge_index[0, j] = k * 64 - 1
#
#         if edge_index[1, j].item() in range(4096 + 65 * 0, 4096 + 65 * 1):
#             edge_index[1, j] = k * 64 - 1
#
#         if edge_index[0, j].item() in range(4096 + 65 * 1, 4096 + 65 * 2):
#             edge_index[0, j] = k
#
#         if edge_index[1, j].item() in range(4096 + 65 * 1, 4096 + 65 * 2):
#             edge_index[1, j] = k
#
#         if edge_index[0, j].item() in range(4096 + 65 * 2, 4096 + 65 * 3):
#             edge_index[0, j] = k * 64 - 64
#
#         if edge_index[1, j].item() in range(4096 + 65 * 2, 4096 + 65 * 3):
#             edge_index[1, j] = k * 64 - 64
#
#         if edge_index[0, j].item() in range(4096 + 65 * 3, 4096 + 65 * 4):
#             edge_index[0, j] = 4032 + k
#
#         if edge_index[1, j].item() in range(4096 + 65 * 3, 4096 + 65 * 4):
#             edge_index[1, j] = 4032 + k


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ adaptive 1dwave @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# args.train_size = 15
# args.test_size = 5
# args.n_time_steps = 400
# args.n_Nodes = 101
# args.niters = 30
# args.max_batch_time = 20
# args.hidden = 16
# args.n_linear_layers = 2  # 1
# args.n_conv_layers = 2
# args.conv_hidden_dim = 4
# args.l_w = 2
# # model.Layer(args, 4, 1)
# model = Net(args, 1, 1).to(device)
#
# def train_adptive__none(**kwargs):  #
#     for B in ['Chebychev']:#, 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#         for n_B in [3]:  # [1, 2, 3, 5, 10]:  # range(10):
#             for train_size in [15]:  # [5, 15, 25, 35, 45, 50]:
#                 for batch_time in [5]:  # [2, 6, 10, 18, 26, 38]:
#                     parser = argparse.ArgumentParser('ODE demo')
#                     parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
#                                         default='rk4')  # dopri5') rk4
#                     # parser.add_argument('--n_time_steps', type=int, default=500)  # i.e. 0th 1th ..999th = 1000 time value
#                     # parser.add_argument('--train_size', type=int, default=0)
#                     # parser.add_argument('--test_size', type=int, default=0)
#                     # parser.add_argument('--batch_time', type=int, default=5)
#                     parser.add_argument('--batch_size', type=int, default=1)
#                     # parser.add_argument('--niters', type=int, default=65)
#                     parser.add_argument('--test_freq', type=int, default=20)
#                     # parser.add_argument('--viz', action='store_true')
#                     parser.add_argument('--viz', default=True)
#                     parser.add_argument('--gpu', type=int, default=0)
#                     parser.add_argument('--adjoint', action='store_true')
#                     parser.add_argument('--tr_style', type=str, choices=['MyclassV2', 'None'], default=None)
#                     parser.add_argument('--myepoch', type=list, default=[3, 4])
#                     parser.add_argument('--N_Basis', type=int, default=n_B)
#                     # parser.add_argument('--N_particles', type=int, default='101')
#                     parser.add_argument('--n_dense_Nodes', type=int, default='1000')
#                     parser.add_argument('--ConvLayer', type=str, choices=['PointNetLayer', 'GATConv'],
#                                         default='PointNetLayer')
#                     parser.add_argument('--data_type', type=str,
#                                         choices=['1Dwave', 'pde_2d', 'TrajectoryExtrapolation'],
#                                         default='1Dwave')
#                     parser.add_argument('--Basis', type=str,
#                                         choices=['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF',
#                                                  'MultiquadRBF'],
#                                         default='{}'.format(B))
#                     parser.add_argument('--op_type', type=str,
#                                         choices=[r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}', 'u*ux', 'ux*ux', 'u*uxx', 'uxx*uxx'],
#                                         default='u*uxx')
#                     parser.add_argument('--adaptive_graph', action='store_true', default=False)
#                     args = parser.parse_args()
#                     args.train_size_used = train_size
#                     args.bt = batch_time
#                     print('Number of Basis:', n_B)
#                     print('Time of integration:', batch_time)
#                     print('Basis:', '{}'.format(B))
#                     model, optimizer, criterion, train_loader, test_loader, get_path, integrator, PlotResults = init_model(
#                         args, **kwargs)
#                     for epoch in range(0, args.niters):
#                         # btt = [2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10,
#                         #        10]
#                         btt = [2,2,2,2,4,4,6,6,8,8,10,10,10, 10, 10, 10,11,11,11,11 ,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13]
#                         batch_time = btt[epoch]
#                         train_error = train(model, optimizer, criterion, train_loader, get_path, integrator,
#                                             PlotResults,
#                                             epoch, batch_time, args, **kwargs)
#                         args.adaptive_graph = True
#                         test_error = test(model, test_loader, get_path, integrator, PlotResults, epoch, 5,
#                                           args, **kwargs)
#                         args.adaptive_graph = False
#                         print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, Test Accuracy: {test_error:.4f}')
#                         # print(f'Epoch: {epoch:02d}, Loss: {test_error:.4f}')
#                         # tt = time.time() - start
#                         # print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}')
#
#
# def adptive__none(**kwargs):
#     # ========================================= adptive__none ==========================================
#     for ad in [False, True]:
#         for B in ['Chebychev']:#, 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#             for n_B in [3]:  # [1, 2, 3, 5, 10]:  # range(10):
#                 for train_size in [15]:  # [5, 15, 25, 35, 45, 50]:
#                     for batch_time in [13]:  # [2, 6, 10, 18, 26, 38]:
#                         for bt_test in [15]:#[5, 10, 15, 20]:
#                             parser = argparse.ArgumentParser('ODE demo')
#                             parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
#                                                 default='euler')  # dopri5') rk4
#                             # parser.add_argument('--n_time_steps', type=int, default=500)  # i.e. 0th 1th ..999th = 1000 time value
#                             # parser.add_argument('--train_size', type=int, default=0)
#                             # parser.add_argument('--test_size', type=int, default=0)
#                             # parser.add_argument('--batch_time', type=int, default=5)
#                             parser.add_argument('--batch_size', type=int, default=1)
#                             # parser.add_argument('--niters', type=int, default=65)
#                             parser.add_argument('--test_freq', type=int, default=20)
#                             # parser.add_argument('--viz', action='store_true')
#                             parser.add_argument('--viz', default=True)
#                             parser.add_argument('--gpu', type=int, default=0)
#                             parser.add_argument('--adjoint', action='store_true')
#                             parser.add_argument('--tr_style', type=str, choices=['MyclassV2', 'None'], default=None)
#                             parser.add_argument('--myepoch', type=list, default=[3, 4])
#                             parser.add_argument('--N_Basis', type=int, default=n_B)
#                             # parser.add_argument('--N_particles', type=int, default='101')
#                             parser.add_argument('--n_dense_Nodes', type=int, default='1000')
#                             parser.add_argument('--ConvLayer', type=str, choices=['PointNetLayer', 'GATConv'],
#                                                 default='PointNetLayer')
#                             parser.add_argument('--data_type', type=str,
#                                                 choices=['1Dwave', 'pde_2d', 'TrajectoryExtrapolation'],
#                                                 default='1Dwave')
#                             parser.add_argument('--Basis', type=str,
#                                                 choices=['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF',
#                                                          'MultiquadRBF'],
#                                                 default='{}'.format(B))
#                             parser.add_argument('--op_type', type=str,
#                                                 choices=[r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}'],
#                                                 default=r'\Delta{u}*\Delta{u}')
#                             parser.add_argument('--adaptive_graph', action='store_true', default=ad)
#                             args = parser.parse_args()
#                             args.train_size_used = train_size
#                             args.bt = batch_time
#                             print('Number of Basis:', n_B)
#                             print('Time of integration:', batch_time)
#                             print('Basis:', '{}'.format(B))
#                             model, optimizer, criterion, train_loader, test_loader, get_path, integrator, PlotResults = init_model(
#                                 args, **kwargs)
#                             path = get_path(args, W=1)
#                             pretrained_weights = osp.join(path, 'weights_bt{}'.format(batch_time))
#                             model.load_state_dict(torch.load(pretrained_weights))
#                             for epoch in [0]:
#                                 # batch_time = 10
#                                 test_error = test(model, test_loader, get_path, integrator, PlotResults, epoch, bt_test,
#                                                   args, **kwargs)
#                                 # print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, Test Accuracy: {test_error:.4f}')
#                                 print(f'Epoch: {epoch:02d}, Loss: {test_error:.4f}')
#
#                             # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#                             if not "er" in vars():
#                                 er = np.array([[test_error]])
#                             else:
#                                 er = np.concatenate((er, np.array([[test_error]])), axis=1)
#         if not "err" in vars():
#             err = er
#         else:
#             err = np.concatenate((err, er), axis=0)
#         del er
#
#     if kwargs.get('save', None):
#         path2 = get_path(args, C1=1)
#         np.savetxt(osp.join(path2, 'adptive__none.csv'), err, delimiter=",")
#     del err
#     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ change()- random @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class change():
#     def __init__(self, args):
#         super(change, self).__init__()
#         self.pos = torch.zeros((args.n_dense_Nodes, 1)).to(args.device)
#         for j in range(args.n_dense_Nodes):
#             self.pos[j, 0] = j
#
#     def findCeil(self, arr, r, l, h):
#         while (l < h):
#             mid = l + ((h - l) >> 1)  # Same as mid = (l+h)/2
#             if r > arr[mid]:
#                 l = mid + 1
#             else:
#                 h = mid
#
#         if arr[l] >= r:
#             return l
#         else:
#             return -1
#
#     def myRand(self, arr, freq, n):
#         # Create and fill prefix array
#         prefix = [0] * n
#         prefix[0] = freq[0]
#         for i in range(n):
#             prefix[i] = prefix[i - 1] + freq[i]
#
#         # prefix[n-1] is sum of all frequencies.
#         # Generate a random number with
#         # value from 1 to this sum
#         r = random.randint(0, prefix[n - 1]) + 1
#
#         # Find index of ceiling of r in prefix arrat
#         indexc = self.findCeil(prefix, r, 0, n - 1)
#         return arr[indexc]
#
#     def new_pos(self, u, n_Nodes, n_dense_Nodes):
#         # Driver code
#         arr = list(range(0, n_dense_Nodes))  # [1, 2, 3, 4]
#         # u = np.abs(u * 10)
#         freq = u.tolist()  # u.reshape((n_x)).astype(int).tolist()  # [10, 5, 20, 100]
#         n = len(arr)
#
#         # position = np.zeros((n_Nodes, 2))
#         position = torch.zeros((n_Nodes, 1)).to(u.device)
#         for i in range(n_Nodes):
#             aaa = self.myRand(arr, freq, n)
#             position[i] = self.pos[aaa]
#
#         return position

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Net with adaptive @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class Net(torch.nn.Module):
#     def __init__(self, args, in_features, out_features):
#         super(Net, self).__init__()
#
#         self.r = 3
#         self.args = args
#         self.del_t = 1
#         self.epoch = 0
#         self.batch = torch.tensor([0])
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         torch.manual_seed(12345)
#         self.Layer(args, in_features, out_features)
#         # self.pos = np.zeros((args.n_dense_Nodes, 2))
#         self.pos = torch.zeros((args.n_dense_Nodes, 1)).to(self.args.device)
#         for j in range(args.n_dense_Nodes):
#             self.pos[j, 0] = j
#
#         self.pos_list = []
#         self.edge_index_list = []
#
#         # self.conv1 = GCN
#         # #self.classifier = Linear(32, dataset.num_classes)
#         # Conv(1, 8)
#         # self.conv2 = GCNConv(8, 1)
#
#     def findCeil(self, arr, r, l, h):
#         while (l < h):
#             mid = l + ((h - l) >> 1);  # Same as mid = (l+h)/2
#             if r > arr[mid]:
#                 l = mid + 1
#             else:
#                 h = mid
#
#         if arr[l] >= r:
#             return l
#         else:
#             return -1
#
#     def myRand(self, arr, freq, n):
#         # Create and fill prefix array
#         prefix = [0] * n
#         prefix[0] = freq[0]
#         for i in range(n):
#             prefix[i] = prefix[i - 1] + freq[i]
#
#         # prefix[n-1] is sum of all frequencies.
#         # Generate a random number with
#         # value from 1 to this sum
#         r = random.randint(0, prefix[n - 1]) + 1
#
#         # Find index of ceiling of r in prefix arrat
#         indexc = self.findCeil(prefix, r, 0, n - 1)
#         return arr[indexc]
#
#     def change(self, u, n_Nodes, n_dense_Nodes):
#         # Driver code
#         arr = list(range(0, n_dense_Nodes))  # [1, 2, 3, 4]
#         # u = np.abs(u * 10)
#         freq = u.tolist()  # u.reshape((n_x)).astype(int).tolist()  # [10, 5, 20, 100]
#         n = len(arr)
#
#         # position = np.zeros((n_Nodes, 2))
#         position = torch.zeros((n_Nodes, 1)).to(self.args.device)
#         for i in range(n_Nodes):
#             aaa = self.myRand(arr, freq, n)
#             position[i] = self.pos[aaa]
#
#         return position
#
#     # def change(self, u, n_Nodes, n_dense_Nodes):
#     #     # Driver code
#     #     arr = list(range(0, n_dense_Nodes))  # [1, 2, 3, 4]
#     #     # u = np.abs(u * 10)
#     #     freq = u[:,0].tolist()  # u.reshape((n_x)).astype(int).tolist()  # [10, 5, 20, 100]
#     #     n = len(arr)
#     #
#     #     position = np.zeros((n_Nodes, 2))
#     #     for i in range(n_Nodes):
#     #         aaa = self.myRand(arr, freq, n)
#     #         position[i] = self.pos[aaa]
#     #
#     #     return position
#
#     def Layer(self, args, in_features, out_features):
#         if args.ConvLayer == 'PointNetLayer':
#             # self.conv1 = PointNetLayer(args, in_features, out_features, basisfunc=self.Basis(args))
#             # self.conv1 = PointNetLayer(args, in_features, 4, basisfunc=self.Basis(args))
#             # self.conv2 = PointNetLayer(args, 4*2+2, 1, basisfunc=self.Basis(args))
#             if hasattr(args, 'conv_hidden_dim'):
#                 self.conv_hidden_dim = args.conv_hidden_dim
#             else:
#                 if self.args.n_conv_layers > 1:
#                     assert 1 == 0, 'number of Neurones in hidden layer is required'
#                 else:
#                     self.conv_hidden_dim = out_features
#
#             self.dilation = True
#             self.shift = True
#             self.basisfunc = self.Basis(args)
#             self.conv = nn.ModuleDict({'1': PointNetLayer(args, in_features, self.conv_hidden_dim,
#                                                           basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)})
#             for c in range(self.args.n_conv_layers - 1):
#                 if c + 2 < self.args.n_conv_layers:
#                     self.conv['{:d}'.format(c + 2)] = PointNetLayer(args, self.conv_hidden_dim,
#                                                                 self.conv_hidden_dim, basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)
#                 else:
#                     self.conv['{:d}'.format(c + 2)] = PointNetLayer(args, self.conv_hidden_dim,
#                                                                 out_features, basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)
#
#         elif args.ConvLayer == 'GATConv':
#             if hasattr(args, 'conv_hidden_dim'):
#                 self.conv_hidden_dim = args.conv_hidden_dim
#             else:
#                 if self.args.n_conv_layers > 1:
#                     assert 1 == 0, 'number of Neurones in hidden layer is required'
#                 else:
#                     self.conv_hidden_dim = out_features
#
#             self.dilation = True
#             self.shift = True
#             self.basisfunc = self.Basis(args)
#             self.conv = nn.ModuleDict({'1': GATConv(args, in_features, self.conv_hidden_dim, heads=2,
#                                                           basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)})
#             for c in range(self.args.n_conv_layers - 1):
#                 if c + 2 < self.args.n_conv_layers:
#                     self.conv['{:d}'.format(c + 2)] = GATConv(args, self.conv_hidden_dim,
#                                                                     self.conv_hidden_dim, heads=1,
#                                                               basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)
#                 else:
#                     self.conv['{:d}'.format(c + 2)] = GATConv(args, self.conv_hidden_dim,
#                                                                     out_features, heads=1,
#                                                               basisfunc=self.basisfunc, dilation=self.dilation, shift=self.shift)
#             # self.conv1 = GATConv(args, in_features, out_features, heads=1, basisfunc=self.Basis(args))
#             # self.conv2 = GATConv(args, 32, 1, basisfunc=self.Basis(args))
#
#     def Basis(self, args):
#         if args.Basis == 'Chebychev':
#             basis = Chebychev(args.N_Basis)
#         elif args.Basis == 'Polynomial':
#             basis = Polynomial(args.N_Basis)
#             self.dilation = False
#             self.shift = False
#         elif args.Basis == 'Fourier':
#             basis = Fourier(args.N_Basis)
#         elif args.Basis == 'VanillaRBF':
#             basis = VanillaRBF(args.N_Basis)
#         elif args.Basis == 'GaussianRBF':
#             basis = GaussianRBF(args.N_Basis)
#         elif args.Basis == 'MultiquadRBF':
#             basis = MultiquadRBF(args.N_Basis)
#         elif args.Basis == 'PiecewiseConstant':
#             basis = PiecewiseConstant(args.N_Basis)
#         return basis
#
#     def get_edge_index(self, hh, pos, batch):
#         if self.args.data_type == '1Dwave':
#             edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)  # k = 5
#
#         elif self.args.data_type == 'pde_2d':
#             edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)  # k = 8
#
#         elif self.args.data_type == 'TrajectoryExtrapolation':
#             # n_Nodes = kwargs.get('n_Nodes', None)
#             for i in range(n_Nodes):
#                 for j in range(n_Nodes):
#                     if 2 * LA.norm(hh[i, :2] - hh[j, :2]) <= self.r:  # and i != j:
#                         # data.edge_index[] =
#                         edge_ = torch.empty(2, 1, dtype=int).to(self.device)
#                         edge_[0] = j
#                         edge_[1] = i
#                         try:
#                             torch.is_tensor(edge_index)
#                             edge_index = torch.cat((edge_index, edge_), dim=1)
#                         except:
#                             edge_index = edge_
#
#         return edge_index
#
#     def retrn(self, h, hh):
#
#         if self.args.data_type == 'TrajectoryExtrapolation':
#             retrn = torch.empty(self.args.n_Nodes, 4).to(self.args.device)
#             retrn[:, :2] = hh[:, 2:4]
#             retrn[:, 2:] = -hh[:, :2] * 2 - h  # -h  #
#             return retrn
#         else:
#             return h
#
#     def forward(self, t, data):  # hh, batch, edge_index, pos, edge_attr):
#
#         self.t = t / self.del_t
#         self.idx = int(t / self.del_t)
#         self.basisfunc.idx = self.idx
#         hh = data
#         n_Nodes = len(hh[:, 0])
#         batch = self.batch.long()
#         # pos = torch.tensor(self.pos_list[self.idx]).to(self.device)
#         # edge_attr = torch.tensor(self.edge_index_list[self.idx]).to(self.device)
#         pos = self.pos_list[self.idx]
#         edge_attr = self.edge_attr  # edge_index_list[self.idx]
#
#         edge_index = self.get_edge_index(hh, pos, batch)
#
#         # h = self.conv1(h=hh, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
#         # h = h.relu()
#         # h = self.conv2(h=h, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
#
#         h = self.conv['{:d}'.format(1)](hh, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
#         for d in range(self.args.n_conv_layers - 1):
#             h = h.relu()
#             h = self.conv['{:d}'.format(d+2)](h, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
#
#         if self.args.adaptive_graph and self.idx != self.previou_idx and not self.training:
#             self.previou_idx = self.idx
#
#             pos_t = hh
#             # aa = np.linspace(0, 100, self.args.n_Nodes)
#             # aa_new = np.linspace(0, 100, self.args.n_dense_Nodes)  # , endpoint=True)
#             aa = torch.linspace(0, 100, self.args.n_Nodes).to(self.args.device)
#             aa_new = torch.linspace(0, 100, self.args.n_dense_Nodes).to(self.args.device)
#             # torchdiffeq.odeint(model, data.x.to(device), batch_t.to(device), method=args.method)
#             pos_t_1 = hh + self.del_t * h
#             # torch.nn.functional.interpolate(pos_t_1, )
#             f1 = Interp1d()
#             # f1 = interp1d(aa, pos_t_1.cpu().detach().numpy().transpose(), kind='cubic')
#             # f2 = interp1d(aa, pos_t.cpu().detach().numpy().transpose(), kind='cubic')
#             # diff = np.abs((f1(aa_new) - f2(aa_new)) * 500 * 7).astype(int) + 1
#             diff = torch.abs((f1(aa, torch.transpose(pos_t_1, 0, 1), aa_new) - f1(aa, torch.transpose(pos_t, 0, 1),
#                                                                                   aa_new)*1.3) * 50 ).long() + 20
#             pos_n = self.change(diff.reshape((1000)), n_Nodes, self.args.n_dense_Nodes) * 100 / 999
#             nmo = 0
#             # self.pos_list.append(pos_n)
#             # self.edge_index_list.append(edge_index.detach().cpu().numpy())
#             self.pos_list.append(pos_n.detach().clone())
#             self.edge_index_list.append(edge_index.detach().clone())
#         else:
#             # self.pos_list.append(pos.detach().cpu().numpy())
#             # self.edge_index_list.append(edge_index.detach().cpu().numpy())
#             self.pos_list.append(pos.detach().clone())
#             self.edge_index_list.append(edge_index.detach().clone())
#
#         return self.retrn(h, hh)
#
#     # def __repr__(self):
#     #     return

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ tt() @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def tt():
#     for B in ['Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#         for n_B in [1, 2, 3, 5, 10]:  # range(10):
#             for train_size in [15]:  # [5, 15, 25, 35, 45, 50]:
#                 for batch_time in [10]:  # [2, 6, 10, 18, 26, 38]:
#                     parser = argparse.ArgumentParser('ODE demo')
#                     parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
#                                         default='rk4')  # dopri5') rk4
#                     # parser.add_argument('--n_time_steps', type=int, default=500)  # i.e. 0th 1th ..999th = 1000 time value
#                     # parser.add_argument('--train_size', type=int, default=0)
#                     # parser.add_argument('--test_size', type=int, default=0)
#                     # parser.add_argument('--batch_time', type=int, default=5)
#                     parser.add_argument('--batch_size', type=int, default=1)
#                     # parser.add_argument('--niters', type=int, default=65)
#                     parser.add_argument('--test_freq', type=int, default=20)
#                     # parser.add_argument('--viz', action='store_true')
#                     parser.add_argument('--viz', default=True)
#                     parser.add_argument('--gpu', type=int, default=0)
#                     parser.add_argument('--adjoint', action='store_true')
#                     parser.add_argument('--tr_style', type=str, choices=['MyclassV2', 'None'], default=None)
#                     parser.add_argument('--myepoch', type=list, default=[3, 4])
#                     parser.add_argument('--N_Basis', type=int, default=n_B)
#                     # parser.add_argument('--N_particles', type=int, default='101')
#                     parser.add_argument('--n_dense_Nodes', type=int, default='1000')
#                     parser.add_argument('--ConvLayer', type=str, choices=['PointNetLayer', 'GATConv'],
#                                         default='PointNetLayer')
#                     parser.add_argument('--data_type', type=str, choices=['1Dwave', 'pde_2d', 'TrajectoryExtrapolation'],
#                                         default='pde_2d')
#                     parser.add_argument('--Basis', type=str,
#                                         choices=['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF',
#                                                  'MultiquadRBF'],
#                                         default='{}'.format(B))
#                     parser.add_argument('--op_type', type=str,
#                                         choices=[r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}'],
#                                         default=r'\Delta{u}*\Delta{u}')
#                     parser.add_argument('--adaptive_graph', action='store_true', default=False)
#                     args = parser.parse_args()
#                     args.train_size_used = train_size
#                     args.bt = batch_time
#                     print('Number of Basis:', n_B)
#                     print('Time of integration:', batch_time)
#                     print('Basis:', '{}'.format(B))
#                     model, optimizer, criterion, train_loader, test_loader, get_path, integrator, PlotResults = init_model(
#                         args)
#                     # path = get_path(args, W=1)
#                     # pretrained_weights = osp.join(path, 'weights_bt{}'.format(batch_time))
#                     # model.load_state_dict(torch.load(pretrained_weights))
#                     for epoch in range(0, args.niters):
#                         # if epoch % 2 == 0:
#                         #     plot = True
#                         batch_time = args.bt
#                         # batch_time = 2 + 1 * int(epoch / 3)
#                         # start = time.time()
#                         train_error = train(model, optimizer, criterion, train_loader, get_path, integrator,
#                                             PlotResults,
#                                             epoch, batch_time, args, plot=0)
#                         # tt = time.time() - start
#                         # batch_time = 10
#                         # test_error = test(model, test_loader, get_path, integrator, PlotResults, epoch, batch_time, args)
#                         # print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, Test Accuracy: {test_acc:.4f}')
#                         print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}')

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PointNetLayer without mutilple layers @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class PointNetLayer(MessagePassing):
#     # def __init__(self, bias=True, basisfunc=Fourier(5), dilation=True, shift=True):
#     #     super().__init__()
#     #     self.dilation = torch.ones(1) if not dilation else nn.Parameter(data=torch.ones(1), requires_grad=True)
#     #     self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
#     #     self.basisfunc = basisfunc
#     #     self.n_eig = n_eig = self.basisfunc.n_eig
#     #     self.deg = deg = self.basisfunc.deg
#
#     def __init__(self, args, in_features, out_features, bias=True, basisfunc=Fourier(5), dilation=True,
#                  shift=True):
#         # Message passing with "max" aggregation.
#         super(PointNetLayer, self).__init__('mean')
#
#         # if
#         self.args = args
#         self.device = args.device
#         self.dilation = torch.ones(1) if not dilation else nn.Parameter(data=torch.ones(1), requires_grad=True)
#         self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
#         self.basisfunc = basisfunc
#         self.n_eig = n_eig = self.basisfunc.n_eig
#         self.deg = deg = self.basisfunc.deg
#
#         self.in_features, self.out_features = in_features, out_features
#
#         try:self.hidden = args.hidden
#         except:self.hidden = out_features
#         # if hidden is None:
#         #     self.hidden = self.out_features  # out_features
#         # else:
#         #     assert 1 == 0, 'number of neuron in hidden layers is required'
#         self.weight = torch.Tensor(out_features, in_features)
#         if bias:
#             self.bias = torch.Tensor(out_features)
#         else:
#             self.register_parameter('bias', None)
#
#         self.coeffs1 = torch.nn.Parameter(torch.Tensor((in_features + 1) * self.hidden, self.deg, self.n_eig))
#         self.coeffs2 = torch.nn.Parameter(torch.Tensor((self.hidden + 1) * out_features, self.deg, self.n_eig))
#         self.reset_parameters()
#
#     # def __init__(self, time_d, in_channels, out_channels, N_Basis):
#     #     # Message passing with "max" aggregation.
#     #     super(PointNetLayer, self).__init__('add')
#     #     self.time_d = time_d
#     #     self.in_channels = in_channels
#     #     self.out_channels = out_channels
#     #
#     #     # Initialization of the MLP:
#     #     # Here, the number of input features correspond to the hidden node
#     #     # dimensionality plus point dimensionality (=3).
#     #     # self.mlp = Sequential(Linear(in_channels * 2, out_channels),
#     #     #                       ReLU(),
#     #     #                       Linear(out_channels, out_channels))
#     #
#     #     # self.weight = torch.nn.Parameter(torch.randn(3, out_channels, in_channels, width, width))
#     #     # self.bias = torch.nn.Parameter(torch.zeros(3, out_channels))
#     #     self.N_Basis = N_Basis
#     #     self.weight = torch.nn.Parameter(torch.randn(N_Basis, out_channels, in_channels * 2))
#     #     self.bias = torch.nn.Parameter(torch.zeros(N_Basis, out_channels))
#     #     kl = 0
#
#     def reset_parameters(self):
#         torch.nn.init.normal_(self.coeffs1)
#         torch.nn.init.normal_(self.coeffs2)
#
#     def calculate_weights(self, s, coeffs):
#         "Expands `s` following the chosen eigenbasis"
#         n_range = torch.linspace(0, self.deg, self.deg).to(coeffs.device)
#         basis = self.basisfunc(n_range, s * self.dilation.to(coeffs.device) + self.shift.to(coeffs.device))
#         B = []
#         for i in range(self.n_eig):
#             Bin = torch.eye(self.deg).to(coeffs.device)
#             Bin[range(self.deg), range(self.deg)] = basis[i]
#             B.append(Bin)
#         B = torch.cat(B, 1).to(coeffs.device)
#         coeffss = torch.cat([coeffs[:, :, i] for i in range(self.n_eig)], 1).transpose(0, 1).to(coeffs.device)
#         X = torch.matmul(B, coeffss)
#         return X.sum(0)
#
#     def forward(self, h, edge_index, pos, edge_attr, t):
#         # Start propagating messages.
#         # print('edge_index ==',edge_index, 'h==',h)
#         return self.propagate(edge_index, pos=pos, edge_attr=edge_attr, h=h,
#                               t=t)  # pos=pos, edge_attr=edge_attr, h=h, t=t)
#
#     # def forward(self, input):
#     #     # For the moment, GalLayers rely on DepthCat to access the `s` variable. A better design would free the user
#     #     # of having to introduce DepthCat(1) every time a GalLayer is used
#     #     s = input[-1, -1]
#     #     input = input[:, :-1]
#     #     w = self.calculate_weights(s)
#     #     self.weight = w[0:self.in_features * self.out_features].reshape(self.out_features, self.in_features)
#     #     self.bias = w[self.in_features * self.out_features:(self.in_features + 1) * self.out_features].reshape(
#     #         self.out_features)
#     #     return torch.nn.functional.linear(input, self.weight, self.bias)
#
#     def message(self, h_j, h_i, pos_j, pos_i, t):
#         s = t
#         diff = h_j - h_i  # Compute spatial relation.
#         Rpos = pos_j - pos_i
#
#         if self.args.data_type == '1Dwave':
#             if h_j is not None:
#                 input = torch.cat([h_j, diff, Rpos], dim=-1)
#                 input = input.float()
#
#         elif self.args.data_type == 'pde_2d':
#             if h_j is not None:
#                 input = torch.cat([h_j, diff, Rpos], dim=-1)
#                 input = input.float()
#
#         elif self.args.data_type == 'TrajectoryExtrapolation':
#             if h_j is not None:
#                 input = torch.cat([h_j, diff], dim=-1)
#                 input = input.float()
#
#         w1 = self.calculate_weights(s, self.coeffs1)
#         self.weight1 = w1[0:self.in_features * self.hidden].reshape(self.hidden, self.in_features)
#         self.bias1 = w1[self.in_features * self.hidden:(self.in_features + 1) * self.hidden].reshape(
#             self.hidden)
#         valu1 = torch.nn.functional.linear(input, self.weight1, self.bias1)
#
#         val1 = valu1.relu()
#         w2 = self.calculate_weights(s, self.coeffs2)
#         self.weight2 = w2[0:self.hidden * self.out_features].reshape(self.out_features, self.hidden)
#         self.bias2 = w2[self.hidden * self.out_features:(self.hidden + 1) * self.out_features].reshape(
#             self.out_features)
#         valu2 = torch.nn.functional.linear(valu1, self.weight2, self.bias2)
#
#         return valu2

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ MyclassV2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# if args.tr_style == 'MyclassV2':
#     myclass = MyclassV2(10, data, 0.0001)
#     batch_t = torch.linspace(0, 0.0009, 10)
#     logits0 = odeint(model, data.x.to(device), data.batch.to(device), data.edge_index.to(device),
#                      data.pos.to(device), data.edge_attr.to(device), batch_t.to(device), method='euler').to(
#         device)  # args.method)
#     myclass.correct(logits0, 10)
#     logits = myclass.solutionList
#     # trgtt_batch = TargetList(data.x, i, True, args)
#     # trgt_batch = np.float32(trgtt_batch.numpy())
#     # trgt_batch = torch.tensor(trgt_batch)
# else:

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Test loops @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# for B in ['Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial']:
#     for n_B in [3]:#[1, 2, 3, 5, 10]:  # range(10):
#         for train_size in [15]:  # [5, 15, 25, 35, 45, 50]:
#             for batch_time in [10]:  # [2, 6, 10, 18, 26, 38]:
#                 for bt_test in [5, 10, 15, 20, 25, 30]:
#                     parser = argparse.ArgumentParser('ODE demo')
#                     parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'euler'],
#                                         default='rk4')  # dopri5') rk4
#                     # parser.add_argument('--n_time_steps', type=int, default=500)  # i.e. 0th 1th ..999th = 1000 time value
#                     # parser.add_argument('--train_size', type=int, default=0)
#                     # parser.add_argument('--test_size', type=int, default=0)
#                     # parser.add_argument('--batch_time', type=int, default=5)
#                     parser.add_argument('--batch_size', type=int, default=1)
#                     # parser.add_argument('--niters', type=int, default=65)
#                     parser.add_argument('--test_freq', type=int, default=20)
#                     # parser.add_argument('--viz', action='store_true')
#                     parser.add_argument('--viz', default=True)
#                     parser.add_argument('--gpu', type=int, default=0)
#                     parser.add_argument('--adjoint', action='store_true')
#                     parser.add_argument('--tr_style', type=str, choices=['MyclassV2', 'None'], default=None)
#                     parser.add_argument('--myepoch', type=list, default=[3, 4])
#                     parser.add_argument('--N_Basis', type=int, default=n_B)
#                     # parser.add_argument('--N_particles', type=int, default='101')
#                     parser.add_argument('--n_dense_Nodes', type=int, default='1000')
#                     parser.add_argument('--ConvLayer', type=str, choices=['PointNetLayer', 'GATConv'],
#                                         default='PointNetLayer')
#                     parser.add_argument('--data_type', type=str, choices=['1Dwave', 'pde_2d', 'TrajectoryExtrapolation'],
#                                         default='pde_2d')
#                     parser.add_argument('--Basis', type=str,
#                                         choices=['Polynomial', 'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF',
#                                                  'MultiquadRBF'],
#                                         default='{}'.format(B))
#                     parser.add_argument('--op_type', type=str,
#                                         choices=[r'\Delta{u}', r'\Delta{u}*\Delta{u}', r'u\Delta{u}'],
#                                         default=r'\Delta{u}*\Delta{u}')
#                     parser.add_argument('--adaptive_graph', action='store_true', default=False)
#                     args = parser.parse_args()
#                     args.train_size_used = train_size
#                     args.bt = batch_time
#                     print('Number of Basis:', n_B)
#                     print('Time of integration:', batch_time)
#                     print('Basis:', '{}'.format(B))
#                     model, optimizer, criterion, train_loader, test_loader, get_path, integrator, PlotResults = init_model(
#                         args)
#                     path = get_path(args, W=1)
#                     pretrained_weights = osp.join(path, 'weights_bt{}'.format(batch_time))
#                     model.load_state_dict(torch.load(pretrained_weights))
#                     for epoch in [0]:  # range(0, args.niters):
#                         # if epoch % 2 == 0:
#                         #     plot = True
#                         # batch_time = args.bt
#                         # batch_time = 2 + 1 * int(epoch / 3)
#                         # train_error = train(model, optimizer, criterion, train_loader, get_path, integrator, PlotResults,
#                         #                     epoch, batch_time, args, plot=0)
#                         # batch_time = 10
#                         test_error = test(model, test_loader, get_path, integrator, PlotResults, epoch, bt_test,
#                                           args)
#                         # print(f'Epoch: {epoch:02d}, Loss: {train_error:.4f}, Test Accuracy: {test_acc:.4f}')
#                         print(f'Epoch: {epoch:02d}, Loss: {test_error:.4f}')
#
#                     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#             #         if not "cr" in vars():
#             #             cr = np.array([[test_error]])
#             #         else:
#             #             cr = np.concatenate((cr, np.array([[test_error]])), axis=1)
#             #     if not "crr" in vars():
#             #         crr = cr
#             #     else:
#             #         crr = np.concatenate((crr, cr), axis=0)
#             #     del cr
#             # path2 = get_path(args, C2=1)
#             # np.savetxt(osp.join(path, 'bt_train__bt_test.csv'), crr, delimiter=",")
#             # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#                     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#                     if not "br" in vars():
#                         br = np.array([[test_error]])
#                     else:
#                         br = np.concatenate((br, np.array([[test_error]])), axis=1)
#     if not "brr" in vars():
#         brr = br
#     else:
#         brr = np.concatenate((brr, br), axis=0)
#     del br
#
# path2 = get_path(args, C2=1)
# np.savetxt(osp.join(path2, 'B__bt_test.csv'), brr, delimiter=",")
# # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#                 # **********************************************************************
#         #         if not "ar" in vars():
#         #             ar = np.array([[test_error]])
#         #         else:
#         #             ar = np.concatenate((ar, np.array([[test_error]])), axis=1)
#         #     if not "arr" in vars():
#         #         arr = ar
#         #     else:
#         #         arr = np.concatenate((arr, ar), axis=0)
#         #     del ar
#         # # dictionary = {'n_B{}'.format(n_B): arr}
#         # path = get_path(args, C1=1)
#         # # scipy.io.savemat('test.mat', dict(x=x, y=y))
#         # # np.save(osp.join(path, 'n_B{}.npy'.format(n_B)), arr)
#         # np.savetxt(osp.join(path, 'n_B{}.csv'.format(n_B)), arr, delimiter=",")
#         # del arr
#         # **********************************************************************
#
#         # =============================================================================
# #         if not "arrr" in vars():
# #             arrr = np.array([[test_error]])
# #         else:
# #             arrr = np.concatenate((arrr, np.array([[test_error]])), axis=1)  # arrr = [1,5]
# #
# #     if not "arrrr" in vars():
# #         arrrr = arrr
# #     else:
# #         arrrr = np.concatenate((arrrr, arrr), axis=0)
# #     del arrr
# #
# # path2 = get_path(args, C2=1)
# # np.savetxt(osp.join(path, 'B__n_B.csv'), arrrr, delimiter=",")
# # =============================================================================


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Net with multiple arguments @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class Net(torch.nn.Module):
#     def __init__(self, args):
#         super(Net, self).__init__()
#
#         self.r = 3
#         # self.N_particles = args.N_particles
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         torch.manual_seed(12345)
#         self.Layer(args)
#         # self.conv1 = GCN
#         # #self.classifier = Linear(32, dataset.num_classes)
#         # Conv(1, 8)
#         # self.conv2 = GCNConv(8, 1)
#
#     def Layer(self, args):
#         if args.ConvLayer == 'PointNetLayer':
#             self.conv1 = PointNetLayer(1, 32, basisfunc=self.Basis(args))
#             self.conv2 = PointNetLayer(32, 1, basisfunc=self.Basis(args))
#         elif args.ConvLayer == 'GATConv':
#             self.conv1 = GATConv(1, 32, basisfunc=self.Basis(args))
#             self.conv2 = GATConv(32, 1, basisfunc=self.Basis(args))
#
#     def Basis(self, args):
#         if args.Basis == 'Chebychev':
#             basis = Chebychev(args.N_Basis)
#         elif args.Basis == 'Polynomial':
#             basis = Polynomial(args.N_Basis)
#         elif args.Basis == 'Fourier':
#             basis = Fourier(args.N_Basis)
#         elif args.Basis == 'VanillaRBF':
#             basis = VanillaRBF(args.N_Basis)
#         return basis
#
#     def forward(self, t, hh, batch, edge_index, pos, edge_attr):
#         # Compute the kNN graph:
#         # Here, we need to pass the batch vector to the function call in order
#         # to prevent creating edges between points of different examples.
#         # We also add `loop=True` which will add self-loops to the graph in
#         # order to preserve central point information.
#
#         # hh = data[0]
#         n_Nodes = len(hh[:, 0])
#         # edge_index = torch.empty(2, 1, dtype=int)
#         # for i in range(n_Nodes):
#         #     for j in range(n_Nodes):
#         #         if 2 * LA.norm(hh[i, :2] - hh[j, :2]) <= self.r:  # and i != j:
#         #             # data.edge_index[] =
#         #             edge_ = torch.empty(2, 1, dtype=int).to(self.device)
#         #             edge_[0] = j
#         #             edge_[1] = i
#         #             try:
#         #                 torch.is_tensor(edge_index)
#         #                 edge_index = torch.cat((edge_index, edge_), dim=1)
#         #             except:
#         #                 edge_index = edge_
#         # try:
#         #     torch.is_tensor(edge_index)
#         # except:
#         #     retrn = torch.empty(self.N_particles, 4).to(self.device)
#         #     retrn[:, :2] = hh[:, 2:4]
#         #     retrn[:, 2:] = -hh[:, :2] * 2
#         #     return retrn
#
#         # edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
#         # edge_index = data[2].long()#.int()
#         # pos = data[3]
#         # edge_attr = data[4]
#
#         # 3. Start bipartite message passing.
#         h = self.conv1(h=hh, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
#         h = h.relu()
#         h = self.conv2(h=h, edge_index=edge_index, pos=pos, edge_attr=edge_attr, t=t)
#
#         # 4. Global Pooling.
#         # h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
#
#         # 5. Classifier.
#         # retrn = torch.empty(self.N_particles, 4).to(self.device)
#         # retrn[:, :2] = hh[:, 2:4]
#         # retrn[:, 2:] = -h  # -hh[:, :2] * 2 - h
#         return h# , -2*h[1] - 2*h[0] + np.cos(2*x)]  # self.classifier(h)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Net with multiple arguments @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# class Net(torch.nn.Module):
#     def __init__(self, args):
#         super(Net, self).__init__()
#
#         self.r = 3
#         self.args = args
#         self.del_t = 1
#         self.epoch = 0
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         torch.manual_seed(12345)
#         self.Layer(args)
#         
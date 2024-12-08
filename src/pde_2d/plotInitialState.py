import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp
import sys
# sys.path.insert(0,'/src/pde2d')
sys.path.insert(0, './')
from Dataset import raw_sequential_data, get_path


def getInitialState(i: int, args):
    """
    Args:
        i (int): initial state index
    Vars:
        x_array (torch.tensor): [n_initial conditions, n_sequence generated with an initial
                                    conditions, n_components(x, y), n_Nodes(64*64)]
    Returns:
        init_state (np.array): [2, 63, 63]
    """
    x_array = raw_sequential_data(args, plot_data=False)
    init_states = x_array[:, 0]
    return init_states[i+15].reshape((2, 63, 63)).numpy()  # starting ploting from 15


if __name__ == '__main__':

    # ---------------------------- Argument Parser -----------------------------
    parser = argparse.ArgumentParser('plot initial states')
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

    args.n_Nodes = 64*64
    args.n_Nodes_d = 128*128
    args.os = 'window'
    if args.op_type == 'burgers2d':
        args.max_batch_time = 101

    # ----------------------- plot initial conditions  -------------------------
    # Create figure
    mpl.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=False)
    cmap = cm.get_cmap('inferno')  # "plasma"

    ncases = 5
    fig, ax = plt.subplots(2, ncases, figsize=(14, 5))
    cmin = -3
    cmax = 3
    for i in range(ncases):
        u = getInitialState(i, args)

        ax[0,i].imshow(u[0], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1], vmin=cmin, vmax=cmax)
        ax[1,i].imshow(u[1], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1], vmin=cmin, vmax=cmax)

        ax[0,i].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[0,i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[1,i].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[1,i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[0,i].set_xticklabels([])
        if(i > 0):
            for ax0 in ax[:,i]:
                ax0.set_yticklabels([])
        else:
            ax[0,i].set_ylabel('y', fontsize=14)
            ax[1,i].set_ylabel('y', fontsize=14)

        ax[1,i].set_xlabel('x', fontsize=14)

    ax[0,2].set_title('u', fontsize=14)
    ax[1,2].set_title('v', fontsize=14)

    p0 = ax[0,-1].get_position().get_points().flatten()
    p1 = ax[1,-1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.01, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(cmin, cmax, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)
    # plt.savefig(file_name+".pdf", bbox_inches='tight')

    name = 'InitialState'
    path = osp.join(prjct_dir, 'src/pde_2d\weights_results', 'burgers2d')
    file_loc = osp.join(path, name + '.pdf')
    plt.savefig(file_loc, format='pdf')
    plt.show()
    plt.close(fig)
    plt.close('all')
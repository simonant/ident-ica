import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np


def plot_exp(name, f):
    data_dir = './experiments/' + name
    plot_dir = './plots/' + name
    if not os.path.exists(data_dir):
        print('Experiment does not exist')
        return
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    with open(data_dir + '/param_dict.pkl', 'rb') as fp:
        param_dict = pickle.load(fp)
    runs = param_dict['runs']
    steps = param_dict['steps']
    dim = param_dict['dim']
    lambd = param_dict['lambda']
    res_reg = []
    res_unreg = []
    print(param_dict)
    for i in range(runs):
        run_dir = data_dir + '/' + str(i)
        if not os.path.exists(run_dir):
            print('Experiment not completed')
            return

        with open(run_dir + '/loss_reg.pkl', 'rb') as fp:
            res_reg.append(np.array(pickle.load(fp)))

        with open(run_dir + '/loss_unreg.pkl', 'rb') as fp:
            res_unreg.append(np.array(pickle.load(fp)))
    res_reg = np.stack(res_reg)
    res_unreg = np.stack(res_unreg)
    step_seq = np.arange(0, steps + 1, 1) / (steps + 1)
    mean_reg = np.mean(res_reg, axis=0).T
    std_reg = np.std(res_reg, axis=0).T
    mean_unreg = np.mean(res_unreg, axis=0).T
    std_unreg = np.std(res_unreg, axis=0).T

    alpha = .5
    fig, ax = plt.subplots(1, 3, figsize=(7.5,2.5))

    ax[0].plot(step_seq, mean_reg[0], label='OCT-regularised')
    ax[0].fill_between(step_seq, mean_reg[0]-std_reg[0], mean_reg[0]+std_reg[0], alpha=alpha)
    ax[0].plot(step_seq, mean_unreg[0], label='Unregularized')
    ax[0].fill_between(step_seq, mean_unreg[0] - std_unreg[0], mean_unreg[0] + std_unreg[0], alpha=alpha)
    ax[0].set_xlabel(r' $t$')
    ax[0].set_ylabel(r'$\mathrm{L}_1$')

    ax[1].plot(step_seq, mean_reg[3]-mean_reg[1], label=r'$\lambda=$' + str(lambd))
    ax[1].fill_between(step_seq,  mean_reg[3]-mean_reg[1] - std_reg[1], mean_reg[3]-mean_reg[1] + std_reg[1], alpha=alpha)
    ax[1].set_ylim([-.1, .1])
    ax[1].plot(step_seq, mean_unreg[3]-mean_unreg[1], label=r'$\lambda=0$')
    ax[1].fill_between(step_seq,mean_unreg[3]-mean_unreg[1] - std_unreg[1], mean_unreg[3]-mean_unreg[1] + std_unreg[1], alpha=alpha)
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$D_{KL}(q_t||p_{\theta_t})$')
    ax[1].legend()

    ax[2].plot(step_seq, mean_reg[2], label='OCT-regularised')
    ax[2].fill_between(step_seq, mean_reg[2] - std_reg[2], mean_reg[2] + std_reg[2], alpha=alpha)
    ax[2].plot(step_seq, mean_unreg[2], label='Unregularized')
    ax[2].fill_between(step_seq, mean_unreg[2] - std_unreg[2], mean_unreg[2] + std_unreg[2], alpha=alpha)
    ax[2].set_xlabel(r' $t$')
    ax[2].set_ylabel(r'$C_{\mathrm{OCT}}$')

    plt.tight_layout()
    plt.savefig(plot_dir + '/graphs.jpg')
    plt.clf()
    plt.close('all')
    plot_model_comp(data_dir, f, dim, steps, i=0, plot_dir=plot_dir)


def plot_model_comp(data_dir, f, dim, step, i=0, plot_dir=None):
    f.t = 1
    model_reg = torch.load(data_dir + '/' + str(i) + '/' + str(step) + '/reg.pt')
    model_unreg = torch.load(data_dir + '/' + str(i) + '/' + str(step) + '/unreg.pt')

    # plot settings
    samples = 30000
    cmap = 'hsv'
    s = 3
    alpha = .75

    input = 2 * torch.rand((samples, dim)) - 1

    # color maps parametrized by angle, x, and y coordinate
    colors_1 = torch.arctan2(input[:, 1], input[:, 0])
    colors_2 = input[:, 0]
    colors_3 = input[:, 1]

    output = f(input)
    reconst_reg = model_reg.transform_to_noise(output).detach()
    reconst_unreg = model_unreg.transform_to_noise(output).detach()
    for j, colors in enumerate([colors_1, colors_2, colors_3]):
        strings = ['', 'hor', 'ver']
        string = strings[j]
        fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5))
        for axis in ax:
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlim([-1.,1.])
            axis.set_ylim([-1.,1.])
        ax[0].scatter(input[:, 0], input[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)
        ax[1].scatter(reconst_reg[:, 0], reconst_reg[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)
        ax[2].scatter(reconst_unreg[:, 0], reconst_unreg[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)

        plt.tight_layout()
        if plot_dir is None:
            plt.show()
        else:
            plt.savefig(plot_dir + '/reconst_{}steps{}_{}.jpg'.format(step, string, i))
        plt.clf()
        plt.close('all')

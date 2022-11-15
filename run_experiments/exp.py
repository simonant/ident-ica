import argparse
import os
import pickle

import torch

from identifiable_ica.mixing_fcts import PolarSampleFct, RotationSampleFct
from identifiable_ica.train import compare_trajectories
from run_experiments.plot import plot_exp


def run_eperiments(f, name, lambd, dim, layers, hidden, steps, runs, seed):
    torch.manual_seed(seed)
    time_values = (torch.arange(0., steps + 1.) / steps).tolist()

    para_dict = {'name': name, 'hidden': hidden, 'steps': steps, 'runs': runs, 'dim': dim, 'layers': layers,
                 'lambda': lambd}
    dir_path = './experiments/' + name
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    with open(dir_path + '/param_dict.pkl', 'wb') as file:
        pickle.dump(para_dict, file)
    for i in range(runs):
        print('Starting run {} of {} runs in total'.format(i + 1, runs))
        run_path = dir_path + '/' + str(i)
        if os.path.exists(run_path):
            continue
        a, b, c, d = compare_trajectories(f, time_values, lambd, dim, layers, hidden)
        os.mkdir(run_path)

        with open(run_path + '/loss_unreg.pkl', 'wb') as file:
            pickle.dump(b, file)

        with open(run_path + '/loss_reg.pkl', 'wb') as file:
            pickle.dump(d, file)

        for j, flow in enumerate(a):
            flow_path = run_path + '/' + str(j)
            os.mkdir(flow_path)
            torch.save(flow, flow_path + '/unreg.pt')
            torch.save(c[j], flow_path + '/reg.pt')


def main(args):
    dim = 2
    f = PolarSampleFct(t=0.)
    name_f = 'polar'
    lambda_f = 2
    g = RotationSampleFct(t=0.)
    name_g = 'rotation'
    lambda_g = 50
    run_eperiments(f, name_f, lambda_f, dim, args.layers, args.hidden, args.steps, args.runs, args.seed)
    run_eperiments(g, name_g, lambda_g, dim, args.layers, args.hidden, args.steps, args.runs, args.seed)
    plot_exp(name_f, f)
    plot_exp(name_g, g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=15)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    main(args)

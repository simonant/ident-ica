import copy

import torch
from torch import optim

from identifiable_ica.ima import ima
from identifiable_ica.flow import get_flow
from identifiable_ica.evaluation import check_faithful, evaluate_flow


def get_initialised_flow(f, layers=3, hidden=20, dim=2):
    # Trains an initial flow that matches the mixing f
    f.t = 0
    for i in range(100):
        flow = get_flow(layers, dim, hidden)
        flow = train_init_model(flow, f, dim)
        if check_faithful(flow, f, dim):
            return flow
        else:
            print('Training of initial flow did not converge, restart training')
    print('No suitable intitialisation found!')
    quit()


def train_init_model(flow, f, dim):
    num_iter = 10000
    batch_size = 64
    loss_target = 1.
    optimizer = optim.Adam(flow.parameters())
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fct = torch.nn.HuberLoss()
    for i in range(num_iter):
        inputs = torch.randn((batch_size, dim), requires_grad=True)
        x = f(inputs)
        optimizer.zero_grad()
        loss = loss_fct(flow.transform_to_noise(inputs=x), inputs)
        loss += loss_fct(flow._transform.inverse(inputs)[0], x)
        # loss += torch.mean(ima(flow, inputs))
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            if loss.item() < loss_target and batch_size < 1024:
                batch_size *= 2
                loss_target /= 2
            # print('Compare measures {}'.format(evaluate_flow(flow, f, dim), params))
            scheduler.step()
            if loss.item() < 0.001:
                return flow
        if i == 2000:
            if loss.item() > .2:
                return flow
    return flow


def train_model(flow, f, dim, lambd=0.):
    num_iter = 100
    batch_size = 256
    optimizer = optim.Adam(flow.parameters())
    for i in range(num_iter):
        inputs = torch.randn((batch_size, dim), requires_grad=True)
        x = f(inputs)
        loss = 0
        optimizer.zero_grad()
        if abs(lambd) > 1e-6:
            loss = lambd * torch.mean(ima(flow, inputs))
        loss -= flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()

    return flow


def train_along_trajectory(flow, f, para_list, dim, lambd=0.):
    f.t = 0
    loss_values = []
    flow_list = [copy.deepcopy(flow)]
    for i, t in enumerate(para_list):
        print('Started step {} of {} steps'.format(i+1, len(para_list)))
        f.t = t
        flow = train_model(flow, f, dim, lambd)
        loss_values.append(evaluate_flow(flow, f, dim))
        print('The loss is: ', loss_values[-1])
        flow_list.append(copy.deepcopy(flow))
    return flow_list, loss_values


def compare_trajectories(f, para_list, lambd, dim, layers, hidden):
    print('Train initial flow')
    flow = get_initialised_flow(f, layers, hidden, dim)
    print('Found flow initialisation')
    flow_copy = copy.deepcopy(flow)
    print('Train unregularized model')
    flow_unreg, loss_unreg = train_along_trajectory(flow_copy, f, para_list, dim, lambd=0.)
    flow_copy = copy.deepcopy(flow)

    print('Train regularized model')
    flow_reg, loss_reg = train_along_trajectory(flow_copy, f, para_list, dim, lambd=lambd)
    return flow_unreg, loss_unreg, flow_reg, loss_reg

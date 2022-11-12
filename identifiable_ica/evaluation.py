import torch

from identifiable_ica.ima import jacobian, ima


def check_faithful(flow, f, dim=2):
    samples = 5000
    inputs = torch.randn((samples, dim))
    x = f(inputs)
    loss_fct = torch.nn.MSELoss(reduction='mean')
    loss = loss_fct(flow._transform.inverse(inputs)[0], x)
    return loss.item() < .01


def evaluate_flow(flow, f, dim=2):
    samples = 10000
    inputs = torch.randn((samples, dim), requires_grad=True)
    out = f(inputs)
    in_out = flow.transform_to_noise(out)
    fidelity = torch.mean(torch.sum(torch.abs(inputs - in_out), axis=1))
    likelihood = torch.mean(flow.log_prob(out))
    likelihood_ground_truth = evaluate_ground_truth_likelihood(f, dim)
    ima_loss = torch.mean(ima(flow, inputs))
    return [fidelity.item(), likelihood.item(), ima_loss.item(), likelihood_ground_truth.item()]


def evaluate_ground_truth_likelihood(f, dim=2):
    samples = 10000
    inputs = torch.randn((samples, dim), requires_grad=True)
    outputs = f(inputs)
    jac = jacobian(torch.sum(outputs, dim=0), inputs)
    m = torch.distributions.normal.Normal(torch.zeros(1), torch.ones(1), validate_args=None)
    log_lik = torch.sum(m.log_prob(inputs), dim=1)
    det = torch.log(torch.abs(torch.linalg.det(jac)))
    return torch.mean(log_lik - det)


import torch


def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    jac = torch.stack(jac).reshape(y.shape + x.shape)
    return torch.permute(jac, (1, 2, 0))


def ima(flow, x):
    out, logabsdet = flow._transform.inverse(x)
    jac = jacobian(torch.sum(out, axis=0), x, True)
    sum_log_norms = torch.sum(torch.log(torch.linalg.norm(jac, axis=-1)), axis=1)
    return sum_log_norms - logabsdet

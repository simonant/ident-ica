from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation


def get_flow(num_layers, features, hidden=20, base_dist=StandardNormal):
    base_dist = base_dist(shape=[features])
    transforms = []
    for _ in range(num_layers):
        transforms.append(RandomPermutation(features=features))
        transforms.append(MaskedAffineAutoregressiveTransform(features=features,
                                                              hidden_features=hidden))
    transform = CompositeTransform(transforms)
    return Flow(transform, base_dist)


import unittest

import torch

from federated.aggregation import batchnorm_state_keys, fedbn


class _BatchNormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.BatchNorm1d(2),
        )


class FedBNTests(unittest.TestCase):
    def test_discovers_all_batchnorm_parameters_and_buffers(self):
        keys = batchnorm_state_keys(_BatchNormModel())

        self.assertEqual(
            keys,
            {
                "encoder.1.weight",
                "encoder.1.bias",
                "encoder.1.running_mean",
                "encoder.1.running_var",
                "encoder.1.num_batches_tracked",
            },
        )

    def test_aggregates_only_non_batchnorm_state(self):
        model = _BatchNormModel()
        global_state = model.state_dict()
        bn_keys = batchnorm_state_keys(model)
        client_states = []

        for linear_value, bn_value in ((1.0, 10.0), (3.0, 30.0)):
            state = {key: value.clone() for key, value in global_state.items()}
            state["encoder.0.weight"].fill_(linear_value)
            state["encoder.1.weight"].fill_(bn_value)
            state["encoder.1.bias"].fill_(bn_value)
            state["encoder.1.running_mean"].fill_(bn_value)
            state["encoder.1.running_var"].fill_(bn_value)
            state["encoder.1.num_batches_tracked"].fill_(int(bn_value))
            client_states.append(state)

        aggregated = fedbn(global_state, client_states, [1, 3], bn_keys)

        self.assertTrue(
            torch.allclose(
                aggregated["encoder.0.weight"],
                torch.full_like(aggregated["encoder.0.weight"], 2.5),
            )
        )
        for key in bn_keys:
            self.assertTrue(torch.equal(aggregated[key], global_state[key]), key)


if __name__ == "__main__":
    unittest.main()

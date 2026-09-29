import unittest

import torch
from torch.utils.data import TensorDataset

from federated.aggregation import fedproc
from federated.fedproc import (
    compute_class_prototypes,
    extract_representation,
    prototype_contrastive_loss,
)
from federated.utils import client_checkpoint_path


class _DummyModel(torch.nn.Module):
    def forward(self, text_x, audio_x, return_all=False):
        outputs = {
            "logits": text_x[:, :2],
            "fusion": text_x,
        }
        return outputs if return_all else outputs["logits"]


def _identity_modality(text_x, audio_x, modality):
    return text_x, audio_x


class FedProcTests(unittest.TestCase):
    def test_aggregates_parameters_and_prototypes_by_sample_count(self):
        states = [
            {"weight": torch.tensor([1.0])},
            {"weight": torch.tensor([3.0])},
        ]
        local_prototypes = [
            {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([2.0, 2.0])},
            {0: torch.tensor([0.0, 1.0])},
        ]
        prototype_counts = [{0: 1, 1: 2}, {0: 3}]
        previous = {2: torch.tensor([4.0, 5.0])}

        state, prototypes = fedproc(
            states,
            [1, 3],
            local_prototypes,
            prototype_counts,
            previous_global_prototypes=previous,
        )

        self.assertTrue(torch.allclose(state["weight"], torch.tensor([2.5])))
        self.assertTrue(torch.allclose(prototypes[0], torch.tensor([0.25, 0.75])))
        self.assertTrue(torch.allclose(prototypes[1], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.equal(prototypes[2], previous[2]))

    def test_contrastive_loss_rewards_the_matching_class_prototype(self):
        prototypes = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.0, 1.0]),
        }
        labels = torch.tensor([0, 1])
        aligned = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        swapped = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        aligned_loss = prototype_contrastive_loss(aligned, labels, prototypes, temperature=0.5)
        swapped_loss = prototype_contrastive_loss(swapped, labels, prototypes, temperature=0.5)

        self.assertLess(aligned_loss.item(), swapped_loss.item())
        aligned_loss.backward()
        self.assertIsNotNone(aligned.grad)

    def test_computes_local_class_prototypes_and_counts(self):
        text = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        audio = torch.zeros_like(text)
        labels = torch.tensor([0, 0, 1])
        confidences = torch.ones(3)
        dataset = TensorDataset(text, audio, labels, confidences)

        prototypes, counts = compute_class_prototypes(
            model=_DummyModel(),
            dataset=dataset,
            batch_size=2,
            device=torch.device("cpu"),
            modality="both",
            modality_mask_fn=_identity_modality,
        )

        self.assertEqual(counts, {0: 2, 1: 1})
        self.assertTrue(torch.allclose(prototypes[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(prototypes[1], torch.tensor([0.0, 1.0])))

    def test_prefers_explicit_fusion_representation(self):
        outputs = {
            "fusion": torch.tensor([[1.0, 2.0]]),
            "text_pool": torch.tensor([[3.0]]),
            "audio_pool": torch.tensor([[4.0]]),
        }
        self.assertTrue(torch.equal(extract_representation(outputs), outputs["fusion"]))

    def test_uses_short_temporary_checkpoint_names(self):
        path = client_checkpoint_path("checkpoints/run/pretrain/_tmp", "client_039", 50)
        self.assertEqual(path.replace("\\", "/"), "checkpoints/run/pretrain/_tmp/c039_r50.pt")


if __name__ == "__main__":
    unittest.main()

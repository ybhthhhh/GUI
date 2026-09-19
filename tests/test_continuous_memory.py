import unittest


try:
    import torch
except ImportError:  # pragma: no cover - package is intentionally dependency-light
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ContinuousMemoryTests(unittest.TestCase):
    def test_compressor_has_fixed_budget_and_gradients(self):
        from gui_memory_specialization.continuous_memory import ContinuousMemoryCompressor

        compressor = ContinuousMemoryCompressor(hidden_size=16, memory_tokens=3, layers=2, heads=4).module()
        history = torch.randn(2, 7, 16, requires_grad=True)
        memory = compressor(history, torch.tensor([[1] * 7, [1] * 5 + [0] * 2], dtype=torch.bool))
        self.assertEqual(tuple(memory.shape), (2, 3, 16))
        memory.square().mean().backward()
        self.assertIsNotNone(compressor.latents.grad)

    def test_stage_one_loss_has_action_and_teacher_terms(self):
        from gui_memory_specialization.continuous_memory import stage_one_distillation_loss

        compressed = torch.randn(1, 4, 11, requires_grad=True)
        teacher = torch.randn(1, 4, 11)
        loss, metrics = stage_one_distillation_loss(compressed, teacher, torch.tensor([[-100, 2, 3, -100]]), kl_weight=0.2)
        self.assertEqual(metrics.action_tokens, 2)
        self.assertGreater(metrics.total, 0.0)
        loss.backward()
        self.assertIsNotNone(compressed.grad)

    def test_stage_two_rejects_offline_labels(self):
        from gui_memory_specialization.continuous_memory import validate_online_rewards

        with self.assertRaises(ValueError):
            validate_online_rewards([{"source": "offline_trajectory", "terminal_reward": 1.0}])
        validate_online_rewards([{"source": "online_execution", "terminal_reward": 1.0}])

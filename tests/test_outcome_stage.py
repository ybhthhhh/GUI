import unittest


class OutcomeStageTests(unittest.TestCase):
    def test_rloo_uses_only_sibling_rollouts(self):
        from gui_memory_specialization.outcome_stage import rloo_advantages

        records = [
            {"source": "online_execution", "rollout_group": "task-a", "terminal_reward": 1.0},
            {"source": "online_execution", "rollout_group": "task-a", "terminal_reward": 0.0},
            {"source": "online_execution", "rollout_group": "task-b", "terminal_reward": 0.5},
            {"source": "online_execution", "rollout_group": "task-b", "terminal_reward": 0.5},
        ]
        self.assertEqual(rloo_advantages(records), [1.0, -1.0, 0.0, 0.0])

    def test_offline_result_is_rejected(self):
        from gui_memory_specialization.outcome_stage import validate_execution_records

        with self.assertRaises(ValueError):
            validate_execution_records([
                {"source": "offline_trajectory", "rollout_group": "task", "terminal_reward": 1.0},
                {"source": "offline_trajectory", "rollout_group": "task", "terminal_reward": 0.0},
            ])

    def test_policy_loss_backpropagates(self):
        import torch
        from gui_memory_specialization.outcome_stage import outcome_policy_loss

        scores = torch.tensor([-1.0, -2.0], requires_grad=True)
        loss, advantages = outcome_policy_loss(scores, [
            {"source": "online_execution", "rollout_group": "task", "terminal_reward": 1.0},
            {"source": "online_execution", "rollout_group": "task", "terminal_reward": 0.0},
        ])
        self.assertEqual(advantages, [1.0, -1.0])
        loss.backward()
        self.assertIsNotNone(scores.grad)


if __name__ == "__main__":
    unittest.main()

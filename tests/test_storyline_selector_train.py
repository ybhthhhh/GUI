import math
import unittest

from examples.train_storyline_selectors import choose, fit_selector


class SelectorTrainingTest(unittest.TestCase):
    def test_selector_fits_downstream_candidate_loss_without_target_access(self):
        import torch

        rows = {}
        rewards = {}
        for index in range(50):
            sample_id = f"row-{index}"
            events = []
            candidate_count = 3 if index % 3 == 0 else 2
            for timestamp in range(candidate_count):
                level = (index + timestamp) % 5 / 5
                events.append({"timestamp": timestamp, "action": "click(1,2)" if timestamp else "scroll(1)",
                               "pre_rgb": [level] * 3, "post_rgb": [level + 0.1] * 3,
                               "outcome": {"observed_rgb_l1": 0.1}})
            rows[sample_id] = {"sample_id": sample_id, "episode_id": f"episode-{index}", "split": "train",
                               "current_rgb": [0.5] * 3, "history": events,
                               "target_action": "not visible to selector"}
            rewards[sample_id] = [1.0, 0.7] + ([0.9] if candidate_count == 3 else [])
        model, mean, std, metadata = fit_selector("qwen25vl", rows, rewards, torch)
        self.assertEqual(metadata["fit_rows"] + metadata["calibration_rows"], len(rows))
        self.assertTrue(1 <= metadata["best_epoch"] <= 300)
        self.assertTrue(math.isfinite(metadata["calibration_action_ce"]))
        for row in rows.values():
            self.assertIn(choose(model, mean, std, row, torch), range(len(row["history"])))


if __name__ == "__main__":
    unittest.main()

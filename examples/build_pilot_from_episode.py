"""Build one screenshot-backed next-action sample from a chosen extracted episode."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("episode", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--target-index", type=int, default=2)
parser.add_argument("--source", required=True)
args = parser.parse_args()

rows = [json.loads(line) for line in (args.episode / "traj.jsonl").read_text().splitlines() if line]
if not 0 < args.target_index < len(rows):
    raise ValueError("target-index must leave at least one history event")
history = [{"timestamp": row["step_num"], "pre_state": "previous GUI state", "action": row["action"],
            "outcome": "observed", "post_state": "next GUI state", "visual_evidence": [str(args.episode / row["screenshot_file"])],
            "subgoal": "unknown", "failure_status": "unknown", "temporal_links": []} for row in rows[:args.target_index]]
target = rows[args.target_index]
sample = {"sample_id": f"{args.source}-{args.episode.name}-step-{target['step_num']}", "task": "OSWorld Writer line spacing",
          "current_observation": "current OSWorld GUI screenshot", "gold_action": target["action"],
          "history": history, "current_screenshot": str(args.episode / target["screenshot_file"])}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(sample) + "\n")
print(json.dumps({"output": str(args.output), "history_events": len(history), "target_step": target["step_num"]}))

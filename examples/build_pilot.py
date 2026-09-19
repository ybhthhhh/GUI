"""Build one real screenshot-backed action-prediction sample from an extracted episode."""
import json
from pathlib import Path

root = Path("data/derived/pilot/jedi-7b-4o-15steps/libreoffice_writer/0810415c-bde4-4443-9047-d5f70165a697")
rows = [json.loads(line) for line in (root / "traj.jsonl").read_text().splitlines() if line]
target_index = 3
history = []
for row in rows[:target_index]:
    history.append({"timestamp": row["step_num"], "pre_state": "previous GUI state", "action": row["action"], "outcome": "observed", "post_state": "next GUI state", "visual_evidence": [str(root / row["screenshot_file"])], "subgoal": "unknown", "failure_status": "unknown", "temporal_links": []})
target = rows[target_index]
sample = {"sample_id": "jedi-writer-pilot-step-4", "task": "unknown", "current_observation": "current OSWorld GUI screenshot", "gold_action": target["action"], "history": history, "current_screenshot": str(root / target["screenshot_file"])}
Path("data/derived/pilot/pilot.jsonl").write_text(json.dumps(sample) + "\n")
print(json.dumps({"history_events": len(history), "target_step": target["step_num"], "screenshot": sample["current_screenshot"]}))

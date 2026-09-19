"""Read-only audit and canonical conversion for OSWorld Verified ZIP archives."""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import PurePosixPath


def audit_archive(path: str) -> dict[str, object]:
    """Inspect metadata inside an archive without extracting it."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        trajs = [name for name in names if name.endswith('/traj.jsonl')]
        results = [name for name in names if name.endswith('/result.txt')]
        apps = Counter(PurePosixPath(name).parts[-3] for name in trajs)
        episodes = 0
        steps = 0
        done_steps = 0
        for name in trajs:
            with archive.open(name) as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            episodes += 1
            steps += len(rows)
            done_steps += sum(bool(row.get('done')) for row in rows)
        return {"archive": path, "episodes": episodes, "steps": steps,
                "done_steps": done_steps, "traj_files": len(trajs),
                "result_files": len(results), "apps": dict(apps)}


def canonical_rows(path: str, limit: int) -> list[dict[str, object]]:
    """Produce action-prediction rows; never invent task or success labels."""
    rows_out: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for traj_name in (n for n in archive.namelist() if n.endswith('/traj.jsonl')):
            episode = str(PurePosixPath(traj_name).parent)
            with archive.open(traj_name) as handle:
                actions = [json.loads(line) for line in handle if line.strip()]
            for index, target in enumerate(actions[1:], start=1):
                history = []
                for previous in actions[:index]:
                    history.append({"timestamp": previous['step_num'], "pre_state": "previous GUI state",
                        "action": previous['action'], "outcome": "observed", "post_state": "next GUI state",
                        "visual_evidence": [f"{episode}/{previous['screenshot_file']}"], "subgoal": "unknown",
                        "failure_status": "unknown", "temporal_links": []})
                rows_out.append({"sample_id": f"{episode}#{target['step_num']}", "task": "unknown",
                    "current_observation": "OSWorld screenshot at current step", "gold_action": target['action'],
                    "history": history, "source_episode": episode,
                    "current_screenshot_member": f"{episode}/{target['screenshot_file']}"})
                if len(rows_out) >= limit:
                    return rows_out
    return rows_out

import json

from PIL import Image

from gui_memory_specialization.storyline_canonical import (
    build,
    candidate_features,
    candidate_messages,
    canonicalize,
)


def test_transition_links_and_same_budget(tmp_path):
    paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (400, 200), color).save(path)
        paths.append(path)
    row = {
        "sample_id": "e1-1", "episode_id": "e1", "split": "train", "task": "Buy a book",
        "history": [{"image": str(paths[0]), "action": "click(1,2)"},
                    {"image": str(paths[1]), "action": "scroll(4)"}],
        "current_image": str(paths[2]), "action": "click(5,6)",
    }
    canonical = canonicalize(row, tmp_path / "cache")
    assert len(canonical["history"]) == 2
    assert canonical["history"][0]["post_state_image"] == canonical["history"][1]["pre_state_image"]
    assert canonical["history"][1]["post_state_image"] == canonical["current_image"]
    assert canonical["history"][0]["outcome"]["success"] is None
    for event in canonical["history"]:
        messages = candidate_messages(canonical, event)
        assert len(messages) == 3
        assert all(sum(part["type"] == "image" for part in message["content"]) == 1 for message in messages)
        assert len(candidate_features(canonical, event)) == 15
    altered = dict(canonical, target_action="secret gold target")
    assert candidate_features(altered, altered["history"][0]) == candidate_features(canonical, canonical["history"][0])
    assert "secret gold target" not in json.dumps(candidate_messages(altered, altered["history"][0]))


def test_episode_split_leakage_is_rejected(tmp_path):
    path = tmp_path / "state.png"
    Image.new("RGB", (10, 10)).save(path)
    row = {"sample_id": "a", "episode_id": "shared", "split": "train", "task": "x",
           "history": [{"image": str(path), "action": "a"}, {"image": str(path), "action": "b"}],
           "current_image": str(path), "action": "c"}
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, sample_id="b", split="validation")) + "\n")
    try:
        build(source, tmp_path / "manifest.jsonl", tmp_path / "cache")
    except ValueError as error:
        assert "leakage" in str(error)
    else:
        raise AssertionError("episode leakage was not detected")

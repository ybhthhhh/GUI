import json

from PIL import Image

from examples.build_storyline_structured import build
from gui_memory_specialization.storyline_actions import executable_action


def test_extracts_only_executable_fields():
    response = "I should scroll.\n```json\n" + json.dumps({
        "name": "Scroll", "arguments": {"direction": "down", "reasoning": "private prose",
                                        "description": "narrative"}}) + "\n```"
    assert executable_action(response) == '{"arguments":{"direction":"down"},"name":"scroll"}'
    assert executable_action("no JSON command here") is None


def test_structured_manifest_filters_unparsed_actions(tmp_path):
    image = tmp_path / "screen.png"
    Image.new("RGB", (20, 10)).save(image)
    command = '```json\n{"name":"click","arguments":{"element_id":1,"reasoning":"omit"}}\n```'
    base = {"episode_id": "episode-a", "split": "train", "task": "Do task",
            "history": [{"image": str(image), "action": command}, {"image": str(image), "action": command}],
            "current_image": str(image), "action": command}
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(dict(base, sample_id="good")) + "\n" +
                      json.dumps(dict(base, sample_id="bad", action="not an action")) + "\n")
    output = tmp_path / "canonical.jsonl"
    result = build(source, output, tmp_path / "images")
    assert result["train"] == 1
    assert result["excluded_unparsed_target"] == 1
    row = json.loads(output.read_text().splitlines()[0])
    assert "reasoning" not in row["target_action"]
    assert "reasoning" not in row["history"][0]["action"]

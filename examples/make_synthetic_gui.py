"""Create a deterministic mock GUI image for a VLM adapter health check.

It is deliberately not benchmark data. Its sole purpose is to prove that a
screenshot is passed through the frozen VLM scorer without a GPU-memory failure.
"""

from pathlib import Path

from PIL import Image, ImageDraw


output = Path(__file__).with_name("synthetic_upload.png")
image = Image.new("RGB", (800, 450), "#f3f5f7")
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((80, 55, 720, 395), radius=12, fill="white", outline="#c9d1d9", width=2)
draw.text((120, 100), "Upload report", fill="#20242a")
draw.text((120, 155), "Network error. The upload did not complete.", fill="#b42318")
draw.rounded_rectangle((120, 260, 280, 315), radius=8, fill="#1769aa")
draw.text((163, 278), "Retry", fill="white")
image.save(output)
print(output)


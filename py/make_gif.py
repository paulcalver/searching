"""
make_gif.py
===========

Standalone GIF encoder, invoked as a subprocess by ai_assess_git.py.

Why a separate script?
----------------------
TouchDesigner ships its own embedded Python which often does not have
Pillow available, and adding new packages to TD's Python is fiddly.
The system Python (`python3`) almost always has Pillow installed, so
we shell out to it instead of trying to encode the GIF inside the TD
process. Keeping the encoder in its own file also makes it trivial to
swap implementations later (imageio, ffmpeg, etc.) without touching
the main controller.

Usage
-----
    python3 make_gif.py <frame_dir> <output_path> <fps>

    frame_dir   : directory containing sequentially-named .jpg frames
    output_path : where to write the resulting .gif
    fps         : playback rate of the GIF in frames per second

Exits with code 1 if no frames are found in frame_dir.
"""

import sys
import os
from PIL import Image  # Pillow — only used here for GIF encoding

# CLI args. No argparse — the caller (ai_assess_git.call to subprocess)
# always passes exactly these three positionally, so keep it simple.
frame_dir = sys.argv[1]
output_path = sys.argv[2]
fps = int(sys.argv[3])

# Load all jpg frames in order
# sorted() works because the caller writes files as frame_0000.jpg,
# frame_0001.jpg, ... so lexical order matches frame order.
frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
images = [Image.open(os.path.join(frame_dir, f)) for f in frames]

# Guard against an empty directory — the caller checks this too, but
# better to fail loudly here than write an invalid GIF.
if not images:
	sys.exit(1)

# Pillow's GIF writer wants per-frame duration in milliseconds, not fps.
duration = int(1000 / fps)  # ms per frame

# Pillow's multi-frame GIF idiom: save the first image with
# save_all=True and pass the rest via append_images. loop=0 means
# the GIF loops forever, which is what the API ingest expects.
images[0].save(
	output_path,
	save_all=True,
	append_images=images[1:],
	duration=duration,
	loop=0
)

# Single-line status that the parent process captures from stdout
# and forwards to the TouchDesigner textport.
print(f'GIF saved: {output_path} ({len(images)} frames at {fps}fps)')

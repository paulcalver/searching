import sys
import os
from PIL import Image

frame_dir = sys.argv[1]
output_path = sys.argv[2]
fps = int(sys.argv[3])

# Load all jpg frames in order
frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
images = [Image.open(os.path.join(frame_dir, f)) for f in frames]

if not images:
	sys.exit(1)

duration = int(1000 / fps)  # ms per frame

images[0].save(
	output_path,
	save_all=True,
	append_images=images[1:],
	duration=duration,
	loop=0
)

print(f'GIF saved: {output_path} ({len(images)} frames at {fps}fps)')
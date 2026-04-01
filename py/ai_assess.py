import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import base64
import json
import subprocess
import threading
import os
import tempfile

API_KEY = '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'

# Paths
FRAME_DIR = '/Users/pc_mbp14/CODE/MA/Term_02/Final_Project/cymatics_frames'
GIF_PATH = '/Users/pc_mbp14/CODE/MA/Term_02/Final_Project/cymatics.gif'
GIF_SCRIPT = '/Users/pc_mbp14/CODE/MA/Term_02/Final_Project/py/make_gif.py'

# Ensure frame directory exists
os.makedirs(FRAME_DIR, exist_ok=True)

# Frequency state
current_freq = 45.0
FREQ_MIN = 32.0
FREQ_MAX = 100.0

# Nudge parameters
BASE_NUDGE = 0.5
MAX_NUDGE = 4.0
HOLD_THRESHOLD = 7.5
SETTLE_THRESHOLD = 6.0

# Gradient memory
last_score = None
last_direction = 1
pending_freq = None
pending_display = None
boundary_hit = False

# Exploration state
settled_count = 0
SETTLED_BAND = 3.0
SETTLED_LIMIT = 8
JUMP_NUDGE = 12.0

# Frame buffer
FRAME_BUFFER_MAX = 60
FRAME_SUBSAMPLE = 3
GIF_FPS = 10
frame_index = 0

# Assessment lock
assessment_in_progress = False


def call_api(payload_str):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(payload_str)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST',
             'https://api.anthropic.com/v1/messages',
             '-H', 'Content-Type: application/json',
             '-H', f'x-api-key: {API_KEY}',
             '-H', 'anthropic-version: 2023-06-01',
             '-d', f'@{temp_path}'],
            capture_output=True, text=True, timeout=60
        )
        parsed = json.loads(result.stdout)
        if 'content' not in parsed:
            print(f'API error response: {parsed}')
        return parsed
    finally:
        os.remove(temp_path)


def capture():
    global frame_index

    top = op('/project1/main_video_feed')
    buf = top.numpyArray()
    img = (buf[:, :, :3] * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_small = cv2.resize(img_bgr, (320, 180))

    frame_path = os.path.join(FRAME_DIR, f'frame_{frame_index:04d}.jpg')
    cv2.imwrite(frame_path, img_small, [cv2.IMWRITE_JPEG_QUALITY, 80])

    frame_index = (frame_index + 1) % FRAME_BUFFER_MAX

    all_frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
    while len(all_frames) > FRAME_BUFFER_MAX:
        os.remove(os.path.join(FRAME_DIR, all_frames.pop(0)))


def make_gif():
    import shutil

    all_frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
    subsampled = all_frames[::FRAME_SUBSAMPLE]

    if len(subsampled) < 3:
        return None

    temp_dir = os.path.join(FRAME_DIR, 'gif_temp')
    os.makedirs(temp_dir, exist_ok=True)

    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))

    for i, fname in enumerate(subsampled):
        src = os.path.join(FRAME_DIR, fname)
        dst = os.path.join(temp_dir, f'frame_{i:04d}.jpg')
        shutil.copy(src, dst)

    result = subprocess.run(
        ['python3', GIF_SCRIPT, temp_dir, GIF_PATH, str(GIF_FPS)],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        print(f'GIF encoding error: {result.stderr}')
        return None

    print(result.stdout.strip())

    with open(GIF_PATH, 'rb') as f:
        gif_bytes = f.read()

    return base64.b64encode(gif_bytes).decode('utf-8')


def run():
    global pending_freq, pending_display, assessment_in_progress

    if pending_freq is not None:
        op('/project1/audio_out_1').par.frequency = pending_freq
        print(f'Frequency set to: {pending_freq:.1f}Hz')
        pending_freq = None

    if pending_display is not None:
        table = op('/project1/readout_data')
        table[0, 0] = pending_display
        pending_display = None

    if assessment_in_progress:
        print('Assessment in progress, skipping cycle')
        return

    all_frames = [f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')]
    if len(all_frames) < FRAME_BUFFER_MAX // 2:
        print(f'Waiting for frame buffer to fill... ({len(all_frames)}/{FRAME_BUFFER_MAX})')
        return

    assessment_in_progress = True
    thread = threading.Thread(target=assess_pattern)
    thread.daemon = True
    thread.start()


def assess_pattern():
    import random
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    global current_freq, pending_freq, pending_display, last_score, last_direction
    global settled_count, boundary_hit, assessment_in_progress

    try:
        # Encode gif from current frame buffer
        gif_b64 = make_gif()
        if gif_b64 is None:
            print('GIF encoding failed, skipping assessment')
            return

        # Call 1 — vision description
        description_payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 150,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/gif",
                            "data": gif_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are watching a short animation of water in a petri dish being "
                            "vibrated at a low frequency. Describe what you see in 2-3 sentences, "
                            "focusing specifically on: whether any bright regions stay fixed in "
                            "position across the animation, whether you can see any lines, ridges "
                            "or geometric arrangement of light and dark zones, and how much the "
                            "surface texture changes between frames. Be precise and observational, "
                            "not interpretive. Do not use headers, titles or markdown formatting. "
                            "Begin your response directly with the description."
                        )
                    }
                ]
            }]
        })

        description_result = call_api(description_payload)
        if 'content' not in description_result:
            print('Description call failed')
            return

        description = description_result['content'][0]['text'].strip()
        # Remove any markdown headers the model adds
        description = '\n'.join(line for line in description.split('\n') if not line.startswith('#')).strip()
        print(f'Description: {description}')

        # Call 2 — rating from description only, no image
        rating_payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 10,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"You are rating cymatics resonance in vibrating water based on this "
                        f"description:\n\n{description}\n\n"
                        "Resonant water produces stable standing waves visible as bright lines, "
                        "ridges or symmetric geometric zones of light and dark that stay fixed "
                        "over time. Non-resonant water shows random rippling with no persistent "
                        "geometry. "
                        "Based solely on the description above, how closely does this match "
                        "resonant standing wave behaviour? Rate 0-10 where 0 is purely chaotic "
                        "and 10 is clearly resonant with stable persistent geometry. Be decisive "
                        "and use the full range of the scale. Return only a number."
                    )
                }]
            }]
        })

        rating_result = call_api(rating_payload)
        if 'content' not in rating_result:
            print('Rating call failed')
            return

        raw = rating_result['content'][0]['text'].strip()
        score = float(raw)
        print(f'Score: {score}')

# Call 3 — short display summary
        summary_payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 40,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"Based on this description of vibrating water and a resonance score of {score}/10, "
                        f"write one sentence of no more than 12 words that states what the system has concluded "
                        f"about the current pattern. Write as if the system is reporting its own assessment. "
                        f"Do not use markdown. Return only the sentence:\n\n{description}"
                    )
                }]
            }]
        })

        summary_result = call_api(summary_payload)
        if 'content' not in summary_result:
            print('Summary call failed, using full description')
            display_text = description
        else:
            display_text = summary_result['content'][0]['text'].strip()
            # Remove any markdown headers
            display_text = '\n'.join(line for line in display_text.split('\n') if not line.startswith('#')).strip()
            print(f'Summary: {display_text}')

        # Check if we've been settled too long
        freq_delta = abs(current_freq - (pending_freq if pending_freq else current_freq))
        if freq_delta < SETTLED_BAND:
            settled_count += 1
        else:
            settled_count = 0

        # Force exploration if settled too long
        if settled_count >= SETTLED_LIMIT:
            settled_count = 0
            last_score = None
            boundary_hit = False
            jump_direction = random.choice([-1, 1])
            new_freq = current_freq + (jump_direction * JUMP_NUDGE)
            new_freq = float(round(max(FREQ_MIN, min(FREQ_MAX, new_freq)) * 2) / 2)
            current_freq = new_freq
            pending_freq = new_freq
            pending_display = f'{display_text}\n\nScore: {score} | jump: {jump_direction * JUMP_NUDGE:+.1f}Hz | freq: {new_freq:.1f}Hz'
            print(f'Score: {score} -> exploration jump: {jump_direction * JUMP_NUDGE:+.1f}Hz -> new freq: {new_freq:.1f}Hz')
            return

        # Determine direction using gradient memory with threshold
        if boundary_hit:
            direction = last_direction
            boundary_hit = False
        elif last_score is None:
            direction = last_direction
        elif score > last_score + 0.5:
            direction = last_direction
        elif score < last_score - 0.5:
            direction = -last_direction
        else:
            direction = last_direction

        # Determine nudge size based on score
        if score >= HOLD_THRESHOLD:
            nudge_size = 0.0
            pending_display = f'{display_text}\n\nScore: {score} | holding at {current_freq:.1f}Hz'
            print(f'Score: {score} -> holding at {current_freq:.1f}Hz')
        elif score >= SETTLE_THRESHOLD:
            nudge_size = BASE_NUDGE
        else:
            nudge_size = BASE_NUDGE + (MAX_NUDGE - BASE_NUDGE) * ((SETTLE_THRESHOLD - score) / SETTLE_THRESHOLD)

        last_score = score
        last_direction = direction

        if nudge_size > 0:
            new_freq = current_freq + (direction * nudge_size)

            if new_freq <= FREQ_MIN:
                new_freq = FREQ_MIN
                last_direction = 1
                boundary_hit = True
            elif new_freq >= FREQ_MAX:
                new_freq = FREQ_MAX
                last_direction = -1
                boundary_hit = True
            else:
                new_freq = float(round(new_freq * 2) / 2)

            current_freq = new_freq
            pending_freq = new_freq
            pending_display = f'{display_text}\n\nScore: {score} | nudge: {direction * nudge_size:+.1f}Hz | freq: {new_freq:.1f}Hz'
            print(f'Score: {score} -> nudge: {direction * nudge_size:+.1f}Hz -> new freq: {new_freq:.1f}Hz')

    except subprocess.TimeoutExpired:
        print('API call timed out')
    except json.JSONDecodeError as e:
        print(f'Could not parse API response: {e}')
    except ValueError as e:
        print(f'Could not parse score: {e}')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        assessment_in_progress = False
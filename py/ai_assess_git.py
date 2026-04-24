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
import time
import math

API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'YOUR-KEY-HERE')

# Paths — replace with the absolute path to this project folder on your machine
PROJECT_DIR = '/path/to/Touch Designer'
FRAME_DIR = f'{PROJECT_DIR}/cymatics_frames'
GIF_PATH = f'{PROJECT_DIR}/cymatics.gif'
GIF_SCRIPT = f'{PROJECT_DIR}/py/make_gif.py'

# Ensure frame directory exists
os.makedirs(FRAME_DIR, exist_ok=True)

# Frequency state
current_freq = 65.0
target_freq = 85.0
shift_start_freq = 65.0
FREQ_MIN = 30.0
FREQ_MAX = 100.0

# Nudge parameters
BASE_NUDGE = 0.5
MAX_NUDGE = 20.0
HOLD_THRESHOLD = 10.0
SETTLE_THRESHOLD = 6.0

# Gradient memory
last_score = None
last_direction = 1
boundary_hit = False

# Exploration state
settled_count = 0
SETTLED_BAND = 3.0
SETTLED_LIMIT = 8
JUMP_NUDGE = 25.0

# Frame buffer
FRAME_BUFFER_MAX = 60
FRAME_SUBSAMPLE = 3
GIF_FPS = 10
frame_index = 0

# Display
pending_display = None
last_display_text = 'initialising...'
last_score_line = ''

# State machine
# States: ASSESSING, RESULTS, SHIFTING, BUFFERING
state = 'BUFFERING'
state_start_time = time.time()
RESULTS_DURATION = 10.0
SHIFT_DURATION = 2.0
BUFFER_WAIT = 2.0

# Assessment timing
last_assessment_time = 0
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


def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def set_display(line1, line2=''):
    global pending_display
    if line2:
        pending_display = f'{line1}\n\n{line2}'
    else:
        pending_display = line1


def capture():
    global frame_index, state, state_start_time, current_freq

    # Handle frequency ramp at capture rate (30fps)
    if state == 'SHIFTING':
        now = time.time()
        elapsed = now - state_start_time
        progress = min(elapsed / SHIFT_DURATION, 1.0)
        eased = ease_in_out(progress)
        interpolated = shift_start_freq + (target_freq - shift_start_freq) * eased

        if abs(interpolated - current_freq) > 0.01:
            current_freq = interpolated
            op('/project1/audio_out_1').par.frequency = current_freq

        if progress >= 1.0:
            current_freq = target_freq
            op('/project1/audio_out_1').par.frequency = current_freq
            print(f'Shift complete, now at {current_freq:.1f}Hz')
            for f in os.listdir(FRAME_DIR):
                if f.endswith('.jpg'):
                    os.remove(os.path.join(FRAME_DIR, f))
            state = 'BUFFERING'
            state_start_time = now
        return

    # Normal frame capture
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
    global state, state_start_time, current_freq, target_freq, shift_start_freq
    global pending_display, assessment_in_progress, last_assessment_time
    global last_display_text, last_score_line

    now = time.time()

    # Apply any pending display update
    if pending_display is not None:
        table = op('/project1/readout_data')
        table[0, 0] = pending_display
        pending_display = None

    # --- STATE: BUFFERING ---
    if state == 'BUFFERING':
        all_frames = [f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')]
        set_display('assessing...')
        elapsed = now - state_start_time
        if len(all_frames) >= FRAME_BUFFER_MAX and elapsed >= BUFFER_WAIT:
            print('Buffer full, starting assessment')
            state = 'ASSESSING'
            state_start_time = now
            last_assessment_time = now
            assessment_in_progress = True
            thread = threading.Thread(target=assess_pattern)
            thread.daemon = True
            thread.start()

    # --- STATE: ASSESSING ---
    elif state == 'ASSESSING':
        set_display('assessing...')

    # --- STATE: RESULTS ---
    elif state == 'RESULTS':
        if state_start_time == 0:
            state_start_time = now
            print('RESULTS: timer started')
        elapsed = now - state_start_time
        set_display(last_display_text, last_score_line)
        if elapsed >= RESULTS_DURATION:
            print('Results displayed, beginning frequency shift')
            shift_start_freq = current_freq
            state = 'SHIFTING'
            state_start_time = now

    # --- STATE: SHIFTING ---
    elif state == 'SHIFTING':
        set_display(f'shifting frequency...\n{current_freq:.1f} Hz → {target_freq:.1f} Hz')


def assess_pattern():
    import random
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    global target_freq, last_score, last_direction
    global settled_count, boundary_hit, assessment_in_progress
    global state, state_start_time, last_display_text, last_score_line

    try:
        gif_b64 = make_gif()
        if gif_b64 is None:
            print('GIF encoding failed, skipping assessment')
            state = 'BUFFERING'
            state_start_time = time.time()
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
            state = 'BUFFERING'
            state_start_time = time.time()
            return

        description = description_result['content'][0]['text'].strip()
        description = '\n'.join(line for line in description.split('\n') if not line.startswith('#')).strip()
        print(f'Description: {description}')

        # Call 2 — rating from description
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
            state = 'BUFFERING'
            state_start_time = time.time()
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
                    # "text": (
                    #     f"Based on this description of vibrating water and a resonance score of {score}/10, "
                    #     f"write one sentence of no more than 12 words that states what the system has concluded "
                    #     f"about the current pattern. Write as if the system is reporting its own assessment. "
                    #     f"Do not use markdown. Return only the sentence:\n\n{description}"
                    # )
#                     "text": (
#                         f"You are an entity searching through vibrating water for moments of hidden order. "
#                         f"Based on this description of what you observed:\n\n{description}\n\n"
#                         f"And a resonance score of {score}/10, write two short sentences in a searching, "
#                         f"uncertain voice that express what you are looking for and whether you feel closer "
#                         f"to or further from finding it. Use elemental language: water, light, stillness, "
#                         f"pattern, order. Do not explain or analyse. Do not use markdown. "
#                         f"Write as if you are genuinely uncertain whether what you saw was meaningful. "
#                         f"Each sentence should be under 10 words."
#                     )
                    "text": (
                        f"Resonance score: {score}/10.\n\n"
                        f"What was observed: {description}\n\n"
                        f"Write two short sentences, each under 8 words. "
                        f"You are searching for order in water and light. "
                        f"Respond poetically and obliquely — suggest rather than describe. "
                        f"Let the score guide your feeling of closeness or distance from what you seek. "
                        f"Write both sentences on the same line with no line breaks between them. "
                        f"No markdown."
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
            display_text = '\n'.join(line for line in display_text.split('\n') if not line.startswith('#')).strip()
            print(f'Summary: {display_text}')

            # TEMP: comment out to restore poetic display
            # display_text = ''

        # Calculate new target frequency
        if settled_count >= SETTLED_LIMIT:
            settled_count = 0
            last_score = None
            boundary_hit = False
            jump_direction = random.choice([-1, 1])
            new_freq = current_freq + (jump_direction * JUMP_NUDGE)
            new_freq = float(round(max(FREQ_MIN, min(FREQ_MAX, new_freq)) * 2) / 2)
            score_line = f'Score: {score} | jump: {jump_direction * JUMP_NUDGE:+.1f}Hz | freq: {new_freq:.1f}Hz'
            print(f'Score: {score} -> exploration jump to {new_freq:.1f}Hz')
        else:
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

            if score >= HOLD_THRESHOLD:
                nudge_size = 0.0
                score_line = f'Score: {score} | holding at {current_freq:.1f}Hz'
                print(f'Score: {score} -> holding at {current_freq:.1f}Hz')
            else:
                # Exponential curve: small nudges near threshold, large when cold
                # Normalise score to 0-1 range below hold threshold
                t = 1.0 - (score / HOLD_THRESHOLD)
                nudge_size = BASE_NUDGE + (MAX_NUDGE - BASE_NUDGE) * (t ** 2)

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

                score_line = f'Score: {score} | nudge: {direction * nudge_size:+.1f}Hz | freq: {new_freq:.1f}Hz'
                print(f'Score: {score} -> nudge: {direction * nudge_size:+.1f}Hz -> new freq: {new_freq:.1f}Hz')
            else:
                new_freq = current_freq
                score_line = f'Score: {score} | holding at {current_freq:.1f}Hz'

        # Check settled count
        freq_delta = abs(current_freq - new_freq)
        if freq_delta < SETTLED_BAND:
            settled_count += 1
        else:
            settled_count = 0

        # Store results and transition to RESULTS state
        target_freq = new_freq
        last_display_text = display_text
        last_score_line = score_line
        state = 'RESULTS'
        state_start_time = 0  # signal run() to set timer on first entry

    except subprocess.TimeoutExpired:
        print('API call timed out')
        state = 'BUFFERING'
        state_start_time = time.time()
    except json.JSONDecodeError as e:
        print(f'Could not parse API response: {e}')
        state = 'BUFFERING'
        state_start_time = time.time()
    except ValueError as e:
        print(f'Could not parse score: {e}')
        state = 'BUFFERING'
        state_start_time = time.time()
    except Exception as e:
        print(f'Error: {e}')
        state = 'BUFFERING'
        state_start_time = time.time()
    finally:
        assessment_in_progress = False
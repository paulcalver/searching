"""
ai_assess_git.py
================

TouchDesigner-side controller for an AI-driven cymatics installation.

Overview
--------
A speaker vibrates a petri dish of water at a chosen frequency. A camera
films the surface of the water and feeds it into TouchDesigner. This
script runs inside that TouchDesigner project and does three things:

    1. Continuously captures frames from the live video feed and keeps a
       rolling buffer of the most recent ones on disk.
    2. Periodically encodes that buffer into a short GIF and asks the
       Claude API to (a) describe what it sees, (b) score how
       "resonant" / standing-wave-like the pattern looks on a 0-10
       scale, and (c) generate a short poetic line for the on-screen
       readout.
    3. Uses the score to nudge the speaker frequency toward more
       resonant patterns — a closed feedback loop where the AI's
       aesthetic judgement steers the physical system toward order.

The script is split across two TouchDesigner Execute DAT entry points:

    - capture(): called every frame (~30 fps). Cheap. Grabs a frame
                 from the video TOP, writes it to disk, and also
                 advances the frequency ramp during a SHIFTING phase.

    - run():     called on a slower timer (~1 Hz). Drives the state
                 machine, updates the on-screen text, and kicks off
                 background assessment threads.

State machine
-------------
The system cycles through four states:

    BUFFERING -> ASSESSING -> RESULTS -> SHIFTING -> BUFFERING -> ...

    BUFFERING : waiting for the rolling frame buffer to refill after a
                frequency change so the GIF reflects the *new* pattern,
                not the transition.
    ASSESSING : a background thread is talking to the Claude API.
                The UI shows "assessing...".
    RESULTS   : the score and poetic line are displayed for a fixed
                duration so the audience can read them.
    SHIFTING  : the speaker frequency is being smoothly ramped from
                its current value to the new target chosen by the
                feedback logic.

Feedback logic
--------------
A simple gradient-follower with two escape hatches:

    - If the score is high enough (>= HOLD_THRESHOLD) the system holds
      the current frequency.
    - Otherwise it nudges in the direction that improved the score
      last time, with nudge size scaled exponentially by how far the
      score is from the hold threshold (cold = big jumps, warm = small).
    - If the system stays in roughly the same frequency band for too
      many assessments in a row (settled_count >= SETTLED_LIMIT) it
      makes a large random jump to escape local maxima.
    - Hitting FREQ_MIN/FREQ_MAX flips direction.
"""

# SSL verification is disabled because the embedded Python that ships
# with TouchDesigner often does not have an up-to-date certificate
# bundle, which makes HTTPS calls to api.anthropic.com fail. We make
# the calls via curl as a subprocess (see call_api) so this monkey
# patch is mostly defensive — but the assess_pattern thread also
# re-applies it because it imports ssl in its own scope.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2          # OpenCV — colour conversion, resizing, JPEG writing
import numpy as np  # for converting the TOP's pixel array
import base64       # GIF -> base64 for the Claude vision payload
import json         # building API request bodies and parsing replies
import subprocess   # shelling out to curl and to the GIF encoder script
import threading    # assessment runs off-thread to keep TD responsive
import os
import tempfile     # API payloads are written to a temp file for curl -d @
import time
import math         # for the cosine ease used during frequency ramps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Anthropic API key. Pulled from the environment if available, otherwise
# falls back to a placeholder so the file is safe to commit.
API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'YOUR-KEY-HERE')

# Paths — replace with the absolute path to this project folder on your machine
PROJECT_DIR = '/path/to/Touch Designer'
FRAME_DIR = f'{PROJECT_DIR}/cymatics_frames'   # rolling buffer of JPEGs
GIF_PATH = f'{PROJECT_DIR}/cymatics.gif'       # output GIF that gets sent to the API
GIF_SCRIPT = f'{PROJECT_DIR}/py/make_gif.py'   # external encoder, run via python3

# Make sure the frame buffer directory exists before anything tries to
# write into it. exist_ok=True so re-runs don't fail.
os.makedirs(FRAME_DIR, exist_ok=True)


# --- Frequency state -------------------------------------------------------
# All values in Hz. The speaker is driven by op('/project1/audio_out_1').
current_freq = 65.0       # what the speaker is actually playing right now
target_freq = 85.0        # where we want to ramp to next
shift_start_freq = 65.0   # value of current_freq at the moment a SHIFT began
                          # (needed to interpolate cleanly with easing)
FREQ_MIN = 30.0           # lower bound — below this is inaudible / unsafe
FREQ_MAX = 100.0          # upper bound — above this the dish stops responding


# --- Nudge parameters ------------------------------------------------------
# How much the frequency moves between assessments.
BASE_NUDGE = 0.5          # smallest step, used when score is just below threshold
MAX_NUDGE = 20.0          # largest step, used when score is 0 (totally chaotic)
HOLD_THRESHOLD = 10.0     # score >= this and the system stops moving
SETTLE_THRESHOLD = 6.0    # (currently unused — kept for future tuning)


# --- Gradient memory -------------------------------------------------------
# Used by the feedback logic to decide which way to step next.
last_score = None         # previous assessment's score (None on first run)
last_direction = 1        # +1 = nudging up in Hz, -1 = nudging down
boundary_hit = False      # set when we clamp at FREQ_MIN/MAX so the next
                          # step keeps the flipped direction


# --- Exploration state -----------------------------------------------------
# When the system stalls in one frequency band, force a big random jump
# so we don't get stuck in a shallow local maximum.
settled_count = 0         # consecutive assessments where freq barely moved
SETTLED_BAND = 3.0        # "barely moved" threshold in Hz
SETTLED_LIMIT = 8         # how many in a row before we trigger a jump
JUMP_NUDGE = 25.0         # size of that escape jump in Hz


# --- Frame buffer ----------------------------------------------------------
FRAME_BUFFER_MAX = 60     # rolling buffer size (frames on disk)
FRAME_SUBSAMPLE = 3       # take every Nth frame when building the GIF
                          # (60 frames / 3 = 20 frames -> ~2s GIF at 10fps)
GIF_FPS = 10              # playback rate of the generated GIF
frame_index = 0           # cyclic counter used for filenames


# --- Display ---------------------------------------------------------------
# We can't safely write to TouchDesigner ops from the assessment thread,
# so the thread stages text in `pending_display` and the main-thread
# run() callback flushes it into the readout DAT.
pending_display = None
last_display_text = 'initialising...'   # latest poetic line from the API
last_score_line = ''                    # debug/numeric line shown below it


# --- State machine ---------------------------------------------------------
# Valid states: ASSESSING, RESULTS, SHIFTING, BUFFERING
state = 'BUFFERING'
state_start_time = time.time()
RESULTS_DURATION = 10.0   # seconds the score/text stays on screen
SHIFT_DURATION = 2.0      # seconds the frequency ramp takes
BUFFER_WAIT = 2.0         # min seconds in BUFFERING before assessing again


# --- Assessment timing -----------------------------------------------------
last_assessment_time = 0
assessment_in_progress = False   # guard so we don't spawn parallel threads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_api(payload_str):
    """
    Send a single request to the Claude messages endpoint via curl.

    We shell out to curl rather than using `requests` or the official
    SDK because TouchDesigner's bundled Python often lacks both, and
    has unreliable SSL. curl is universally available on macOS.

    The JSON payload is written to a temp file and passed to curl with
    -d @<file> so we don't have to worry about shell-escaping the
    (often large, base64-laden) body.
    """
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
        # Successful responses contain a 'content' array; anything else
        # is an error envelope from the API (rate limit, bad key, etc.)
        if 'content' not in parsed:
            print(f'API error response: {parsed}')
        return parsed
    finally:
        # Always clean up the temp file even if curl/json blew up.
        os.remove(temp_path)


def ease_in_out(t):
    """
    Smooth 0->1 easing curve based on a half-cosine.

    Used to make the frequency ramp feel organic rather than linear:
    slow at the start, fastest in the middle, slow at the end.
    """
    return 0.5 - 0.5 * math.cos(math.pi * t)


def set_display(line1, line2=''):
    """
    Stage text to be written to the on-screen readout DAT.

    The actual write happens in run() on the main thread; this just
    stores the desired text so it's safe to call from any thread.
    """
    global pending_display
    if line2:
        pending_display = f'{line1}\n\n{line2}'
    else:
        pending_display = line1


# ---------------------------------------------------------------------------
# capture() — called every frame from a TouchDesigner Execute DAT
# ---------------------------------------------------------------------------

def capture():
    """
    Per-frame work. Two responsibilities:

    1. While in SHIFTING state, drive the frequency ramp at full frame
       rate so the audio output changes smoothly. This has to happen
       here (not in run()) because run() is called too infrequently
       for the ramp to look continuous.
    2. Otherwise, grab the latest video frame, downscale it, and write
       it to the rolling buffer on disk.
    """
    global frame_index, state, state_start_time, current_freq

    # --- Frequency ramp branch --------------------------------------------
    # During SHIFTING we don't capture frames — we just interpolate
    # current_freq from shift_start_freq toward target_freq using a
    # cosine ease, and push the new value into the audio op every
    # frame. When the ramp completes, we wipe the frame buffer (so the
    # post-shift assessment isn't contaminated by old frames at the
    # old frequency) and drop into BUFFERING.
    if state == 'SHIFTING':
        now = time.time()
        elapsed = now - state_start_time
        progress = min(elapsed / SHIFT_DURATION, 1.0)
        eased = ease_in_out(progress)
        interpolated = shift_start_freq + (target_freq - shift_start_freq) * eased

        # Only push to the audio op when the value actually changes by
        # something audibly meaningful — avoids spamming the param.
        if abs(interpolated - current_freq) > 0.01:
            current_freq = interpolated
            op('/project1/audio_out_1').par.frequency = current_freq

        if progress >= 1.0:
            # Snap exactly to target to avoid floating-point drift over
            # many ramps.
            current_freq = target_freq
            op('/project1/audio_out_1').par.frequency = current_freq
            print(f'Shift complete, now at {current_freq:.1f}Hz')

            # Clear the stale frame buffer so the next GIF only contains
            # frames captured at the new frequency.
            for f in os.listdir(FRAME_DIR):
                if f.endswith('.jpg'):
                    os.remove(os.path.join(FRAME_DIR, f))

            state = 'BUFFERING'
            state_start_time = now
        return

    # --- Normal frame capture branch --------------------------------------
    # Pull the latest frame from the live video TOP. numpyArray() returns
    # a float32 RGBA array in [0,1]; we throw away alpha, scale to 8-bit,
    # and convert to BGR for OpenCV. Then resize down — the API doesn't
    # need full-res frames and smaller files mean faster encodes/uploads.
    top = op('/project1/main_video_feed')
    buf = top.numpyArray()
    img = (buf[:, :, :3] * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_small = cv2.resize(img_bgr, (320, 180))

    # Cyclic filename so we overwrite old frames once frame_index wraps.
    frame_path = os.path.join(FRAME_DIR, f'frame_{frame_index:04d}.jpg')
    cv2.imwrite(frame_path, img_small, [cv2.IMWRITE_JPEG_QUALITY, 80])

    frame_index = (frame_index + 1) % FRAME_BUFFER_MAX

    # Belt-and-braces: if for some reason the directory has more than
    # FRAME_BUFFER_MAX files (e.g. stale frames from a previous run),
    # trim the oldest until we're back at the cap.
    all_frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
    while len(all_frames) > FRAME_BUFFER_MAX:
        os.remove(os.path.join(FRAME_DIR, all_frames.pop(0)))


# ---------------------------------------------------------------------------
# make_gif() — encode the frame buffer into a single GIF for the API
# ---------------------------------------------------------------------------

def make_gif():
    """
    Build a GIF from the current frame buffer and return it as a
    base64 string ready to drop into the Claude vision payload.

    The actual GIF encoding happens in a separate script (make_gif.py)
    invoked via subprocess. That keeps the heavyweight imageio /
    Pillow imports out of TouchDesigner's Python and makes it easy to
    swap encoders.

    Returns None if the buffer doesn't have enough frames yet, or if
    the encoder script fails.
    """
    import shutil

    # Subsample so we get ~2 seconds of motion rather than ~6, which
    # keeps the GIF small and the API call fast while still giving the
    # model enough frames to judge whether features are stable.
    all_frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
    subsampled = all_frames[::FRAME_SUBSAMPLE]

    if len(subsampled) < 3:
        # Not enough motion to assess — bail and let BUFFERING continue.
        return None

    # Stage the subsampled frames into a clean temp dir with sequential
    # names so the encoder doesn't have to deal with our cyclic naming.
    temp_dir = os.path.join(FRAME_DIR, 'gif_temp')
    os.makedirs(temp_dir, exist_ok=True)

    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))

    for i, fname in enumerate(subsampled):
        src = os.path.join(FRAME_DIR, fname)
        dst = os.path.join(temp_dir, f'frame_{i:04d}.jpg')
        shutil.copy(src, dst)

    # Hand off to the external encoder. python3 (system Python) is
    # used rather than TD's Python so imageio is available.
    result = subprocess.run(
        ['python3', GIF_SCRIPT, temp_dir, GIF_PATH, str(GIF_FPS)],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        print(f'GIF encoding error: {result.stderr}')
        return None

    print(result.stdout.strip())

    # Read the encoded GIF back off disk and base64-encode it for the
    # Claude image payload.
    with open(GIF_PATH, 'rb') as f:
        gif_bytes = f.read()

    return base64.b64encode(gif_bytes).decode('utf-8')


# ---------------------------------------------------------------------------
# run() — slow-tick state machine, called from a Timer/Execute DAT
# ---------------------------------------------------------------------------

def run():
    """
    Main-thread state machine driver.

    Responsibilities:
      - Flush any text staged by other threads into the readout DAT.
      - Decide whether to enter ASSESSING (if buffer is full and we've
        waited long enough).
      - Update the on-screen text appropriate to the current state.
      - Trigger the transition from RESULTS into SHIFTING after the
        result has been on screen long enough.
    """
    global state, state_start_time, current_freq, target_freq, shift_start_freq
    global pending_display, assessment_in_progress, last_assessment_time
    global last_display_text, last_score_line

    now = time.time()

    # Flush any text staged by set_display() (potentially from the
    # assessment thread) into the readout table on the main thread.
    if pending_display is not None:
        table = op('/project1/readout_data')
        table[0, 0] = pending_display
        pending_display = None

    # --- STATE: BUFFERING -------------------------------------------------
    # Wait until the rolling buffer is full AND a minimum dwell time
    # has passed, then start an assessment in a background thread.
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
            # Daemon thread so it dies with TD instead of blocking exit.
            thread = threading.Thread(target=assess_pattern)
            thread.daemon = True
            thread.start()

    # --- STATE: ASSESSING -------------------------------------------------
    # The thread is doing its thing; we just keep the placeholder text
    # on screen. The thread itself flips us into RESULTS when done.
    elif state == 'ASSESSING':
        set_display('assessing...')

    # --- STATE: RESULTS ---------------------------------------------------
    # The thread sets state_start_time = 0 as a sentinel meaning "start
    # the display timer from the moment run() first sees the new state".
    # This avoids race conditions where the thread's `now` is stale by
    # the time we get here.
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

    # --- STATE: SHIFTING --------------------------------------------------
    # The actual ramp is driven by capture() at frame rate; here we
    # just keep the readout updated with the current value.
    elif state == 'SHIFTING':
        set_display(f'shifting frequency...\n{current_freq:.1f} Hz → {target_freq:.1f} Hz')


# ---------------------------------------------------------------------------
# assess_pattern() — runs in a background thread
# ---------------------------------------------------------------------------

def assess_pattern():
    """
    The slow path: encode a GIF, send three calls to the Claude API
    (description -> score -> poetic summary), then run the feedback
    logic to pick the next target frequency.

    On any failure we fall back to BUFFERING and try again next cycle
    rather than getting stuck.
    """
    import random
    # Re-apply the SSL patch in this thread's import scope. ssl module
    # state is process-wide, but reasserting it here makes the
    # dependency explicit and survives any other code that might
    # reset the default context.
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    global target_freq, last_score, last_direction
    global settled_count, boundary_hit, assessment_in_progress
    global state, state_start_time, last_display_text, last_score_line

    try:
        # Encode the buffer. If we don't even have enough frames, drop
        # back to BUFFERING and let it refill.
        gif_b64 = make_gif()
        if gif_b64 is None:
            print('GIF encoding failed, skipping assessment')
            state = 'BUFFERING'
            state_start_time = time.time()
            return

        # ------------------------------------------------------------------
        # Call 1 — vision description
        # ------------------------------------------------------------------
        # We deliberately split observation from judgement: this call
        # is asked to *describe* what's visible (stability, geometry,
        # texture change) without interpreting it. Keeping this purely
        # observational makes the scoring in call 2 more consistent
        # because the model isn't second-guessing its own assessment
        # in the same prompt.
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

        # Strip any stray markdown headers the model might have added
        # despite our instructions.
        description = description_result['content'][0]['text'].strip()
        description = '\n'.join(line for line in description.split('\n') if not line.startswith('#')).strip()
        print(f'Description: {description}')

        # ------------------------------------------------------------------
        # Call 2 — score the description
        # ------------------------------------------------------------------
        # Text-only call. The model is given the description from
        # call 1 (NOT the image) and asked to rate resonance 0-10.
        # Going text-only here is intentional: it forces the score to
        # be derived from the textual observations, which makes the
        # whole pipeline auditable — we can read the description and
        # check whether the score is justified.
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

        # max_tokens is 10 but we still float() the whole reply in case
        # the model returns "7.5" or " 8 ". A ValueError here is caught
        # by the outer except.
        raw = rating_result['content'][0]['text'].strip()
        score = float(raw)
        print(f'Score: {score}')

        # ------------------------------------------------------------------
        # Call 3 — poetic display line
        # ------------------------------------------------------------------
        # Purely aesthetic. Generates the short on-screen line that
        # the audience reads. The system is framed as an entity
        # searching for order, with the score modulating its tone
        # (closer to / further from what it seeks). Two earlier prompt
        # iterations are kept commented out below for reference and
        # easy A/B testing.
        summary_payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 40,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",

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
            # If the poetic call fails, fall back to showing the raw
            # description rather than going blank.
            print('Summary call failed, using full description')
            display_text = description
        else:
            display_text = summary_result['content'][0]['text'].strip()
            display_text = '\n'.join(line for line in display_text.split('\n') if not line.startswith('#')).strip()
            print(f'Summary: {display_text}')

            # TEMP: comment out to restore poetic display
            # display_text = ''

        # ------------------------------------------------------------------
        # Feedback logic — pick the next target frequency
        # ------------------------------------------------------------------

        # Escape hatch: if we've been hovering in the same band for too
        # long, abandon the gradient and make a big random jump. This
        # prevents the system from camping at a mediocre local maximum
        # for the entire performance.
        if settled_count >= SETTLED_LIMIT:
            settled_count = 0
            last_score = None
            boundary_hit = False
            jump_direction = random.choice([-1, 1])
            new_freq = current_freq + (jump_direction * JUMP_NUDGE)
            # Clamp to range, then snap to the nearest 0.5 Hz so the
            # speaker is always on a "round" frequency.
            new_freq = float(round(max(FREQ_MIN, min(FREQ_MAX, new_freq)) * 2) / 2)
            score_line = f'Score: {score} | jump: {jump_direction * JUMP_NUDGE:+.1f}Hz | freq: {new_freq:.1f}Hz'
            print(f'Score: {score} -> exploration jump to {new_freq:.1f}Hz')
        else:
            # Choose direction to step in.
            if boundary_hit:
                # We hit FREQ_MIN/FREQ_MAX last time; last_direction
                # was already flipped at the boundary so just keep it.
                direction = last_direction
                boundary_hit = False
            elif last_score is None:
                # First assessment — no gradient info yet, use default.
                direction = last_direction
            elif score > last_score + 0.5:
                # Score went up: keep going the same way.
                direction = last_direction
            elif score < last_score - 0.5:
                # Score went down: reverse.
                direction = -last_direction
            else:
                # Within noise floor — preserve direction so we're not
                # jittering back and forth.
                direction = last_direction

            if score >= HOLD_THRESHOLD:
                # Good enough. Stop moving and let the audience see
                # this pattern for a while.
                nudge_size = 0.0
                score_line = f'Score: {score} | holding at {current_freq:.1f}Hz'
                print(f'Score: {score} -> holding at {current_freq:.1f}Hz')
            else:
                # Exponential curve: small nudges near threshold, large when cold
                # Normalise score to 0-1 range below hold threshold
                t = 1.0 - (score / HOLD_THRESHOLD)
                nudge_size = BASE_NUDGE + (MAX_NUDGE - BASE_NUDGE) * (t ** 2)

            # Remember this assessment for next time's gradient calc.
            last_score = score
            last_direction = direction

            if nudge_size > 0:
                new_freq = current_freq + (direction * nudge_size)

                # Clamp at boundaries and flip direction so we bounce
                # off the edge of the frequency range instead of
                # getting stuck pushing against it.
                if new_freq <= FREQ_MIN:
                    new_freq = FREQ_MIN
                    last_direction = 1
                    boundary_hit = True
                elif new_freq >= FREQ_MAX:
                    new_freq = FREQ_MAX
                    last_direction = -1
                    boundary_hit = True
                else:
                    # Snap to nearest 0.5 Hz.
                    new_freq = float(round(new_freq * 2) / 2)

                score_line = f'Score: {score} | nudge: {direction * nudge_size:+.1f}Hz | freq: {new_freq:.1f}Hz'
                print(f'Score: {score} -> nudge: {direction * nudge_size:+.1f}Hz -> new freq: {new_freq:.1f}Hz')
            else:
                # Holding — frequency stays where it is.
                new_freq = current_freq
                score_line = f'Score: {score} | holding at {current_freq:.1f}Hz'

        # Track how many assessments in a row have stayed in the same
        # band, so the SETTLED_LIMIT escape hatch above can fire when
        # we genuinely seem stuck.
        freq_delta = abs(current_freq - new_freq)
        if freq_delta < SETTLED_BAND:
            settled_count += 1
        else:
            settled_count = 0

        # Stage results for run() to consume on the main thread, then
        # transition into RESULTS. state_start_time = 0 is a sentinel
        # that tells run() to start its display timer the next time it
        # sees this state — see RESULTS handling in run().
        target_freq = new_freq
        last_display_text = display_text
        last_score_line = score_line
        state = 'RESULTS'
        state_start_time = 0  # signal run() to set timer on first entry

    # All failures route back to BUFFERING so the system keeps
    # trying. We log distinct exceptions so it's obvious from the
    # console what kind of failure occurred.
    except subprocess.TimeoutExpired:
        print('API call timed out')
        state = 'BUFFERING'
        state_start_time = time.time()
    except json.JSONDecodeError as e:
        print(f'Could not parse API response: {e}')
        state = 'BUFFERING'
        state_start_time = time.time()
    except ValueError as e:
        # Raised by float(raw) when the rating call returns something
        # non-numeric (e.g. an apology or refusal).
        print(f'Could not parse score: {e}')
        state = 'BUFFERING'
        state_start_time = time.time()
    except Exception as e:
        print(f'Error: {e}')
        state = 'BUFFERING'
        state_start_time = time.time()
    finally:
        # Always clear the in-progress flag so run() can start the
        # next assessment when conditions are right.
        assessment_in_progress = False

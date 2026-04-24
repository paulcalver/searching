# Searching...

A cymatics installation in which an AI watches vibrating water and searches aloud for beauty.

Final project, MA Computational Arts, Goldsmiths. By Paul Calver.

- Video: https://vimeo.com/1182923519
- Written documentation: `docs/Searching_Paul_Calver.pdf`
- Pipeline diagram: `docs/searching_system_pipeline.png`

## Requirements

- TouchDesigner 2023+
- Python packages in TD's bundled Python: `opencv-python`, `Pillow`
- Anthropic API key (model: `claude-haiku-4-5-20251001`)
- Sony A7R V + 90mm macro, LED ring light, 3" 4Ω driver, PAM8406 amp, 15cm perspex cube, petri dish (10–20% glycerine/water), MDF plinth

Install packages into TD's Python:

```bash
/Applications/TouchDesigner.app/Contents/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m pip install opencv-python Pillow
```

## Running it

1. Open `searching.toe`.
2. Add your Anthropic API key to `py/ai_assess.py` (line 14).
3. Update paths at the top of `py/ai_assess.py` to match your system.
4. Select the Sony A7R V in the Video Device In TOP.
5. Start the capture timer. The loop runs autonomously.

## Structure

```
├── searching.toe
├── td_screenshots/        Commented screenshots of the TD network
├── py/
│   ├── ai_assess.py       State machine, 3-call API pipeline, frequency control
│   └── make_gif.py        GIF encoder (subprocess to avoid TD SSL issues)
├── arduino/               HC-SR04 sketch, developed but not used in final piece
├── docs/                  Written documentation and pipeline diagram
└── README.md
```

## Credits

Blob tracking trialled early in development followed [Bileam Tschepe, TouchDesigner Tutorial 76](https://youtu.be/D5N1R5CVMkc). None of it remains in the final pipeline.

Anthropic's Claude (Opus 4.7) was used throughout development as a coding and writing assistant. All design decisions and final code were reviewed and edited by the author.

**The committed `ai_assess.py` has no API key. Add your own before running.**

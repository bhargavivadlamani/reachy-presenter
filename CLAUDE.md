# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # then add OPENAI_API_KEY
```

## Running

```bash
python -m app.main <path-to-file.pdf|pptx>
```

No test suite exists. No build step required.

## Architecture

**reachy-presenter** animates a Reachy Mini robot to deliver presentations from PDF/PPTX files.

Data flow:
1. `app/main.py` parses the file via `app/parsers/` (pdfplumber for PDF, python-pptx for PPTX)
2. All slides are pre-processed in bulk: GPT-4o generates a spoken script and classifies emotional tone (excited/neutral/questioning/serious) for each slide
3. The robot connects, then for each slide (user-paced via Enter):
   - `robot/gestures.py` executes a slide-transition movement, then an emotion-based gesture
   - `robot/tts.py` runs two threads in parallel: OpenAI TTS audio playback (24kHz→robot sample rate) and MediaPipe face tracking at 4Hz (`robot/vision.py`)
   - Face tracking looks at detected faces or scans left/right if none found

Key points:
- Pre-computation (LLM calls) happens before robot connection so the robot isn't idle
- Audio and face-tracking threads are synchronized to audio duration
- `load_data()` in `main.py` uses lazy imports based on file extension

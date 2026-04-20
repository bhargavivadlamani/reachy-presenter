# Reachy Audience — Project Context

## What This Project Does
Reachy Mini audience robot powered by Google ADK + Gemini Bidi streaming. It:
1. Watches for visitors via camera face detection (YOLO)
2. Runs a persistent Gemini Live (Bidi) audio session — hears and reacts to the presenter robot naturally
3. Looks toward the presenter robot when a visitor is present; scans otherwise
4. Receives slide events from the presenter robot (HTTP POST /slide) and asks questions between slides

Run: `python -m app.main_audience`

## Project Layout
```
app/
  main_audience.py          # Entry point — wires everything together
  audio_helpers/
    __init__.py             # MIC_SR=16000, GEM_SR=24000, to_pcm_bytes(), decode_pcm()
  audience/
    agent.py                # AudienceAgent — slide reactions + TTS questions via Gemini TTS
    bidi_conversation.py    # BidiConversationSession — persistent Gemini Live audio (natural convo)
    presence_detector.py    # PresenceDetector — wraps ScanBehavior, fires on_arrived/on_left
    server.py               # HTTP server — receives /slide events from presenter robot
  robot/
    attention_classifier.py # AttentionClassifier — DOA + gaze + VAD → SILENT/TO_HUMAN/TO_COMPUTER
    attentive_listener.py   # AttentiveListener — gaze control: Robot1 when visitor, DOA when speaking
    doa.py                  # DOAEstimator — Direction of Arrival + VAD from stereo mic
    gaze.py                 # GazeEstimator — face gaze direction from camera frame
    gestures.py             # emotion_gesture() — excited/questioning/serious/neutral
    head_tracker.py         # HeadTracker — YOLO face detection (YOLOv11n)
    scan_behavior.py        # ScanBehavior — scan left/center/right until face found
```

## Environment (.env)
```
GOOGLE_API_KEY=...          # Gemini API key
PRESENTER_ROBOT_URL=http://<presenter-ip>:8000
AUDIENCE_ROBOT_URL=http://<this-robot-ip>:5001
ROBOT1_YAW_DEG=-45          # Yaw to presenter robot (negative = left)
```

## Architecture

### Startup
`main_audience.py` boots everything in order:
1. `emotion_gesture(excited)` — startup wiggle
2. `AudienceAgent.start()` — initialises TTS pipeline for slide questions
3. `AttentionClassifier.start()` — DOA + VAD running
4. `PresenceDetector` — owns `ScanBehavior` internally
5. `AttentiveListener.start()` — gaze control thread
6. `BidiConversationSession.start(mini)` — Gemini Live session opens
7. `AudienceServer.start()` — HTTP /slide server on port 5001

### Conversation Flow
```
Visitor arrives → set_visitor_present(True) → scanner pauses, look at Robot1
               → greet_visitor() → injects prompt into live Bidi session
               → Bidi session hears presenter robot + visitor via mic
               → Gemini generates natural responses → push_audio_sample()

Visitor leaves → set_visitor_present(False) — conversation keeps running

Slide event → mute_output() → AudienceAgent speaks TTS question → unmute_output()
```

### Head Gaze Logic (AttentiveListener)
- `_speaking=True`        → look at ROBOT1_YAW_DEG (Robot 1 direction)
- `rms >= threshold`      → look toward DOA angle (whoever is speaking)
- visitor present + silent → nudge toward Robot 1 every 3s, scanner stays paused
- no visitor + silent     → resume ScanBehavior (scan left/center/right)

### BidiConversationSession
- Starts once at boot, auto-restarts on session close (Gemini has ~15min limit)
- `greet_visitor()` — injects greeting text into live_queue without restarting
- `mute_output()` / `unmute_output()` — suppress TTS during slide playback
- `shutdown()` — permanent stop on app exit

## Common Issues
- **Robot 1 yaw wrong**: adjust `ROBOT1_YAW_DEG` in `.env` (negative = left, positive = right)
- **DOA device fails at boot**: normal — falls back to device None, VAD still works
- **Bidi session silent**: `GOOGLE_API_KEY` not set, or model name mismatch
- **Head jitter**: `_HEAD_DURATION` too short vs `_POLL` interval — currently 0.6s/0.3s

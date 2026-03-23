"""Face tracking for Reachy Mini presenter."""

import threading

import numpy as np

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


class FaceTracker:
    """Detects faces using MediaPipe BlazeFace."""

    def __init__(self):
        if _MEDIAPIPE_AVAILABLE:
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,          # 0 = short range; full-range (1) has protobuf issue on aarch64
                min_detection_confidence=0.5,
            )
        else:
            self._detector = None

    def get_face_pixels(self, frame_bgr):
        """Return (cx, cy) pixel center of first detected face, or None."""
        if self._detector is None:
            return None
        frame_rgb = frame_bgr[:, :, ::-1].copy()   # BGR → RGB, must be C-contiguous for MediaPipe
        results = self._detector.process(frame_rgb)
        if not results.detections:
            return None
        h, w = frame_bgr.shape[:2]
        bbox = results.detections[0].location_data.relative_bounding_box
        cx = int((bbox.xmin + bbox.width / 2) * w)
        cy = int((bbox.ymin + bbox.height / 2) * h)
        return cx, cy

    def close(self):
        if self._detector is not None:
            self._detector.close()


def track_faces_during_speech(mini, stop_event: threading.Event) -> None:
    """Track audience faces and move head for natural eye contact.

    Runs at 4 Hz until stop_event is set — called in a thread alongside
    the Gemini Live audio session.
    """
    from reachy_mini.utils import create_head_pose

    tracker = FaceTracker()
    scan_t = 0.0

    try:
        while not stop_event.is_set():
            frame = mini.media.get_frame()
            if frame is not None:
                face = tracker.get_face_pixels(frame)
                if face is not None:
                    # Look directly at the detected face
                    mini.look_at_image(face[0], face[1], duration=0.25)
                else:
                    # No face found — slow audience scan left/right
                    scan_t += 0.25
                    yaw = np.deg2rad(25 * np.sin(2 * np.pi * 0.05 * scan_t))
                    mini.set_target(head=create_head_pose(yaw=yaw))
            stop_event.wait(timeout=0.25)  # 4 Hz, exits early when stopped
    finally:
        tracker.close()

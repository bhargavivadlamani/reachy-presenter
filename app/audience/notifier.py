"""Notifies the audience robot when a slide has been presented.

Set AUDIENCE_ROBOT_URL in .env to enable, e.g.:
    AUDIENCE_ROBOT_URL=http://192.168.1.42:5001

If the env var is not set, notifications are silently skipped.
"""

import json
import logging
import os
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


def notify_slide_presented(slide_number: int, script: str) -> None:
    """Fire-and-forget POST to the audience robot. Never raises."""
    url = os.environ.get("AUDIENCE_ROBOT_URL", "").rstrip("/")
    if not url:
        return
    endpoint = f"{url}/slide"
    payload = json.dumps({"slide_number": slide_number, "script": script}).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            logger.debug("[notifier] audience robot responded %d", resp.status)
    except urllib.error.URLError as e:
        logger.warning("[notifier] could not reach audience robot at %s: %s", endpoint, e)
    except Exception as e:
        logger.warning("[notifier] unexpected error notifying audience robot: %s", e)

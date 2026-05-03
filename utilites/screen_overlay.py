from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class OverlayStyle:
    line_width: int = 3


def show_bbox_overlay(
    *,
    bboxes_px: list[tuple[int, int, int, int]],
    screen_width: int,
    screen_height: int,
    duration_s: float = 2.5,
    style: OverlayStyle | None = None,
) -> None:
    """
    Draw bounding boxes on top of the *live* screen using an OpenCV fullscreen window.

    - `bboxes_px`: list of (x1, y1, x2, y2)
    - closes automatically after `duration_s` seconds or when user presses Esc.
    """
    import time
    import cv2
    import numpy as np

    st = style or OverlayStyle()

    def _clamp(n: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(n)))

    frame = np.zeros((int(screen_height), int(screen_width), 3), dtype=np.uint8)

    # Always red (BGR in OpenCV).
    color = (0, 0, 255)

    for (x1, y1, x2, y2) in bboxes_px:
        x1c = _clamp(x1, 0, screen_width - 1)
        y1c = _clamp(y1, 0, screen_height - 1)
        x2c = _clamp(x2, 0, screen_width - 1)
        y2c = _clamp(y2, 0, screen_height - 1)
        if x2c <= x1c or y2c <= y1c:
            continue
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, int(st.line_width))

    window_name = "AI-Agent Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Best-effort: make it fullscreen + topmost.
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    t0 = time.time()
    while True:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):  # Esc or q
            break
        if (time.time() - t0) >= float(duration_s):
            break

    try:
        cv2.destroyWindow(window_name)
    except Exception:
        cv2.destroyAllWindows()


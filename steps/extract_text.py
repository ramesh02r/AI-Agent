from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from utilites.call_to_llm import VisionLLM  # noqa: E402


def _b64_png_from_screenshot(pil_img) -> str:
    """
    Convert a PIL screenshot to base64 PNG without touching disk.
    This is faster and avoids filesystem permission surprises.
    """
    import io

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _norm_item(s: str) -> str:
    # Normalize for de-duping across minor whitespace/punctuation differences.
    # Keep it conservative so we don't over-collapse distinct items.
    t = (s or "").strip()
    t = " ".join(t.split())
    return t.lower()


def _safe_int(x: Any, default: int) -> int:
    try:
        if isinstance(x, bool):
            return default
        return int(float(x))
    except Exception:
        return default


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


@dataclass
class _Decision:
    status: str
    action: str | None
    pixels: int
    extracted_data: list[str]
    reason: str


def _parse_decision(obj: dict[str, Any] | None) -> _Decision | None:
    if not isinstance(obj, dict):
        return None

    status = obj.get("status")
    action = obj.get("action")
    pixels = obj.get("pixels")
    extracted = obj.get("extracted_data")
    reason = obj.get("reason")

    if status not in {"continue", "success", "fail"}:
        return None

    if action is not None and action not in {"scroll_down", "scroll_up", "wait"}:
        return None

    if not isinstance(extracted, list) or any(not isinstance(x, str) for x in extracted):
        return None

    if not isinstance(reason, str):
        reason = str(reason or "")

    px = _safe_int(pixels, 0)
    if action in {"scroll_down", "scroll_up"}:
        # Don't hardcode large clamps here; final scroll bounds depend on viewport height.
        # We only sanitize to a non-negative integer; run() will clamp proportionally.
        px = abs(int(px))
    elif action == "wait":
        px = 0
    else:
        px = 0

    return _Decision(status=status, action=action, pixels=px, extracted_data=extracted, reason=reason)


def _screenshot_thumb_gray(pil_img):
    """
    Downscale screenshot and convert to grayscale numpy array for fast comparisons.
    Used to detect end-of-scroll / no-change loops.
    """
    import numpy as np
    import cv2

    try:
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    target_w = 320
    scale = target_w / float(w)
    target_h = max(1, int(h * scale))
    small = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _screens_equalish(prev_gray_small, curr_gray_small, mean_absdiff_threshold: float) -> bool:
    import cv2

    if prev_gray_small is None or curr_gray_small is None:
        return False
    if prev_gray_small.shape != curr_gray_small.shape:
        return False
    diff = cv2.absdiff(prev_gray_small, curr_gray_small)
    mean_diff = float(diff.mean())
    return mean_diff <= float(mean_absdiff_threshold)


class ExtractTextStep:
    """
    LLM-driven, human-like visual extraction loop:
    screenshot -> LLM decides (extract vs scroll/wait) -> act via pyautogui -> repeat.

    Constraints:
    - No DOM access
    - No Selenium
    - No API usage
    - Screenshot-only understanding + pyautogui actions
    """

    def __init__(
        self,
        *,
        user_task: str,
        target_count: int | None,
        out_dir: Path,
        max_iters: int = 40,
        post_action_sleep_s: float = 0.7,
        max_wait_s: float = 8.0,
        stop_when_same_screenshot: bool = True,
        same_screenshot_mean_diff_threshold: float = 1.25,
        llm_retries: int = 2,
        pixels_per_scroll_click: int = 30,
    ) -> None:
        self.user_task = user_task
        self.target_count = target_count
        self.out_dir = out_dir
        self.max_iters = max_iters
        self.post_action_sleep_s = post_action_sleep_s
        self.max_wait_s = max_wait_s
        self.stop_when_same_screenshot = stop_when_same_screenshot
        self.same_screenshot_mean_diff_threshold = same_screenshot_mean_diff_threshold
        self.llm_retries = llm_retries
        self.pixels_per_scroll_click = max(5, int(pixels_per_scroll_click))
        self.llm = VisionLLM()

        self.extracted: list[str] = []
        self._extracted_norm: set[str] = set()

    def _remaining_target(self) -> int | None:
        if self.target_count is None:
            return None
        return max(0, int(self.target_count) - len(self.extracted))

    def _add_extracted(self, items: list[str]) -> int:
        added = 0
        for it in items:
            t = (it or "").strip()
            if not t:
                continue
            k = _norm_item(t)
            if not k or k in self._extracted_norm:
                continue
            self._extracted_norm.add(k)
            self.extracted.append(t)
            added += 1
        return added

    def _execute_action(self, action: str | None, scroll_clicks: int) -> None:
        if pyautogui is None:  # pragma: no cover
            raise RuntimeError("pyautogui is not installed. Install dependencies from requirements.txt to run this step.")
        if action == "scroll_down":
            # pyautogui: negative scroll value scrolls *down* (content moves up).
            pyautogui.scroll(-scroll_clicks)
        elif action == "scroll_up":
            pyautogui.scroll(scroll_clicks)
        elif action == "wait":
            time.sleep(min(self.max_wait_s, max(0.2, float(self.post_action_sleep_s))))
        else:
            return

    def run(self) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        prev_thumb = None
        same_screen_streak = 0
        no_progress_streak = 0

        base = Path(__file__).stem

        for i in range(1, self.max_iters + 1):
            screenshot_path = self.out_dir / f"{base}_input.png"
            llm_raw_path = self.out_dir / f"{base}_llm_raw.txt"
            llm_json_path = self.out_dir / f"{base}_output.json"
            context_path = self.out_dir / f"{base}_context.json"

            if pyautogui is None:  # pragma: no cover
                raise RuntimeError("pyautogui is not installed. Install dependencies from requirements.txt to run this step.")

            img = pyautogui.screenshot()
            width, height = img.size
            img.save(screenshot_path)

            if self.stop_when_same_screenshot:
                curr_thumb = _screenshot_thumb_gray(img)
                if _screens_equalish(
                    prev_thumb,
                    curr_thumb,
                    mean_absdiff_threshold=self.same_screenshot_mean_diff_threshold,
                ):
                    same_screen_streak += 1
                else:
                    same_screen_streak = 0
                prev_thumb = curr_thumb

            remaining = self._remaining_target()
            context_path.write_text(
                json.dumps(
                    {
                        "user_task": self.user_task,
                        "target_count": self.target_count,
                        "already_extracted": self.extracted,
                        "remaining_target": remaining,
                        "iteration": i,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            image_b64 = _b64_png_from_screenshot(img)

            decision: _Decision | None = None
            last_raw = ""
            last_json: dict[str, Any] | None = None
            last_err: str | None = None
            for attempt in range(self.llm_retries + 1):
                try:
                    llm_resp = self.llm.extract_text_step_from_base64(
                        image_b64=image_b64,
                        user_task=self.user_task,
                        already_extracted=self.extracted,
                        remaining_target=remaining,
                        image_width_px=int(width),
                        image_height_px=int(height),
                        mime_type="image/png",
                    )
                    last_raw = llm_resp.raw_text or ""
                    last_json = llm_resp.json
                    decision = _parse_decision(last_json)
                    if decision is not None:
                        break
                    last_err = "LLM JSON missing/invalid schema"
                except Exception as e:
                    last_err = str(e)

                # Small backoff before retrying the same screenshot.
                time.sleep(0.35)

            llm_raw_path.write_text(last_raw, encoding="utf-8")
            llm_json_path.write_text(json.dumps(last_json or {}, ensure_ascii=False, indent=2), encoding="utf-8")

            if decision is None:
                return {
                    "status": "fail",
                    "reason": f"LLM decision invalid after retries: {last_err}",
                    "extracted_data": self.extracted,
                    "out_dir": str(self.out_dir),
                }

            added = self._add_extracted(decision.extracted_data)
            if added > 0:
                no_progress_streak = 0
            else:
                no_progress_streak += 1

            remaining = self._remaining_target()
            if remaining == 0:
                return {
                    "status": "success",
                    "reason": "Target count reached.",
                    "extracted_data": self.extracted,
                    "out_dir": str(self.out_dir),
                }

            if decision.status == "success":
                # If user didn't request a count, success means "task satisfied".
                # If they did request a count but we didn't reach it, treat as continue.
                if self.target_count is None:
                    return {
                        "status": "success",
                        "reason": decision.reason or "LLM indicated success.",
                        "extracted_data": self.extracted,
                        "out_dir": str(self.out_dir),
                    }

            if decision.status == "fail":
                return {
                    "status": "fail",
                    "reason": decision.reason or "LLM indicated failure.",
                    "extracted_data": self.extracted,
                    "out_dir": str(self.out_dir),
                }

            # Loop safety: if we keep seeing the same screen and aren't extracting anything,
            # we are likely stuck at end-of-page or on a loading state.
            if self.stop_when_same_screenshot and same_screen_streak >= 2 and no_progress_streak >= 2:
                return {
                    "status": "fail",
                    "reason": "Stuck: screen not changing after actions and no new items extracted.",
                    "extracted_data": self.extracted,
                    "out_dir": str(self.out_dir),
                }

            if decision.action is None:
                # LLM should only return continue with an action or success.
                # If it didn't, we fail fast to avoid random / undefined behavior.
                return {
                    "status": "fail",
                    "reason": "LLM returned status=continue but action=null; refusing to take random actions.",
                    "extracted_data": self.extracted,
                    "out_dir": str(self.out_dir),
                }
            px_to_use = decision.pixels
            if decision.action in {"scroll_down", "scroll_up"}:
                # Use a consistent "human" scroll amount: 45% of viewport height.
                # This avoids skipping content due to overly large model-chosen values.
                px_to_use = _clamp(int(height * 0.45), 80, 1600)

            # Convert our "pixel-like" intent into pyautogui scroll wheel clicks.
            scroll_clicks = max(1, int(round(px_to_use / float(self.pixels_per_scroll_click))))

            print(f"Executing action: {decision.action} with scroll_clicks: {scroll_clicks} (intended_px≈{px_to_use})")
            print(f"Image height: {height}, image width: {width}")
            self._execute_action(decision.action, int(scroll_clicks))
            time.sleep(float(self.post_action_sleep_s))

        return {
            "status": "fail",
            "reason": f"Max iterations ({self.max_iters}) reached without satisfying the task.",
            "extracted_data": self.extracted,
            "out_dir": str(self.out_dir),
        }


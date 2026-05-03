from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image, ImageDraw
from pathlib import Path

try:
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None  # type: ignore[assignment]



DEFAULT_OMNIPARSER_URL = "http://34.59.239.39:8000/parse/"


@dataclass(frozen=True)
class MatchResult:
    element: dict[str, Any]
    score: float


class ScreenParser:
    """
    Minimal wrapper around an OmniParser-style API that accepts:
      POST { "base64_image": "..." }
    and returns JSON with `parsed_content_list` where each item may include:
      - bbox: [x1, y1, x2, y2] normalized (0..1)
      - content: string
      - type: string
      - interactivity: bool
    """

    def __init__(self, omniparser_url: str = DEFAULT_OMNIPARSER_URL, timeout_s: int = 30) -> None:
        self.omniparser_url = omniparser_url
        self.timeout_s = timeout_s

    @staticmethod
    def encode_image_file_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call_omniparser_api(self, base64_image: str) -> dict[str, Any]:
        response = requests.post(
            self.omniparser_url,
            json={"base64_image": base64_image},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def parse_image_file(self, image_path: str, annotated_out_path: str | None = None) -> dict[str, Any]:
        base64_image = self.encode_image_file_to_base64(image_path)
        response = self.call_omniparser_api(base64_image)
        if annotated_out_path:
            self.save_annotated_image(image_path, response, annotated_out_path)
        return response

    def save_annotated_image(
        self,
        image_path: str,
        omniparser_result: dict[str, Any],
        out_path: str,
    ) -> None:
        """
        Draw OmniParser `parsed_content_list` bounding boxes on the image and save.

        - `bbox` is expected to be normalized [x1,y1,x2,y2] in 0..1.
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        elements = omniparser_result.get("parsed_content_list", []) or []

        color_map = {
            "button": "red",
            "text": "blue",
            "input": "green",
            "link": "orange",
            "image": "purple",
            "icon": "pink",
            "checkbox": "cyan",
            "radio": "magenta",
            "select": "yellow",
        }

        for el in elements:
            bbox = el.get("bbox", [])
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            try:
                x1n, y1n, x2n, y2n = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except Exception:
                continue

            x1, y1 = int(x1n * w), int(y1n * h)
            x2, y2 = int(x2n * w), int(y2n * h)

            el_type = str(el.get("type", "unknown")).lower()
            color = color_map.get(el_type, "gray")

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = str(el.get("type", "unknown"))
            content = el.get("content")
            if content:
                content = str(content)
                preview = content[:20] + "..." if len(content) > 20 else content
                label += f": {preview}"

            # Simple label background
            try:
                tb = draw.textbbox((x1, max(0, y1 - 15)), label)
                draw.rectangle(tb, fill=color)
                draw.text((x1, max(0, y1 - 15)), label, fill="white")
            except Exception:
                # textbbox may not exist on very old Pillow; ignore labels in that case
                pass

        img.save(out_path)

    def show_overlay_on_screen(
        self,
        *,
        image_path: str,
        omniparser_result: dict[str, Any],
        duration_s: float = 2.5,
    ) -> None:
        """
        Show a temporary on-screen overlay (transparent topmost window) with all bboxes.

        Notes:
        - Works best when `image_path` is a *full-screen* screenshot from the same display setup.
        - If your screenshot dimensions differ from current screen resolution, boxes are scaled.
        """
        if pyautogui is None:  # pragma: no cover
            raise RuntimeError("pyautogui is required for on-screen overlay. Install dependencies from requirements.txt.")

        try:
            screen_w, screen_h = pyautogui.size()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Could not read screen size via pyautogui: {e}")

        img = Image.open(image_path)
        img_w, img_h = img.size
        if img_w <= 0 or img_h <= 0 or screen_w <= 0 or screen_h <= 0:
            return

        sx = float(screen_w) / float(img_w)
        sy = float(screen_h) / float(img_h)

        elements = omniparser_result.get("parsed_content_list", []) or []
        bboxes_px: list[list[int]] = []

        for el in elements:
            bbox = el.get("bbox", [])
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            try:
                x1n, y1n, x2n, y2n = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except Exception:
                continue

            # bbox is normalized in 0..1 relative to the screenshot.
            x1_img, y1_img = int(x1n * img_w), int(y1n * img_h)
            x2_img, y2_img = int(x2n * img_w), int(y2n * img_h)
            x1 = int(round(x1_img * sx))
            y1 = int(round(y1_img * sy))
            x2 = int(round(x2_img * sx))
            y2 = int(round(y2_img * sy))
            bboxes_px.append([x1, y1, x2, y2])

        if not bboxes_px:
            return

    @staticmethod
    def element_center_px(element: dict[str, Any], width: int, height: int) -> tuple[int, int]:
        bbox = element.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValueError("Element does not contain bbox=[x1,y1,x2,y2].")

        x1, y1, x2, y2 = bbox
        cx_n = (float(x1) + float(x2)) / 2.0
        cy_n = (float(y1) + float(y2)) / 2.0
        return (int(cx_n * width), int(cy_n * height))

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @classmethod
    def find_best_match(cls, elements: list[dict[str, Any]], query: str) -> MatchResult | None:
        q = cls._normalize_text(query)
        if not q:
            return None

        q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if t]
        best: MatchResult | None = None

        for el in elements:
            content = el.get("content") or ""
            text = cls._normalize_text(str(content))
            if not text:
                continue

            score = 0.0

            # Strong signal: substring match
            if q in text:
                score += 5.0

            # Token overlap
            for t in q_tokens:
                if t and t in text:
                    score += 1.0

            # Slight preference for interactive targets if tie
            if el.get("interactivity"):
                score += 0.1

            if score <= 0:
                continue

            if best is None or score > best.score:
                best = MatchResult(element=el, score=score)

        return best
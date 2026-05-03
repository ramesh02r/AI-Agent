from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import pyautogui

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from utilites.screen_parser import ScreenParser  # noqa: E402
from utilites.call_to_llm import VisionLLM  # noqa: E402


def _norm(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def _bbox_norm_to_px(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    try:
        x1n, y1n, x2n, y2n = map(float, bbox)
    except Exception:
        return None
    x1 = max(0, min(width - 1, int(x1n * width)))
    y1 = max(0, min(height - 1, int(y1n * height)))
    x2 = max(0, min(width - 1, int(x2n * width)))
    y2 = max(0, min(height - 1, int(y2n * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _draw_ids_overlay(
    *,
    screenshot_path: Path,
    out_path: Path,
    elements_with_ids: list[dict],
    width: int,
    height: int,
) -> bool:
    bgr = cv2.imread(str(screenshot_path))
    if bgr is None:
        return False

    for el in elements_with_ids:
        el_id = el.get("id")
        bbox = el.get("bbox")
        if not isinstance(el_id, int):
            continue
        if not isinstance(bbox, list):
            continue
        px = _bbox_norm_to_px(bbox, width, height)
        if not px:
            continue
        x1, y1, x2, y2 = px

        # Draw a subtle box and a high-contrast ID tag.
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (40, 200, 255), 2)

        label = str(el_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pad = 4
        tx1 = x1
        ty1 = max(0, y1 - th - pad * 2)
        tx2 = min(width - 1, x1 + tw + pad * 2)
        ty2 = min(height - 1, ty1 + th + pad * 2)
        cv2.rectangle(bgr, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
        cv2.putText(
            bgr,
            label,
            (tx1 + pad, ty2 - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), bgr)
    return True


def _cluster_rows(elements: list[dict], height: int, row_tol_px: int = 14) -> list[list[dict]]:
    """Group OCR elements into rows by y-center proximity."""
    items: list[tuple[int, dict]] = []
    for el in elements:
        bbox = el.get("bbox")
        content = el.get("content")
        if not content:
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            _, y1n, _, y2n = map(float, bbox)
        except Exception:
            continue
        cy = int(((y1n + y2n) / 2.0) * height)
        items.append((cy, el))

    items.sort(key=lambda t: t[0])
    rows: list[list[dict]] = []
    row_centers: list[int] = []

    def x_center(el: dict) -> float:
        bbox2 = el.get("bbox") or [0, 0, 0, 0]
        try:
            x1n, _, x2n, _ = map(float, bbox2)
        except Exception:
            return 0.0
        return (x1n + x2n) / 2.0

    for cy, el in items:
        placed = False
        for i, rcy in enumerate(row_centers):
            if abs(cy - rcy) <= row_tol_px:
                rows[i].append(el)
                row_centers[i] = int((rcy + cy) / 2)
                placed = True
                break
        if not placed:
            rows.append([el])
            row_centers.append(cy)

    for row in rows:
        row.sort(key=x_center)

    return rows


def _row_bbox_px(row: list[dict], width: int, height: int) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for el in row:
        bbox = el.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1n, y1n, x2n, y2n = map(float, bbox)
        xs.extend([int(x1n * width), int(x2n * width)])
        ys.extend([int(y1n * height), int(y2n * height)])
    if not xs or not ys:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def _cluster_ids_into_rows(
    *,
    ids: list[int],
    elements_with_ids: list[dict],
    width: int,
    height: int,
    row_tol_px: int,
) -> list[list[int]]:
    id_set = set(ids)
    picked: list[dict] = []
    for el in elements_with_ids:
        el_id = el.get("id")
        if isinstance(el_id, int) and el_id in id_set:
            picked.append(el)

    def cy(el: dict) -> int:
        px = _bbox_norm_to_px(el.get("bbox") or [], width, height)
        if not px:
            return 0
        _, y1, _, y2 = px
        return int((y1 + y2) / 2)

    def cx(el: dict) -> int:
        px = _bbox_norm_to_px(el.get("bbox") or [], width, height)
        if not px:
            return 0
        x1, _, x2, _ = px
        return int((x1 + x2) / 2)

    picked.sort(key=lambda e: cy(e))
    rows: list[list[dict]] = []
    row_centers: list[int] = []
    for el in picked:
        y = cy(el)
        placed = False
        for i, rcy in enumerate(row_centers):
            if abs(y - rcy) <= row_tol_px:
                rows[i].append(el)
                row_centers[i] = int((rcy + y) / 2)
                placed = True
                break
        if not placed:
            rows.append([el])
            row_centers.append(y)

    rows_ids: list[list[int]] = []
    for r in rows:
        r.sort(key=lambda e: cx(e))
        rows_ids.append([int(e["id"]) for e in r if isinstance(e.get("id"), int)])
    return rows_ids


def _screenshot_thumb_gray(img) -> "cv2.Mat | None":
    """
    Convert a PIL screenshot to a small grayscale OpenCV image.
    Using a thumbnail makes comparisons fast and robust to minor noise.
    """
    try:
        rgb = cv2.cvtColor(__import__("numpy").array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return None
    target_w = 320
    scale = target_w / float(w)
    target_h = max(1, int(h * scale))
    small = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _screens_equalish(prev_gray_small, curr_gray_small, mean_absdiff_threshold: float) -> bool:
    """
    Return True if two screenshots are effectively identical.
    We compare mean absolute pixel difference on a downscaled grayscale view.
    """
    if prev_gray_small is None or curr_gray_small is None:
        return False
    if prev_gray_small.shape != curr_gray_small.shape:
        return False
    diff = cv2.absdiff(prev_gray_small, curr_gray_small)
    mean_diff = float(diff.mean())
    return mean_diff <= float(mean_absdiff_threshold)


class ScrollFindRowsStep:
    def __init__(
        self,
        value_query: str,
        out_dir: Path,
        omniparser_url: str | None = None,
        max_scrolls: int = 15,
        scroll_amount: int = 700,
        row_tol_px: int = 14,
        stop_when_same_screenshot: bool = True,
        same_screenshot_mean_diff_threshold: float = 1.25,
    ) -> None:
        self.value_query = value_query
        self.out_dir = out_dir
        self.max_scrolls = max_scrolls
        self.scroll_amount = scroll_amount
        self.row_tol_px = row_tol_px
        self.stop_when_same_screenshot = stop_when_same_screenshot
        self.same_screenshot_mean_diff_threshold = same_screenshot_mean_diff_threshold
        self.parser = ScreenParser(omniparser_url=omniparser_url) if omniparser_url else ScreenParser()
        self.llm = VisionLLM()

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        q = _norm(self.value_query)
        prev_thumb = None

        base = Path(__file__).stem

        for i in range(self.max_scrolls + 1):
            screenshot_path = self.out_dir / f"{base}_input.png"
            parsed_path = self.out_dir / f"{base}_parsed.json"
            annotated_path = self.out_dir / f"{base}_output.png"
            ids_annotated_path = self.out_dir / f"{base}_ids.png"

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
                    print("Reached end-of-scroll (two consecutive screenshots are the same).")
                    break
                prev_thumb = curr_thumb

            result = self.parser.parse_image_file(str(screenshot_path))
            elements = result.get("parsed_content_list", []) or []

            # Assign stable per-screenshot IDs and persist them in JSON.
            elements_with_ids: list[dict] = []
            for idx, el in enumerate(elements, start=1):
                if isinstance(el, dict):
                    el2 = dict(el)
                    el2["id"] = idx
                    elements_with_ids.append(el2)
            parsed_path.write_text(json.dumps(elements_with_ids, ensure_ascii=False, indent=2), encoding="utf-8")

            # Create an ID-annotated image so the LLM can return IDs without seeing the JSON list.
            _draw_ids_overlay(
                screenshot_path=screenshot_path,
                out_path=ids_annotated_path,
                elements_with_ids=elements_with_ids,
                width=width,
                height=height,
            )

            with open(screenshot_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            with open(ids_annotated_path, "rb") as f:
                ids_image_b64 = base64.b64encode(f.read()).decode("utf-8")

            llm_resp = self.llm.match_ids_from_two_images_base64(
                normal_image_b64=image_b64,
                ids_annotated_image_b64=ids_image_b64,
                user_description=self.value_query,
                mime_type="image/png",
            )

            llm_path = self.out_dir / f"{base}_llm.json"
            llm_path.write_text(llm_resp.raw_text or "", encoding="utf-8")

            rows_obj = (llm_resp.json or {}).get("rows") if llm_resp.json else None
            ids_obj = (llm_resp.json or {}).get("ids") if llm_resp.json else None  # backward/robustness

            rows_ids: list[list[int]] = []
            if isinstance(rows_obj, list) and rows_obj:
                for row in rows_obj:
                    if not isinstance(row, list):
                        continue
                    row_ids = [
                        int(x)
                        for x in row
                        if isinstance(x, (int, float, str)) and str(x).strip().isdigit()
                    ]
                    if row_ids:
                        rows_ids.append(row_ids)
            elif isinstance(ids_obj, list) and ids_obj:
                flat_ids = [
                    int(x)
                    for x in ids_obj
                    if isinstance(x, (int, float, str)) and str(x).strip().isdigit()
                ]
                if flat_ids:
                    rows_ids = _cluster_ids_into_rows(
                        ids=flat_ids,
                        elements_with_ids=elements_with_ids,
                        width=width,
                        height=height,
                        row_tol_px=self.row_tol_px,
                    )

            if rows_ids:
                id_to_el = {el.get("id"): el for el in elements_with_ids if isinstance(el.get("id"), int)}
                matches_out_rows: list[list[dict]] = []
                for row_ids in rows_ids:
                    row_els = [id_to_el[i2] for i2 in row_ids if i2 in id_to_el]
                    if row_els:
                        matches_out_rows.append(row_els)

                matches_out = self.out_dir / f"{base}_matches.json"
                matches_out.write_text(json.dumps(matches_out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

                # Draw a single rectangle around each matched row on the *input* screenshot,
                # and save as `{base}_output.png`.
                bgr = cv2.imread(str(screenshot_path))
                if bgr is not None and matches_out_rows:
                    for row in matches_out_rows:
                        x1, y1, x2, y2 = _row_bbox_px(row, width, height)
                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.imwrite(str(annotated_path), bgr)

                print(f"user_description={self.value_query!r}")
                print(f"found_rows={len(matches_out_rows)}")
                print(f"saved_screenshot={str(screenshot_path)}")
                print(f"saved_parsed_json={str(parsed_path)}")
                print(f"saved_ids_annotated={str(ids_annotated_path)}")
                print(f"saved_llm_raw={str(llm_path)}")
                print(f"saved_matches_json={str(matches_out)}")
                print(f"saved_output={str(annotated_path)}")
                print(json.dumps(matches_out_rows, ensure_ascii=False, indent=2))
                return

            rows = _cluster_rows(elements_with_ids, height, row_tol_px=self.row_tol_px)
            matches: list[list[dict]] = []
            for row in rows:
                row_text = " | ".join(_norm(str(el.get("content") or "")) for el in row if el.get("content"))
                if q and q in row_text:
                    matches.append(row)

            if matches:
                bgr = cv2.imread(str(screenshot_path))
                if bgr is not None:
                    for row in matches:
                        x1, y1, x2, y2 = _row_bbox_px(row, width, height)
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.imwrite(str(annotated_path), bgr)

                x1, y1, x2, y2 = _row_bbox_px(matches[0], width, height)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                print(f"query={self.value_query!r}")
                print(f"found_rows={len(matches)}")
                print(f"x={cx}, y={cy}")
                print(f"saved_screenshot={str(screenshot_path)}")
                print(f"saved_parsed_json={str(parsed_path)}")
                print(f"saved_annotated={str(annotated_path)}")
                return

            if i < self.max_scrolls:
                pyautogui.scroll(-self.scroll_amount)
                time.sleep(0.6)

        print("No matching row found after scrolling.")
        print(f"query={self.value_query!r}")


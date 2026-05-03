from __future__ import annotations

import json
from pathlib import Path

import cv2
import pyautogui

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from utilites.screen_parser import ScreenParser  # noqa: E402
from utilites.call_to_llm import VisionLLM # noqa: E402


class FindCoordinatesStep:
    def __init__(self, description: str, out_dir: Path, omniparser_url: str | None = None) -> None:
        self.description = description
        self.out_dir = out_dir
        self.parser = ScreenParser(omniparser_url=omniparser_url) if omniparser_url else ScreenParser()
        self.llm: VisionLLM | None = None

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        base = Path(__file__).stem
        screenshot_path = self.out_dir / f"{base}_input.png"
        annotated_path = self.out_dir / f"{base}_output.png"
        omniparser_dump_path = self.out_dir / f"{base}_parsed.json"

        img = pyautogui.screenshot()
        width, height = img.size
        img.save(screenshot_path)

        result = self.parser.parse_image_file(str(screenshot_path))
        elements = result.get("parsed_content_list", []) or []
        omniparser_dump_path.write_text(json.dumps(result["parsed_content_list"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_omniparser_json={str(omniparser_dump_path)}")

        # Optional: call LLM using ONLY OmniParser JSON (no screenshot to the LLM).
        # Enable by setting LLM_API_KEY in your environment.
        try:
            self.llm = VisionLLM()
            llm_resp = None
            llm_resp = self.llm.select_best_omniparser_element(
                omniparser_result=elements,
                user_description=self.description,
            )
            print("\n--- LLM (raw) ---")
            print(llm_resp.raw_text)
            if llm_resp.json is not None:
                print("\n--- LLM (json) ---")
                print(llm_resp.json)

                # Convert LLM bbox_norm -> pixel bbox and annotate screenshot.
                llm_el = llm_resp.json
                bbox = llm_el.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1n, y1n, x2n, y2n = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    x1, y1 = int(x1n * width), int(y1n * height)
                    x2, y2 = int(x2n * width), int(y2n * height)

                    bgr = cv2.imread(str(screenshot_path))
                    if bgr is not None:
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = str(llm_el.get("content") or llm_el.get("type") or "target")
                        label = label[:40]
                        cv2.putText(
                            bgr,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imwrite(str(annotated_path), bgr)
                        cx = int(((x1n + x2n) / 2.0) * width)
                        cy = int(((y1n + y2n) / 2.0) * height)
                        print(f"\nLLM pixel bbox: {(x1, y1, x2, y2)}")
                        print(f"LLM click center: x={cx}, y={cy}")
                        print(f"annotated_screenshot={str(annotated_path)}")
        except Exception as e:
            # If no key/provider is set, we skip LLM without failing the step.
            print(f"\n[LLM skipped] {e}")

        match = self.parser.find_best_match(elements, self.description)
        if match is None:
            print("No match found.")
            print(f"Saved screenshot: {screenshot_path}")
            return

        x, y = self.parser.element_center_px(match.element, width, height)
        print(f"query={self.description!r}")
        print(f"x={x}, y={y}")
        print(f"match_score={match.score}")
        print(f"match_type={match.element.get('type')!r}")
        print(f"match_content={match.element.get('content')!r}")
        print(f"match_bbox={match.element.get('bbox')!r}  # normalized [x1,y1,x2,y2]")
        print(f"saved_screenshot={str(screenshot_path)}")
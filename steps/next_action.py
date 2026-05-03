from __future__ import annotations

import json
from pathlib import Path

import base64
import pyautogui

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from utilites.call_to_llm import VisionLLM  # noqa: E402


class NextActionStep:
    def __init__(self, goal: str, out_dir: Path) -> None:
        self.goal = goal
        self.out_dir = out_dir
        self.llm = VisionLLM()

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(__file__).stem
        screenshot_path = self.out_dir / f"{base}_input.png"
        llm_json_path = self.out_dir / f"{base}_output.json"

        img = pyautogui.screenshot()
        width, height = img.size
        img.save(screenshot_path)

        # Pass base64 image to the LLM
        with open(screenshot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        llm_resp = self.llm.decide_next_action_from_base64(
            image_b64=image_b64,
            user_description=self.goal,
        )

        print("\n--- LLM (raw) ---")
        print(llm_resp.raw_text)
        if llm_resp.json is None:
            print("\nNo JSON parsed from LLM output.")
            return

        print("\n--- LLM (json) ---")
        print(llm_resp.json)
        llm_json_path.write_text(json.dumps(llm_resp.json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_llm_json={str(llm_json_path)}")

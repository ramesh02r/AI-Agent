from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import io
import base64

import requests


def _load_dotenv_if_available() -> None:
    """
    Python does not auto-load `.env`. If `python-dotenv` is installed, load project `.env`.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except Exception:
        return

    # utilites/call_to_llm.py -> repo root is parents[1]
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)


@dataclass(frozen=True)
class LlmResponse:
    raw_text: str
    json: dict[str, Any] | None


class VisionLLM:
    """
    Text-only LLM helper (name kept for imports).

    Supports:
    - OpenAI-compatible Chat Completions API (no image attachment)
    - Gemini Generative Language API (text-only)

    Environment variables:
    - LLM_PROVIDER (default: openai)  # openai | gemini
    - LLM_BASE_URL (openai only; default: https://api.openai.com/v1)
    - LLM_API_KEY  (openai key OR gemini key; required)
    - LLM_MODEL    (default depends on provider)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: int = 60,
    ) -> None:
        _load_dotenv_if_available()
        self.provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL") or (("gemini-2.0-flash" if self.provider == "gemini" else "gpt-4o-mini"))
        self.timeout_s = timeout_s

        if not self.api_key:
            raise RuntimeError("Missing LLM_API_KEY environment variable.")

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """
        Best-effort parse: models sometimes wrap JSON in markdown fences or add prose.
        """
        t = (text or "").strip()
        if not t:
            return None

        # Strip ```json ... ``` fences if present
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
        if fence:
            t = fence.group(1).strip()

        # If extra prose exists, try the first {...} block
        if not (t.startswith("{") and t.endswith("}")):
            m = re.search(r"\{[\s\S]*\}", t)
            if m:
                t = m.group(0).strip()

        try:
            obj = json.loads(t)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _chat_openai(self, *, system: str, user: str) -> LlmResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        parsed = self._extract_json_object(text)
        return LlmResponse(raw_text=text, json=parsed)

    def _chat_openai_with_image_b64(
        self,
        *,
        system: str,
        user: str,
        image_b64: str,
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """
        OpenAI-compatible vision call using Chat Completions with an image_url data URL.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        data_url = f"data:{mime_type};base64,{image_b64}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.2,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        parsed = self._extract_json_object(text)
        return LlmResponse(raw_text=text, json=parsed)

    def _chat_openai_with_images_b64(
        self,
        *,
        system: str,
        user: str,
        images_b64: list[str],
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """
        OpenAI-compatible vision call with multiple images.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for b64 in images_b64:
            data_url = f"data:{mime_type};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        parsed = self._extract_json_object(text)
        return LlmResponse(raw_text=text, json=parsed)

    def _chat_gemini(self, *, system: str, user: str) -> LlmResponse:
        # Gemini Generative Language API (AI Studio key)
        # POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=API_KEY
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        prompt = f"{system}\n\n{user}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }

        resp = requests.post(url, params=params, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        # Typical response: candidates[0].content.parts[0].text
        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0].get("text", "")  # type: ignore[index]
        except Exception:
            text = json.dumps(data, ensure_ascii=False)

        parsed = self._extract_json_object(text)
        return LlmResponse(raw_text=text, json=parsed)

    def _chat_gemini_with_image_b64(self, *, system: str, user: str, image_b64: str, mime_type: str = "image/png") -> LlmResponse:
        """
        Gemini generateContent with an image part (inlineData).
        `image_b64` is the raw base64 string (no data: prefix).
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        prompt = f"{system}\n\n{user}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.2},
        }

        resp = requests.post(url, params=params, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0].get("text", "")  # type: ignore[index]
        except Exception:
            text = json.dumps(data, ensure_ascii=False)

        parsed = self._extract_json_object(text)
        return LlmResponse(raw_text=text, json=parsed)

    def _chat_gemini_with_images_b64(
        self,
        *,
        system: str,
        user: str,
        images_b64: list[str],
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """
        Gemini generateContent with multiple image parts (inlineData).
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        prompt = f"{system}\n\n{user}"
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for b64 in images_b64:
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2},
        }

        resp = requests.post(url, params=params, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0].get("text", "")  # type: ignore[index]
        except Exception:
            text = json.dumps(data, ensure_ascii=False)

        parsed = self._extract_json_object(text)
        return LlmResponse(raw_text=text, json=parsed)

    @staticmethod
    def encode_image_file_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _chat(self, *, system: str, user: str) -> LlmResponse:
        if self.provider == "gemini":
            return self._chat_gemini(system=system, user=user)
        return self._chat_openai(system=system, user=user)

    

    def elements_to_csv(self, elements: list[dict]) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)

        w.writerow(["type", "content", "interactivity", "source", "x1", "y1", "x2", "y2"])

        for el in elements:
            bbox = el.get("bbox") or [None, None, None, None]
            x1, y1, x2, y2 = (bbox + [None, None, None, None])[:4]
            w.writerow([
                el.get("type", ""),
                (el.get("content") or "").replace("\n", " "),
                bool(el.get("interactivity", False)),
                el.get("source", ""),
                x1, y1, x2, y2,
            ])

        return buf.getvalue()

    def select_best_omniparser_element(
        self,
        *,
        omniparser_result: dict[str, Any],
        user_description: str,
    ) -> LlmResponse:
        """
        Given only OmniParser JSON + a user description, return ONE element object from
        `parsed_content_list` that should be clicked (usually `type: icon` for app icons).

        Output must be EXACTLY one OmniParser element dict (copied from the array).
        """
        # Accept either:
        # - the full OmniParser response dict (with parsed_content_list)
        # - or directly a list of element dicts
        if isinstance(omniparser_result, dict):
            elements = omniparser_result.get("parsed_content_list", []) or []
        else:
            elements = omniparser_result or []
        elements_csv = self.elements_to_csv(elements)
        
        system = (
            "You are a GUI automation planner. You ONLY receive OmniParser parsed results (no screenshot).\n"
            "You will be given the elements ONLY as CSV.\n"
            "Each CSV row represents one element with columns:\n"
            "  idx,type,content,interactivity,source,x1,y1,x2,y2\n"
            "Where x1..y2 are normalized bbox floats in 0..1.\n\n"
            "Each element has fields like: bbox, content, interactivity, source, type.\n"
            "`type` is commonly 'text' or 'icon'.\n\n"
            "Your job: choose the SINGLE best element to click for the user's intent.\n\n"
            "Rules:\n"
            "- If the user asks to click an *icon* (e.g. 'Chrome icon', 'Safari icon'), "
            "FIRST restrict candidates to items where type is exactly 'icon' (case-sensitive value 'icon').\n"
            "- Among icon candidates, pick the one whose `content` best matches the user's words "
            "(e.g. 'Chrome' matches 'Chrome').\n"
            "- If there are NO icon candidates, you may consider other types ONLY if the user intent is clearly not an icon.\n"
            "- You MUST return exactly one object constructed STRICTLY from a single CSV row "
            "(do not invent values that are not present in that row).\n"
            "- Output MUST be valid JSON only (a single object). No markdown, no code fences, no commentary.\n"
            "- Output JSON shape must be:\n"
            "  {\"bbox\":[x1,y1,x2,y2],\"content\":\"...\",\"interactivity\":false,\"source\":\"...\",\"type\":\"...\"}\n"
        )

        user = (
            f"user_description: {user_description!r}\n\n"
            "parsed_content_list (CSV):\n"
            f"{elements_csv}\n\n"
        )

        return self._chat(system=system, user=user)

    def analyze(
        self,
        *,
        omniparser_result: dict[str, Any],
        prompt: str,
    ) -> LlmResponse:
        """
        Back-compat wrapper: sends your custom prompt plus the full OmniParser JSON as text.
        Expected model output: a single OmniParser element object (JSON only).
        """
        system = (
            "You are a GUI automation planner. You ONLY receive OmniParser JSON (no screenshot).\n"
            "Follow the user's instructions exactly.\n"
            "Return valid JSON only: a single object copied verbatim from parsed_content_list.\n"
            "No markdown, no code fences, no commentary.\n"
        )
        user = (
            f"{prompt}\n\n"
            "omniparser_result (JSON):\n"
            f"{json.dumps(omniparser_result, ensure_ascii=False)}\n"
        )
        return self._chat(system=system, user=user)

    def decide_next_action_from_screenshot(
        self,
        *,
        image_path: str,
        user_description: str,
    ) -> LlmResponse:
        """
        Vision-based "next action" planner.

        Returns JSON:
        {
          "next_action": {
            "action_type": "click" | "type" | "scroll" | "none",
            "target": "short description of what to click/type/scroll"
          }
        }
        """
        system = (
            "You are an expert GUI automation agent.\n"
            "You will receive:\n"
            "- a screenshot of the current screen\n"
            "- a user task description\n\n"
            "Goal:\n"
            "Determine the SINGLE next action that most helps complete the task.\n\n"
            "Examples:\n"
            "- If user_description says: \"I'm in Google Docs and I want to share this doc\" and the screenshot shows a Share button,\n"
            "  output: {\"next_action\": {\"action_type\": \"click\", \"target\": \"Share button\"}}\n"
            "- If user_description says: \"Save this Google Doc\" and the screenshot shows the top menu with \"File\",\n"
            "  output: {\"next_action\": {\"action_type\": \"click\", \"target\": \"File menu\"}}\n\n"
            "- If user_description says: \"open youtube\" and you are already in a browser,\n"
            "  a human often presses Ctrl+L (or Cmd+L on macOS) to focus the address bar.\n"
            "  output: {\"next_action\": {\"action_type\": \"shortcut\", \"target\": \"Focus address bar\", \"keys\": \"Ctrl+L\"}}\n"
            "- If user_description says: \"open youtube\" and a human prefers a new tab first,\n"
            "  output: {\"next_action\": {\"action_type\": \"shortcut\", \"target\": \"New tab\", \"keys\": \"Ctrl+T\"}}\n\n"
            "Screen-dependent rule (VERY IMPORTANT):\n"
            "- Your output MUST depend on what is currently visible in the screenshot.\n"
            "- You may use browser shortcuts (Cmd+L/Ctrl+L, Cmd+T/Ctrl+T) ONLY if the screenshot clearly shows a browser UI (tabs + address bar / URL bar).\n"
            "- If a browser UI is NOT clearly visible (e.g. you are in an IDE, desktop, or another app), the next action for 'open youtube' should be to open a browser first (e.g. click Chrome icon in Dock/launcher).\n"
            "- If multiple browser icons are visible (e.g. Chrome and Safari), prefer clicking Chrome unless the user explicitly asked for a different browser.\n"
            "- Do NOT suggest clicking a \"YouTube tab\" unless a YouTube tab is clearly visible.\n\n"
            "Allowed actions:\n"
            "- click\n"
            "- type\n"
            "- scroll\n"
            "- shortcut (keyboard shortcut like Ctrl+T, Ctrl+L, Cmd+L)\n"
            "- none: if the next action cannot be determined\n\n"
            "Rules:\n"
            "- Think step-by-step internally, but output ONLY JSON.\n"
            "- Output format must be EXACTLY:\n"
            "  {\"next_action\": {\"action_type\": \"click|type|scroll|shortcut|none\", \"target\": \"...\", \"keys\": \"...\"}}\n"
            "- `keys` must be present ONLY when action_type is \"shortcut\"; otherwise set it to null.\n"
            "- Do NOT include x/y, bbox, confidence, reason, or any extra fields.\n"
        )

        user = (
            "Task: first read the user_description, then inspect the screenshot, then decide the next action.\n"
            f"user_description: {user_description!r}\n"
        )

        if self.provider == "gemini":
            b64 = self.encode_image_file_to_base64(image_path)
            return self._chat_gemini_with_image_b64(system=system, user=user, image_b64=b64, mime_type="image/png")
        # For openai-compatible providers, send screenshot via image_url data URL.
        b64 = self.encode_image_file_to_base64(image_path)
        return self._chat_openai_with_image_b64(system=system, user=user, image_b64=b64, mime_type="image/png")

    def decide_next_action_from_base64(
        self,
        *,
        image_b64: str,
        user_description: str,
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """Same as decide_next_action_from_screenshot, but caller provides base64 image."""
        system = (
            "You are an expert GUI automation agent.\n"
            "You will receive:\n"
            "- a screenshot of the current screen\n"
            "- a user task description\n\n"
            "Goal:\n"
            "Determine the SINGLE next action that most helps complete the task.\n\n"
            "Examples:\n"
            "- If user_description says: \"I'm in Google Docs and I want to share this doc\" and the screenshot shows a Share button,\n"
            "  output: {\"next_action\": {\"action_type\": \"click\", \"target\": \"Share button\"}}\n"
            "- If user_description says: \"Save this Google Doc\" and the screenshot shows the top menu with \"File\",\n"
            "  output: {\"next_action\": {\"action_type\": \"click\", \"target\": \"File menu\"}}\n\n"
            "- If user_description says: \"open youtube\" and you are already in a browser,\n"
            "  a human often presses Ctrl+L (or Cmd+L on macOS) to focus the address bar.\n"
            "  output: {\"next_action\": {\"action_type\": \"shortcut\", \"target\": \"Focus address bar\", \"keys\": \"Ctrl+L\"}}\n"
            "- If user_description says: \"open youtube\" and a human prefers a new tab first,\n"
            "  output: {\"next_action\": {\"action_type\": \"shortcut\", \"target\": \"New tab\", \"keys\": \"Ctrl+T\"}}\n\n"
            "Screen-dependent rule (VERY IMPORTANT):\n"
            "- Your output MUST depend on what is currently visible in the screenshot.\n"
            "- You may use browser shortcuts (Cmd+L/Ctrl+L, Cmd+T/Ctrl+T) ONLY if the screenshot clearly shows a browser UI (tabs + address bar / URL bar).\n"
            "- If a browser UI is NOT clearly visible (e.g. you are in an IDE, desktop, or another app), the next action for 'open youtube' should be to open a browser first (e.g. click Chrome icon in Dock/launcher).\n"
            "- If multiple browser icons are visible (e.g. Chrome and Safari), prefer clicking Chrome unless the user explicitly asked for a different browser.\n"
            "- Do NOT suggest clicking a \"YouTube tab\" unless a YouTube tab is clearly visible.\n\n"
            "Allowed actions:\n"
            "- click\n"
            "- type\n"
            "- scroll\n"
            "- shortcut (keyboard shortcut like Ctrl+T, Ctrl+L, Cmd+L)\n"
            "- none: if the next action cannot be determined\n\n"
            "Rules:\n"
            "- Think step-by-step internally, but output ONLY JSON.\n"
            "- Output format must be EXACTLY:\n"
            "  {\"next_action\": {\"action_type\": \"click|type|scroll|shortcut|none\", \"target\": \"...\", \"keys\": \"...\"}}\n"
            "- `keys` must be present ONLY when action_type is \"shortcut\"; otherwise set it to null.\n"
            "- Do NOT include x/y, bbox, confidence, reason, or any extra fields.\n"
        )

        user = (
            "Task: first read the user_description, then inspect the screenshot, then decide the next action.\n"
            f"user_description: {user_description!r}\n"
        )

        if self.provider == "gemini":
            return self._chat_gemini_with_image_b64(system=system, user=user, image_b64=image_b64, mime_type=mime_type)
        return self._chat_openai_with_image_b64(system=system, user=user, image_b64=image_b64, mime_type=mime_type)

    def match_elements_from_base64(
        self,
        *,
        image_b64: str,
        user_description: str,
        elements: list[dict[str, Any]],
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """
        Given a screenshot + a simplified elements list (no bbox), return the subset that matches the user description.

        Expected output JSON:
          {"matches": [<element>, <element>, ...]}
        Where each <element> is copied from the provided `elements` array.
        """
        system = (
            "You are a GUI element selector.\n"
            "You will receive:\n"
            "- a screenshot of the current screen\n"
            "- a user_description\n"
            "- a list of UI elements extracted from the screen (JSON array). Each element includes an integer `id`.\n\n"
            "Task:\n"
            "- Read user_description.\n"
            "- Look at the screenshot for context.\n"
            "- Select ALL elements from the provided array that best match the user_description.\n\n"
            "Rules:\n"
            "- Only select from the provided elements array.\n"
            "- Do NOT invent new elements.\n"
            "- Do NOT add bbox or any new keys.\n"
            "- Return ONLY valid JSON in this exact shape:\n"
            "  {\"matches\": [ ... ]}\n"
            "- Each match must be copied verbatim from the provided array.\n"
        )

        user = (
            f"user_description: {user_description!r}\n\n"
            "elements (JSON array):\n"
            f"{json.dumps(elements, ensure_ascii=False)}\n"
        )

        if self.provider == "gemini":
            return self._chat_gemini_with_image_b64(system=system, user=user, image_b64=image_b64, mime_type=mime_type)
        return self._chat_openai_with_image_b64(system=system, user=user, image_b64=image_b64, mime_type=mime_type)

    def match_elements_from_two_images_base64(
        self,
        *,
        normal_image_b64: str,
        ids_annotated_image_b64: str,
        user_description: str,
        elements: list[dict[str, Any]],
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """
        Same as `match_elements_from_base64`, but provides two images:
        1) normal screenshot
        2) the same screenshot annotated with element IDs
        """
        system = (
            "You are a GUI element selector.\n"
            "You will receive:\n"
            "- image_1: normal screenshot\n"
            "- image_2: the same screenshot annotated with element IDs (numbers)\n"
            "- a user_description\n"
            "- a list of UI elements extracted from the screen (JSON array). Each element includes an integer `id`.\n\n"
            "Task:\n"
            "- Read user_description.\n"
            "- Use image_1 for visual context.\n"
            "- Use image_2 to map what you see to numeric IDs.\n"
            "- Select ALL elements from the provided array that best match the user_description.\n\n"
            "Rules:\n"
            "- Only select from the provided elements array.\n"
            "- Do NOT invent new elements.\n"
            "- Do NOT add bbox or any new keys.\n"
            "- Return ONLY valid JSON in this exact shape:\n"
            "  {\"matches\": [ ... ]}\n"
            "- Each match must be copied verbatim from the provided array.\n"
        )

        user = (
            f"user_description: {user_description!r}\n\n"
            "elements (JSON array):\n"
            f"{json.dumps(elements, ensure_ascii=False)}\n"
        )

        images = [normal_image_b64, ids_annotated_image_b64]
        if self.provider == "gemini":
            return self._chat_gemini_with_images_b64(system=system, user=user, images_b64=images, mime_type=mime_type)
        return self._chat_openai_with_images_b64(system=system, user=user, images_b64=images, mime_type=mime_type)

    def match_ids_from_two_images_base64(
        self,
        *,
        normal_image_b64: str,
        ids_annotated_image_b64: str,
        user_description: str,
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """
        Determine matching element IDs using two images only:
        1) normal screenshot
        2) the same screenshot annotated with numeric IDs (next to/inside elements)

        Expected output JSON:
          {"rows": [[1,2,3],[10,11]]}
        """
        system = (
            "You are a GUI element locator.\n"
            "You will receive:\n"
            "- image_1: a normal screenshot\n"
            "- image_2: the same screenshot annotated with numeric IDs\n"
            "- a user_description\n\n"
            "Task:\n"
            "- Read user_description.\n"
            "- Use image_1 for visual understanding.\n"
            "- Use image_2 to read the numeric IDs corresponding to the matching elements.\n"
            "- Return ALL matching IDs, grouped by row (top-to-bottom).\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON.\n"
            "- Output format must be EXACTLY:\n"
            "  {\"rows\": [[...],[...]]}\n"
            "- Each inner array is one row (left-to-right order if possible).\n"
            "- IDs must be integers.\n"
            "- If nothing matches, return: {\"rows\": []}\n"
            "- Do NOT include any other keys.\n"
        )

        user = f"user_description: {user_description!r}"
        images = [normal_image_b64, ids_annotated_image_b64]
        if self.provider == "gemini":
            return self._chat_gemini_with_images_b64(system=system, user=user, images_b64=images, mime_type=mime_type)
        return self._chat_openai_with_images_b64(system=system, user=user, images_b64=images, mime_type=mime_type)

    def extract_text_step_from_base64(
        self,
        *,
        image_b64: str,
        user_task: str,
        already_extracted: list[str],
        remaining_target: int | None,
        image_width_px: int | None = None,
        image_height_px: int | None = None,
        mime_type: str = "image/png",
    ) -> LlmResponse:
        """
        Single-step screenshot-driven extraction + next action.

        Expected output JSON:
          {
            "status": "continue" | "success" | "fail",
            "action": "scroll_down" | "scroll_up" | "wait" | null,
            "pixels": 0,
            "extracted_data": ["..."],
            "reason": "..."
          }
        """
        system = (
            "You are a human-like visual extraction agent.\n"
            "You ONLY have access to the current screenshot, the user task, and prior extracted context.\n"
            "No DOM, no APIs, no hidden metadata.\n\n"
            "Goal:\n"
            "- Decide if the requested content is currently visible.\n"
            "- If visible: extract it as text.\n"
            "- If not visible: choose ONE intentional next action (scroll_up/scroll_down/wait).\n"
            "- Scrolling MUST be non-random: choose a pixel amount that would realistically reveal new relevant content.\n"
            "- Avoid duplicates: do NOT repeat anything in already_extracted.\n\n"
            "Large targets (e.g. 100 items):\n"
            "- Extract as many NEW items as are clearly visible now.\n"
            "- If remaining_target is still > 0, return status=continue with an intentional scroll.\n\n"
            "Output rules (CRITICAL):\n"
            "- Output MUST be valid JSON only. No markdown, no commentary.\n"
            "- Output MUST contain EXACTLY these keys: status, action, pixels, extracted_data, reason.\n"
            "- status meanings:\n"
            "  - \"continue\": perform the action, then take another screenshot\n"
            "  - \"success\": task satisfied / remaining_target reached\n"
            "  - \"fail\": cannot proceed (e.g. wrong page/app)\n"
            "- action must be one of: \"scroll_down\", \"scroll_up\", \"wait\", or null.\n"
            "- pixels must be an integer. Use 0 if action is null or wait.\n"
            "- extracted_data must be an array of strings (may be empty).\n"
        )

        user = (
            f"user_task: {user_task!r}\n"
            f"remaining_target: {remaining_target!r}\n"
            f"image_width_px: {image_width_px!r}\n"
            f"image_height_px: {image_height_px!r}\n"
            f"already_extracted_count: {len(already_extracted)}\n"
            "already_extracted (last 40):\n"
            f"{json.dumps(already_extracted[-40:], ensure_ascii=False)}\n"
        )

        if self.provider == "gemini":
            return self._chat_gemini_with_image_b64(system=system, user=user, image_b64=image_b64, mime_type=mime_type)
        return self._chat_openai_with_image_b64(system=system, user=user, image_b64=image_b64, mime_type=mime_type)

import json
from pathlib import Path

import cv2

from utilites.screen_parser import ScreenParser

# If needed on Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_IMAGE = "input.png"
OUTPUT_IMAGE = "output.png"
OUTPUT_JSON = "output.json"

# Example condition:
# Draw red box around rows where numeric value < 50

# call the screen parser to get the data
parser = ScreenParser()
result = parser.parse_image_file(INPUT_IMAGE, annotated_out_path="out/extract_table_output.png")
elements = result.get("parsed_content_list", []) or []

# Show boxes on your *live screen* (temporary overlay) instead of only saving annotated.png.
# This assumes `INPUT_IMAGE` is a full-screen screenshot from this machine.
try:
    parser.show_overlay_on_screen(image_path=INPUT_IMAGE, omniparser_result=result, duration_s=2.5)
except Exception as e:
    print(f"Overlay skipped: {e}")

# Save parsed JSON for debugging / downstream use
Path(OUTPUT_JSON).write_text(json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved parsed result: {OUTPUT_JSON}")

img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise SystemExit(f"Could not read image: {INPUT_IMAGE}")

# Draw all OmniParser boxes (normalized bbox -> pixel bbox)
h, w = img.shape[:2]
for el in elements:
    bbox = el.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        continue
    try:
        x1n, y1n, x2n, y2n = map(float, bbox)
    except Exception:
        continue

    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    content = str(el.get("content") or "")
    if content:
        cv2.putText(
            img,
            content[:20],
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    
    

# Save result
cv2.imwrite(OUTPUT_IMAGE, img)

print(f"Done. Saved as {OUTPUT_IMAGE}")
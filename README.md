# AI-Agent

## Screenshot omni-parser

This repo includes a small “omni parser” that accepts a screenshot and outputs:

- **`parsed.json`**: OCR text + bounding boxes (and heuristic UI boxes)
- **`annotated.png`**: an overlay image (OCR boxes in red, UI boxes in green)

### Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Install one OCR engine (open-source) of your choice:

- **PaddleOCR** (best overall on UI screenshots):

```bash
pip install paddleocr
```

- **EasyOCR**:

```bash
pip install easyocr
```

- **Tesseract** (requires OS binary + Python wrapper):

```bash
pip install pytesseract
```

### Run

```bash
. .venv/bin/activate
python main.py parse --image /path/to/screenshot.png --out out
```

Outputs:

- `out/parsed.json`
- `out/annotated.png`

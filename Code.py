
import os
import re
import cv2
import fitz
import math
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from rapidfuzz import process, fuzz

import pytesseract
from pytesseract import Output

from config import (
    INPUT_PDF,
    OUTPUT_DIR,
    OUTPUT_EXCEL,
    OUTPUT_REVIEW,
    DEBUG_DIR,
    DPI,
    OCR_LANG,
    MIN_CONFIDENCE,
    TESSERACT_PATH,
    UPSCALE,
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    EXCEL_SHEET_NAME,
)


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# INDIAN STATES / UTs
# ============================================================

INDIAN_STATES = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",

    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
]


# ============================================================
# PADDLE OCR
# ============================================================

print("\nLoading PaddleOCR...")

try:
    from paddleocr import PaddleOCR

    ocr_engine = PaddleOCR(
        lang=OCR_LANG,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    PADDLE_AVAILABLE = True
    print("PaddleOCR loaded successfully.")

except Exception as e:

    print("WARNING: PaddleOCR could not be loaded.")
    print(e)

    ocr_engine = None
    PADDLE_AVAILABLE = False


# ============================================================
# PDF → IMAGE
# ============================================================

def pdf_to_images(pdf_path, dpi=400):

    print("\nConverting PDF to images...")

    doc = fitz.open(pdf_path)

    images = []

    zoom = dpi / 72.0

    matrix = fitz.Matrix(zoom, zoom)

    for page_number, page in enumerate(doc):

        print(f"Rendering page {page_number + 1}/{len(doc)}")

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        img = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            pix.n
        )

        if pix.n == 4:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGBA2BGR
            )
        else:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR
            )

        images.append(img)

    doc.close()

    return images


# ============================================================
# DESKEW
# ============================================================

def deskew(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Binary image
    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(
        np.where(thresh > 0)
    )

    if len(coords) < 100:

        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:

        angle = -(90 + angle)

    else:

        angle = -angle

    # Ignore tiny rotations
    if abs(angle) < 0.1:

        return image

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ============================================================
# TABLE LINE DETECTION
# ============================================================

def detect_lines(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    h, w = binary.shape

    # Horizontal
    horizontal_kernel_size = max(
        20,
        w // 30
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_size, 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Vertical
    vertical_kernel_size = max(
        20,
        h // 30
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_kernel_size)
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    return horizontal, vertical


# ============================================================
# REMOVE TABLE LINES
# ============================================================

def remove_table_lines(image):

    horizontal, vertical = detect_lines(image)

    lines = cv2.bitwise_or(
        horizontal,
        vertical
    )

    # Expand slightly to completely remove lines
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    lines = cv2.dilate(
        lines,
        kernel,
        iterations=1
    )

    result = image.copy()

    result[lines > 0] = 255

    return result


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_cell(cell):

    if cell is None or cell.size == 0:

        return None

    # Add white border
    cell = cv2.copyMakeBorder(
        cell,
        10,
        10,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # Upscale
    cell = cv2.resize(
        cell,
        None,
        fx=UPSCALE,
        fy=UPSCALE,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )

    # Slight denoise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    # Otsu
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adaptive
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    return {
        "gray": gray,
        "sharp": sharpened,
        "otsu": otsu,
        "adaptive": adaptive,
    }


# ============================================================
# TESSERACT OCR
# ============================================================

def tesseract_ocr(image, psm=6):

    if image is None:

        return "", 0.0

    try:

        data = pytesseract.image_to_data(
            image,
            lang=OCR_LANG,
            config=f"--psm {psm}",
            output_type=Output.DICT
        )

        texts = []
        confidences = []

        for text, conf in zip(
            data["text"],
            data["conf"]
        ):

            text = text.strip()

            try:
                conf = float(conf)
            except:
                conf = -1

            if text and conf >= 0:

                texts.append(text)
                confidences.append(conf)

        if not texts:

            return "", 0.0

        result = " ".join(texts)

        confidence = np.mean(
            confidences
        ) / 100.0

        return result, confidence

    except Exception as e:

        return "", 0.0


# ============================================================
# PADDLE OCR
# ============================================================

def paddle_ocr(image):

    if not PADDLE_AVAILABLE:

        return "", 0.0

    if image is None:

        return "", 0.0

    try:

        results = ocr_engine.predict(image)

        all_text = []
        all_scores = []

        for result in results:

            # PaddleOCR 3.x result object
            data = None

            if hasattr(result, "json"):
                try:
                    data = result.json
                except:
                    pass

            if callable(data):
                data = data()

            if data is None and hasattr(
                result,
                "to_dict"
            ):
                try:
                    data = result.to_dict()
                except:
                    pass

            if isinstance(data, str):

                import json

                try:
                    data = json.loads(data)
                except:
                    data = None

            if not isinstance(data, dict):
                continue

            # Find recognition fields recursively
            rec_texts = data.get(
                "rec_texts",
                []
            )

            rec_scores = data.get(
                "rec_scores",
                []
            )

            if rec_texts:

                all_text.extend(
                    [
                        str(x)
                        for x in rec_texts
                        if str(x).strip()
                    ]
                )

            if rec_scores:

                all_scores.extend(
                    [
                        float(x)
                        for x in rec_scores
                    ]
                )

        if not all_text:

            return "", 0.0

        text = " ".join(all_text)

        confidence = (
            float(np.mean(all_scores))
            if all_scores
            else 0.0
        )

        return text, confidence

    except Exception as e:

        print(
            "PaddleOCR error:",
            str(e)[:200]
        )

        return "", 0.0


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_text(text):

    if text is None:

        return ""

    text = str(text)

    # Remove weird whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([,.;:])",
        r"\1",
        text
    )

    # Remove duplicate punctuation
    text = re.sub(
        r",{2,}",
        ",",
        text
    )

    text = re.sub(
        r"\.{3,}",
        "...",
        text
    )

    return text.strip()


# ============================================================
# STATE NORMALIZATION
# ============================================================

def normalize_state(text):

    original = clean_text(text)

    if not original:

        return ""

    # Direct exact match
    for state in INDIAN_STATES:

        if original.lower() == state.lower():

            return state

    # Fuzzy matching
    match = process.extractOne(
        original,
        INDIAN_STATES,
        scorer=fuzz.ratio
    )

    if match:

        state, score, _ = match

        if score >= 75:

            return state

    return original


# ============================================================
# SERIAL NUMBER CLEANING
# ============================================================

def clean_serial(text):

    text = clean_text(text)

    if not text:

        return None

    # Common OCR substitutions
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "B": "8",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Find first integer
    match = re.search(
        r"\d+",
        text
    )

    if not match:

        return None

    try:

        return int(
            match.group()
        )

    except:

        return None


# ============================================================
# PIN VALIDATION
# ============================================================

def find_pin(text):

    matches = re.findall(
        r"\b[1-9][0-9]{5}\b",
        text
    )

    if matches:

        return matches[-1]

    return None


# ============================================================
# OCR SINGLE CELL
# ============================================================

def ocr_cell(cell, column_type):

    processed = preprocess_cell(
        cell
    )

    if not processed:

        return "", 0.0, "none"

    candidates = []

    # --------------------------------------------------------
    # PaddleOCR
    # --------------------------------------------------------

    for name in [
        "gray",
        "sharp",
        "otsu",
        "adaptive"
    ]:

        image = processed[name]

        text, confidence = paddle_ocr(
            image
        )

        if text:

            candidates.append(
                (
                    clean_text(text),
                    confidence,
                    "paddle_" + name
                )
            )

    # --------------------------------------------------------
    # Tesseract
    # --------------------------------------------------------

    psm_values = [6, 7, 11]

    if column_type == "serial":

        psm_values = [7, 8]

    for psm in psm_values:

        text, confidence = tesseract_ocr(
            processed["sharp"],
            psm=psm
        )

        if text:

            candidates.append(
                (
                    clean_text(text),
                    confidence,
                    f"tesseract_{psm}"
                )
            )

    if not candidates:

        return "", 0.0, "none"

    # --------------------------------------------------------
    # Special handling for serial
    # --------------------------------------------------------

    if column_type == "serial":

        valid = []

        for text, conf, engine in candidates:

            number = clean_serial(text)

            if number is not None:

                valid.append(
                    (
                        str(number),
                        conf,
                        engine
                    )
                )

        if valid:

            return max(
                valid,
                key=lambda x: x[1]
            )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    if column_type == "state":

        valid = []

        for text, conf, engine in candidates:

            state = normalize_state(
                text
            )

            valid.append(
                (
                    state,
                    conf,
                    engine
                )
            )

        if valid:

            return max(
                valid,
                key=lambda x: x[1]
            )

    # --------------------------------------------------------
    # Generic
    # --------------------------------------------------------

    return max(
        candidates,
        key=lambda x: x[1]
    )


# ============================================================
# FIND HORIZONTAL ROW LINES
# ============================================================

def find_horizontal_lines(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    h, w = binary.shape

    kernel_width = max(
        30,
        w // 20
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Count black pixels in each row
    row_counts = np.sum(
        horizontal > 0,
        axis=1
    )

    threshold = w * 0.20

    ys = np.where(
        row_counts > threshold
    )[0]

    if len(ys) == 0:

        return []

    # Group adjacent rows
    groups = []

    start = ys[0]
    previous = ys[0]

    for y in ys[1:]:

        if y <= previous + 2:

            previous = y

        else:

            groups.append(
                (start, previous)
            )

            start = y
            previous = y

    groups.append(
        (start, previous)
    )

    centers = [
        int((a + b) / 2)
        for a, b in groups
    ]

    return centers


# ============================================================
# FIND VERTICAL LINES
# ============================================================

def find_vertical_lines(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    h, w = binary.shape

    kernel_height = max(
        30,
        h // 15
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, kernel_height)
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    col_counts = np.sum(
        vertical > 0,
        axis=0
    )

    threshold = h * 0.15

    xs = np.where(
        col_counts > threshold
    )[0]

    if len(xs) == 0:

        return []

    groups = []

    start = xs[0]
    previous = xs[0]

    for x in xs[1:]:

        if x <= previous + 2:

            previous = x

        else:

            groups.append(
                (start, previous)
            )

            start = x
            previous = x

    groups.append(
        (start, previous)
    )

    centers = [
        int((a + b) / 2)
        for a, b in groups
    ]

    return centers


# ============================================================
# GROUP CLOSE LINES
# ============================================================

def merge_close(values, tolerance=15):

    if not values:

        return []

    values = sorted(values)

    groups = [
        [values[0]]
    ]

    for value in values[1:]:

        if abs(
            value - np.mean(groups[-1])
        ) <= tolerance:

            groups[-1].append(value)

        else:

            groups.append(
                [value]
            )

    return [
        int(np.mean(group))
        for group in groups
    ]


# ============================================================
# FIND TABLE BLOCKS
# ============================================================

def detect_table_blocks(image):

    h, w = image.shape[:2]

    # We use vertical lines to locate table boundaries.
    vertical_lines = find_vertical_lines(
        image
    )

    vertical_lines = merge_close(
        vertical_lines,
        tolerance=20
    )

    # Need enough lines to make a table.
    if len(vertical_lines) < 4:

        # fallback: divide page into thirds
        third = w // 3

        return [
            (0, 0, third, h),
            (third, 0, third * 2, h),
            (third * 2, 0, w, h)
        ]

    blocks = []

    # Look for large gaps between vertical lines.
    starts = []

    for i in range(
        len(vertical_lines) - 1
    ):

        x1 = vertical_lines[i]
        x2 = vertical_lines[i + 1]

        gap = x2 - x1

        if gap > w * 0.15:

            starts.append(
                (x1, x2)
            )

    # Usually each table block has several internal columns.
    # A robust fallback is based on page thirds.
    if len(starts) < 3:

        third = w // 3

        blocks = [
            (
                max(0, third * i - 5),
                0,
                min(w, third * (i + 1) + 5),
                h
            )
            for i in range(3)
        ]

    else:

        # Use thirds because your Bandhan PDF has 3 blocks.
        third = w // 3

        blocks = [
            (0, 0, third, h),
            (third, 0, third * 2, h),
            (third * 2, 0, w, h)
        ]

    return blocks


# ============================================================
# COLUMN BOUNDARIES
# ============================================================

def detect_columns(block):

    h, w = block.shape[:2]

    vertical_lines = find_vertical_lines(
        block
    )

    vertical_lines = merge_close(
        vertical_lines,
        tolerance=15
    )

    # Add page boundaries
    boundaries = [0]

    # Keep internal lines
    for x in vertical_lines:

        if 5 < x < w - 5:

            boundaries.append(x)

    boundaries.append(w)

    boundaries = sorted(
        list(set(boundaries))
    )

    # We need exactly 5 boundaries for 4 columns.
    #
    # If table detection sees extra lines, select
    # the 5 strongest/most useful boundaries.

    if len(boundaries) >= 5:

        # In your layout there are four columns.
        # Select approximately:
        # 0%, ~15%, ~45%, ~65%, 100%
        expected = [
            0,
            int(w * 0.16),
            int(w * 0.43),
            int(w * 0.62),
            w
        ]

        selected = []

        for target in expected:

            nearest = min(
                boundaries,
                key=lambda x: abs(x - target)
            )

            if nearest not in selected:

                selected.append(nearest)

        if len(selected) == 5:

            return sorted(selected)

    # Fallback based on visual proportions.
    return [
        0,
        int(w * 0.16),
        int(w * 0.43),
        int(w * 0.62),
        w
    ]


# ============================================================
# EXTRACT ROWS
# ============================================================

def extract_rows(block):

    h, w = block.shape[:2]

    horizontal_lines = find_horizontal_lines(
        block
    )

    horizontal_lines = merge_close(
        horizontal_lines,
        tolerance=10
    )

    # Need at least header + some rows.
    if len(horizontal_lines) < 4:

        # Fallback using connected text projection
        return extract_rows_projection(
            block
        )

    rows = []

    for i in range(
        len(horizontal_lines) - 1
    ):

        y1 = horizontal_lines[i]
        y2 = horizontal_lines[i + 1]

        if y2 - y1 < 15:

            continue

        rows.append(
            (
                max(0, y1 + 2),
                min(h, y2 - 2)
            )
        )

    return rows


# ============================================================
# PROJECTION-BASED ROW EXTRACTION
# ============================================================

def extract_rows_projection(block):

    gray = cv2.cvtColor(
        block,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Remove vertical lines
    _, vertical = detect_lines(block)

    text_mask = cv2.subtract(
        binary,
        vertical
    )

    row_projection = np.sum(
        text_mask > 0,
        axis=1
    )

    # Text-containing rows
    threshold = max(
        5,
        block.shape[1] * 0.005
    )

    ys = np.where(
        row_projection > threshold
    )[0]

    if len(ys) == 0:

        return []

    groups = []

    start = ys[0]
    previous = ys[0]

    for y in ys[1:]:

        if y <= previous + 4:

            previous = y

        else:

            groups.append(
                (start, previous)
            )

            start = y
            previous = y

    groups.append(
        (start, previous)
    )

    # Add small padding
    rows = []

    for y1, y2 in groups:

        if y2 - y1 >= 10:

            rows.append(
                (
                    max(0, y1 - 4),
                    min(
                        block.shape[0],
                        y2 + 4
                    )
                )
            )

    return rows


# ============================================================
# PROCESS ONE TABLE BLOCK
# ============================================================

def process_block(
    block,
    page_number,
    block_number
):

    print(
        f"\nProcessing page {page_number}, "
        f"block {block_number}"
    )

    columns = detect_columns(
        block
    )

    print(
        "Column boundaries:",
        columns
    )

    rows = extract_rows(
        block
    )

    print(
        "Detected rows:",
        len(rows)
    )

    results = []

    # Skip first row because it is probably header.
    # We will later detect whether it actually is.
    for row_index, (y1, y2) in enumerate(rows):

        row = block[
            y1:y2,
            :
        ]

        # Ignore very small rows
        if row.shape[0] < 15:

            continue

        cells = []

        for col_index in range(4):

            x1 = columns[col_index]
            x2 = columns[col_index + 1]

            # Padding
            px = 3

            x1 = max(
                0,
                x1 + px
            )

            x2 = min(
                block.shape[1],
                x2 - px
            )

            cell = row[
                :,
                x1:x2
            ]

            # Remove table lines
            cell = remove_table_lines(
                cell
            )

            cells.append(
                cell
            )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        serial, serial_conf, serial_engine = ocr_cell(
            cells[0],
            "serial"
        )

        state, state_conf, state_engine = ocr_cell(
            cells[1],
            "state"
        )

        branch, branch_conf, branch_engine = ocr_cell(
            cells[2],
            "branch"
        )

        address, address_conf, address_engine = ocr_cell(
            cells[3],
            "address"
        )

        serial_num = clean_serial(
            serial
        )

        state = normalize_state(
            state
        )

        address = clean_text(
            address
        )

        branch = clean_text(
            branch
        )

        # ----------------------------------------------------
        # Header detection
        # ----------------------------------------------------

        combined = (
            f"{serial} {state} "
            f"{branch} {address}"
        ).lower()

        if (
            "state" in combined
            or "branch" in combined
            or "address" in combined
            or "sl.no" in combined
        ):

            print(
                f"Skipping header row {row_index}"
            )

            continue

        # ----------------------------------------------------
        # Suspicion scoring
        # ----------------------------------------------------

        problems = []

        if serial_num is None:

            problems.append(
                "invalid_serial"
            )

        if not state:

            problems.append(
                "missing_state"
            )

        if not branch:

            problems.append(
                "missing_branch"
            )

        if not address:

            problems.append(
                "missing_address"
            )

        if serial_conf < MIN_CONFIDENCE:

            problems.append(
                "low_serial_confidence"
            )

        if state_conf < MIN_CONFIDENCE:

            problems.append(
                "low_state_confidence"
            )

        if branch_conf < MIN_CONFIDENCE:

            problems.append(
                "low_branch_confidence"
            )

        if address_conf < MIN_CONFIDENCE:

            problems.append(
                "low_address_confidence"
            )

        pin = find_pin(
            address
        )

        if not pin:

            problems.append(
                "no_pin_detected"
            )

        result = {
            "Sl. No.": serial_num,
            "State": state,
            "Branch Name": branch,
            "Branch Address": address,

            "_serial_conf": serial_conf,
            "_state_conf": state_conf,
            "_branch_conf": branch_conf,
            "_address_conf": address_conf,

            "_serial_engine": serial_engine,
            "_state_engine": state_engine,
            "_branch_engine": branch_engine,
            "_address_engine": address_engine,

            "_page": page_number,
            "_block": block_number,
            "_row": row_index,

            "_problems": ";".join(problems)
        }

        results.append(
            result
        )

    return results


# ============================================================
# FIX SERIAL NUMBERS
# ============================================================

def fix_serial_numbers(df):

    if df.empty:

        return df

    df = df.copy()

    # Sort by page/block/row first
    df = df.sort_values(
        [
            "_page",
            "_block",
            "_row"
        ]
    )

    previous = None

    for index in df.index:

        current = df.loc[
            index,
            "Sl. No."
        ]

        if current is None:

            continue

        if previous is not None:

            # If OCR gives a suspicious jump
            if (
                current < previous
                or current > previous + 3
            ):

                expected = previous + 1

                # Only correct if extremely likely
                # to be OCR error.
                if abs(
                    current - expected
                ) <= 2:

                    df.loc[
                        index,
                        "Sl. No."
                    ] = expected

                    df.loc[
                        index,
                        "_problems"
                    ] += ";serial_sequence_corrected"

        previous = df.loc[
            index,
            "Sl. No."
        ]

    return df


# ============================================================
# REMOVE DUPLICATE ROWS
# ============================================================

def remove_duplicates(df):

    if df.empty:

        return df

    # Remove exact duplicate records
    df = df.drop_duplicates(
        subset=[
            "Sl. No.",
            "State",
            "Branch Name",
            "Branch Address"
        ]
    )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    print("\nRunning dataset validation...")

    if df.empty:

        return df

    # --------------------------------------------------------
    # Duplicate serial numbers
    # --------------------------------------------------------

    duplicate_serials = df[
        df["Sl. No."].duplicated(
            keep=False
        )
    ]

    if not duplicate_serials.empty:

        print(
            "\nWARNING: Duplicate serial numbers:"
        )

        print(
            duplicate_serials[
                [
                    "Sl. No.",
                    "State",
                    "Branch Name"
                ]
            ].to_string(
                index=False
            )
        )

        for idx in duplicate_serials.index:

            df.loc[
                idx,
                "_problems"
            ] += ";duplicate_serial"

    # --------------------------------------------------------
    # Empty values
    # --------------------------------------------------------

    for column in [
        "Sl. No.",
        "State",
        "Branch Name",
        "Branch Address"
    ]:

        empty = (
            df[column]
            .isna()
            |
            (
                df[column]
                .astype(str)
                .str.strip()
                == ""
            )
        )

        for idx in df[
            empty
        ].index:

            df.loc[
                idx,
                "_problems"
            ] += (
                f";empty_{column}"
            )

    return df


# ============================================================
# SAVE DEBUG IMAGE
# ============================================================

def save_debug(
    image,
    page_number,
    block_number
):

    path = (
        DEBUG_DIR
        /
        f"page_{page_number}_block_{block_number}.jpg"
    )

    cv2.imwrite(
        str(path),
        image
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BANDHAN BANK PDF OCR → EXCEL")
    print("=" * 70)

    if not INPUT_PDF.exists():

        raise FileNotFoundError(
            f"PDF not found:\n{INPUT_PDF}"
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pages = pdf_to_images(
        INPUT_PDF,
        DPI
    )

    all_rows = []

    # --------------------------------------------------------
    # PAGE LOOP
    # --------------------------------------------------------

    for page_number, image in enumerate(
        pages,
        start=1
    ):

        print("\n" + "=" * 60)

        print(
            f"PAGE {page_number}"
        )

        print("=" * 60)

        # Deskew
        image = deskew(
            image
        )

        # Save page
        cv2.imwrite(
            str(
                DEBUG_DIR
                /
                f"page_{page_number}.jpg"
            ),
            image
        )

        # Detect table blocks
        blocks = detect_table_blocks(
            image
        )

        print(
            "Detected table blocks:",
            len(blocks)
        )

        # ----------------------------------------------------
        # BLOCK LOOP
        # ----------------------------------------------------

        for block_number, (
            x1,
            y1,
            x2,
            y2
        ) in enumerate(
            blocks,
            start=1
        ):

            block = image[
                y1:y2,
                x1:x2
            ]

            if block.size == 0:

                continue

            save_debug(
                block,
                page_number,
                block_number
            )

            rows = process_block(
                block,
                page_number,
                block_number
            )

            all_rows.extend(
                rows
            )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    print("\nCreating dataframe...")

    if not all_rows:

        print(
            "No rows detected."
        )

        return

    df = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # FIX SERIALS
    # --------------------------------------------------------

    df = fix_serial_numbers(
        df
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    df = validate_dataset(
        df
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "Sl. No."
        ],
        na_position="last"
    )

    # --------------------------------------------------------
    # MANUAL REVIEW
    # --------------------------------------------------------

    review_mask = (
        df["_problems"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    )

    review_df = df[
        review_mask
    ].copy()

    # --------------------------------------------------------
    # SAVE REVIEW FILE
    # --------------------------------------------------------

    review_columns = [
        "Sl. No.",
        "State",
        "Branch Name",
        "Branch Address",

        "_serial_conf",
        "_state_conf",
        "_branch_conf",
        "_address_conf",

        "_page",
        "_block",
        "_row",
        "_problems"
    ]

    review_df[
        review_columns
    ].to_csv(
        OUTPUT_REVIEW,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # FINAL EXCEL
    # --------------------------------------------------------

    final_columns = [
        "Sl. No.",
        "State",
        "Branch Name",
        "Branch Address"
    ]

    final_df = df[
        final_columns
    ].copy()

    # --------------------------------------------------------
    # EXCEL FORMATTING
    # --------------------------------------------------------

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        final_df.to_excel(
            writer,
            sheet_name=EXCEL_SHEET_NAME,
            index=False
        )

        worksheet = writer[
            EXCEL_SHEET_NAME
        ]

        # Freeze header
        worksheet.freeze_panes = "A2"

        # Widths
        worksheet.column_dimensions[
            "A"
        ].width = 10

        worksheet.column_dimensions[
            "B"
        ].width = 25

        worksheet.column_dimensions[
            "C"
        ].width = 30

        worksheet.column_dimensions[
            "D"
        ].width = 80

        # Wrap address
        for row in worksheet.iter_rows():

            for cell in row:

                cell.alignment = (
                    cell.alignment.copy(
                        wrap_text=True,
                        vertical="top"
                    )
                )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("EXTRACTION COMPLETE")

    print("=" * 70)

    print(
        f"Total rows: {len(final_df)}"
    )

    print(
        f"Rows needing review: "
        f"{len(review_df)}"
    )

    print(
        f"\nExcel:\n{OUTPUT_EXCEL}"
    )

    print(
        f"\nReview file:\n{OUTPUT_REVIEW}"
    )

    print(
        f"\nDebug images:\n{DEBUG_DIR}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

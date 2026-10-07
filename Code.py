import cv2
import pytesseract
import os
import re
import pandas as pd

INPUT_FOLDER = r"C:\your\folder"
OUTPUT_FILE = "ocr_numbers.xlsx"

# Windows: change this if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

results = []

for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    path = os.path.join(INPUT_FOLDER, filename)

    img = cv2.imread(path)

    if img is None:
        continue

    # Upscale
    img = cv2.resize(
        img,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR - digits only
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"

    text = pytesseract.image_to_string(
        thresh,
        config=config
    )

    # Extract numbers
    numbers = re.findall(r"\d+", text)

    detected = numbers[0] if numbers else ""

    results.append({
        "File Name": filename,
        "Detected Number": detected
    })

df = pd.DataFrame(results)

df.to_excel(OUTPUT_FILE, index=False)

print("Done!")
print(f"Saved to: {OUTPUT_FILE}")

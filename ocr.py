# comic_translate.py
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import re

# ← 請替換為你的 Gemini API KEY
GEMINI_API_KEY = '#####'

def translate_with_gemini(text, api_key, target_lang="Japanese"):
    """
    將輸入的繁體中文 text 翻譯成 target_lang（預設 Japanese），
    僅回傳翻譯後的純文字，不包含任何額外內容。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = f"請將以下繁體中文翻譯成{target_lang}，只輸出翻譯結果，不要有任何解釋或額外文字：{text}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        try:
            translated = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            # 去除所有可能的前導文字，只保留實際翻譯
            clean_result = re.sub(r'^[^a-zA-Z\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]+', '', translated).strip()
            return clean_result
        except (KeyError, IndexError):
            return text
    else:
        print("⚠️ 翻譯失敗:", r.text)
        return text

def remove_text_with_inpainting(img, boxes):
    """
    根據 OCR 回傳的 boxes 四點座標，製作 mask 並 inpaint。
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def draw_translated_text(pil_img, boxes, texts, font_path):
    """
    在 PIL Image 上，依照原本 boxes 的位置貼回翻譯後的文本，
    動態計算字型大小、換行，確保文字置入框內。
    """
    draw = ImageDraw.Draw(pil_img)
    for box, txt in zip(boxes, texts):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        box_w, box_h = x1 - x0, y1 - y0

        font_size = max(int(box_h * 0.8), 12)
        font = ImageFont.truetype(font_path, font_size)

        max_chars = max(int(box_w / (font_size * 0.6)), 1)
        lines = [txt[i:i+max_chars] for i in range(0, len(txt), max_chars)]

        y = y0
        for line in lines:
            draw.text((x0, y), line, font=font, fill=(0, 0, 0))
            y += font_size
    return pil_img

def process_image(image_path, output_path):
    # 讀取圖片
    img = cv2.imread(image_path)
    # OCR 僅偵測繁體中文
    reader = easyocr.Reader(['ch_tra'], gpu=False)
    results = reader.readtext(img)

    boxes, orig_texts = [], []
    for box, text, conf in results:
        if conf > 0.4:
            boxes.append(box)
            orig_texts.append(text)

    # 移除原文文字
    img_clean = remove_text_with_inpainting(img, boxes)

    # 翻譯所有文字
    translated = [translate_with_gemini(t, GEMINI_API_KEY, target_lang="Japanese") for t in orig_texts]

    # 轉為 PIL 進行貼字
    pil_img = Image.fromarray(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB))
    # 日文字型
    font_jp = "NotoSansCJKjp-Regular.otf"

    final = draw_translated_text(pil_img, boxes, translated, font_jp)
    final.save(output_path)
    print(f"✅ 輸出完成：{output_path}")

if __name__ == "__main__":
    process_image("slides_output/slide_01.jpg", "output_episode1.jpg")
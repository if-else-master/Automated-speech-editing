import cv2
import easyocr
import cv2
import numpy as np
import imagehash
from PIL import Image, ImageDraw, ImageFont
import requests 
import re
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

def lock_api():
    global GEMINI_API_KEY
    GEMINI_API_KEY = api_key_entry.get()
    print("🔑 鎖定金鑰:", GEMINI_API_KEY)
    
GEMINI_API_KEY = ""

def extract_slides(video_path, output_dir, frame_interval, hash_diff_threshold, status_label, counter_label):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    prev_hash = None
    slide_index = 0
    output_img = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            curr_hash = imagehash.phash(pil_image)

            if prev_hash is None or abs(curr_hash - prev_hash) > hash_diff_threshold:
                filename = os.path.join(output_dir, f'slide_{slide_index:02d}.jpg')
                pil_image.save(filename)
                process_image(filename,f'ocr_output/out_{output_img:02d}.jpg')
                slide_index += 1
                output_img += 1
                prev_hash = curr_hash
                counter_label.config(text=f"已擷取頁數：{slide_index}")

        frame_count += 1

    cap.release()
    status_label.config(text="🎉 擷取完成！")
    messagebox.showinfo("完成", f"成功擷取 {slide_index} 張投影片！")

def browse_file(entry):
    path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    entry.delete(0, tk.END)
    entry.insert(0, path)

def start_process(video_entry, interval_entry, threshold_entry, status_label, counter_label):
    video_path = video_entry.get()
    if not os.path.isfile(video_path):
        messagebox.showerror("錯誤", "影片路徑錯誤或不存在！")
        return

    try:
        interval = int(interval_entry.get())
        threshold = int(threshold_entry.get())
    except ValueError:
        messagebox.showerror("錯誤", "請輸入有效的數字參數")
        return

    status_label.config(text="🚀 正在擷取中...")
    counter_label.config(text="")

    output_dir = "slides_output"
    threading.Thread(target=extract_slides, args=(video_path, output_dir, interval, threshold, status_label, counter_label)).start()
    
    
## OCR##############################
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




# === GUI 設計 ===
root = tk.Tk()
root.title("投影片擷取器")
root.geometry("500x400")

tk.Label(root, text="Gemini API 金鑰：").pack()
api_key_entry = tk.Entry(root, width=50)
api_key_entry.pack()
tk.Button(root, text="鎖定金鑰", command=lock_api).pack(pady=10)

tk.Label(root, text="🎞️ 選擇影片檔案：").pack()
video_entry = tk.Entry(root, width=50)
video_entry.pack()
tk.Button(root, text="瀏覽...", command=lambda: browse_file(video_entry)).pack()

tk.Label(root, text="🔁 幀間隔 (frame_interval)：").pack()
interval_entry = tk.Entry(root)
interval_entry.insert(0, "15")
interval_entry.pack()

tk.Label(root, text="⚙️ Hash 差異門檻：").pack()
threshold_entry = tk.Entry(root)
threshold_entry.insert(0, "5")
threshold_entry.pack()

status_label = tk.Label(root, text="", fg="blue")
status_label.pack(pady=10)

counter_label = tk.Label(root, text="", fg="green")
counter_label.pack()

tk.Button(root, text="開始擷取", command=lambda: start_process(video_entry, interval_entry, threshold_entry, status_label, counter_label)).pack(pady=10)

root.mainloop()

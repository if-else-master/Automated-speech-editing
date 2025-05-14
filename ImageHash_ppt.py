import cv2
import easyocr
import numpy as np
import imagehash
from PIL import Image, ImageDraw, ImageFont
import requests 
import re
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # 新增，用於標籤頁

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
    if path:
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


## 新增功能：將翻譯好的投影片合成為影片 ##############
def create_translated_video(video_path, translated_slides_dir, output_video_path, frame_interval, hash_diff_threshold, status_label, progress_label):
    """
    將翻譯好的投影片圖片合成到原始影片中
    
    參數:
        video_path: 原始影片路徑
        translated_slides_dir: 翻譯後投影片的目錄
        output_video_path: 輸出影片路徑
        frame_interval: 檢查影格的間隔
        hash_diff_threshold: 判斷幀變化的閾值
        status_label: 狀態標籤
        progress_label: 進度標籤
    """
    # 創建輸出目錄（如果不存在）
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # 讀取所有翻譯好的投影片
    translated_slides = {}
    for filename in os.listdir(translated_slides_dir):
        if filename.startswith('out_') and filename.endswith('.jpg'):
            slide_index = int(filename.split('_')[1].split('.')[0])
            slide_path = os.path.join(translated_slides_dir, filename)
            translated_slides[slide_index] = slide_path
    
    if not translated_slides:
        messagebox.showerror("錯誤", "找不到翻譯後的投影片！")
        status_label.config(text="❌ 找不到翻譯後的投影片")
        return
    
    # 打開原始影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("錯誤", "無法開啟影片檔案！")
        status_label.config(text="❌ 無法開啟影片檔案")
        return
    
    # 取得影片參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定輸出影片編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 初始化幀計數和雜湊值
    frame_count = 0
    prev_hash = None
    current_slide_index = 0
    current_slide = None
    
    # 更新進度標籤
    progress_label.config(text=f"進度: 0/{total_frames} 幀")
    
    # 處理每一幀
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 檢查是否需要更新投影片
        if frame_count % frame_interval == 0:
            # 計算目前幀的哈希值
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            curr_hash = imagehash.phash(pil_image)
            
            # 如果哈希值與前一幀差異較大，則判定為新投影片
            if prev_hash is None or abs(curr_hash - prev_hash) > hash_diff_threshold:
                if current_slide_index in translated_slides:
                    # 讀取對應的翻譯投影片
                    slide_path = translated_slides[current_slide_index]
                    current_slide = cv2.imread(slide_path)
                    # 確保尺寸與影片一致
                    if current_slide.shape[0] != height or current_slide.shape[1] != width:
                        current_slide = cv2.resize(current_slide, (width, height))
                
                current_slide_index += 1
                prev_hash = curr_hash
        
        # 如果有對應的翻譯投影片，則替換當前幀
        if current_slide is not None:
            output_frame = current_slide.copy()
        else:
            output_frame = frame.copy()
        
        # 寫入輸出影片
        out.write(output_frame)
        
        # 更新進度
        if frame_count % 30 == 0:  # 每30幀更新一次進度，以免UI更新過於頻繁
            progress = int((frame_count / total_frames) * 100)
            progress_label.config(text=f"進度: {frame_count}/{total_frames} 幀 ({progress}%)")
        
        frame_count += 1
    
    # 釋放資源
    cap.release()
    out.release()
    
    # 更新完成狀態
    status_label.config(text="✅ 影片合成完成！")
    progress_label.config(text=f"完成處理 {frame_count} 幀")
    messagebox.showinfo("完成", f"成功將翻譯投影片合成到影片中！\n輸出路徑: {output_video_path}")

def browse_video_file(entry):
    """選擇影片檔案"""
    path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def browse_slides_dir(entry):
    """選擇翻譯後投影片目錄"""
    path = filedialog.askdirectory()
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def browse_output_file(entry):
    """選擇輸出影片路徑"""
    path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def start_video_process(video_entry, slides_entry, output_entry, interval_entry, threshold_entry, status_label, progress_label):
    """開始處理影片"""
    video_path = video_entry.get()
    slides_dir = slides_entry.get()
    output_path = output_entry.get()
    
    # 檢查輸入
    if not os.path.isfile(video_path):
        messagebox.showerror("錯誤", "影片路徑錯誤或不存在！")
        return
    
    if not os.path.isdir(slides_dir):
        messagebox.showerror("錯誤", "翻譯投影片目錄不存在！")
        return
    
    if not output_path:
        messagebox.showerror("錯誤", "請指定輸出影片路徑！")
        return
    
    try:
        interval = int(interval_entry.get())
        threshold = int(threshold_entry.get())
    except ValueError:
        messagebox.showerror("錯誤", "請輸入有效的數字參數")
        return
    
    status_label.config(text="🚀 正在處理影片...")
    progress_label.config(text="準備中...")
    
    # 在新執行緒中處理影片，以免阻塞UI
    threading.Thread(target=create_translated_video, 
                     args=(video_path, slides_dir, output_path, interval, threshold, status_label, progress_label)).start()


# === 創建原始功能的標籤頁 ===
def create_extraction_tab(notebook):
    """創建投影片擷取與翻譯的標籤頁"""
    tab = ttk.Frame(notebook)
    
    tk.Label(tab, text="Gemini API 金鑰：").pack()
    api_key_entry = tk.Entry(tab, width=50)
    api_key_entry.pack()
    tk.Button(tab, text="鎖定金鑰", command=lock_api).pack(pady=10)

    tk.Label(tab, text="🎞️ 選擇影片檔案：").pack()
    video_entry = tk.Entry(tab, width=50)
    video_entry.pack()
    tk.Button(tab, text="瀏覽...", command=lambda: browse_file(video_entry)).pack()

    tk.Label(tab, text="🔁 幀間隔 (frame_interval)：").pack()
    interval_entry = tk.Entry(tab)
    interval_entry.insert(0, "15")
    interval_entry.pack()

    tk.Label(tab, text="⚙️ Hash 差異門檻：").pack()
    threshold_entry = tk.Entry(tab)
    threshold_entry.insert(0, "5")
    threshold_entry.pack()

    status_label = tk.Label(tab, text="", fg="blue")
    status_label.pack(pady=10)

    counter_label = tk.Label(tab, text="", fg="green")
    counter_label.pack()

    tk.Button(tab, text="開始擷取", 
              command=lambda: start_process(video_entry, interval_entry, threshold_entry, status_label, counter_label),
              bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)
    
    return tab, api_key_entry

# === 創建影片合成的標籤頁 ===
def create_video_composition_tab(notebook):
    """創建翻譯影片合成的標籤頁"""
    tab = ttk.Frame(notebook)
    
    # 原始影片
    tk.Label(tab, text="🎞️ 選擇原始影片：").pack(pady=(10, 0))
    video_entry = tk.Entry(tab, width=50)
    video_entry.pack()
    tk.Button(tab, text="瀏覽...", command=lambda: browse_video_file(video_entry)).pack()
    
    # 翻譯後投影片目錄
    tk.Label(tab, text="📁 翻譯後投影片目錄：").pack(pady=(10, 0))
    slides_entry = tk.Entry(tab, width=50)
    slides_entry.insert(0, "ocr_output")
    slides_entry.pack()
    tk.Button(tab, text="瀏覽...", command=lambda: browse_slides_dir(slides_entry)).pack()
    
    # 輸出影片路徑
    tk.Label(tab, text="💾 輸出影片路徑：").pack(pady=(10, 0))
    output_entry = tk.Entry(tab, width=50)
    output_entry.insert(0, "output_video.mp4")
    output_entry.pack()
    tk.Button(tab, text="瀏覽...", command=lambda: browse_output_file(output_entry)).pack()
    
    # 參數設定
    params_frame = tk.Frame(tab)
    params_frame.pack(pady=10)
    
    tk.Label(params_frame, text="🔁 幀間隔：").grid(row=0, column=0, padx=5)
    interval_entry = tk.Entry(params_frame, width=10)
    interval_entry.insert(0, "15")
    interval_entry.grid(row=0, column=1, padx=5)
    
    tk.Label(params_frame, text="⚙️ Hash 差異門檻：").grid(row=0, column=2, padx=5)
    threshold_entry = tk.Entry(params_frame, width=10)
    threshold_entry.insert(0, "5")
    threshold_entry.grid(row=0, column=3, padx=5)
    
    # 狀態和進度標籤
    status_label = tk.Label(tab, text="", fg="blue")
    status_label.pack(pady=10)
    
    progress_label = tk.Label(tab, text="", fg="green")
    progress_label.pack()
    
    # 開始處理按鈕
    tk.Button(tab, text="開始合成影片", 
              command=lambda: start_video_process(video_entry, slides_entry, output_entry, 
                                                 interval_entry, threshold_entry,
                                                 status_label, progress_label),
              bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=20)
    
    return tab

# === 主程式 ===
def main():
    global api_key_entry  # 使其成為全域變數，以便在 lock_api 函數中使用
    
    root = tk.Tk()
    root.title("投影片翻譯與影片合成工具")
    root.geometry("600x500")
    
    # 創建選項卡
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # 添加原始功能的標籤頁
    extraction_tab, api_key_entry = create_extraction_tab(notebook)
    notebook.add(extraction_tab, text="投影片擷取與翻譯")
    
    # 添加影片合成功能的標籤頁
    composition_tab = create_video_composition_tab(notebook)
    notebook.add(composition_tab, text="翻譯影片合成")
    
    # 確保輸出目錄存在
    os.makedirs("slides_output", exist_ok=True)
    os.makedirs("ocr_output", exist_ok=True)
    
    root.mainloop()

if __name__ == "__main__":
    main()
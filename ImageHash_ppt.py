import cv2
import imagehash
from PIL import Image
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

def extract_slides(video_path, output_dir, frame_interval, hash_diff_threshold, status_label, counter_label):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    prev_hash = None
    slide_index = 0

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
                slide_index += 1
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

# === GUI 設計 ===
root = tk.Tk()
root.title("投影片擷取器")
root.geometry("500x300")

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

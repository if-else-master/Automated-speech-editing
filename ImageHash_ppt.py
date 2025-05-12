import cv2
import imagehash
from PIL import Image
import os

# === 設定 ===
video_path = 'aa.mp4'
output_dir = 'slides_output'
frame_interval = 15           # 每幾幀檢查一次
hash_diff_threshold = 5       # Hash 差異門檻

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 開啟影片
cap = cv2.VideoCapture(video_path)
frame_count = 0
prev_hash = None
slide_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # 將 BGR 圖片轉為 RGB，再轉為 PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = imagehash.phash(pil_image)

        # 第一次或發生換頁
        if prev_hash is None or abs(curr_hash - prev_hash) > hash_diff_threshold:
            filename = os.path.join(output_dir, f'slide_{slide_index:02d}.jpg')
            pil_image.save(filename)
            print(f"偵測換頁，儲存：{filename}")
            slide_index += 1
            prev_hash = curr_hash

    frame_count += 1

cap.release()
print("完成換頁擷取與儲存。")

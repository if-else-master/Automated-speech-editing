import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import re
import threading
import sys
import subprocess
import numpy as np

# 檢查並安裝必要套件
print("正在檢查必要套件...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy", "pillow", "numpy"])
print("套件檢查完成")

# 直接導入moviepy模組（不使用moviepy.editor）
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from PIL import Image, ImageTk

class VideoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("影片時間段替換工具")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # 設置變數
        self.video_path = ""
        self.image_path = ""
        self.start_time = tk.StringVar()
        self.end_time = tk.StringVar()
        self.output_path = ""
        self.video_duration = 0
        
        # 創建UI元素
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 視頻選擇區域
        video_frame = ttk.LabelFrame(main_frame, text="影片選擇", padding=10)
        video_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame, text="尚未選擇影片")
        self.video_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        video_btn = ttk.Button(video_frame, text="選擇影片", command=self.select_video)
        video_btn.pack(side=tk.RIGHT, padx=5)
        
        # 圖片選擇區域
        image_frame = ttk.LabelFrame(main_frame, text="圖片選擇", padding=10)
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.image_label = ttk.Label(image_frame, text="尚未選擇圖片")
        self.image_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        image_btn = ttk.Button(image_frame, text="選擇圖片", command=self.select_image)
        image_btn.pack(side=tk.RIGHT, padx=5)
        
        # 預覽區域
        preview_frame = ttk.LabelFrame(main_frame, text="預覽", padding=10)
        preview_frame.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # 左側顯示影片預覽圖
        self.video_preview_frame = ttk.Frame(preview_frame)
        self.video_preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_preview_label = ttk.Label(self.video_preview_frame, text="影片預覽")
        self.video_preview_label.pack(pady=5)
        
        self.video_thumbnail = ttk.Label(self.video_preview_frame)
        self.video_thumbnail.pack(fill=tk.BOTH, expand=True)
        
        # 右側顯示圖片預覽
        self.image_preview_frame = ttk.Frame(preview_frame)
        self.image_preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_preview_label = ttk.Label(self.image_preview_frame, text="圖片預覽")
        self.image_preview_label.pack(pady=5)
        
        self.image_thumbnail = ttk.Label(self.image_preview_frame)
        self.image_thumbnail.pack(fill=tk.BOTH, expand=True)
        
        # 時間區域
        time_frame = ttk.LabelFrame(main_frame, text="時間設定", padding=10)
        time_frame.pack(fill=tk.X, padx=5, pady=5)
        
        start_label = ttk.Label(time_frame, text="開始時間 (分:秒):")
        start_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        start_entry = ttk.Entry(time_frame, textvariable=self.start_time, width=10)
        start_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        end_label = ttk.Label(time_frame, text="結束時間 (分:秒):")
        end_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        end_entry = ttk.Entry(time_frame, textvariable=self.end_time, width=10)
        end_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # 影片資訊顯示
        self.video_info_label = ttk.Label(time_frame, text="影片長度: 0:00")
        self.video_info_label.grid(row=0, column=4, padx=20, pady=5, sticky=tk.E)
        
        # 輸出區域
        output_frame = ttk.LabelFrame(main_frame, text="輸出設定", padding=10)
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.output_label = ttk.Label(output_frame, text="輸出檔案位置: 未設定")
        self.output_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        output_btn = ttk.Button(output_frame, text="選擇輸出位置", command=self.select_output)
        output_btn.pack(side=tk.RIGHT, padx=5)
        
        # 進度條
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="就緒")
        self.status_label.pack(pady=5)
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        process_btn = ttk.Button(button_frame, text="開始處理", command=self.start_processing)
        process_btn.pack(side=tk.RIGHT, padx=5)
    
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=os.path.basename(file_path))
            self.update_video_info()
            self.load_video_thumbnail()
    
    def update_video_info(self):
        # 使用MoviePy獲取影片資訊
        video = VideoFileClip(self.video_path)
        duration = video.duration
        self.video_duration = duration
        
        # 格式化顯示時間
        mins = int(duration // 60)
        secs = int(duration % 60)
        self.video_info_label.config(text=f"影片長度: {mins}:{secs:02d}")
        
        video.close()
    
    def load_video_thumbnail(self):
        # 使用MoviePy讀取影片的第一幀作為縮略圖
        video = VideoFileClip(self.video_path)
        thumb = video.get_frame(0)  # 獲取第一幀
        
        # 轉換為PIL圖片
        pil_img = Image.fromarray(thumb)
        
        # 調整大小以適應UI
        pil_img = self.resize_image(pil_img, 300)
        
        # 轉換為Tkinter圖片
        tk_img = ImageTk.PhotoImage(pil_img)
        self.video_thumbnail.config(image=tk_img)
        self.video_thumbnail.image = tk_img  # 保持引用
        
        video.close()
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="選擇圖片檔案",
            filetypes=[("圖片檔案", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=os.path.basename(file_path))
            self.load_image_thumbnail()
    
    def load_image_thumbnail(self):
        # 使用PIL讀取圖片
        pil_img = Image.open(self.image_path)
        
        # 調整大小以適應UI
        pil_img = self.resize_image(pil_img, 300)
        
        # 轉換為Tkinter圖片
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_thumbnail.config(image=tk_img)
        self.image_thumbnail.image = tk_img  # 保持引用
    
    def resize_image(self, img, max_size):
        # 等比例調整圖片大小
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def select_output(self):
        default_filename = "output.mp4"
        if self.video_path:
            # 從原始影片檔案名生成預設輸出檔案名
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            default_filename = f"{base_name}_edited.mp4"
        
        file_path = filedialog.asksaveasfilename(
            title="選擇輸出位置",
            defaultextension=".mp4",
            initialfile=default_filename,
            filetypes=[("MP4 檔案", "*.mp4")]
        )
        
        if file_path:
            self.output_path = file_path
            self.output_label.config(text=f"輸出檔案位置: {os.path.basename(file_path)}")
    
    def parse_time(self, time_str):
        # 解析時間字符串 (分:秒)
        pattern = r"(\d+):(\d+)"
        match = re.match(pattern, time_str)
        
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
        else:
            # 如果無法解析，嘗試直接轉換為秒數
            try:
                return float(time_str)
            except ValueError:
                return None
    
    def validate_times(self):
        # 檢查時間格式和範圍
        start_time_str = self.start_time.get()
        end_time_str = self.end_time.get()
        
        start_seconds = self.parse_time(start_time_str)
        end_seconds = self.parse_time(end_time_str)
        
        # 不使用錯誤處理，直接進行條件檢查
        if start_seconds is None or end_seconds is None:
            # 不顯示消息框，直接返回False
            return False
        
        if start_seconds >= end_seconds:
            return False
        
        if start_seconds < 0 or end_seconds > self.video_duration:
            return False
        
        return True
    
    def start_processing(self):
        # 檢查是否已選擇所有必要的檔案和設定
        if not self.video_path or not self.image_path or not self.output_path:
            return
        
        if not self.validate_times():
            return
        
        # 在新線程中處理影片，避免阻塞UI
        threading.Thread(target=self.process_video).start()
    
    def process_video(self):
        # 更新UI
        self.status_label.config(text="正在處理影片...")
        self.progress['value'] = 0
        self.root.update()
        
        # 解析時間
        start_seconds = self.parse_time(self.start_time.get())
        end_seconds = self.parse_time(self.end_time.get())
        
        # 讀取影片
        self.progress['value'] = 10
        self.root.update()
        
        video = VideoFileClip(self.video_path)
        
        # 讀取圖片
        self.progress['value'] = 20
        self.root.update()
        
        # 使用moviepy的ImageClip直接處理
        pil_img = Image.open(self.image_path)
        image = ImageClip(np.array(pil_img))
        image = image.with_duration(end_seconds - start_seconds)
        
        # 設定圖片位置和大小
        image = image.resized(width=video.w)  # 調整圖片寬度與影片相同
        image = image.with_position(('center', 'center'))  # 將圖片置中
        
        # 創建第一部分 (0 到 start_seconds)
        self.progress['value'] = 30
        self.root.update()
        
        if start_seconds > 0:
            part1 = video.subclipped(0, start_seconds)
        else:
            part1 = None
        
        # 創建替換部分
        self.progress['value'] = 40
        self.root.update()
        
        # 使用將圖片添加至原始視頻片段上的方式
        image = image.with_start(start_seconds)
        
        # 創建第三部分 (end_seconds 到 end)
        self.progress['value'] = 50
        self.root.update()
        
        if end_seconds < video.duration:
            part3 = video.subclipped(end_seconds, video.duration)
        else:
            part3 = None
        
        # 組合所有部分
        self.progress['value'] = 60
        self.root.update()
        
        # 創建合成影片
        clips = [video]  # 基礎影片
        clips.append(image)  # 添加圖片覆蓋層
        
        final_clip = CompositeVideoClip(clips)
        
        # 輸出最終影片
        self.progress['value'] = 70
        self.root.update()
        
        self.status_label.config(text="正在輸出影片...")
        
        final_clip.write_videofile(
            self.output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            preset='medium',
            threads=4,
            logger=None  # 禁用MoviePy的控制台輸出
        )
        
        # 清理
        self.progress['value'] = 90
        self.root.update()
        
        video.close()
        final_clip.close()
        
        # 完成
        self.progress['value'] = 100
        self.status_label.config(text=f"處理完成! 已儲存至 {os.path.basename(self.output_path)}")
        
        messagebox.showinfo("成功", f"影片處理完成!\n已儲存至 {self.output_path}")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorApp(root)
    root.mainloop()
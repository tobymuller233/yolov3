import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

def highlight_suppression(img, threshold=0.9, decay_factor=0.5, smoothness=0.2):
    """
    改进版高光区域抑制算法（处理速度优化）
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:,:,2] / 255.0
    
    # 向量化运算优化
    over = np.clip((v - threshold) / (1 - threshold), 0, 1)
    weight = 1 - (1 - decay_factor) * (1 - np.exp(-over/smoothness))
    adjusted_v = v * (1 - over * decay_factor) + v * over * weight
    
    hsv[:,:,2] = np.clip(adjusted_v * 255, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 使用快速锐化核
    return cv2.filter2D(result, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))

def process_entire_video(input_path, output_path, threshold=0.85, crf=18, decay=0.6, smooth=0.15):
    """
    全视频处理函数
    :param crf: 输出视频质量 (0-51, 越小质量越高)
    """
    # 输入视频检查
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入视频不存在: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件，可能是不支持的格式")
    
    # 获取视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 配置视频编码器
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建输出视频文件，检查路径权限和编码器")
    
    try:
        # 进度条显示
        pbar = tqdm(total=total_frames, desc="处理进度", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理当前帧
            processed = highlight_suppression(frame, 
                                            threshold=threshold,
                                            decay_factor=decay,
                                            smoothness=smooth)
            
            # 写入视频
            writer.write(processed)
            pbar.update(1)
        
        pbar.close()
        print(f"处理完成！输出视频已保存至: {output_path}")
    
    finally:
        cap.release()
        writer.release()
        # 使用FFmpeg二次编码提升压缩率（可选）
        if os.path.exists(output_path):
            os.system(f"ffmpeg -y -i {output_path} -c:v libx264 -crf {crf} -preset fast {output_path}_compressed.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频高光抑制处理器")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", required=True, help="输出视频路径")
    parser.add_argument("-t", "--threshold", type=float, default=0.85,
                       help="高光阈值 (0-1, 默认0.85)")
    parser.add_argument("-d", "--decay", type=float, default=0.6,
                       help="衰减强度 (0-1, 默认0.6)")
    parser.add_argument("-s", "--smooth", type=float, default=0.15,
                       help="平滑度 (0-1, 默认0.15)")
    parser.add_argument("--crf", type=int, default=18,
                       help="输出视频质量 (0-51, 默认18)")
    
    args = parser.parse_args()
    
    try:
        process_entire_video(
            input_path=args.input,
            output_path=args.output,
            threshold=args.threshold,
            crf=args.crf,
            decay=args.decay,
            smooth=args.smooth
        )
    except Exception as e:
        print(f"处理失败: {str(e)}")

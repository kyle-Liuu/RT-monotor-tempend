import asyncio
import websockets
import cv2
import base64
import json
from ultralytics import YOLO
import torch

# 检查GPU是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载YOLO模型并移至GPU
model = YOLO('yolo11n.pt')
model.to(device)

# 跳帧设置
SKIP_FRAMES = 2  # 每3帧处理1帧
frame_count = 0

def draw_detections(frame, detections):
    """在后端绘制检测框，使用四角标记样式"""
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        cls = det['class']
        
        # 计算四个角的坐标
        corners = [
            (int(x1), int(y1)),  # 左上
            (int(x2), int(y1)),  # 右上
            (int(x1), int(y2)),  # 左下
            (int(x2), int(y2))   # 右下
        ]
        
        # 绘制四个角的标记
        corner_length = 20  # 角标记的长度
        thickness = 1      # 线条粗细
        color = (0, 0, 255)  # 红色
        
        # 左上角
        cv2.line(frame, corners[0], (corners[0][0] + corner_length, corners[0][1]), color, thickness)
        cv2.line(frame, corners[0], (corners[0][0], corners[0][1] + corner_length), color, thickness)
        
        # 右上角
        cv2.line(frame, corners[1], (corners[1][0] - corner_length, corners[1][1]), color, thickness)
        cv2.line(frame, corners[1], (corners[1][0], corners[1][1] + corner_length), color, thickness)
        
        # 左下角
        cv2.line(frame, corners[2], (corners[2][0] + corner_length, corners[2][1]), color, thickness)
        cv2.line(frame, corners[2], (corners[2][0], corners[2][1] - corner_length), color, thickness)
        
        # 右下角
        cv2.line(frame, corners[3], (corners[3][0] - corner_length, corners[3][1]), color, thickness)
        cv2.line(frame, corners[3], (corners[3][0], corners[3][1] - corner_length), color, thickness)
        
        # 绘制标签
        label = f"{cls} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # 绘制标签文本
        cv2.putText(frame, label,
                    (corners[0][0], corners[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

async def process_video(websocket):
    global frame_count
    # 打开视频流
    cap = cv2.VideoCapture("rtsp://192.168.1.74:9554/live/test")
    # cap = cv2.VideoCapture("rtsp://admin:kaoe.robot@192.168.1.169:554/Streaming/Channels/101")
    
    # 设置缓冲区大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 跳帧处理
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue
            
        # 调整图像大小以提高性能
        frame = cv2.resize(frame, (640, 480))
        
        # YOLO检测
        results = model(frame, device=device, conf=0.35)
        
        # 处理检测结果
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': model.names[cls]
                })
        
        # 在后端绘制检测框
        frame = draw_detections(frame, detections)
        
        # 将处理后的帧转换为base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 发送数据
        await websocket.send(json.dumps({
            'type': 'frame',
            'frame': frame_base64
        }))
        
        # 控制帧率
        await asyncio.sleep(0.03)  # 约30FPS

async def main():
    # 启动检测视频流的WebSocket服务器
    async with websockets.serve(process_video, "localhost", 8000):
        print("检测视频流WebSocket服务器启动在 ws://localhost:8000")
        await asyncio.Future()  # 运行永久

if __name__ == "__main__":
    asyncio.run(main())
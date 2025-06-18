from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import subprocess
import threading
import time
import torch
from ultralytics import YOLO
import queue
import numpy as np

app = Flask(__name__)
CORS(app)

# --- Configuration ---
input_rtsp_url = 'rtsp://192.168.1.74:9554/live/test'
output_rtsp_url = "rtsp://192.168.1.161:554/live/con54351"
FFMPEG_BIN = "ffmpeg"

# --- YOLO模型配置 ---
YOLO_MODELS = {
    'detect': YOLO('yolov8n.pt'),           # 目标检测
    'obb': YOLO('yolov8n-obb.pt'),          # 旋转边界框检测
    'pose': YOLO('yolov8n-pose.pt'),        # 姿态估计
    'segment': YOLO('yolov8n-seg.pt'),      # 分割模型
    'classify': YOLO('yolov8n-cls.pt'),     # 分类模型
}

# 当前活跃的任务类型
CURRENT_TASKS = ['detect']  # 默认只开启检测，可以同时开启多个任务

# 优化后的帧跳过配置
FRAME_SKIP_CONFIG = {
    'detect': 5,      # 检测相对轻量
    'obb': 3,         # OBB检测
    'pose': 3,        # 姿态检测
    'segment': 5,     # 分割计算量大，跳过更多帧
    'classify': 6,    # 分类任务跳过最多帧
}

TOKEN = "123456"

# 设备信息
data = [
    {
        "deviceid": "cam54351",
        "devicename": "模拟摄像头A",
        "groupid": "0",
        "groupname": "未分组设备",
        "wsurl": "",
        "online": False,
        "started": False,
        "tasks": CURRENT_TASKS
    }
]

# 多线程相关
input_frame_queue = queue.Queue(maxsize=30)
output_frame_queue = queue.Queue(maxsize=30)
stop_event = threading.Event()
reader_thread = None
detector_thread = None
writer_thread = None

# --- 优化后的多任务检测可视化函数 ---

def draw_detection_results(frame, results, model, task_type):
    """根据不同任务类型绘制检测结果"""
    
    if task_type == 'detect':
        return draw_detection_boxes(frame, results, model)
    elif task_type == 'segment':
        return draw_segmentation_masks(frame, results, model)
    elif task_type == 'obb':
        return draw_obb_boxes(frame, results, model)
    elif task_type == 'pose':
        return draw_pose_keypoints(frame, results, model)
    elif task_type == 'classify':
        return draw_classification_results(frame, results, model)
    
    return frame

def draw_detection_boxes(frame, results, model):
    """绘制检测边界框"""
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, clss):
            if conf > 0.5:  # 添加置信度阈值
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def draw_segmentation_masks(frame, results, model):
    """优化后的分割掩码绘制 - 减少计算量"""
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        
        # 预定义颜色，避免每次随机生成
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        for i, (mask, box, conf, cls) in enumerate(zip(masks, boxes, confs, clss)):
            if conf > 0.5:  # 只处理高置信度的结果
                # 只在边界框区域内处理掩码，减少计算量
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # 调整掩码尺寸到边界框大小而不是整个图像
                roi_height, roi_width = y2 - y1, x2 - x1
                if roi_height > 0 and roi_width > 0:
                    mask_resized = cv2.resize(mask, (roi_width, roi_height))
                    mask_bool = mask_resized > 0.5
                    
                    # 应用掩码到ROI区域
                    color = colors[i % len(colors)]
                    roi = frame[y1:y2, x1:x2]
                    roi[mask_bool] = roi[mask_bool] * 0.7 + np.array(color) * 0.3
                    
                    # 绘制边界框和标签
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def draw_obb_boxes(frame, results, model):
    """绘制旋转边界框(OBB)"""
    if results[0].obb is not None:
        obb_boxes = results[0].obb.xyxyxyxy.cpu().numpy()
        confs = results[0].obb.conf.cpu().numpy()
        clss = results[0].obb.cls.cpu().numpy()
        
        for obb, conf, cls in zip(obb_boxes, confs, clss):
            if conf > 0.5:  # 添加置信度阈值
                points = obb.reshape(4, 2).astype(int)
                cv2.polylines(frame, [points], True, (255, 0, 0), 2)
                
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

def draw_pose_keypoints_optimized(frame, results, model, detection_results=None):
    """优化后的姿态关键点绘制 - 先检测人体再做姿态估计"""
    
    # 如果有检测结果，只在检测到人的区域做姿态估计
    if detection_results is not None:
        person_boxes = []
        if detection_results[0].boxes is not None:
            boxes = detection_results[0].boxes.xyxy.cpu().numpy()
            confs = detection_results[0].boxes.conf.cpu().numpy()
            clss = detection_results[0].boxes.cls.cpu().numpy()
            
            # 只保留人类检测结果（COCO数据集中人类的类别ID是0）
            for box, conf, cls in zip(boxes, confs, clss):
                if int(cls) == 0 and conf > 0.5:  # 人类检测
                    person_boxes.append(box)
        
        # 如果没有检测到人，直接返回原图
        if not person_boxes:
            return frame
    
    # 绘制姿态关键点
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf.cpu().numpy()
        
        # 简化的骨架连接（只保留主要连接）
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 头部和躯干
            [6, 12], [7, 13], [6, 8], [7, 9], [8, 10], [9, 11]  # 主要肢体
        ]
        
        for person_kpts, person_conf in zip(keypoints, confs):
            # 绘制关键点（只绘制高置信度的）
            for i, (kpt, conf) in enumerate(zip(person_kpts, person_conf)):
                if conf > 0.6:  # 提高置信度阈值
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            
            # 绘制简化的骨架连接线
            for connection in skeleton:
                kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1
                if kpt1_idx < len(person_kpts) and kpt2_idx < len(person_kpts):
                    if person_conf[kpt1_idx] > 0.6 and person_conf[kpt2_idx] > 0.6:
                        pt1 = tuple(map(int, person_kpts[kpt1_idx]))
                        pt2 = tuple(map(int, person_kpts[kpt2_idx]))
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
    
    return frame

def draw_pose_keypoints(frame, results, model):
    """原始姿态关键点绘制函数的包装"""
    return draw_pose_keypoints_optimized(frame, results, model)

def draw_classification_results(frame, results, model):
    """优化后的分类结果绘制 - 只显示top3"""
    if results[0].probs is not None:
        probs = results[0].probs.data.cpu().numpy()
        top3_indices = np.argsort(probs)[-3:][::-1]  # 只显示top3
        
        # 在图像右上角显示分类结果
        y_offset = 30
        for i, idx in enumerate(top3_indices):
            prob = probs[idx]
            if prob > 0.05:  # 提高显示阈值
                label = f"{model.names[idx]}: {prob:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                x_pos = frame.shape[1] - text_size[0] - 10  # 右对齐
                cv2.putText(frame, label, (x_pos, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return frame

# --- 优化后的多任务检测线程 ---

def multi_task_yolo_detector_thread(models_dict, active_tasks):
    """优化后的多任务YOLO检测线程"""
    print(f"多任务YOLO检测线程启动，活跃任务: {active_tasks}")
    
    # 为每个任务单独记录帧计数
    frame_counts = {task: 0 for task in active_tasks}
    last_results = {task: None for task in active_tasks}
    
    # 预热GPU（如果可用）
    if torch.cuda.is_available():
        for task in active_tasks:
            if task in models_dict:
                models_dict[task].cuda()
        print("GPU预热完成")

    while not stop_event.is_set():
        try:
            frame = input_frame_queue.get(timeout=1)
        except queue.Empty:
            if stop_event.is_set():
                break
            time.sleep(0.01)
            continue

        output_frame = frame.copy()
        
        # 检测结果用于优化姿态估计
        detection_results = None
        
        # 按任务处理，使用不同的帧跳过策略
        for task in active_tasks:
            frame_counts[task] += 1
            skip_frames = FRAME_SKIP_CONFIG.get(task, 3)
            
            # 按配置跳帧执行检测
            if frame_counts[task] % skip_frames == 0:
                try:
                    if task in models_dict:
                        model = models_dict[task]
                        
                        # 对于分类任务，降低输入分辨率
                        if task == 'classify':
                            # 缩小图像尺寸以加速分类
                            small_frame = cv2.resize(frame, (224, 224))
                            results = model(small_frame, verbose=False)
                        else:
                            results = model(frame, verbose=False)
                        
                        last_results[task] = results
                        
                        # 保存检测结果用于姿态估计优化
                        if task == 'detect':
                            detection_results = results
                        
                        # 绘制当前任务的检测结果
                        if task == 'pose' and detection_results is not None:
                            # 使用优化后的姿态检测
                            output_frame = draw_pose_keypoints_optimized(output_frame, results, model, detection_results)
                        else:
                            output_frame = draw_detection_results(output_frame, results, model, task)
                        
                except Exception as e:
                    print(f"{task}任务检测错误: {e}")
            else:
                # 使用缓存结果绘制
                if last_results[task] is not None:
                    try:
                        model = models_dict[task]
                        if task == 'pose' and 'detect' in active_tasks and last_results.get('detect') is not None:
                            # 姿态估计使用检测结果优化
                            output_frame = draw_pose_keypoints_optimized(output_frame, last_results[task], model, last_results['detect'])
                        else:
                            output_frame = draw_detection_results(output_frame, last_results[task], model, task)
                    except Exception as e:
                        print(f"{task}任务绘制错误: {e}")

        # 添加性能信息显示
        task_info = f"Tasks: {', '.join(active_tasks)}"
        fps_info = f"Skip: {', '.join([f'{t}:{FRAME_SKIP_CONFIG.get(t,3)}' for t in active_tasks])}"
        
        cv2.putText(output_frame, task_info, (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, fps_info, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        try:
            output_frame_queue.put(output_frame, timeout=1)
        except queue.Full:
            print("多任务检测器: 输出队列满，丢弃处理帧")
        
        input_frame_queue.task_done()

    print("多任务YOLO检测线程停止")
    stop_event.set()

# --- 其他线程函数保持不变 ---
def frame_reader_thread(rtsp_url_in):
    """帧读取线程"""
    cap = cv2.VideoCapture(rtsp_url_in)
    if not cap.isOpened():
        print(f"错误: 无法打开视频流 {rtsp_url_in}")
        stop_event.set()
        return

    print(f"帧读取线程启动: {rtsp_url_in}")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("帧读取器: 流结束或读取错误，尝试重连...")
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(rtsp_url_in)
            if not cap.isOpened():
                print(f"错误: 重连失败 {rtsp_url_in}")
                break
            continue

        try:
            input_frame_queue.put(frame, timeout=1)
        except queue.Full:
            print("帧读取器: 输入队列满，丢弃帧")
        time.sleep(0.001)

    cap.release()
    print("帧读取线程停止")
    stop_event.set()

def ffmpeg_writer_thread(output_rtsp_url, frame_width, frame_height, frame_rate):
    """FFmpeg写入线程"""
    print(f"FFmpeg写入线程启动: {output_rtsp_url}")
    ffmpeg_command = [
        FFMPEG_BIN, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{frame_width}x{frame_height}',
        '-r', str(frame_rate), '-i', '-', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-preset', 'ultrafast',
        '-tune', 'zerolatency', '-f', 'rtsp', output_rtsp_url
    ]
    
    pipe = None
    try:
        pipe = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"FFmpeg命令: {' '.join(ffmpeg_command)}")

        while not stop_event.is_set():
            try:
                frame = output_frame_queue.get(timeout=1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                time.sleep(0.01)
                continue
            try:
                pipe.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as e:
                print(f"FFmpeg写入器: 管道错误 {e}")
                break
            except Exception as e:
                print(f"FFmpeg写入器: 写入错误 {e}")
                break
            output_frame_queue.task_done()
    except Exception as e:
        print(f"FFmpeg写入器: 启动失败 {e}")
    finally:
        if pipe:
            if pipe.stdin:
                try:
                    pipe.stdin.close()
                except BrokenPipeError:
                    pass
            pipe.wait(timeout=5)
            if pipe.poll() is None:
                pipe.terminate()
                pipe.wait(timeout=2)
                if pipe.poll() is None:
                    pipe.kill()
        print("FFmpeg写入线程停止")
        stop_event.set()

# --- API端点 ---

@app.route('/get_devList', methods=['GET'])
def get_data():
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401
    return jsonify(data)

@app.route('/set_tasks', methods=['POST'])
def set_tasks():
    """设置要执行的YOLO任务"""
    global CURRENT_TASKS
    
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401
    
    request_data = request.get_json()
    if not request_data or 'tasks' not in request_data:
        return jsonify({"error": "请提供tasks参数"}), 400
    
    new_tasks = request_data['tasks']
    valid_tasks = ['detect', 'segment', 'obb', 'pose', 'classify']
    
    # 验证任务有效性
    invalid_tasks = [task for task in new_tasks if task not in valid_tasks]
    if invalid_tasks:
        return jsonify({"error": f"无效任务: {invalid_tasks}"}), 400
    
    CURRENT_TASKS = new_tasks
    
    # 更新设备信息
    for d in data:
        if d["deviceid"] == "cam54351":
            d["tasks"] = CURRENT_TASKS
    
    return jsonify({"status": f"任务已设置为: {CURRENT_TASKS}，跳帧配置: {FRAME_SKIP_CONFIG}"})

@app.route('/get_performance_config', methods=['GET'])
def get_performance_config():
    """获取性能配置信息"""
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401
    
    return jsonify({
        "frame_skip_config": FRAME_SKIP_CONFIG,
        "current_tasks": CURRENT_TASKS,
        "gpu_available": torch.cuda.is_available()
    })

@app.route('/set_frame_skip', methods=['POST'])
def set_frame_skip():
    """设置帧跳过配置"""
    global FRAME_SKIP_CONFIG
    
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401
    
    request_data = request.get_json()
    if not request_data or 'config' not in request_data:
        return jsonify({"error": "请提供config参数"}), 400
    
    new_config = request_data['config']
    
    # 验证配置
    valid_tasks = ['detect', 'segment', 'obb', 'pose', 'classify']
    for task, skip_frames in new_config.items():
        if task not in valid_tasks:
            return jsonify({"error": f"无效任务: {task}"}), 400
        if not isinstance(skip_frames, int) or skip_frames < 1:
            return jsonify({"error": f"跳帧数必须是正整数: {task}={skip_frames}"}), 400
    
    FRAME_SKIP_CONFIG.update(new_config)
    
    return jsonify({"status": f"帧跳过配置已更新: {FRAME_SKIP_CONFIG}"})

@app.route('/start_multi_yolo', methods=['POST'])
def start_multi_yolo():
    """启动多任务YOLO检测"""
    global reader_thread, detector_thread, writer_thread
    
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401

    # 检查是否已运行
    if (reader_thread and reader_thread.is_alive() and 
        detector_thread and detector_thread.is_alive() and 
        writer_thread and writer_thread.is_alive()):
        return jsonify({"status": "多任务YOLO检测已在运行"}), 200

    # 更新设备状态
    for d in data:
        if d["deviceid"] == "cam54351":
            d["wsurl"] = output_rtsp_url.replace("rtsp://", "ws://").replace(":554/", ":80/") + ".live.mp4"
            d["online"] = True
            d["started"] = True
            d["tasks"] = CURRENT_TASKS

    # 获取流属性
    temp_cap = cv2.VideoCapture(input_rtsp_url)
    if not temp_cap.isOpened():
        return jsonify({"status": "错误: 无法打开输入流"}), 500

    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = temp_cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        frame_rate = 25
    temp_cap.release()

    # 清理队列和重置事件
    with input_frame_queue.mutex:
        input_frame_queue.queue.clear()
    with output_frame_queue.mutex:
        output_frame_queue.queue.clear()
    stop_event.clear()

    # 启动线程
    reader_thread = threading.Thread(target=frame_reader_thread, args=(input_rtsp_url,), name="FrameReader")
    detector_thread = threading.Thread(target=multi_task_yolo_detector_thread, args=(YOLO_MODELS, CURRENT_TASKS), name="MultiYOLODetector")
    writer_thread = threading.Thread(target=ffmpeg_writer_thread, args=(output_rtsp_url, frame_width, frame_height, frame_rate), name="FFmpegWriter")

    reader_thread.daemon = True
    detector_thread.daemon = True
    writer_thread.daemon = True

    reader_thread.start()
    detector_thread.start()
    writer_thread.start()

    print(f"多任务YOLO检测线程启动，任务: {CURRENT_TASKS}")
    return jsonify({
        "status": f"多任务YOLO检测已启动", 
        "tasks": CURRENT_TASKS,
        "frame_skip_config": FRAME_SKIP_CONFIG,
        "gpu_enabled": torch.cuda.is_available()
    })

@app.route('/stop_multi_yolo', methods=['POST'])
def stop_multi_yolo():
    """停止多任务YOLO检测"""
    global reader_thread, detector_thread, writer_thread
    
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401

    if not stop_event.is_set():
        stop_event.set()
        print("已发送停止信号给多任务YOLO检测线程")
        time.sleep(2)

    # 更新设备状态
    for d in data:
        if d["deviceid"] == "cam54351":
            d["online"] = False
            d["started"] = False

    return jsonify({"status": "多任务YOLO检测正在停止"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=False)

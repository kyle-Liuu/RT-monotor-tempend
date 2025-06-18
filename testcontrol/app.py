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
import random
import string

app = Flask(__name__)
CORS(app)

# --- Configuration ---
FFMPEG_BIN = "ffmpeg"
TOKEN = "123456" # Simple token for API authentication

# --- In-memory Databases ---
# stream_db stores camera information and their streaming status
stream_db = {}
# control_db stores control (AI task) configurations linked to devices
control_db = {}

# --- YOLO Model Configuration (from mutliyolo.py) ---
YOLO_MODELS = {
    'detect': YOLO('yolov8n.pt'),           # 目标检测
    'obb': YOLO('yolov8n-obb.pt'),          # 旋转边界框检测
    'pose': YOLO('yolov8n-pose.pt'),        # 姿态估计
    'segment': YOLO('yolov8n-seg.pt'),      # 分割模型
    'classify': YOLO('yolov8n-cls.pt'),     # 分类模型
}

# Mapping of control_tag integers to YOLO task names
CONTROL_TAG_MAP = {
    1: 'detect',
    2: 'segment',
    3: 'obb',
    4: 'pose',
    5: 'classify'
}
REVERSE_CONTROL_TAG_MAP = {v: k for k, v in CONTROL_TAG_MAP.items()}


# --- Global state for the single active YOLO stream (inherent limitation from mutliyolo.py structure) ---
current_active_stream = None # Stores the device_id of the currently active stream
current_active_control = None # Stores the control_id of the currently active control

# Multi-threading related for the SINGLE active stream
input_frame_queue = queue.Queue(maxsize=30)
output_frame_queue = queue.Queue(maxsize=30)
stop_event = threading.Event()
reader_thread = None
detector_thread = None
writer_thread = None

# Frame skip configuration (from mutliyolo.py)
FRAME_SKIP_CONFIG = {
    'detect': 3,
    'obb': 3,
    'pose': 3,
    'segment': 5,
    'classify': 6,
}


# --- Helper Functions (from mutliyolo.py) ---

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


# --- Multi-task YOLO Detector Thread (from mutliyolo.py, modified to use passed active_tasks) ---

def multi_task_yolo_detector_thread(models_dict, active_tasks):
    """优化后的多任务YOLO检测线程"""
    print(f"多任务YOLO检测线程启动，活跃任务: {active_tasks}")
    
    frame_counts = {task: 0 for task in active_tasks}
    last_results = {task: None for task in active_tasks}
    
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
        
        detection_results = None
        
        for task in active_tasks:
            frame_counts[task] += 1
            skip_frames = FRAME_SKIP_CONFIG.get(task, 3)
            
            if frame_counts[task] % skip_frames == 0:
                try:
                    if task in models_dict:
                        model = models_dict[task]
                        
                        if task == 'classify':
                            small_frame = cv2.resize(frame, (224, 224))
                            results = model(small_frame, verbose=False)
                        else:
                            results = model(frame, verbose=False)
                        
                        last_results[task] = results
                        
                        if task == 'detect':
                            detection_results = results
                        
                        if task == 'pose' and detection_results is not None:
                            output_frame = draw_pose_keypoints_optimized(output_frame, results, model, detection_results)
                        else:
                            output_frame = draw_detection_results(output_frame, results, model, task)
                        
                except Exception as e:
                    print(f"{task}任务检测错误: {e}")
            else:
                if last_results[task] is not None:
                    try:
                        model = models_dict[task]
                        if task == 'pose' and 'detect' in active_tasks and last_results.get('detect') is not None:
                            output_frame = draw_pose_keypoints_optimized(output_frame, last_results[task], model, last_results['detect'])
                        else:
                            output_frame = draw_detection_results(output_frame, last_results[task], model, task)
                    except Exception as e:
                        print(f"{task}任务绘制错误: {e}")

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

# --- Other Thread Functions (from mutliyolo.py) ---
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


# --- Utility Functions ---
def generate_unique_id(prefix, existing_ids_set):
    """Generates a unique ID with a given prefix and 7 random digits."""
    while True:
        suffix = ''.join(random.choices(string.digits, k=7))
        new_id = prefix + suffix
        if new_id not in existing_ids_set:
            return new_id

def authenticate_token(token):
    if token != TOKEN:
        return False
    return True

def stop_current_yolo_stream():
    """Stops the currently running YOLO stream and updates status."""
    global reader_thread, detector_thread, writer_thread, current_active_stream, current_active_control
    if not stop_event.is_set():
        stop_event.set()
        print("Stopping current YOLO stream...")
        time.sleep(2) # Give threads some time to react

        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=5)
        if detector_thread and detector_thread.is_alive():
            detector_thread.join(timeout=5)
        if writer_thread and writer_thread.is_alive():
            writer_thread.join(timeout=5)
        
        # Clear queues
        with input_frame_queue.mutex:
            input_frame_queue.queue.clear()
        with output_frame_queue.mutex:
            output_frame_queue.queue.clear()
        
        print("Current YOLO stream stopped.")
    
    # Update status in DB
    if current_active_stream and current_active_stream in stream_db:
        stream_db[current_active_stream]['status'] = 'offline'
    if current_active_control and current_active_control in control_db:
        control_db[current_active_control]['status'] = 'standby'

    current_active_stream = None
    current_active_control = None
    stop_event.clear() # Reset event for future runs


# --- API Endpoints ---

@app.route('/get_devList', methods=['GET'])
def get_devList():
    """Backend's version of /get_devList for compatibility with original code if needed"""
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    # This might be used by the original mutliyolo.py's internal data.
    # For external use, we'll expose a more structured get_cameras.
    return jsonify([stream_db[device_id] for device_id in stream_db])

@app.route('/api/add_camera', methods=['POST'])
def add_camera():
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    device_name = data.get('device_name')
    pull_stream_type = data.get('pull_stream_type')
    pull_stream_url = data.get('pull_stream_url')
    device_remark = data.get('device_remark', '')
    is_audio = data.get('is_audio', 0)

    if not all([device_name, pull_stream_type, pull_stream_url]):
        return jsonify({"error": "Missing required fields: device_name, pull_stream_type, pull_stream_url"}), 400
    
    device_id = generate_unique_id("cam", set(stream_db.keys()))
    
    stream_db[device_id] = {
        "device_id": device_id,
        "device_name": device_name,
        "pull_stream_type": pull_stream_type,
        "pull_stream_url": pull_stream_url,
        "device_remark": device_remark,
        "is_audio": is_audio,
        "status": "offline", # Initial status
        "output_rtsp_url": f"rtsp://localhost:554/live/{device_id}" # Placeholder, actual depends on ffmpeg server
    }
    
    return jsonify({"status": "Camera added successfully", "camera": stream_db[device_id]}), 201

@app.route('/api/get_cameras', methods=['GET'])
def get_cameras():
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    return jsonify(list(stream_db.values()))

@app.route('/api/edit_camera/<device_id>', methods=['PUT'])
def edit_camera(device_id):
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    if device_id not in stream_db:
        return jsonify({"error": "Camera not found"}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Only allow updating specific fields
    stream_db[device_id]['device_name'] = data.get('device_name', stream_db[device_id]['device_name'])
    stream_db[device_id]['pull_stream_type'] = data.get('pull_stream_type', stream_db[device_id]['pull_stream_type'])
    stream_db[device_id]['pull_stream_url'] = data.get('pull_stream_url', stream_db[device_id]['pull_stream_url'])
    stream_db[device_id]['device_remark'] = data.get('device_remark', stream_db[device_id]['device_remark'])
    stream_db[device_id]['is_audio'] = data.get('is_audio', stream_db[device_id]['is_audio'])
    
    return jsonify({"status": "Camera updated successfully", "camera": stream_db[device_id]})

@app.route('/api/delete_camera/<device_id>', methods=['DELETE'])
def delete_camera(device_id):
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    if device_id not in stream_db:
        return jsonify({"error": "Camera not found"}), 404
    
    # If the camera is currently streaming, stop it first
    if current_active_stream == device_id:
        stop_current_yolo_stream()
    
    # Also delete any controls associated with this camera
    controls_to_delete = [cid for cid, ctrl in control_db.items() if ctrl['device_id'] == device_id]
    for cid in controls_to_delete:
        del control_db[cid]

    del stream_db[device_id]
    
    return jsonify({"status": "Camera and associated controls deleted successfully"}), 200

@app.route('/api/start_stream/<device_id>', methods=['POST'])
def start_stream(device_id):
    global reader_thread, detector_thread, writer_thread, current_active_stream, current_active_control
    
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401

    camera_info = stream_db.get(device_id)
    if not camera_info:
        return jsonify({"error": "Camera not found"}), 404
    
    if camera_info['status'] == 'online':
        return jsonify({"status": "Stream is already running for this camera"}), 200

    # Stop any currently active stream/control before starting a new one
    stop_current_yolo_stream()

    input_rtsp_url = camera_info['pull_stream_url']
    output_rtsp_url = camera_info['output_rtsp_url'] # Use the generated output URL

    # Get stream properties for FFmpeg writer
    temp_cap = cv2.VideoCapture(input_rtsp_url)
    if not temp_cap.isOpened():
        return jsonify({"status": "Error: Could not open input stream. Check URL or camera."}), 500

    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = temp_cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        frame_rate = 25 # Default frame rate if not detected
    temp_cap.release()

    # Reset global state for the new stream
    stop_event.clear()
    with input_frame_queue.mutex:
        input_frame_queue.queue.clear()
    with output_frame_queue.mutex:
        output_frame_queue.queue.clear()

    # Start threads without any YOLO tasks initially, just a passthrough stream
    reader_thread = threading.Thread(target=frame_reader_thread, args=(input_rtsp_url,), name="FrameReader")
    # For initial passthrough, we might use a dummy detector or just a writer.
    # To use the mutliyolo.py detector, we need to pass CURRENT_TASKS.
    # For a simple "forward", we will start with no tasks, or a default minimal task if needed.
    # Let's set CURRENT_TASKS to an empty list for simple forwarding.
    # The detector will then simply pass frames if no tasks are active.
    # This requires a slight modification to mutliyolo_detector_thread if it blocks on empty tasks.
    # For now, let's assume it handles empty tasks gracefully.
    detector_thread = threading.Thread(target=multi_task_yolo_detector_thread, args=(YOLO_MODELS, []), name="MultiYOLODetector") # No tasks active yet
    writer_thread = threading.Thread(target=ffmpeg_writer_thread, args=(output_rtsp_url, frame_width, frame_height, frame_rate), name="FFmpegWriter")

    reader_thread.daemon = True
    detector_thread.daemon = True
    writer_thread.daemon = True

    reader_thread.start()
    detector_thread.start()
    writer_thread.start()

    camera_info['status'] = 'online'
    current_active_stream = device_id
    current_active_control = None # No control active yet
    
    return jsonify({"status": f"Stream started for camera {device_id}", "output_rtsp_url": output_rtsp_url})

@app.route('/api/stop_stream/<device_id>', methods=['POST'])
def stop_stream(device_id):
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    camera_info = stream_db.get(device_id)
    if not camera_info:
        return jsonify({"error": "Camera not found"}), 404
    
    if camera_info['status'] == 'offline':
        return jsonify({"status": "Stream is already stopped for this camera"}), 200

    if current_active_stream == device_id:
        stop_current_yolo_stream()
    else:
        return jsonify({"error": "This camera's stream is not the currently active one managed by YOLO tasks."}), 400
    
    camera_info['status'] = 'offline'
    return jsonify({"status": f"Stream stopped for camera {device_id}"})

@app.route('/api/add_control', methods=['POST'])
def add_control():
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    device_id = data.get('device_id')
    control_name = data.get('control_name')
    control_tag = data.get('control_tag') # e.g., "1,2,3"
    control_remark = data.get('control_remark', '')

    if not all([device_id, control_name, control_tag is not None]):
        return jsonify({"error": "Missing required fields: device_id, control_name, control_tag"}), 400
    
    if device_id not in stream_db:
        return jsonify({"error": "Associated camera not found"}), 404

    try:
        # Convert comma-separated string to list of integers
        control_tag_list = [int(x) for x in str(control_tag).split(',') if x.strip()]
        # Validate control tags
        valid_yolo_tasks = [CONTROL_TAG_MAP[tag] for tag in control_tag_list if tag in CONTROL_TAG_MAP]
        if len(valid_yolo_tasks) != len(control_tag_list):
            return jsonify({"error": "Invalid control_tag value provided"}), 400
    except ValueError:
        return jsonify({"error": "Invalid control_tag format. Must be comma-separated integers."}), 400
    
    # Generate control_id: "con" + 7 random digits for simplicity,
    # or "con" + last 7 digits of device_id if strict naming is preferred (but might not be unique)
    # Let's stick to system generated unique random for robust id.
    control_id = generate_unique_id("con", set(control_db.keys()))

    control_db[control_id] = {
        "control_id": control_id,
        "device_id": device_id,
        "control_name": control_name,
        "control_tag": control_tag_list, # Storing as list of integers
        "control_remark": control_remark,
        "status": "standby" # Initial status
    }
    
    return jsonify({"status": "Control added successfully", "control": control_db[control_id]}), 201

@app.route('/api/get_controls', methods=['GET'])
def get_controls():
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    return jsonify(list(control_db.values()))

@app.route('/api/edit_control/<control_id>', methods=['PUT'])
def edit_control(control_id):
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    if control_id not in control_db:
        return jsonify({"error": "Control not found"}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    control_name = data.get('control_name', control_db[control_id]['control_name'])
    control_tag = data.get('control_tag')
    control_remark = data.get('control_remark', control_db[control_id]['control_remark'])

    if control_tag is not None:
        try:
            control_tag_list = [int(x) for x in str(control_tag).split(',') if x.strip()]
            valid_yolo_tasks = [CONTROL_TAG_MAP[tag] for tag in control_tag_list if tag in CONTROL_TAG_MAP]
            if len(valid_yolo_tasks) != len(control_tag_list):
                return jsonify({"error": "Invalid control_tag value provided"}), 400
            control_db[control_id]['control_tag'] = control_tag_list
        except ValueError:
            return jsonify({"error": "Invalid control_tag format. Must be comma-separated integers."}), 400

    control_db[control_id]['control_name'] = control_name
    control_db[control_id]['control_remark'] = control_remark
    
    # If this control is currently running, changing its tasks would require restarting it.
    # For simplicity, we'll let the user manually stop/start after editing, or force a stop here.
    if current_active_control == control_id:
        # Option 1: Force stop (simpler for this example)
        stop_current_yolo_stream() 
        # Option 2: Reconfigure and restart (more complex) - would require passing new tasks to detector thread
        # For now, let's just update the DB and expect user to restart
        control_db[control_id]['status'] = 'standby'


    return jsonify({"status": "Control updated successfully", "control": control_db[control_id]})

@app.route('/api/delete_control/<control_id>', methods=['DELETE'])
def delete_control(control_id):
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    if control_id not in control_db:
        return jsonify({"error": "Control not found"}), 404
    
    if current_active_control == control_id:
        stop_current_yolo_stream() # Stop if this control is active

    del control_db[control_id]
    
    return jsonify({"status": "Control deleted successfully"}), 200

@app.route('/api/execute_control/<control_id>', methods=['POST'])
def execute_control(control_id):
    global reader_thread, detector_thread, writer_thread, current_active_stream, current_active_control
    
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    control_info = control_db.get(control_id)
    if not control_info:
        return jsonify({"error": "Control not found"}), 404
    
    device_id = control_info['device_id']
    camera_info = stream_db.get(device_id)
    if not camera_info:
        return jsonify({"error": "Associated camera not found. Please check camera ID."}), 404

    if control_info['status'] == 'running' and current_active_control == control_id:
        return jsonify({"status": "Control is already running"}), 200
    
    # Stop any previously active stream/control
    stop_current_yolo_stream()

    input_rtsp_url = camera_info['pull_stream_url']
    output_rtsp_url = camera_info['output_rtsp_url'] # Use the generated output URL

    # Get stream properties for FFmpeg writer
    temp_cap = cv2.VideoCapture(input_rtsp_url)
    if not temp_cap.isOpened():
        return jsonify({"status": "Error: Could not open input stream for camera. Check URL or camera."}), 500

    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = temp_cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        frame_rate = 25
    temp_cap.release()

    # Determine active tasks based on control_tag
    active_tasks_from_control = [CONTROL_TAG_MAP[tag] for tag in control_info['control_tag'] if tag in CONTROL_TAG_MAP]
    if not active_tasks_from_control:
        return jsonify({"error": "No valid YOLO tasks configured for this control."}), 400

    # Reset global state for the new stream
    stop_event.clear()
    with input_frame_queue.mutex:
        input_frame_queue.queue.clear()
    with output_frame_queue.mutex:
        output_frame_queue.queue.clear()

    # Start threads with the selected YOLO tasks
    reader_thread = threading.Thread(target=frame_reader_thread, args=(input_rtsp_url,), name="FrameReader")
    detector_thread = threading.Thread(target=multi_task_yolo_detector_thread, args=(YOLO_MODELS, active_tasks_from_control), name="MultiYOLODetector")
    writer_thread = threading.Thread(target=ffmpeg_writer_thread, args=(output_rtsp_url, frame_width, frame_height, frame_rate), name="FFmpegWriter")

    reader_thread.daemon = True
    detector_thread.daemon = True
    writer_thread.daemon = True

    reader_thread.start()
    detector_thread.start()
    writer_thread.start()

    # Update statuses
    camera_info['status'] = 'online' # Camera is now actively streaming
    control_info['status'] = 'running'
    current_active_stream = device_id
    current_active_control = control_id
    
    return jsonify({
        "status": f"Control {control_id} started for camera {device_id}",
        "tasks": active_tasks_from_control,
        "output_rtsp_url": output_rtsp_url
    })

@app.route('/api/stop_control/<control_id>', methods=['POST'])
def stop_control(control_id):
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    control_info = control_db.get(control_id)
    if not control_info:
        return jsonify({"error": "Control not found"}), 404
    
    if control_info['status'] == 'standby':
        return jsonify({"status": "Control is already stopped"}), 200

    if current_active_control == control_id:
        stop_current_yolo_stream()
    else:
        return jsonify({"error": "This control is not the currently active one."}), 400
    
    control_info['status'] = 'standby'
    # The associated camera's status will be set to 'offline' by stop_current_yolo_stream
    return jsonify({"status": f"Control {control_id} stopped."})


@app.route('/api/get_yolo_config', methods=['GET'])
def get_yolo_config():
    """Returns the YOLO models and their mapped tags."""
    token = request.args.get('token')
    if not authenticate_token(token):
        return jsonify({"error": "Invalid token"}), 401
    
    yolo_tasks = {}
    for tag_id, task_name in CONTROL_TAG_MAP.items():
        yolo_tasks[tag_id] = task_name
    
    return jsonify({
        "yolo_tasks": yolo_tasks,
        "frame_skip_config": FRAME_SKIP_CONFIG,
        "gpu_available": torch.cuda.is_available(),
        "current_active_stream": current_active_stream,
        "current_active_control": current_active_control
    })


if __name__ == '__main__':
    # Initial population for testing
    stream_db['cam1234567'] = {
        "device_id": "cam1234567",
        "device_name": "TestCam",
        "pull_stream_type": 1,
        "pull_stream_url": "rtsp://localhost:8554/mystream", # Replace with your test RTSP stream
        "device_remark": "A test camera",
        "is_audio": 0,
        "status": "offline",
        "output_rtsp_url": "rtsp://localhost:554/live/cam1234567"
    }
    control_db['con1234567'] = {
        "control_id": "con1234567",
        "device_id": "cam1234567",
        "control_name": "Initial Detect",
        "control_tag": [1], # Detect only
        "control_remark": "Initial detection task",
        "status": "standby"
    }

    app.run(host='0.0.0.0', port=5001, debug=True, threaded=False)
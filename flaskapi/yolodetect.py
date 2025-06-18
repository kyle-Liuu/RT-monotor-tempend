from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import subprocess
import threading
import time
from ultralytics import YOLO
import queue

app = Flask(__name__)
CORS(app)

# --- Configuration ---
# Input RTSP stream to be detected
input_rtsp_url = 'rtsp://192.168.1.74:9554/live/test'
# Output RTSP stream for the YOLO processed frames
output_rtsp_url = "rtsp://192.168.1.161:554/live/con54351"
FFMPEG_BIN = "ffmpeg" # Path to your ffmpeg executable

# Load YOLO model once globally
model = YOLO('yolov8n.pt')

# Frame skip interval for YOLO detection (e.g., detect every 3rd frame)
FRAME_SKIP = 3

# --- Global Data and State Management ---
TOKEN = "123456"
# Simplified data structure for demonstration
data = [
    {
        "deviceid": "cam54351",
        "devicename": "YOLO Detection Stream",
        "wsurl": "", # This will be updated when stream starts
        "online": False,
        "started": False
    }
]

# --- Multi-threading Globals ---
# Queues for inter-thread communication
# Maxsize can be adjusted based on memory and desired latency vs. dropped frames
input_frame_queue = queue.Queue(maxsize=30)  # Buffer for raw frames
output_frame_queue = queue.Queue(maxsize=30) # Buffer for processed frames

# Event to signal threads to stop gracefully
stop_event = threading.Event()

# Thread references for management (optional, for debugging/status checks)
reader_thread = None
detector_thread = None
writer_thread = None

# --- Flask Endpoints ---

@app.route('/get_devList', methods=['GET'])
def get_data():
    """Returns the list of devices."""
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401
    return jsonify(data)

@app.route('/add_device', methods=['POST'])
def add_device():
    """Adds a placeholder device if not already present."""
    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401
    for d in data:
        if d["deviceid"] == "cam54351":
            return jsonify({"error": "设备已存在"}), 400
    new_device = {
        "wsurl": input_rtsp_url,
        "devicename": "YOLO Detection Stream",
        "groupid": 0,
        "deviceid": "cam54351",
        "groupname": "未分组设备",
        "online": False,
        "started": False
    }
    data.append(new_device)
    return jsonify({"status": "添加成功"})

# --- Helper Functions for Multi-threaded Streaming ---

def frame_reader_thread(rtsp_url_in):
    """
    Reads frames from the RTSP input stream and puts them into the input_frame_queue.
    Includes basic reconnect logic.
    """
    cap = cv2.VideoCapture(rtsp_url_in)
    if not cap.isOpened():
        print(f"Error: Frame reader could not open video stream {rtsp_url_in}. Signaling stop.")
        stop_event.set()
        return

    print(f"Frame reader thread started for {rtsp_url_in}")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Frame reader: End of stream or error reading frame, attempting reconnect...")
            cap.release()
            time.sleep(5) # Wait before attempting to reconnect
            cap = cv2.VideoCapture(rtsp_url_in)
            if not cap.isOpened():
                print(f"Error: Frame reader failed to reconnect to {rtsp_url_in}. Stopping.")
                break # Exit loop if reconnect fails
            continue # Try reading again after reconnect

        try:
            input_frame_queue.put(frame, timeout=1)
        except queue.Full:
            print("Frame reader: Input queue full, dropping frame.")
        time.sleep(0.001) # Small sleep to yield CPU

    cap.release()
    print("Frame reader thread stopped.")
    stop_event.set() # Ensure all threads are signaled to stop on exit

def yolo_detector_thread(model_instance):
    """
    Pulls frames from input_frame_queue, performs YOLO detection, and
    puts annotated frames into output_frame_queue.
    """
    print("YOLO detector thread started.")
    frame_count = 0
    last_boxes, last_confs, last_clss = [], [], [] # Store last detection results

    while not stop_event.is_set():
        try:
            frame = input_frame_queue.get(timeout=1)
        except queue.Empty:
            if stop_event.is_set():
                break # Exit if signaled to stop and queue is empty
            time.sleep(0.01) # Wait briefly before retrying
            continue

        frame_count += 1
        # Default to last known detection results for frames where YOLO is skipped
        current_boxes, current_confs, current_clss = last_boxes, last_confs, last_clss

        # Only run YOLO detection every FRAME_SKIP frames to reduce load
        if frame_count % FRAME_SKIP == 0:
            try:
                results = model_instance(frame, verbose=False) # verbose=False suppresses YOLO output
                current_boxes = results[0].boxes.xyxy.cpu().numpy()
                current_confs = results[0].boxes.conf.cpu().numpy()
                current_clss = results[0].boxes.cls.cpu().numpy()
                # Update last known good results after a successful detection
                last_boxes, last_confs, last_clss = current_boxes, current_confs, current_clss
            except Exception as e:
                print(f"YOLO detection error: {e}")
                # Fallback to last valid results if detection fails
                current_boxes, current_confs, current_clss = last_boxes, last_confs, last_clss

        # Draw bounding boxes and labels on the frame
        for box, conf, cls in zip(current_boxes, current_confs, current_clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model_instance.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        try:
            output_frame_queue.put(frame, timeout=1)
        except queue.Full:
            print("YOLO detector: Output queue full, dropping processed frame.")
        input_frame_queue.task_done() # Mark item as processed from input queue

    print("YOLO detector thread stopped.")
    stop_event.set() # Ensure all threads are signaled to stop on exit


def ffmpeg_writer_thread(output_rtsp_url, frame_width, frame_height, frame_rate):
    """
    Pulls processed frames from output_frame_queue and writes them to FFmpeg's stdin
    to stream them out as RTSP. Includes robust error handling for FFmpeg process.
    """
    print(f"FFmpeg writer thread started for {output_rtsp_url}")
    ffmpeg_command = [
        FFMPEG_BIN,
        '-y', # Overwrite output stream
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', # Input pixel format from OpenCV
        '-s', f'{frame_width}x{frame_height}',
        '-r', str(frame_rate), # Input frame rate for FFmpeg
        '-i', '-', # Read input from stdin
        '-c:v', 'libx264', # H.264 video codec for output
        '-pix_fmt', 'yuv420p', # Pixel format for H.264
        '-preset', 'ultrafast', # Fastest encoding preset
        '-tune', 'zerolatency', # Crucial for low-latency streaming
        '-f', 'rtsp', # Output format is RTSP
        output_rtsp_url
    ]
    pipe = None
    try:
        pipe = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"FFmpeg command: {' '.join(ffmpeg_command)}")

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
                print(f"FFmpeg writer: Broken pipe or OS error writing to pipe: {e}. FFmpeg process likely terminated.")
                break
            except Exception as e:
                print(f"FFmpeg writer: Unexpected error writing to pipe: {e}")
                break
            output_frame_queue.task_done()
    except Exception as e:
        print(f"FFmpeg writer: Failed to start FFmpeg process: {e}")
    finally:
        if pipe:
            if pipe.stdin:
                try:
                    pipe.stdin.close() # Close stdin to signal EOF to FFmpeg
                except BrokenPipeError:
                    print("FFmpeg writer: Pipe stdin already closed or broken.")
            pipe.wait(timeout=5) # Wait for ffmpeg to terminate
            if pipe.poll() is None: # If still running, terminate forcefully
                print("FFmpeg writer: FFmpeg process did not terminate gracefully, killing.")
                pipe.terminate()
                pipe.wait(timeout=2)
                if pipe.poll() is None:
                    pipe.kill()
        print("FFmpeg writer thread stopped.")
        stop_event.set() # Ensure all threads are signaled to stop on exit

# --- API Endpoints for Stream Control ---

@app.route('/yolo_detect', methods=['POST'])
def yolo_detect_api():
    """
    Starts the multi-threaded YOLO detection and RTSP streaming pipeline.
    Ensures only one instance runs at a time.
    """
    global reader_thread, detector_thread, writer_thread # Declare as global to modify

    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401

    # Check if a stream is already running
    # This check is more robust by checking thread's aliveness
    if reader_thread and reader_thread.is_alive() and \
       detector_thread and detector_thread.is_alive() and \
       writer_thread and writer_thread.is_alive():
        return jsonify({"status": "YOLO detection stream is already running"}), 200

    # Update device info
    for d in data:
        if d["deviceid"] == "cam54351":
            # Assuming the output RTSP stream can be converted to a WebSocket URL for client display
            # Adjust this conversion logic based on your specific media server setup
            d["wsurl"] = output_rtsp_url.replace("rtsp://", "ws://").replace(":554/", ":80/") + ".live.mp4"
            d["online"] = True
            d["started"] = True

    # --- Start the multi-threaded YOLO detection and streaming process ---
    # Get initial stream properties from input to configure FFmpeg output
    temp_cap = cv2.VideoCapture(input_rtsp_url)
    if not temp_cap.isOpened():
        print(f"Error: Could not open input stream {input_rtsp_url} for initial properties.")
        return jsonify({"status": "Error: Could not open input stream for properties"}), 500

    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = temp_cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0: frame_rate = 25 # Default to 25 FPS if not detected
    temp_cap.release()

    # Clear queues and reset stop event for a fresh start
    with input_frame_queue.mutex:
        input_frame_queue.queue.clear()
    with output_frame_queue.mutex:
        output_frame_queue.queue.clear()
    stop_event.clear() # Clear the stop event before starting new threads

    # Initialize and start threads
    reader_thread = threading.Thread(target=frame_reader_thread, args=(input_rtsp_url,), name="FrameReader")
    detector_thread = threading.Thread(target=yolo_detector_thread, args=(model,), name="YOLODetector")
    writer_thread = threading.Thread(target=ffmpeg_writer_thread, args=(output_rtsp_url, frame_width, frame_height, frame_rate), name="FFmpegWriter")

    # Set threads as daemon so they don't block the main program exit
    reader_thread.daemon = True
    detector_thread.daemon = True
    writer_thread.daemon = True

    reader_thread.start()
    detector_thread.start()
    writer_thread.start()

    print("Optimized YOLO detection and streaming threads started.")
    return jsonify({"status": "YOLO检测推流已启动"})

@app.route('/stop_yolo_detect', methods=['POST'])
def stop_yolo_detect_api():
    """
    Signals the running YOLO detection threads to stop gracefully.
    """
    global reader_thread, detector_thread, writer_thread # Declare as global

    token = request.args.get('token')
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401

    if not stop_event.is_set():
        stop_event.set() # Signal all threads to stop
        print("Signal sent to stop YOLO detection threads.")
        # Optional: Add a brief wait for threads to begin shutting down
        time.sleep(2) # Give threads some time to react to the stop signal

        # Join threads if they are alive (optional, to ensure full cleanup before response)
        # This can block the API call if threads take too long to stop
        # if reader_thread and reader_thread.is_alive(): reader_thread.join(timeout=5)
        # if detector_thread and detector_thread.is_alive(): detector_thread.join(timeout=5)
        # if writer_thread and writer_thread.is_alive(): writer_thread.join(timeout=5)

    # Update device status
    for d in data:
        if d["deviceid"] == "cam54351":
            d["online"] = False
            d["started"] = False

    return jsonify({"status": "信号已发送，YOLO检测推流正在停止"})


if __name__ == '__main__':
    # Flask app will run in the main thread
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=False)
    # Using threaded=False for Flask is generally recommended when managing
    # your own threads to avoid conflicts with Flask's internal threading.
import asyncio
import websockets
import cv2
import base64
import json

# 跳帧设置
SKIP_FRAMES = 2  # 每3帧处理1帧
frame_count = 0

async def process_video(websocket):
    global frame_count
    try:
        # 打开视频流
        # cap = cv2.VideoCapture("rtsp://192.168.1.74:9554/live/test")
        cap = cv2.VideoCapture("rtsp://admin:kaoe.robot@192.168.1.169:554/Streaming/Channels/101")
        
        # 设置缓冲区大小
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取视频帧")
                    break
                
                frame_count += 1
                
                # 跳帧处理
                if frame_count % (SKIP_FRAMES + 1) != 0:
                    continue
                    
                # 调整图像大小
                frame = cv2.resize(frame, (640, 480))
                
                # 将帧转换为base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 发送数据
                await websocket.send(json.dumps({
                    'type': 'frame',
                    'frame': frame_base64
                }))
                
                # 控制帧率
                await asyncio.sleep(0.03)  # 约30FPS
                
            except websockets.exceptions.ConnectionClosedOK:
                print("客户端正常断开连接")
                break
            except websockets.exceptions.ConnectionClosedError:
                print("客户端异常断开连接")
                break
            except Exception as e:
                print(f"处理帧时发生错误: {str(e)}")
                break
            
    except Exception as e:
        print(f"处理视频时发生错误: {str(e)}")
    finally:
        # 确保释放视频捕获资源
        if 'cap' in locals():
            cap.release()
        print("视频处理已停止")

async def main():
    # 启动视频流WebSocket服务器
    async with websockets.serve(process_video, "localhost", 8001):
        print("视频流WebSocket服务器启动在 ws://localhost:8001")
        await asyncio.Future()  # 运行永久

if __name__ == "__main__":
    asyncio.run(main())
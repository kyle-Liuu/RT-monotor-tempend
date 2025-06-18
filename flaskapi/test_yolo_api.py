import requests
import time

# API配置
BASE_URL = "http://localhost:5001"  # 根据实际情况修改
TOKEN = "123456"

def add_device():
    """添加设备"""
    url = f"{BASE_URL}/add_device"
    params = {"token": TOKEN}
    
    try:
        response = requests.post(url, params=params)
        print(f"添加设备响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"添加设备失败: {str(e)}")
        return False

def start_yolo_detection():
    """启动YOLO检测"""
    url = f"{BASE_URL}/yolo_detect"
    params = {"token": TOKEN}
    
    try:
        response = requests.post(url, params=params)
        print(f"启动响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"启动失败: {str(e)}")
        return False



if __name__ == "__main__":
    # 1. 添加设备
        
    # 2. 启动YOLO检测
    if start_yolo_detection():
        print("YOLO检测已成功启动")
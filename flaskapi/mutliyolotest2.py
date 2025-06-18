#!/usr/bin/env python3
"""
Multi-YOLO API 测试程序
用于测试 mutliyolo.py 服务的所有功能
"""

import requests
import json
import time
import threading
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import os
from datetime import datetime

class MultiYOLOTester:
    def __init__(self, base_url="http://localhost:5001", token="123456"):
        self.base_url = base_url
        self.token = token
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name, status, message="", response_time=None):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "status": status,  # "PASS", "FAIL", "SKIP"
            "message": message,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️"
        time_info = f" ({response_time:.3f}s)" if response_time else ""
        print(f"{status_icon} {test_name}: {message}{time_info}")
    
    def test_server_connection(self):
        """测试服务器连接"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/get_devList", 
                                      params={"token": self.token}, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test("服务器连接", "PASS", "连接成功", response_time)
                return True
            else:
                self.log_test("服务器连接", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("服务器连接", "FAIL", f"连接失败: {str(e)}")
            return False
    
    def test_token_validation(self):
        """测试Token验证"""
        # 测试正确token
        try:
            response = self.session.get(f"{self.base_url}/get_devList", 
                                      params={"token": self.token})
            if response.status_code == 200:
                self.log_test("Token验证-正确", "PASS", "正确token通过验证")
            else:
                self.log_test("Token验证-正确", "FAIL", f"正确token被拒绝: {response.status_code}")
        except Exception as e:
            self.log_test("Token验证-正确", "FAIL", f"请求失败: {str(e)}")
        
        # 测试错误token
        try:
            response = self.session.get(f"{self.base_url}/get_devList", 
                                      params={"token": "wrong_token"})
            if response.status_code == 401:
                self.log_test("Token验证-错误", "PASS", "错误token被正确拒绝")
            else:
                self.log_test("Token验证-错误", "FAIL", f"错误token未被拒绝: {response.status_code}")
        except Exception as e:
            self.log_test("Token验证-错误", "FAIL", f"请求失败: {str(e)}")
    
    def test_get_device_list(self):
        """测试获取设备列表"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/get_devList", 
                                      params={"token": self.token})
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    device = data[0]
                    required_fields = ["deviceid", "devicename", "wsurl", "online", "started", "tasks"]
                    missing_fields = [field for field in required_fields if field not in device]
                    
                    if not missing_fields:
                        self.log_test("获取设备列表", "PASS", 
                                    f"返回设备数量: {len(data)}, 当前任务: {device.get('tasks', [])}", 
                                    response_time)
                    else:
                        self.log_test("获取设备列表", "FAIL", 
                                    f"缺少字段: {missing_fields}")
                else:
                    self.log_test("获取设备列表", "FAIL", "返回数据格式错误")
            else:
                self.log_test("获取设备列表", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("获取设备列表", "FAIL", f"请求失败: {str(e)}")
    
    def test_set_tasks(self):
        """测试设置任务"""
        test_cases = [
            # (任务列表, 期望结果, 测试描述)
            (["detect"], "PASS", "单一检测任务"),
            (["detect", "segment"], "PASS", "检测+分割任务"),
            (["detect", "pose", "segment"], "PASS", "多任务组合"),
            (["classify"], "PASS", "单一分类任务"),
            (["invalid_task"], "FAIL", "无效任务"),
            ([], "PASS", "空任务列表"),
            (["detect", "segment", "obb", "pose", "classify"], "PASS", "全部任务"),
        ]
        
        for tasks, expected, description in test_cases:
            try:
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/set_tasks",
                                           params={"token": self.token},
                                           json={"tasks": tasks})
                response_time = time.time() - start_time
                
                if expected == "PASS":
                    if response.status_code == 200:
                        self.log_test(f"设置任务-{description}", "PASS", 
                                    f"任务设置成功: {tasks}", response_time)
                    else:
                        self.log_test(f"设置任务-{description}", "FAIL", 
                                    f"设置失败: HTTP {response.status_code}")
                else:  # expected == "FAIL"
                    if response.status_code == 400:
                        self.log_test(f"设置任务-{description}", "PASS", 
                                    "无效任务被正确拒绝", response_time)
                    else:
                        self.log_test(f"设置任务-{description}", "FAIL", 
                                    f"无效任务未被拒绝: HTTP {response.status_code}")
                        
            except Exception as e:
                self.log_test(f"设置任务-{description}", "FAIL", f"请求失败: {str(e)}")
    
    def test_performance_config(self):
        """测试性能配置相关接口"""
        # 获取性能配置
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/get_performance_config",
                                      params={"token": self.token})
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["frame_skip_config", "current_tasks", "gpu_available"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test("获取性能配置", "PASS", 
                                f"GPU可用: {data['gpu_available']}", response_time)
                else:
                    self.log_test("获取性能配置", "FAIL", f"缺少字段: {missing_fields}")
            else:
                self.log_test("获取性能配置", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("获取性能配置", "FAIL", f"请求失败: {str(e)}")
        
        # 设置帧跳过配置
        test_configs = [
            ({"detect": 2, "segment": 4}, "PASS", "有效配置"),
            ({"detect": -1}, "FAIL", "无效跳帧数"),
            ({"invalid_task": 2}, "FAIL", "无效任务名"),
        ]
        
        for config, expected, description in test_configs:
            try:
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/set_frame_skip",
                                           params={"token": self.token},
                                           json={"config": config})
                response_time = time.time() - start_time
                
                if expected == "PASS":
                    if response.status_code == 200:
                        self.log_test(f"设置帧跳过-{description}", "PASS", 
                                    f"配置成功: {config}", response_time)
                    else:
                        self.log_test(f"设置帧跳过-{description}", "FAIL", 
                                    f"设置失败: HTTP {response.status_code}")
                else:
                    if response.status_code == 400:
                        self.log_test(f"设置帧跳过-{description}", "PASS", 
                                    "无效配置被正确拒绝", response_time)
                    else:
                        self.log_test(f"设置帧跳过-{description}", "FAIL", 
                                    f"无效配置未被拒绝: HTTP {response.status_code}")
            except Exception as e:
                self.log_test(f"设置帧跳过-{description}", "FAIL", f"请求失败: {str(e)}")
    
    def test_yolo_start_stop(self):
        """测试YOLO启动和停止"""
        # 测试启动
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/start_multi_yolo",
                                       params={"token": self.token})
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "status" in data and "tasks" in data:
                    self.log_test("启动YOLO检测", "PASS", 
                                f"启动成功, 任务: {data['tasks']}", response_time)
                    
                    # 等待一段时间让系统稳定
                    time.sleep(3)
                    
                    # 检查设备状态
                    self.check_device_status_after_start()
                else:
                    self.log_test("启动YOLO检测", "FAIL", "返回数据格式错误")
            else:
                self.log_test("启动YOLO检测", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("启动YOLO检测", "FAIL", f"请求失败: {str(e)}")
        
        # 测试停止
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/stop_multi_yolo",
                                       params={"token": self.token})
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test("停止YOLO检测", "PASS", "停止信号发送成功", response_time)
                
                # 等待系统停止
                time.sleep(3)
                
                # 检查设备状态
                self.check_device_status_after_stop()
            else:
                self.log_test("停止YOLO检测", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("停止YOLO检测", "FAIL", f"请求失败: {str(e)}")
    
    def check_device_status_after_start(self):
        """检查启动后的设备状态"""
        try:
            response = self.session.get(f"{self.base_url}/get_devList",
                                      params={"token": self.token})
            if response.status_code == 200:
                data = response.json()
                device = data[0] if data else {}
                
                if device.get("started") and device.get("online"):
                    self.log_test("设备状态检查-启动后", "PASS", 
                                "设备状态正确: online=True, started=True")
                else:
                    self.log_test("设备状态检查-启动后", "FAIL", 
                                f"设备状态错误: online={device.get('online')}, started={device.get('started')}")
        except Exception as e:
            self.log_test("设备状态检查-启动后", "FAIL", f"检查失败: {str(e)}")
    
    def check_device_status_after_stop(self):
        """检查停止后的设备状态"""
        try:
            response = self.session.get(f"{self.base_url}/get_devList",
                                      params={"token": self.token})
            if response.status_code == 200:
                data = response.json()
                device = data[0] if data else {}
                
                if not device.get("started") and not device.get("online"):
                    self.log_test("设备状态检查-停止后", "PASS", 
                                "设备状态正确: online=False, started=False")
                else:
                    self.log_test("设备状态检查-停止后", "FAIL", 
                                f"设备状态错误: online={device.get('online')}, started={device.get('started')}")
        except Exception as e:
            self.log_test("设备状态检查-停止后", "FAIL", f"检查失败: {str(e)}")
    
    def test_concurrent_requests(self):
        """测试并发请求"""
        def make_request(request_id):
            try:
                start_time = time.time()
                response = self.session.get(f"{self.base_url}/get_devList",
                                          params={"token": self.token})
                response_time = time.time() - start_time
                return request_id, response.status_code == 200, response_time
            except Exception as e:
                return request_id, False, 0
        
        print("\n🔄 执行并发测试 (10个并发请求)...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = []
            
            for future in as_completed(futures):
                results.append(future.result())
        
        successful_requests = sum(1 for _, success, _ in results if success)
        avg_response_time = sum(time for _, _, time in results) / len(results)
        
        if successful_requests == 10:
            self.log_test("并发请求测试", "PASS", 
                        f"10/10 请求成功, 平均响应时间: {avg_response_time:.3f}s")
        else:
            self.log_test("并发请求测试", "FAIL", 
                        f"只有 {successful_requests}/10 请求成功")
    
    def test_api_error_handling(self):
        """测试API错误处理"""
        error_test_cases = [
            # (endpoint, method, params, json_data, expected_status, description)
            ("/get_devList", "GET", {}, None, 401, "缺少token"),
            ("/set_tasks", "POST", {"token": self.token}, {}, 400, "缺少tasks参数"),
            ("/set_frame_skip", "POST", {"token": self.token}, {}, 400, "缺少config参数"),
            ("/nonexistent", "GET", {"token": self.token}, None, 404, "不存在的端点"),
        ]
        
        for endpoint, method, params, json_data, expected_status, description in error_test_cases:
            try:
                if method == "GET":
                    response = self.session.get(f"{self.base_url}{endpoint}", params=params)
                else:
                    response = self.session.post(f"{self.base_url}{endpoint}", 
                                               params=params, json=json_data)
                
                if response.status_code == expected_status:
                    self.log_test(f"错误处理-{description}", "PASS", 
                                f"正确返回状态码 {expected_status}")
                else:
                    self.log_test(f"错误处理-{description}", "FAIL", 
                                f"期望 {expected_status}, 实际 {response.status_code}")
            except Exception as e:
                self.log_test(f"错误处理-{description}", "FAIL", f"请求失败: {str(e)}")
    
    def test_rtsp_stream_accessibility(self):
        """测试RTSP流的可访问性"""
        input_rtsp = 'rtsp://192.168.1.74:9554/live/test'
        output_rtsp = "rtsp://192.168.1.186:554/live/con54351"
        
        def test_rtsp_url(url, name):
            try:
                # 使用OpenCV测试RTSP流
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        self.log_test(f"RTSP流测试-{name}", "PASS", f"流可访问: {url}")
                        return True
                    else:
                        self.log_test(f"RTSP流测试-{name}", "FAIL", f"无法读取帧: {url}")
                        return False
                else:
                    self.log_test(f"RTSP流测试-{name}", "FAIL", f"无法打开流: {url}")
                    return False
            except Exception as e:
                self.log_test(f"RTSP流测试-{name}", "FAIL", f"测试失败: {str(e)}")
                return False
        
        # 注意：这些测试可能会失败，因为RTSP URL是硬编码的
        test_rtsp_url(input_rtsp, "输入流")
        # 输出流测试通常需要先启动服务才能测试
    
    def performance_benchmark(self):
        """性能基准测试"""
        print("\n📊 执行性能基准测试...")
        
        endpoints = [
            ("/get_devList", "GET", {"token": self.token}, None),
            ("/get_performance_config", "GET", {"token": self.token}, None),
        ]
        
        for endpoint, method, params, json_data in endpoints:
            response_times = []
            failed_requests = 0
            
            for i in range(20):  # 每个端点测试20次
                try:
                    start_time = time.time()
                    if method == "GET":
                        response = self.session.get(f"{self.base_url}{endpoint}", params=params)
                    else:
                        response = self.session.post(f"{self.base_url}{endpoint}", 
                                                   params=params, json=json_data)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        response_times.append(response_time)
                    else:
                        failed_requests += 1
                except Exception:
                    failed_requests += 1
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                min_time = min(response_times)
                max_time = max(response_times)
                
                self.log_test(f"性能测试{endpoint}", "PASS", 
                            f"平均: {avg_time:.3f}s, 最小: {min_time:.3f}s, 最大: {max_time:.3f}s, 失败: {failed_requests}/20")
            else:
                self.log_test(f"性能测试{endpoint}", "FAIL", "所有请求都失败了")
    
    def generate_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed_tests = sum(1 for result in self.test_results if result["status"] == "FAIL")
        skipped_tests = sum(1 for result in self.test_results if result["status"] == "SKIP")
        
        print("\n" + "="*60)
        print("📋 测试报告")
        print("="*60)
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"跳过: {skipped_tests} ⏭️")
        print(f"通过率: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test_name']}: {result['message']}")
        
        # 保存详细报告到文件
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        return passed_tests == total_tests
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始执行 Multi-YOLO API 测试套件")
        print("="*60)
        
        # 基础连接测试
        print("\n🔍 基础功能测试")
        if not self.test_server_connection():
            print("❌ 服务器连接失败，停止测试")
            return False
        
        self.test_token_validation()
        self.test_get_device_list()
        
        # API功能测试
        print("\n⚙️ API功能测试")
        self.test_set_tasks()
        self.test_performance_config()
        
        # 核心功能测试
        print("\n🎯 核心功能测试")
        self.test_yolo_start_stop()
        
        # 错误处理和边界情况测试
        print("\n🛡️ 错误处理测试")
        self.test_api_error_handling()
        
        # 性能和并发测试
        print("\n⚡ 性能测试")
        self.test_concurrent_requests()
        self.performance_benchmark()
        
        # RTSP流测试 (可能失败，但仍然测试)
        print("\n📺 RTSP流测试")
        self.test_rtsp_stream_accessibility()
        
        # 生成报告
        return self.generate_report()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-YOLO API 测试程序')
    parser.add_argument('--host', default='localhost', help='服务器地址 (默认: localhost)')
    parser.add_argument('--port', default='5001', help='服务器端口 (默认: 5001)')
    parser.add_argument('--token', default='123456', help='API Token (默认: 123456)')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（跳过性能测试）')
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    tester = MultiYOLOTester(base_url, args.token)
    
    print(f"🎯 目标服务器: {base_url}")
    print(f"🔑 使用Token: {args.token}")
    
    if args.quick:
        print("⚡ 快速测试模式")
        # 只运行基础测试
        tester.test_server_connection()
        tester.test_get_device_list()
        tester.test_set_tasks()
        success = tester.generate_report()
    else:
        success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
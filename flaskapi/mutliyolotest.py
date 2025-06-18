"""
多任务YOLO系统完整测试代码
支持单任务、多任务组合测试，性能监控和自动化测试
"""

import requests
import json
import time
import threading
import signal
import sys
from datetime import datetime
import psutil
import cv2

class MultiYOLOTester:
    def __init__(self, base_url="http://localhost:5001", token="123456"):
        self.base_url = base_url
        self.token = token
        self.current_test = None
        self.test_start_time = None
        self.performance_data = []
        self.stop_monitoring = False
        
        # 测试配置
        self.test_configs = {
            "单任务测试": {
                "detect_only": ["detect"],
                "segment_only": ["segment"],
                "obb_only": ["obb"],
                "pose_only": ["pose"],
                "classify_only": ["classify"]
            },
            "双任务组合": {
                "detect_segment": ["detect", "segment"],
                "detect_pose": ["detect", "pose"],
                "detect_obb": ["detect", "obb"],
                "segment_pose": ["segment", "pose"],
            },
            "三任务组合": {
                "detect_segment_pose": ["detect", "segment", "pose"],
                "detect_obb_pose": ["detect", "obb", "pose"],
            },
            "全任务模式": {
                "all_tasks": ["detect", "segment", "obb", "pose", "classify"]
            }
        }
        
        # 性能基准配置（不同硬件的推荐跳帧设置）
        self.performance_configs = {
            "高性能GPU": {
                "detect": 1, "segment": 2, "obb": 2, 
                "pose": 2, "classify": 3
            },
            "中等GPU": {
                "detect": 2, "segment": 3, "obb": 3, 
                "pose": 3, "classify": 5
            },
            "CPU模式": {
                "detect": 5, "segment": 8, "obb": 6, 
                "pose": 7, "classify": 10
            }
        }
        
        # 注册信号处理器用于优雅退出
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """处理退出信号"""
        print(f"\n收到退出信号 {signum}，正在停止测试...")
        self.stop_monitoring = True
        if self.current_test:
            self.stop_detection()
        sys.exit(0)

    def api_request(self, endpoint, method="GET", data=None):
        """统一的API请求方法"""
        url = f"{self.base_url}/{endpoint}"
        params = {"token": self.token}
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=10)
            elif method == "POST":
                if data:
                    response = requests.post(url, params=params, json=data, timeout=10)
                else:
                    response = requests.post(url, params=params, timeout=10)
            
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.RequestException as e:
            return False, f"请求错误: {e}"
        except json.JSONDecodeError as e:
            return False, f"JSON解析错误: {e}"

    def set_tasks(self, tasks):
        """设置YOLO任务"""
        success, result = self.api_request("set_tasks", "POST", {"tasks": tasks})
        if success:
            print(f"✓ 任务设置成功: {tasks}")
            return True
        else:
            print(f"✗ 任务设置失败: {result}")
            return False

    def start_detection(self):
        """启动检测"""
        success, result = self.api_request("start_multi_yolo", "POST")
        if success:
            print(f"✓ 检测启动成功")
            self.test_start_time = time.time()
            return True
        else:
            print(f"✗ 检测启动失败: {result}")
            return False

    def stop_detection(self):
        """停止检测"""
        success, result = self.api_request("stop_multi_yolo", "POST")
        if success:
            print(f"✓ 检测停止成功")
        else:
            print(f"✗ 检测停止失败: {result}")
        return success

    def get_device_status(self):
        """获取设备状态"""
        success, result = self.api_request("get_devList", "GET")
        if success and result:
            return result[0] if isinstance(result, list) else result
        return None

    def get_performance_config(self):
        """获取性能配置"""
        success, result = self.api_request("get_performance_config", "GET")
        if success:
            return result
        return None

    def set_frame_skip_config(self, config):
        """设置帧跳过配置"""
        success, result = self.api_request("set_frame_skip", "POST", {"config": config})
        if success:
            print(f"✓ 帧跳过配置更新成功: {config}")
            return True
        else:
            print(f"✗ 帧跳过配置更新失败: {result}")
            return False

    def monitor_performance(self, duration=30):
        """性能监控线程"""
        print(f"开始性能监控，持续 {duration} 秒...")
        
        start_time = time.time()
        sample_count = 0
        
        while not self.stop_monitoring and (time.time() - start_time) < duration:
            try:
                # 获取系统性能指标
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # 尝试获取GPU信息（需要安装nvidia-ml-py）
                gpu_info = "N/A"
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info = f"{gpu_util.gpu}%"
                except:
                    pass
                
                # 获取设备状态
                device_status = self.get_device_status()
                
                performance_sample = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "gpu_percent": gpu_info,
                    "device_online": device_status.get("online", False) if device_status else False,
                    "device_started": device_status.get("started", False) if device_status else False,
                    "current_tasks": device_status.get("tasks", []) if device_status else []
                }
                
                self.performance_data.append(performance_sample)
                sample_count += 1
                
                # 每10秒打印一次性能信息
                if sample_count % 10 == 0:
                    print(f"性能监控 [{performance_sample['timestamp']}]: "
                          f"CPU: {cpu_percent:.1f}%, "
                          f"内存: {memory.percent:.1f}%, "
                          f"GPU: {gpu_info}")
                
            except Exception as e:
                print(f"性能监控错误: {e}")
            
            time.sleep(1)
        
        print(f"性能监控结束，共收集 {len(self.performance_data)} 个样本")

    def run_single_test(self, test_name, tasks, duration=60, frame_skip_config={'detect': 5}):
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"开始测试: {test_name}")
        print(f"任务配置: {tasks}")
        print(f"测试时长: {duration} 秒")
        print(f"{'='*60}")
        
        self.current_test = test_name
        self.performance_data = []
        self.stop_monitoring = False
        
        # 设置帧跳过配置（如果提供）
        if frame_skip_config:
            if not self.set_frame_skip_config(frame_skip_config):
                return False
        
        # 设置任务
        if not self.set_tasks(tasks):
            return False
        
        # 启动检测
        if not self.start_detection():
            return False
        
        # 启动性能监控线程
        monitor_thread = threading.Thread(
            target=self.monitor_performance, 
            args=(duration,), 
            daemon=True
        )
        monitor_thread.start()
        
        try:
            # 等待测试完成
            print(f"测试运行中... (按 Ctrl+C 可以提前结束)")
            monitor_thread.join()
            
        except KeyboardInterrupt:
            print(f"\n用户中断测试")
            self.stop_monitoring = True
        
        # 停止检测
        self.stop_detection()
        
        # 等待一段时间确保系统稳定
        time.sleep(3)
        
        # 生成测试报告
        self.generate_test_report(test_name, tasks)
        
        self.current_test = None
        return True

    def generate_test_report(self, test_name, tasks):
        """生成测试报告"""
        if not self.performance_data:
            print("⚠️  没有性能数据，跳过报告生成")
            return
        
        print(f"\n📊 测试报告: {test_name}")
        print(f"{'='*50}")
        
        # 计算性能统计
        cpu_values = [d["cpu_percent"] for d in self.performance_data if isinstance(d["cpu_percent"], (int, float))]
        memory_values = [d["memory_percent"] for d in self.performance_data if isinstance(d["memory_percent"], (int, float))]
        
        if cpu_values:
            print(f"CPU 使用率: 平均 {sum(cpu_values)/len(cpu_values):.1f}%, "
                  f"最大 {max(cpu_values):.1f}%, 最小 {min(cpu_values):.1f}%")
        
        if memory_values:
            print(f"内存使用率: 平均 {sum(memory_values)/len(memory_values):.1f}%, "
                  f"最大 {max(memory_values):.1f}%, 最小 {min(memory_values):.1f}%")
        
        # 系统稳定性分析
        online_count = sum(1 for d in self.performance_data if d["device_online"])
        stability = (online_count / len(self.performance_data)) * 100 if self.performance_data else 0
        
        print(f"系统稳定性: {stability:.1f}% ({online_count}/{len(self.performance_data)} 样本在线)")
        
        # 性能评级
        avg_cpu = sum(cpu_values)/len(cpu_values) if cpu_values else 0
        avg_memory = sum(memory_values)/len(memory_values) if memory_values else 0
        
        if avg_cpu < 50 and avg_memory < 70 and stability > 95:
            rating = "🟢 优秀"
        elif avg_cpu < 70 and avg_memory < 80 and stability > 90:
            rating = "🟡 良好"
        elif avg_cpu < 85 and avg_memory < 90 and stability > 80:
            rating = "🟠 一般"
        else:
            rating = "🔴 需要优化"
        
        print(f"性能评级: {rating}")
        
        # 优化建议
        self.generate_optimization_suggestions(tasks, avg_cpu, avg_memory, stability)

    def generate_optimization_suggestions(self, tasks, avg_cpu, avg_memory, stability):
        """生成优化建议"""
        print(f"\n💡 优化建议:")
        
        suggestions = []
        
        if avg_cpu > 80:
            suggestions.append("- CPU使用率过高，建议增加帧跳过数或减少并发任务")
        
        if avg_memory > 80:
            suggestions.append("- 内存使用率过高，建议减少模型并发数或使用更轻量的模型")
        
        if stability < 90:
            suggestions.append("- 系统稳定性较低，建议检查网络连接和硬件状态")
        
        if len(tasks) > 3:
            suggestions.append("- 多任务并发较多，建议根据实际需求选择核心任务")
        
        if "segment" in tasks and avg_cpu > 60:
            suggestions.append("- 分割任务计算量大，建议增加跳帧数或使用GPU加速")
        
        if "pose" in tasks and avg_cpu > 60:
            suggestions.append("- 姿态估计任务较重，建议结合目标检测优化处理区域")
        
        if not suggestions:
            suggestions.append("- 当前配置表现良好，可以考虑适当降低跳帧数提高检测频率")
        
        for suggestion in suggestions:
            print(suggestion)

    def run_performance_comparison_test(self):
        """运行性能对比测试"""
        print(f"\n🚀 开始性能对比测试")
        print(f"将测试不同硬件配置下的性能表现")
        
        # 获取当前系统信息
        perf_config = self.get_performance_config()
        if perf_config:
            has_gpu = perf_config.get("gpu_available", False)
            print(f"GPU 可用: {'是' if has_gpu else '否'}")
        
        # 选择合适的配置进行测试
        if has_gpu:
            configs_to_test = ["高性能GPU", "中等GPU"]
        else:
            configs_to_test = ["CPU模式"]
        
        test_tasks = ["detect", "segment"]  # 使用基础任务组合进行对比
        
        for config_name in configs_to_test:
            config = self.performance_configs[config_name]
            print(f"\n测试配置: {config_name}")
            
            success = self.run_single_test(
                f"性能对比-{config_name}",
                test_tasks,
                duration=30,  # 缩短测试时间
                frame_skip_config=config
            )
            
            if not success:
                print(f"配置 {config_name} 测试失败")
                continue
            
            # 测试间隔
            time.sleep(5)

    def run_comprehensive_test_suite(self):
        """运行完整测试套件"""
        print(f"\n🎯 开始综合测试套件")
        print(f"这将测试所有任务组合的性能表现")
        
        total_tests = sum(len(configs) for configs in self.test_configs.values())
        current_test = 0
        
        for category, test_cases in self.test_configs.items():
            print(f"\n📂 测试类别: {category}")
            
            for test_name, tasks in test_cases.items():
                current_test += 1
                print(f"\n进度: {current_test}/{total_tests}")
                
                # 根据任务复杂度调整测试时长
                if len(tasks) == 1:
                    duration = 30  # 单任务测试30秒
                elif len(tasks) <= 3:
                    duration = 45  # 多任务测试45秒
                else:
                    duration = 60  # 全任务测试60秒
                
                success = self.run_single_test(test_name, tasks, duration)
                
                if not success:
                    print(f"⚠️  测试 {test_name} 失败，继续下一个测试")
                
                # 测试间隔，让系统稳定
                if current_test < total_tests:
                    print("等待系统稳定...")
                    time.sleep(5)
        
        print(f"\n🎉 综合测试套件完成！")

    def run_interactive_test(self):
        """交互式测试模式"""
        while True:
            print(f"\n🎮 交互式测试模式")
            print(f"请选择要测试的任务类型:")
            print(f"1. 目标检测 (detect)")
            print(f"2. 实例分割 (segment)")
            print(f"3. 旋转框检测 (obb)")
            print(f"4. 姿态估计 (pose)")
            print(f"5. 图像分类 (classify)")
            print(f"6. 自定义任务组合")
            print(f"7. 性能对比测试")
            print(f"8. 综合测试套件")
            print(f"0. 退出")
            
            try:
                choice = input("\n请输入选择 (0-8): ").strip()
                
                if choice == "0":
                    print("退出交互式测试模式")
                    break
                elif choice == "1":
                    self.run_single_test("交互-目标检测", ["detect"], 600)
                elif choice == "2":
                    self.run_single_test("交互-实例分割", ["segment"], 60)
                elif choice == "3":
                    self.run_single_test("交互-旋转框检测", ["obb"], 60)
                elif choice == "4":
                    self.run_single_test("交互-姿态估计", ["pose"], 60)
                elif choice == "5":
                    self.run_single_test("交互-图像分类", ["classify"], 60)
                elif choice == "6":
                    tasks = self.get_custom_tasks()
                    if tasks:
                        duration = int(input("测试时长(秒，默认60): ") or "60")
                        self.run_single_test("交互-自定义组合", tasks, duration)
                elif choice == "7":
                    self.run_performance_comparison_test()
                elif choice == "8":
                    confirm = input("综合测试需要较长时间，确认继续? (y/N): ")
                    if confirm.lower() == 'y':
                        self.run_comprehensive_test_suite()
                else:
                    print("无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n\n测试被用户中断")
                break
            except Exception as e:
                print(f"输入错误: {e}")

    def get_custom_tasks(self):
        """获取用户自定义任务组合"""
        print("\n可选任务: detect, segment, obb, pose, classify")
        task_input = input("请输入任务列表 (用逗号分隔): ").strip()
        
        if not task_input:
            return None
        
        tasks = [task.strip() for task in task_input.split(",")]
        valid_tasks = ["detect", "segment", "obb", "pose", "classify"]
        
        invalid_tasks = [task for task in tasks if task not in valid_tasks]
        if invalid_tasks:
            print(f"无效任务: {invalid_tasks}")
            return None
        
        return tasks

def main():
    """主函数"""
    print("🤖 多任务YOLO系统测试工具")
    print("="*60)
    
    # 初始化测试器
    tester = MultiYOLOTester()
    
    # 检查API连接
    success, result = tester.api_request("get_devList", "GET")
    if not success:
        print(f"❌ 无法连接到API服务器: {result}")
        print("请确保多任务YOLO服务器正在运行在 http://localhost:5001")
        return
    
    print("✅ API连接成功")
    
    # 获取系统信息
    perf_config = tester.get_performance_config()
    if perf_config:
        print(f"GPU 可用: {'是' if perf_config.get('gpu_available') else '否'}")
        print(f"当前任务: {perf_config.get('current_tasks', [])}")
    
    # 命令行参数处理
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "auto":
            tester.run_comprehensive_test_suite()
        elif mode == "performance":
            tester.run_performance_comparison_test()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: auto (自动测试), performance (性能对比)")
    else:
        # 交互模式
        tester.run_interactive_test()
    
    print("\n👋 测试完成，感谢使用！")

if __name__ == "__main__":
    main()
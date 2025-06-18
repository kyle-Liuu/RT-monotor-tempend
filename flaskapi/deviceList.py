from flask import Flask, jsonify, request
from flask_cors import CORS  # 添加CORS支持

app = Flask(__name__)
CORS(app)  # 启用CORS

# 定义 token
TOKEN = "123456"

# 原始数据
data = [
    {
        "wsurl": "ws://192.168.1.74:9002/live/cam515301fd29.live.mp4",
        "devicename": "小米监测站点",
        "groupid": 1,
        "deviceid": "3224",
        "groupname": "区域一",
        "online": True,
        "started": False
    },
    {
        "wsurl": "ws://192.168.1.74:9002/live/cam200375b758.live.mp4",
        "devicename": "工程头站口",
        "groupid": 1,
        "deviceid": "2421",
        "groupname": "区域一",
        "online": True,
        "started": True
    },
    {
        "wsurl": "",
        "devicename": "幼儿园展示",
        "groupid": 1,
        "deviceid": "546",
        "groupname": "区域一",
        "online": False,
        "started": False
    },
    {
        "wsurl": "ws://192.168.1.74:9002/live/cam8047f64981.live.mp4",
        "devicename": "摄像头",
        "groupid": 2,
        "deviceid": "23557",
        "groupname": "区域二",
        "online": True,
        "started": True
    },
    {
        "wsurl": "",
        "devicename": "测试设备",
        "groupid": 2,
        "deviceid": "273254",
        "groupname": "区域二",
        "online": False,
        "started": False
    },
    {
        "wsurl": "ws://192.168.1.161:80/live/con54351.live.mp4",
        "devicename": "未分组设备A",
        "groupid": 0,
        "deviceid": "54351",
        "groupname": "未分组设备",
        "online": True,
        "started": False
    }
]

@app.route('/get_devList', methods=['GET'])
def get_data():
    # 获取请求中的 token
    token = request.args.get('token')
    
    # 验证 token
    if token != TOKEN:
        return jsonify({"error": "Invalid token"}), 401
    
    # 返回处理后的数据
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
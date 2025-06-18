from flask import Flask, jsonify, request
from flask_cors import CORS
from models import db, User
import json
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

# 配置SQLite数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# 创建数据库表
with app.app_context():
    db.create_all()
    
    # 检查是否需要初始化数据
    if User.query.count() == 0:
        # 生成15条测试数据
        first_names = ['张', '王', '李', '赵', '刘', '陈', '杨', '黄', '周', '吴']
        last_names = ['伟', '芳', '娜', '秀英', '敏', '静', '强', '磊', '军', '洋']
        statuses = ['1', '2', '3', '4']
        roles = ['R_USER', 'R_ADMIN', 'R_SUPER']
        
        for i in range(15):
            user = User(
                create_by=random.choice(first_names) + random.choice(last_names),
                create_time=datetime.now(),
                update_by=random.choice(first_names) + random.choice(last_names),
                update_time=datetime.now(),
                status=random.choice(statuses),
                user_name=f'user{i+1}',
                user_gender=random.randint(0, 1),
                nick_name=random.choice(first_names) + random.choice(last_names),
                user_phone=f'1{random.randint(3,9)}{"".join([str(random.randint(0,9)) for _ in range(9)])}',
                user_email=f'user{i+1}@example.com',
                user_roles=json.dumps([random.choice(roles)]),
                avatar=f'http://dummyimage.com/100x100/50B347/FFF&text=Avatar{i+1}'
            )
            db.session.add(user)
        db.session.commit()

@app.route('/api/user/list', methods=['GET'])
def get_user_list():
    current = int(request.args.get('current', 1))
    size = int(request.args.get('size', 20))
    
    # 计算分页
    start = (current - 1) * size
    end = start + size
    
    # 获取总记录数
    total = User.query.count()
    
    # 获取分页数据
    users = User.query.offset(start).limit(size).all()
    
    # 转换为字典列表
    records = [user.to_dict() for user in users]
    
    return jsonify({
        'code': 200,
        'msg': '请求成功',
        'data': {
            'records': records,
            'current': current,
            'size': size,
            'total': total
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

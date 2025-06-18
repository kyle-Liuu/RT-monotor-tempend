from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    create_by = db.Column(db.String(50))
    create_time = db.Column(db.DateTime, default=datetime.now)
    update_by = db.Column(db.String(50))
    update_time = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    status = db.Column(db.String(10))
    user_name = db.Column(db.String(50))
    user_gender = db.Column(db.Integer)
    nick_name = db.Column(db.String(50))
    user_phone = db.Column(db.String(20))
    user_email = db.Column(db.String(100))
    user_roles = db.Column(db.String(200))  # 存储为JSON字符串
    avatar = db.Column(db.String(200))

    def to_dict(self):
        return {
            'id': self.id,
            'createBy': self.create_by,
            'createTime': self.create_time.strftime('%Y-%m-%d %H:%M:%S'),
            'updateBy': self.update_by,
            'updateTime': self.update_time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': self.status,
            'userName': self.user_name,
            'userGender': self.user_gender,
            'nickName': self.nick_name,
            'userPhone': self.user_phone,
            'userEmail': self.user_email,
            'userRoles': eval(self.user_roles) if self.user_roles else [],
            'avatar': self.avatar
        }

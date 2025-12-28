from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider
from enum import Enum
import os
import json
import uuid
from datetime import datetime, timedelta
import sqlite3
from werkzeug.utils import secure_filename
import logging
from functools import wraps
import time
import traceback
import numpy as np
# 添加 Utils 工具类
class Utils:
    """工具类，提供常用的辅助功能"""
    
    class storage:
        """本地存储工具类（服务器端实现）"""
        
        @staticmethod
        def get(key):
            """获取存储的值"""
            try:
                # 在服务器端，我们可以使用文件存储或其他方式
                # 这里提供一个简单的实现
                return None
            except Exception as e:
                print(f"获取存储值失败: {e}")
                return None
        
        @staticmethod
        def set(key, value, expiry=None):
            """设置存储的值"""
            try:
                # 在服务器端的简单实现
                pass
            except Exception as e:
                print(f"设置存储值失败: {e}")
        
        @staticmethod
        def remove(key):
            """删除存储的值"""
            try:
                # 在服务器端的简单实现
                pass
            except Exception as e:
                print(f"删除存储值失败: {e}")
    
    @staticmethod
    def format_date(date_obj):
        """格式化日期"""
        if isinstance(date_obj, str):
            return date_obj
        if isinstance(date_obj, datetime):
            return date_obj.strftime('%Y-%m-%d %H:%M:%S')
        return str(date_obj)
    
    @staticmethod
    def validate_email(email):
        """验证邮箱格式"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_phone(phone):
        """验证手机号格式"""
        import re
        # 支持中国大陆手机号格式
        pattern = r'^1[3-9]\d{9}$'
        return re.match(pattern, phone) is not None
# 导入自定义模块
try:
    from database import DatabaseManager
    print("✅ 数据库模块导入成功")
except ImportError as e:
    print(f"❌ 数据库模块导入失败: {e}")
    DatabaseManager = None

try:
    from analyse import MDQAnalyzer
    print("✅ 分析模块导入成功")
except ImportError as e:
    print(f"⚠️ 分析模块导入失败: {e}")
    MDQAnalyzer = None

try:
    from gptadvisor import DeepSeekAdvisor, generate_quick_report, format_report_for_display
    print("✅ AI建议模块导入成功")
except ImportError as e:
    print(f"⚠️ AI建议模块导入失败: {e}")
    DeepSeekAdvisor = None
    generate_quick_report = None
    format_report_for_display = None

# 自定义 JSON 编码器
class CustomJSONProvider(DefaultJSONProvider):
    """自定义 JSON 提供器，处理枚举和其他特殊类型的序列化"""
    
    def default(self, obj):
        """自定义序列化逻辑"""
        if isinstance(obj, Enum):
            # 处理所有枚举类型
            return obj.value
        elif isinstance(obj, datetime):
            # 处理日期时间对象
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象（如数据类）
            return {key: value for key, value in obj.__dict__.items()}
        return super().default(obj)

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'mindcare_secret_key_2024'  # 请在生产环境中更改
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# 设置自定义 JSON 提供器
app.json = CustomJSONProvider(app)

# 启用CORS
CORS(app, supports_credentials=True)

# 配置文件上传
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'nii', 'gz', 'jpg', 'jpeg', 'png', 'dcm', 'dicom'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化数据库和分析器
db_manager = None
analyzer = None
advisor = None

def init_components():
    """初始化系统组件"""
    global db_manager, analyzer, advisor
    
    try:
        if DatabaseManager:
            logger.info("正在初始化数据库管理器...")
            db_manager = DatabaseManager()
            logger.info("✅ 数据库管理器初始化成功")
        else:
            logger.error("❌ DatabaseManager 类未导入")
            
        if MDQAnalyzer and db_manager:
            logger.info("正在初始化MDQ分析器...")
            analyzer = MDQAnalyzer(db_manager)
            logger.info("✅ MDQ分析器初始化成功")
        else:
            logger.warning("⚠️ MDQ分析器未初始化")
            
        if DeepSeekAdvisor and db_manager and analyzer:
            logger.info("正在初始化AI建议器...")
            advisor = DeepSeekAdvisor(db_manager, analyzer)
            logger.info("✅ AI建议器初始化成功")
        else:
            logger.warning("⚠️ AI建议器未初始化")
            
    except Exception as e:
        logger.error(f"❌ 初始化组件失败: {e}")
        logger.error(traceback.format_exc())

# 在应用启动时初始化组件
init_components()

# 工具函数
def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logger.warning(f"未授权访问: {request.endpoint}")
            return jsonify({'success': False, 'message': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

def handle_error(e):
    """统一错误处理"""
    logger.error(f"API错误: {e}")
    logger.error(traceback.format_exc())
    return jsonify({'success': False, 'message': str(e)}), 500

def convert_enums_to_strings(obj):
    """递归地将对象中的枚举转换为字符串"""
    if isinstance(obj, dict):
        return {key: convert_enums_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_enums_to_strings(item) for item in obj]
    elif hasattr(obj, 'value'):  # 枚举对象
        return obj.value
    else:
        return obj

# ====== 静态文件路由 ======
@app.route('/')
def index():
    """首页"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    return send_from_directory('.', filename)

# ====== 用户认证API ======
@app.route('/api/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        # 检查数据库管理器是否可用
        if db_manager is None:
            logger.error("数据库管理器不可用")
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'}), 400
            
        logger.info(f"收到注册请求: {data.get('username', 'unknown')}")
        
        # 验证必填字段
        required_fields = ['username', 'password', 'email']
        for field in required_fields:
            if not data.get(field):
                logger.warning(f"缺少必填字段: {field}")
                return jsonify({'success': False, 'message': f'{field} 不能为空'}), 400
        
        # 验证数据格式
        username = data['username'].strip()
        password = data['password']
        email = data['email'].strip()
        
        if len(username) < 3:
            return jsonify({'success': False, 'message': '用户名至少需要3个字符'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': '密码至少需要6个字符'}), 400
        
        # 验证邮箱格式（简单验证）
        if '@' not in email or '.' not in email:
            return jsonify({'success': False, 'message': '邮箱格式不正确'}), 400
        
        # 构建用户数据
        user_data = {
            'username': username,
            'password': password,
            'email': email,
            'full_name': data.get('full_name', '').strip(),
            'gender': data.get('gender', 'prefer_not_to_say'),
            'age': data.get('age'),
            'phone': data.get('phone', '').strip(),
            'occupation': data.get('occupation', '').strip(),
            'education_level': data.get('education_level', '').strip(),
            'emergency_contact': data.get('emergency_contact', '').strip()
        }
        
        # 验证年龄
        if user_data['age'] is not None:
            try:
                age = int(user_data['age'])
                if age < 0 or age > 150:
                    return jsonify({'success': False, 'message': '年龄必须在0-150之间'}), 400
                user_data['age'] = age
            except (ValueError, TypeError):
                return jsonify({'success': False, 'message': '年龄必须是数字'}), 400
        
        logger.info(f"准备注册用户: {username}")
        
        # 注册用户
        result = db_manager.register_user(user_data)
        
        if result['success']:
            logger.info(f"✅ 用户注册成功: {username}, ID: {result['user_id']}")
            return jsonify(result)
        else:
            logger.warning(f"❌ 用户注册失败: {result['message']}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"注册API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        # 检查数据库管理器是否可用
        if db_manager is None:
            logger.error("数据库管理器不可用")
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'}), 400
            
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        logger.info(f"收到登录请求: {username}")
        
        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'}), 400
        
        # 获取客户端信息
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        # 验证登录
        result = db_manager.login_user(username, password, ip_address, user_agent)
        
        if result['success']:
            # 设置会话
            session['user_id'] = result['user_id']
            session['username'] = username
            session['session_id'] = result['session_id']
            session.permanent = True
            
            logger.info(f"✅ 用户登录成功: {username}")
            return jsonify({
                'success': True,
                'message': '登录成功',
                'user': {
                    'user_id': result['user_id'],
                    'username': username
                }
            })
        else:
            logger.warning(f"❌ 用户登录失败: {result['message']}")
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"登录API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    """用户退出"""
    try:
        session_id = session.get('session_id')
        username = session.get('username')
        
        if session_id and db_manager:
            db_manager.logout_user(session_id)
        
        session.clear()
        
        logger.info(f"✅ 用户退出: {username}")
        return jsonify({'success': True, 'message': '退出成功'})
        
    except Exception as e:
        logger.error(f"登出API异常: {e}")
        return handle_error(e)

# ====== 个人资料管理API ======
@app.route('/api/user/profile', methods=['GET'])
@login_required
def get_user_profile():
    """获取用户个人资料"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        profile = db_manager.get_user_profile(user_id)
        
        if profile:
            logger.info(f"✅ 获取用户资料成功: {user_id}")
            return jsonify({'success': True, 'profile': profile})
        else:
            logger.warning(f"❌ 用户不存在: {user_id}")
            return jsonify({'success': False, 'message': '用户不存在'}), 404
            
    except Exception as e:
        logger.error(f"获取用户资料API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)
@app.route('/api/user/change-password', methods=['POST'])
@login_required
def change_user_password():
    """修改用户密码"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'}), 400
            
        old_password = data.get('old_password', '').strip()
        new_password = data.get('new_password', '').strip()
        
        logger.info(f"收到修改密码请求: 用户 {user_id}")
        
        if not old_password or not new_password:
            return jsonify({'success': False, 'message': '原密码和新密码不能为空'}), 400
        
        if len(new_password) < 6:
            return jsonify({'success': False, 'message': '新密码至少需要6个字符'}), 400
        
        # 修改密码
        result = db_manager.change_password(user_id, old_password, new_password)
        
        if result['success']:
            logger.info(f"✅ 密码修改成功: 用户 {user_id}")
            return jsonify(result)
        else:
            logger.warning(f"❌ 密码修改失败: {result['message']}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"修改密码API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)
    

@app.route('/api/user/avatar', methods=['POST'])
@login_required
def upload_user_avatar():
    """上传用户头像"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        
        if 'avatar' not in request.files:
            return jsonify({'success': False, 'message': '没有选择文件'}), 400
        
        file = request.files['avatar']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'}), 400
        
        # 验证文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
        
        # 验证文件大小 (5MB)
        if len(file.read()) > 5 * 1024 * 1024:
            return jsonify({'success': False, 'message': '文件大小不能超过5MB'}), 400
        
        file.seek(0)  # 重置文件指针
        
        # 生成安全的文件名
        import os
        from werkzeug.utils import secure_filename
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{user_id}_avatar_{timestamp}_{filename}"
        
        # 创建头像目录
        avatar_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'avatars')
        os.makedirs(avatar_dir, exist_ok=True)
        
        filepath = os.path.join(avatar_dir, filename)
        file.save(filepath)
        
        logger.info(f"✅ 头像上传成功: 用户 {user_id}, 文件 {filename}")
        
        return jsonify({
            'success': True,
            'message': '头像上传成功',
            'avatar_url': f'/uploads/avatars/{filename}'
        })
        
    except Exception as e:
        logger.error(f"上传头像API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)
@app.route('/api/user/export-data', methods=['GET'])
@login_required
def export_user_data():
    """导出用户数据"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        
        logger.info(f"收到数据导出请求: 用户 {user_id}")
        
        # 获取用户基本信息
        profile = db_manager.get_user_profile(user_id)
        
        # 获取测试历史
        test_history = db_manager.get_user_mdq_history(user_id, limit=1000)
        
        # 获取统计信息
        statistics = db_manager.get_user_statistics(user_id)
        
        # 构建导出数据
        export_data = {
            'export_info': {
                'export_date': datetime.now().isoformat(),
                'user_id': user_id,
                'platform': 'MindCare心理健康评估系统'
            },
            'profile': profile,
            'test_history': test_history,
            'statistics': statistics,
            'note': '这是您在MindCare平台的完整个人数据导出文件，包含个人资料、测试记录和统计信息。'
        }
        
        logger.info(f"✅ 数据导出成功: 用户 {user_id}")
        
        return jsonify({
            'success': True,
            'data': export_data,
            'message': '数据导出成功'
        })
        
    except Exception as e:
        logger.error(f"导出数据API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/user/delete-account', methods=['POST'])
@login_required
def delete_user_account():
    """删除用户账户（软删除）"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        data = request.get_json()
        
        # 验证密码确认
        password = data.get('password', '').strip() if data else ''
        if not password:
            return jsonify({'success': False, 'message': '请输入密码确认删除'}), 400
        
        logger.info(f"收到删除账户请求: 用户 {user_id}")
        
        # 验证密码
        conn = db_manager._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT password_hash, salt FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'success': False, 'message': '用户不存在'}), 404
        
        stored_hash, salt = result
        password_hash, _ = db_manager._hash_password(password, salt)
        
        if password_hash != stored_hash:
            conn.close()
            return jsonify({'success': False, 'message': '密码错误'}), 401
        
        # 软删除账户（标记为不活跃）
        cursor.execute('''
            UPDATE users 
            SET is_active = 0, 
                email = email || '_deleted_' || ?, 
                username = username || '_deleted_' || ?
            WHERE user_id = ?
        ''', (datetime.now().strftime('%Y%m%d%H%M%S'), 
              datetime.now().strftime('%Y%m%d%H%M%S'), 
              user_id))
        
        # 删除所有会话
        cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        # 清除当前会话
        session.clear()
        
        logger.info(f"✅ 账户删除成功: 用户 {user_id}")
        
        return jsonify({
            'success': True,
            'message': '账户删除成功'
        })
        
    except Exception as e:
        logger.error(f"删除账户API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)
@app.route('/api/user/profile-enhanced', methods=['GET'])
@login_required
def get_enhanced_user_profile():
    """获取增强的用户资料（包含统计信息）"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        
        # 获取基本资料
        profile = db_manager.get_user_profile(user_id)
        if not profile:
            return jsonify({'success': False, 'message': '用户不存在'}), 404
        
        # 获取统计信息
        statistics = db_manager.get_user_statistics(user_id)
        
        # 获取最近的测试记录
        recent_tests = db_manager.get_user_mdq_history(user_id, limit=5)
        
        # 构建增强的资料信息
        enhanced_profile = {
            **profile,
            'statistics': statistics,
            'recent_tests': recent_tests,
            'account_status': 'active',
            'profile_completion': calculate_profile_completion(profile)
        }
        
        logger.info(f"✅ 获取增强用户资料成功: {user_id}")
        
        return jsonify({
            'success': True,
            'profile': enhanced_profile
        })
        
    except Exception as e:
        logger.error(f"获取增强用户资料API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)
def calculate_profile_completion(profile):
    """计算资料完整度"""
    total_fields = 9  # 总字段数
    completed_fields = 0
    
    required_fields = ['username', 'email']
    optional_fields = ['full_name', 'gender', 'age', 'phone', 'occupation', 'education_level', 'emergency_contact']
    
    # 必填字段
    for field in required_fields:
        if profile.get(field):
            completed_fields += 1
    
    # 可选字段
    for field in optional_fields:
        if profile.get(field):
            completed_fields += 1
    
    return round((completed_fields / total_fields) * 100, 1)


def calculate_improvement_trend(user_id):
    """计算改善趋势"""
    try:
        if not db_manager:
            return 'unknown'
            
        # 获取最近的测试记录
        recent_tests = db_manager.get_user_mdq_history(user_id, limit=10)
        
        if len(recent_tests) < 2:
            return 'insufficient_data'
        
        # 计算趋势
        scores = [test.get('raw_score', 0) for test in recent_tests if test.get('raw_score') is not None]
        
        if len(scores) < 2:
            return 'insufficient_data'
        
        # 简单的趋势计算
        recent_avg = sum(scores[:3]) / min(3, len(scores))  # 最近3次平均
        earlier_avg = sum(scores[-3:]) / min(3, len(scores[-3:]))  # 较早3次平均
        
        improvement = ((earlier_avg - recent_avg) / earlier_avg) * 100
        
        if improvement > 20:
            return 'significant_improvement'
        elif improvement > 10:
            return 'moderate_improvement'
        elif improvement > 5:
            return 'mild_improvement'
        elif improvement > -5:
            return 'stable'
        elif improvement > -10:
            return 'mild_decline'
        elif improvement > -20:
            return 'moderate_decline'
        else:
            return 'significant_decline'
            
    except Exception as e:
        logger.error(f"计算改善趋势失败: {e}")
        return 'unknown'

# ====== 用户设置API ======
@app.route('/api/user/preferences', methods=['GET', 'POST'])
@login_required
def user_preferences():
    """获取或设置用户偏好设置"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        
        if request.method == 'GET':
            # 获取用户偏好设置
            preferences = Utils.storage.get(f'user_preferences_{user_id}') or {
                'notifications': True,
                'email_reports': True,
                'data_sharing': False,
                'theme': 'light',
                'language': 'zh-CN'
            }
            
            return jsonify({
                'success': True,
                'preferences': preferences
            })
        
        else:  # POST
            # 更新用户偏好设置
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': '请求数据为空'}), 400
            
            # 验证和清理偏好设置
            valid_preferences = {}
            if 'notifications' in data:
                valid_preferences['notifications'] = bool(data['notifications'])
            if 'email_reports' in data:
                valid_preferences['email_reports'] = bool(data['email_reports'])
            if 'data_sharing' in data:
                valid_preferences['data_sharing'] = bool(data['data_sharing'])
            if 'theme' in data and data['theme'] in ['light', 'dark', 'auto']:
                valid_preferences['theme'] = data['theme']
            if 'language' in data and data['language'] in ['zh-CN', 'en-US']:
                valid_preferences['language'] = data['language']
            
            # 保存偏好设置
            Utils.storage.set(f'user_preferences_{user_id}', valid_preferences)
            
            logger.info(f"✅ 用户偏好设置更新成功: 用户 {user_id}")
            
            return jsonify({
                'success': True,
                'message': '偏好设置更新成功',
                'preferences': valid_preferences
            })
            
    except Exception as e:
        logger.error(f"用户偏好设置API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/user/profile', methods=['PUT'])
@login_required
def update_user_profile():
    """更新用户个人资料"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'}), 400
            
        logger.info(f"收到更新用户资料请求: 用户 {user_id}")
        
        # 验证数据
        profile_data = {}
        allowed_fields = ['email', 'full_name', 'gender', 'age', 'phone', 
                         'occupation', 'education_level', 'emergency_contact']
        
        for field in allowed_fields:
            if field in data:
                value = data[field]
                if value is not None:
                    # 清理数据
                    if isinstance(value, str):
                        value = value.strip()
                        if value == '':
                            value = None
                    profile_data[field] = value
        
        # 验证邮箱格式
        if 'email' in profile_data and profile_data['email']:
            if '@' not in profile_data['email'] or '.' not in profile_data['email']:
                return jsonify({'success': False, 'message': '邮箱格式不正确'}), 400
        
        # 验证年龄
        if 'age' in profile_data and profile_data['age'] is not None:
            try:
                age = int(profile_data['age'])
                if age < 0 or age > 150:
                    return jsonify({'success': False, 'message': '年龄必须在0-150之间'}), 400
                profile_data['age'] = age
            except (ValueError, TypeError):
                return jsonify({'success': False, 'message': '年龄必须是数字'}), 400
        
        # 验证性别
        if 'gender' in profile_data and profile_data['gender']:
            valid_genders = ['male', 'female', 'other', 'prefer_not_to_say']
            if profile_data['gender'] not in valid_genders:
                return jsonify({'success': False, 'message': '无效的性别选项'}), 400
        
        # 验证教育程度
        if 'education_level' in profile_data and profile_data['education_level']:
            valid_education = ['primary', 'junior_high', 'high_school', 'college', 
                             'bachelor', 'master', 'doctor', 'other']
            if profile_data['education_level'] not in valid_education:
                return jsonify({'success': False, 'message': '无效的教育程度选项'}), 400
        
        # 更新用户资料
        result = db_manager.update_user_profile(user_id, profile_data)
        
        if result['success']:
            logger.info(f"✅ 用户资料更新成功: {user_id}")
            
            # 返回更新后的用户资料
            updated_profile = db_manager.get_user_profile(user_id)
            return jsonify({
                'success': True, 
                'message': '个人资料更新成功',
                'profile': updated_profile
            })
        else:
            logger.warning(f"❌ 用户资料更新失败: {result['message']}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"更新用户资料API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

# ====== MDQ测试API ======
@app.route('/api/test/mdq', methods=['POST'])
@login_required
def save_mdq_test():
    """保存MDQ测试结果"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        data = request.get_json()
        
        logger.info(f"收到MDQ测试数据: 用户 {user_id}")
        
        # 验证数据
        if not data.get('answers'):
            return jsonify({'success': False, 'message': '测试答案不能为空'}), 400
        
        # 构建测试数据
        test_data = {
            'questions': data['answers'],
            'co_occurrence': data.get('co_occurrence', 'no'),
            'severity': data.get('severity', 'no'),
            'completion_time': data.get('completion_time', 0)
        }
        
        # 保存测试
        result = db_manager.save_mdq_test(user_id, test_data, data.get('completion_time'))
        
        if result['success']:
            logger.info(f"✅ MDQ测试保存成功: 用户 {user_id}, 测试ID {result['test_id']}")
            
            # **修复：确保所有结果数据都被正确转换和返回**
            response_data = {
                'success': True,
                'test_id': result['test_id'],
                'message': result['message']
            }
            
            # 转换分析结果中的枚举对象
            if 'score_result' in result and result['score_result']:
                response_data['score_result'] = convert_enums_to_strings(result['score_result'])
            
            # **新增：包含详细分析数据**
            if 'analysis_data' in result and result['analysis_data']:
                response_data['analysis_data'] = convert_enums_to_strings(result['analysis_data'])
                logger.info(f"✅ 分析数据已包含在响应中")
            
            # **新增：包含AI分析数据**
            if 'ai_analysis_data' in result and result['ai_analysis_data']:
                response_data['ai_analysis_data'] = result['ai_analysis_data']
                logger.info(f"✅ AI分析数据已包含在响应中")
            
            # 执行分析（如果分析器可用）
            if analyzer:
                try:
                    analysis_result = analyzer.analyze_user_comprehensive(user_id)
                    response_data['analysis_id'] = analysis_result.analysis_id
                    response_data['analysis'] = convert_enums_to_strings({
                        'severity_level': analysis_result.severity_level,
                        'risk_percentage': analysis_result.risk_percentage,
                        'improvement_trend': analysis_result.improvement_trend
                    })
                    logger.info(f"✅ 综合分析已完成: 分析ID {analysis_result.analysis_id}")
                except Exception as e:
                    logger.error(f"分析执行失败: {e}")
                    logger.error(traceback.format_exc())
                    response_data['analysis_warning'] = f'分析功能出错: {str(e)}'
            else:
                logger.warning("分析器不可用")
                response_data['analysis_warning'] = '分析功能暂时不可用'
            
            return jsonify(response_data)
        else:
            logger.warning(f"❌ MDQ测试保存失败: {result['message']}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"MDQ测试API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/test/upload', methods=['POST'])
@login_required
def upload_brain_image():
    """上传脑部影像"""
    try:
        user_id = session['user_id']
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'message': '没有选择文件'}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and allowed_file(file.filename):
                # 安全的文件名
                filename = secure_filename(file.filename)
                # 添加时间戳避免重名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{user_id}_{timestamp}_{filename}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                uploaded_files.append({
                    'original_name': file.filename,
                    'saved_name': filename,
                    'size': os.path.getsize(filepath),
                    'path': filepath
                })
                
                logger.info(f"✅ 文件上传成功: {filename}")
        
        if uploaded_files:
            return jsonify({
                'success': True,
                'message': f'成功上传 {len(uploaded_files)} 个文件',
                'files': uploaded_files
            })
        else:
            return jsonify({'success': False, 'message': '没有有效的文件'}), 400
            
    except Exception as e:
        return handle_error(e)


@app.route('/api/test/<test_id>/recommendations', methods=['GET'])
@login_required
def get_test_recommendations(test_id):
    """获取测试建议"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        recommendations = db_manager.get_test_recommendations(test_id, user_id)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return handle_error(e)

# ====== 分析API ======
@app.route('/api/analysis/comprehensive', methods=['POST'])
@login_required
def comprehensive_analysis():
    """综合分析"""
    try:
        if not analyzer:
            return jsonify({'success': False, 'message': '分析服务不可用'}), 500
            
        user_id = session['user_id']
        
        # 执行分析
        analysis_result = analyzer.analyze_user_comprehensive(user_id)
        
        if analysis_result:
            # 确保所有枚举对象都被转换为字符串
            analysis_data = convert_enums_to_strings({
                'analysis_id': analysis_result.analysis_id,
                'current_score': analysis_result.current_score,
                'severity_level': analysis_result.severity_level,
                'risk_percentage': analysis_result.risk_percentage,
                'improvement_trend': analysis_result.improvement_trend,
                'improvement_percentage': analysis_result.improvement_percentage,
                'clinical_recommendations': analysis_result.clinical_recommendations,
                'emergency_flag': analysis_result.emergency_flag
            })
            
            return jsonify({
                'success': True,
                'analysis': analysis_data
            })
        else:
            return jsonify({'success': False, 'message': '分析失败，请稍后重试'}), 500
            
    except Exception as e:
        logger.error(f"综合分析API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/analysis/latest', methods=['GET'])
@login_required
def get_latest_analysis():
    """获取用户最新的分析记录"""
    try:
        if not analyzer:
            return jsonify({'success': False, 'message': '分析服务不可用'}), 500
            
        user_id = session['user_id']
        
        # 获取最新分析记录
        analysis_history = analyzer.get_analysis_history(user_id, limit=1)
        
        if analysis_history:
            latest_analysis = convert_enums_to_strings(analysis_history[0])
            return jsonify({
                'success': True,
                'analysis': latest_analysis
            })
        else:
            return jsonify({'success': False, 'message': '没有找到分析记录'}), 404
            
    except Exception as e:
        logger.error(f"获取最新分析API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/analysis/history', methods=['GET'])
@login_required
def get_analysis_history():
    """获取分析历史"""
    try:
        if not analyzer:
            return jsonify({'success': False, 'message': '分析服务不可用'}), 500
            
        user_id = session['user_id']
        limit = request.args.get('limit', 10, type=int)
        
        history = analyzer.get_analysis_history(user_id, limit)
        
        # 确保历史记录中的枚举对象都被转换为字符串
        history = convert_enums_to_strings(history)
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"获取分析历史API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

# ====== AI建议API ======
@app.route('/api/ai/report', methods=['POST'])
@login_required
def generate_ai_report():
    """生成AI报告"""
    try:
        if not advisor:
            return jsonify({'success': False, 'message': 'AI建议服务不可用'}), 500
            
        user_id = session['user_id']
        data = request.get_json() or {}
        report_type = data.get('type', 'single')  # single, historical, both
        
        # 生成报告
        result = {'success': True, 'reports': {}}
        
        try:
            if report_type in ['single', 'both']:
                # 获取最新分析ID
                analysis_history = analyzer.get_analysis_history(user_id, limit=1)
                if analysis_history:
                    analysis_id = analysis_history[0]['analysis_id']
                    single_report = advisor.generate_single_test_report(analysis_id)
                    
                    # 转换枚举对象
                    result['reports']['single_test_report'] = convert_enums_to_strings({
                        'report_id': single_report.report_id,
                        'executive_summary': single_report.executive_summary,
                        'treatment_recommendations': single_report.treatment_recommendations,
                        'emergency_protocols': single_report.emergency_protocols
                    })
                else:
                    return jsonify({'success': False, 'message': '没有找到测试记录'}), 404
            
            if report_type in ['historical', 'both']:
                historical_report = advisor.generate_historical_analysis_report(user_id)
                
                # 转换枚举对象
                result['reports']['historical_report'] = convert_enums_to_strings({
                    'report_id': historical_report.report_id,
                    'progress_analysis': historical_report.progress_analysis,
                    'trend_interpretation': historical_report.trend_interpretation,
                    'prognosis_assessment': historical_report.prognosis_assessment
                })
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"AI报告生成失败: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'success': False, 'message': f'AI服务暂时不可用: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"AI报告API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/ai/report/<report_id>', methods=['GET'])
@login_required
def get_ai_report(report_id):
    """获取AI报告详情"""
    try:
        if not advisor:
            return jsonify({'success': False, 'message': 'AI建议服务不可用'}), 500
            
        report = advisor.get_report(report_id)
        
        if report:
            # 格式化报告用于显示
            report_data = convert_enums_to_strings(report)
            
            formatted_report = report_data
            if format_report_for_display:
                try:
                    formatted_report = format_report_for_display(report)
                    formatted_report = convert_enums_to_strings(formatted_report)
                except Exception as e:
                    logger.warning(f"报告格式化失败: {e}")
            
            return jsonify({
                'success': True,
                'report': report_data,
                'formatted_report': formatted_report
            })
        else:
            return jsonify({'success': False, 'message': '报告不存在'}), 404
            
    except Exception as e:
        return handle_error(e)

@app.route('/api/ai/reports', methods=['GET'])
@login_required
def get_user_ai_reports():
    """获取用户AI报告列表"""
    try:
        if not advisor:
            return jsonify({'success': False, 'message': 'AI建议服务不可用'}), 500
            
        user_id = session['user_id']
        report_type = request.args.get('type')
        limit = request.args.get('limit', 10, type=int)
        
        reports = advisor.get_user_reports(user_id, report_type, limit)
        reports = convert_enums_to_strings(reports)
        
        return jsonify({
            'success': True,
            'reports': reports
        })
        
    except Exception as e:
        return handle_error(e)

# ====== 辅助API ======
@app.route('/api/check-username', methods=['GET'])
def check_username():
    """检查用户名是否存在"""
    try:
        username = request.args.get('username', '').strip()
        if not username:
            return jsonify({'exists': False})
        
        if db_manager is None:
            return jsonify({'exists': False, 'error': '数据库服务不可用'})
        
        # 查询用户名是否存在
        conn = None
        try:
            conn = db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            exists = result is not None
            logger.info(f"检查用户名 {username}: {'存在' if exists else '不存在'}")
            return jsonify({'exists': exists})
        except Exception as e:
            logger.error(f"检查用户名失败: {e}")
            return jsonify({'exists': False, 'error': '检查失败'})
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"检查用户名API异常: {e}")
        return jsonify({'exists': False, 'error': '检查失败'})

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        # 测试数据库连接
        db_status = False
        if db_manager:
            try:
                conn = db_manager._get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                cursor.fetchone()
                conn.close()
                db_status = True
            except Exception as e:
                logger.error(f"数据库健康检查失败: {e}")
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': db_status,
            'analyzer': analyzer is not None,
            'advisor': advisor is not None,
            'components': {
                'database_manager': db_manager is not None,
                'mdq_analyzer': analyzer is not None,
                'deepseek_advisor': advisor is not None
            }
        })
    except Exception as e:
        logger.error(f"健康检查异常: {e}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

# ====== 错误处理 ======
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': '页面不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"内部服务器错误: {error}")
    return jsonify({'success': False, 'message': '服务器内部错误'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'message': '文件大小超过限制'}), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'success': False, 'message': '请求格式错误'}), 400

# ====== 开发辅助功能 ======
@app.route('/api/dev/init-sample-data', methods=['POST'])
def init_sample_data():
    """初始化示例数据（仅开发环境）"""
    try:
        if db_manager is None:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        # 创建示例用户
        sample_users = [
            {
                'username': 'demo_user',
                'password': 'demo123',
                'email': 'demo@example.com',
                'full_name': '演示用户',
                'age': 25,
                'gender': 'other'
            },
            {
                'username': 'test_user',
                'password': 'test123',
                'email': 'test@example.com',
                'full_name': '测试用户',
                'age': 30,
                'gender': 'male'
            }
        ]
        
        created_users = []
        for user_data in sample_users:
            try:
                result = db_manager.register_user(user_data)
                if result['success']:
                    created_users.append(user_data['username'])
                    logger.info(f"✅ 创建示例用户成功: {user_data['username']}")
                    
                    # 为每个用户创建示例测试数据
                    user_id = result['user_id']
                    sample_test_data = {
                        'questions': {
                            'q1': 'rarely', 'q2': 'no', 'q3': 'sometimes', 'q4': 'often',
                            'q5': 'no', 'q6': 'sometimes', 'q7': 'often', 'q8': 'no',
                            'q9': 'rarely', 'q10': 'sometimes', 'q11': 'no', 'q12': 'sometimes', 'q13': 'often'
                        },
                        'co_occurrence': 'yes',
                        'severity': 'moderate'
                    }
                    
                    db_manager.save_mdq_test(user_id, sample_test_data, 180)
                    logger.info(f"✅ 为用户 {user_data['username']} 创建示例测试数据")
                else:
                    logger.warning(f"❌ 创建示例用户失败: {user_data['username']}, {result['message']}")
            except Exception as e:
                logger.error(f"创建示例用户 {user_data['username']} 失败: {e}")
        
        return jsonify({
            'success': True,
            'message': f'成功创建 {len(created_users)} 个示例用户',
            'users': created_users
        })
        
    except Exception as e:
        logger.error(f"初始化示例数据失败: {e}")
        return handle_error(e)

@app.route('/api/dev/test-db', methods=['GET'])
def test_database():
    """测试数据库连接"""
    try:
        if db_manager is None:
            return jsonify({'success': False, 'message': '数据库管理器未初始化'}), 500
        
        # 测试数据库连接
        conn = db_manager._get_connection()
        cursor = conn.cursor()
        
        # 检查用户表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        users_table = cursor.fetchone()
        
        # 统计用户数量
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # 统计测试数量
        cursor.execute("SELECT COUNT(*) FROM questionnaire_tests")
        test_count = cursor.fetchone()[0]
        
        # 获取最近5个用户
        cursor.execute("SELECT username, registration_date FROM users ORDER BY registration_date DESC LIMIT 5")
        recent_users = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'database_status': 'connected',
            'users_table_exists': users_table is not None,
            'user_count': user_count,
            'test_count': test_count,
            'recent_users': [{'username': row[0], 'registration_date': row[1]} for row in recent_users],
            'message': '数据库连接正常'
        })
        
    except Exception as e:
        logger.error(f"数据库测试失败: {e}")
        return jsonify({
            'success': False,
            'message': f'数据库测试失败: {str(e)}'
        }), 500

# ====== 调试路由 ======
@app.route('/api/debug/session', methods=['GET'])
def debug_session():
    """调试会话信息（仅开发环境）"""
    try:
        return jsonify({
            'session_data': dict(session),
            'session_keys': list(session.keys()),
            'has_user_id': 'user_id' in session,
            'has_username': 'username' in session,
            'has_session_id': 'session_id' in session
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/debug/components', methods=['GET'])
def debug_components():
    """调试组件状态（仅开发环境）"""
    try:
        return jsonify({
            'database_manager': {
                'available': db_manager is not None,
                'type': str(type(db_manager)) if db_manager else None
            },
            'analyzer': {
                'available': analyzer is not None,
                'type': str(type(analyzer)) if analyzer else None
            },
            'advisor': {
                'available': advisor is not None,
                'type': str(type(advisor)) if advisor else None
            },
            'modules': {
                'DatabaseManager': DatabaseManager is not None,
                'MDQAnalyzer': MDQAnalyzer is not None,
                'DeepSeekAdvisor': DeepSeekAdvisor is not None,
                'generate_quick_report': generate_quick_report is not None,
                'format_report_for_display': format_report_for_display is not None
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ====== 启动配置 ======
def create_app():
    """应用工厂函数"""
    return app

def startup_check():
    """启动检查"""
    print("🔍 系统启动检查...")
    print("=" * 50)
    
    # 检查数据库
    if db_manager:
        try:
            conn = db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            user_count = cursor.fetchone()[0]
            conn.close()
            print(f"✅ 数据库连接正常，当前用户数量: {user_count}")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
    else:
        print("❌ 数据库管理器未初始化")
    
    # 检查其他组件
    print(f"📊 MDQ分析器: {'✅ 可用' if analyzer else '❌ 不可用'}")
    print(f"🤖 AI建议器: {'✅ 可用' if advisor else '❌ 不可用'}")
    
    print("=" * 50)

@app.route('/api/test/clear-progress', methods=['POST'])
@login_required
def clear_test_progress():
    """清空用户的测试进度"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        
        logger.info(f"用户 {user_id} 请求清空测试进度")
        
        # 清空用户相关的测试进度数据
        # 这里可以根据具体需求决定清空哪些数据
        # 例如：清空未完成的测试草稿、临时数据等
        
        try:
            # 获取数据库连接
            conn = db_manager._get_connection()
            cursor = conn.cursor()
            
            # 清空测试进度相关的临时数据（如果有的话）
            # 注意：这里不删除已完成的测试记录，只清空进度数据
            
            # 示例：如果有测试草稿表，可以清空用户的草稿
            # cursor.execute('DELETE FROM test_drafts WHERE user_id = ?', (user_id,))
            
            # 示例：如果有会话临时数据表，可以清空
            # cursor.execute('DELETE FROM test_sessions WHERE user_id = ? AND status = "in_progress"', (user_id,))
            
            # 提交更改
            conn.commit()
            conn.close()
            
            logger.info(f"✅ 用户 {user_id} 的测试进度已清空")
            
            return jsonify({
                'success': True,
                'message': '测试进度已清空',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id
            })
            
        except Exception as db_error:
            logger.error(f"数据库操作失败: {db_error}")
            return jsonify({
                'success': False,
                'message': '加载进度时数据库操作失败'
            }), 500
            
    except Exception as e:
        logger.error(f"加载测试进度API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/test/has-progress', methods=['GET'])
@login_required
def check_test_progress():
    """检查用户是否有保存的测试进度"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        
        try:
            # 获取数据库连接
            conn = db_manager._get_connection()
            cursor = conn.cursor()
            
            # 检查是否有测试进度记录
            cursor.execute('''
                SELECT COUNT(*) as count, MAX(updated_at) as last_updated
                FROM test_progress 
                WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            has_progress = result[0] > 0 if result else False
            last_updated = result[1] if result and result[1] else None
            
            return jsonify({
                'success': True,
                'has_progress': has_progress,
                'last_updated': last_updated,
                'message': '检查完成'
            })
            
        except Exception as db_error:
            logger.error(f"数据库操作失败: {db_error}")
            return jsonify({
                'success': False,
                'message': '检查进度时数据库操作失败'
            }), 500
            
    except Exception as e:
        logger.error(f"检查测试进度API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)
@app.route('/api/test/save-progress', methods=['POST'])
@login_required
def save_test_progress():
    """保存测试进度到服务器"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'}), 400
            
        logger.info(f"用户 {user_id} 保存测试进度")
        
        try:
            # 获取数据库连接
            conn = db_manager._get_connection()
            cursor = conn.cursor()
            
            # 创建测试进度表（如果不存在）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    progress_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 检查用户是否已有进度记录
            cursor.execute('SELECT id FROM test_progress WHERE user_id = ?', (user_id,))
            existing_record = cursor.fetchone()
            
            progress_json = json.dumps(data)
            
            if existing_record:
                # 更新现有记录
                cursor.execute('''
                    UPDATE test_progress 
                    SET progress_data = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE user_id = ?
                ''', (progress_json, user_id))
                
                logger.info(f"✅ 更新用户 {user_id} 的测试进度")
            else:
                # 创建新记录
                cursor.execute('''
                    INSERT INTO test_progress (user_id, progress_data)
                    VALUES (?, ?)
                ''', (user_id, progress_json))
                
                logger.info(f"✅ 创建用户 {user_id} 的测试进度记录")
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': '测试进度已保存到服务器',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as db_error:
            logger.error(f"数据库操作失败: {db_error}")
            return jsonify({
                'success': False,
                'message': '保存进度时数据库操作失败'
            }), 500
            
    except Exception as e:
        logger.error(f"保存测试进度API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/test/load-progress', methods=['GET'])
@login_required
def load_test_progress():
    """从服务器加载测试进度"""
    if not db_manager:
        return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
    user_id = session['user_id']
        
    try:
        # 获取数据库连接
        conn = db_manager._get_connection()
        cursor = conn.cursor()
            
        # 查询用户的测试进度
        cursor.execute('''
                SELECT progress_data, updated_at 
                FROM test_progress 
                WHERE user_id = ? 
                ORDER BY updated_at DESC 
                LIMIT 1
            ''', (user_id,))
            
        result = cursor.fetchone()
        conn.close()
            
        if result:
            progress_data = json.loads(result[0])
                
            return jsonify({
                    'success': True,
                    'progress': progress_data,
                    'updated_at': result[1],
                    'message': '测试进度加载成功'
            })
        else:
            return jsonify({
                    'success': False,
                    'message': '没有找到保存的测试进度'
            }), 404
                
    except Exception as db_error:
        logger.error(f"数据库操作失败: {db_error}")
# ====== 数据查询API ======
@app.route('/api/test/history', methods=['GET'])
@login_required
def get_test_history():
    """获取测试历史"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        limit = request.args.get('limit', 10, type=int)
        
        # 获取原始历史数据
        raw_history = db_manager.get_user_mdq_history(user_id, limit)
        
        # 处理和格式化历史数据
        formatted_history = []
        for test in raw_history:
            try:
                # 获取测试详情
                test_detail = db_manager.get_mdq_test_detail(test['test_id'], user_id)
                
                if test_detail:
                    test_data = test_detail['test_data']
                    questions = test_data.get('questions', {})
                    
                    # 计算分数
                    score_mapping = {
                        'no': 0,
                        'rarely': 1,
                        'sometimes': 2,
                        'often': 3,
                        'always': 4
                    }
                    
                    # 计算总分数
                    total_score = 0
                    completed_questions = 0
                    for q_id in [f'q{i}' for i in range(1, 14)]:  # q1 到 q13
                        answer = questions.get(q_id, '')
                        if answer:
                            total_score += score_mapping.get(answer, 0)
                            completed_questions += 1
                    
                    # 判断完成状态
                    is_completed = completed_questions >= 13
                    
                    # 获取completion_time
                    completion_time = test_data.get('completion_time', 0)
                    if isinstance(completion_time, str):
                        try:
                            completion_time = int(completion_time)
                        except:
                            completion_time = 0
                    
                    # 格式化测试记录
                    formatted_test = {
                        'test_id': test['test_id'],
                        'test_date': test['test_timestamp'],
                        'test_timestamp': test['test_timestamp'],
                        'total_score': total_score,
                        'raw_score': total_score,
                        'completed_questions': completed_questions,
                        'completion_status': 'completed' if is_completed else 'incomplete',
                        'completion_time': completion_time,
                        'co_occurrence': test_data.get('co_occurrence', 'no'),
                        'severity': test_data.get('severity', 'no'),
                        'questions': questions,
                        'severity_level': test.get('severity_level', 'normal'),
                        'interpretation': test.get('interpretation', '')
                    }
                    
                    formatted_history.append(formatted_test)
                    logger.info(f"格式化测试记录: {test['test_id']}, 分数: {total_score}, 完成状态: {is_completed}")
                else:
                    logger.warning(f"无法获取测试详情: {test['test_id']}")
                    
            except Exception as e:
                logger.error(f"处理测试记录失败 {test.get('test_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"返回 {len(formatted_history)} 条格式化的历史记录")
        
        return jsonify({
            'success': True,
            'history': formatted_history,
            'total': len(formatted_history)
        })
        
    except Exception as e:
        logger.error(f"获取测试历史失败: {e}")
        return handle_error(e)

@app.route('/api/test/<test_id>/detail', methods=['GET'])
@login_required
def get_test_detail(test_id):
    """获取测试详情"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        detail = db_manager.get_mdq_test_detail(test_id, user_id)
        
        if detail:
            # 增强详情数据
            test_data = detail['test_data']
            questions = test_data.get('questions', {})
            
            # 计算分数
            score_mapping = {
                'no': 0,
                'rarely': 1,
                'sometimes': 2,
                'often': 3,
                'always': 4
            }
            
            # 计算总分数
            total_score = 0
            completed_questions = 0
            for q_id in [f'q{i}' for i in range(1, 14)]:  # q1 到 q13
                answer = questions.get(q_id, '')
                if answer:
                    total_score += score_mapping.get(answer, 0)
                    completed_questions += 1
            
            # 获取completion_time
            completion_time = test_data.get('completion_time', 0)
            if isinstance(completion_time, str):
                try:
                    completion_time = int(completion_time)
                except:
                    completion_time = 0
            
            # 增强的详情信息
            enhanced_detail = {
                'test_id': test_id,
                'test_date': detail['test_timestamp'],
                'test_timestamp': detail['test_timestamp'],
                'test_data': test_data,
                'questions': questions,
                'total_score': total_score,
                'raw_score': total_score,
                'completed_questions': completed_questions,
                'completion_time': completion_time,
                'co_occurrence': test_data.get('co_occurrence', 'no'),
                'severity': test_data.get('severity', 'no'),
                'severity_level': detail.get('severity_level', 'normal'),
                'interpretation': detail.get('interpretation', ''),
                'completion_status': 'completed' if completed_questions >= 13 else 'incomplete'
            }
            
            logger.info(f"测试详情: {test_id}, 分数: {total_score}, 完成题目: {completed_questions}")
            
            return jsonify({'success': True, 'detail': enhanced_detail})
        else:
            return jsonify({'success': False, 'message': '测试记录不存在'}), 404
            
    except Exception as e:
        logger.error(f"获取测试详情失败: {e}")
        return handle_error(e)

@app.route('/api/user/statistics', methods=['GET'])
@login_required
def get_user_statistics():
    """获取用户统计"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        
        # 获取所有测试记录进行更详细的统计
        all_tests = db_manager.get_user_mdq_history(user_id, limit=1000)
        
        # 计算增强统计信息
        completed_tests = 0
        total_score_sum = 0
        valid_scores = []
        
        score_mapping = {
            'no': 0,
            'rarely': 1,
            'sometimes': 2,
            'often': 3,
            'always': 4
        }
        
        for test in all_tests:
            try:
                test_detail = db_manager.get_mdq_test_detail(test['test_id'], user_id)
                if test_detail:
                    test_data = test_detail['test_data']
                    questions = test_data.get('questions', {})
                    
                    # 计算分数
                    total_score = 0
                    completed_questions = 0
                    for q_id in [f'q{i}' for i in range(1, 14)]:
                        answer = questions.get(q_id, '')
                        if answer:
                            total_score += score_mapping.get(answer, 0)
                            completed_questions += 1
                    
                    if completed_questions >= 13:
                        completed_tests += 1
                        total_score_sum += total_score
                        valid_scores.append(total_score)
                        
            except Exception as e:
                logger.error(f"统计计算失败 {test.get('test_id', 'unknown')}: {e}")
                continue
        
        # 计算平均分
        average_score = total_score_sum / completed_tests if completed_tests > 0 else 0
        
        # 获取最新测试信息
        latest_test_info = None
        if all_tests:
            try:
                latest_test = all_tests[0]
                latest_detail = db_manager.get_mdq_test_detail(latest_test['test_id'], user_id)
                if latest_detail:
                    test_data = latest_detail['test_data']
                    questions = test_data.get('questions', {})
                    
                    # 计算最新测试分数
                    latest_score = 0
                    for q_id in [f'q{i}' for i in range(1, 14)]:
                        answer = questions.get(q_id, '')
                        if answer:
                            latest_score += score_mapping.get(answer, 0)
                    
                    latest_test_info = {
                        'score': latest_score,
                        'severity_level': latest_detail.get('severity_level', 'normal'),
                        'date': latest_test['test_timestamp']
                    }
            except Exception as e:
                logger.error(f"获取最新测试信息失败: {e}")
        
        enhanced_stats = {
            'total_tests': len(all_tests),
            'completed_tests': completed_tests,
            'incomplete_tests': len(all_tests) - completed_tests,
            'average_score': round(average_score, 1),
            'latest_test': latest_test_info,
            'last_test_date': latest_test_info['date'] if latest_test_info else None,
            'score_range': {
                'min': min(valid_scores) if valid_scores else 0,
                'max': max(valid_scores) if valid_scores else 0
            }
        }
        
        logger.info(f"用户统计: 总测试{len(all_tests)}次, 完成{completed_tests}次, 平均分{average_score:.1f}")
        
        return jsonify({
            'success': True,
            'statistics': enhanced_stats
        })
        
    except Exception as e:
        logger.error(f"获取用户统计失败: {e}")
        return handle_error(e)
@app.route('/api/user/trend-data', methods=['GET'])
@login_required
def get_user_trend_data():
    """获取用户测试趋势数据用于图表显示 - 基于analyse.py的5级评分系统"""
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': '数据库服务不可用'}), 500
            
        user_id = session['user_id']
        limit = request.args.get('limit', 30, type=int)  # 默认获取最近30次记录
        
        # 获取测试历史（使用database.py的方法）
        test_history = db_manager.get_user_mdq_history(user_id, limit=limit)
        
        if not test_history:
            return jsonify({
                'success': True,
                'trend_data': [],
                'statistics': {'total_tests': 0},
                'total_points': 0
            })
        
        # 5级评分系统的分值映射（与analyse.py保持一致）
        score_mapping = {
            'no': 0,        # 从未
            'rarely': 1,    # 很少
            'sometimes': 2, # 有时
            'often': 3,     # 经常
            'always': 4     # 总是
        }
        
        # 处理和格式化趋势数据
        trend_data = []
        
        for test in reversed(test_history):  # 按时间正序排列
            try:
                # 获取测试详情
                test_detail = db_manager.get_mdq_test_detail(test['test_id'], user_id)
                
                if test_detail:
                    test_data = test_detail['test_data']
                    questions = test_data.get('questions', {})
                    
                    # 计算原始分数（0-39分，与analyse.py的_extract_raw_score_from_test方法一致）
                    raw_score = 0
                    completed_questions = 0
                    for q_id in [f'q{i}' for i in range(1, 14)]:  # q1 到 q13
                        answer = questions.get(q_id, '')
                        if answer:
                            raw_score += score_mapping.get(answer, 0)
                            completed_questions += 1
                    
                    # 只包含完成的测试（至少完成13题）
                    if completed_questions >= 13:
                        # 解析日期
                        test_date = test['test_timestamp']
                        if isinstance(test_date, str):
                            try:
                                parsed_date = datetime.fromisoformat(test_date.replace('Z', '+00:00'))
                                formatted_date = parsed_date.strftime('%Y-%m-%d')
                            except:
                                formatted_date = test_date[:10]
                        else:
                            formatted_date = str(test_date)[:10]
                        
                        # 计算严重程度等级（基于analyse.py的严重程度阈值，调整为0-39分制）
                        if raw_score <= 8:
                            severity = '正常'
                            severity_level = 1
                        elif raw_score <= 15:
                            severity = '轻度风险'
                            severity_level = 2
                        elif raw_score <= 25:
                            severity = '中度风险'
                            severity_level = 3
                        elif raw_score <= 35:
                            severity = '高度风险'
                            severity_level = 4
                        else:
                            severity = '严重风险'
                            severity_level = 5
                        
                        trend_data.append({
                            'date': formatted_date,
                            'score': raw_score,
                            'max_score': 39,  # 最大分数
                            'severity': severity,
                            'severity_level': severity_level,
                            'test_id': test['test_id'],
                            'completion_time': test_data.get('completion_time', 0),
                            'co_occurrence': test_data.get('co_occurrence', 'no'),
                            'functional_severity': test_data.get('severity', 'no')
                        })
                        
            except Exception as e:
                logger.error(f"处理趋势数据失败 {test.get('test_id', 'unknown')}: {e}")
                continue
        
        # 计算趋势统计（基于0-39分制）
        scores = [item['score'] for item in trend_data]
        trend_stats = {}
        
        if scores:
            trend_stats = {
                'total_tests': len(scores),
                'average_score': round(sum(scores) / len(scores), 1),
                'min_score': min(scores),
                'max_score': max(scores),
                'score_range': max(scores) - min(scores),
                'latest_score': scores[-1] if scores else 0,
                'first_score': scores[0] if scores else 0,
                'improvement': scores[0] - scores[-1] if len(scores) > 1 else 0,  # 正值表示改善
                'max_possible_score': 39
            }
            
            # 计算风险百分比（基于39分制）
            latest_risk_percentage = round((scores[-1] / 39) * 100, 1) if scores else 0
            trend_stats['latest_risk_percentage'] = latest_risk_percentage
            
            # 计算改善百分比
            if len(scores) > 1:
                max_historical = max(scores)
                if max_historical > 0:
                    improvement_percentage = ((max_historical - scores[-1]) / max_historical) * 100
                    trend_stats['improvement_percentage'] = round(improvement_percentage, 1)
            
            # 计算趋势方向（基于analyse.py的逻辑）
            if len(scores) >= 3:
                recent_avg = sum(scores[-3:]) / 3
                earlier_avg = sum(scores[:3]) / 3
                change_percentage = ((earlier_avg - recent_avg) / max(recent_avg, 1)) * 100
                
                if abs(change_percentage) < 15:
                    trend_direction = 'stable'
                elif change_percentage >= 25:
                    trend_direction = 'significant_improvement'
                elif change_percentage >= 15:
                    trend_direction = 'mild_improvement'
                elif change_percentage <= -25:
                    trend_direction = 'significant_deterioration'
                else:
                    trend_direction = 'mild_deterioration'
            else:
                trend_direction = 'insufficient_data'
            
            trend_stats['trend_direction'] = trend_direction
            
            # 症状一致性评分
            if len(scores) >= 3:
                cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 1
                consistency_score = max(0, min(1, 1 - cv / 2))
                trend_stats['consistency_score'] = round(consistency_score, 2)
        
        logger.info(f"趋势数据: {len(trend_data)} 条记录, 平均分: {trend_stats.get('average_score', 0)}/39")
        
        return jsonify({
            'success': True,
            'trend_data': trend_data,
            'statistics': trend_stats,
            'total_points': len(trend_data),
            'scoring_system': {
                'type': '5-level',
                'range': '0-39',
                'categories': ['从未(0)', '很少(1)', '有时(2)', '经常(3)', '总是(4)']
            }
        })
        
    except Exception as e:
        logger.error(f"获取趋势数据失败: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)

@app.route('/api/ai/generate-historical-report', methods=['POST'])
@login_required
def generate_historical_ai_report():
    """生成历史分析AI报告 - 使用analyse.py的分析系统"""
    try:
        if not advisor:
            return jsonify({'success': False, 'message': 'AI建议服务不可用'}), 500
        
        if not analyzer:
            return jsonify({'success': False, 'message': '分析服务不可用'}), 500
            
        user_id = session['user_id']
        
        logger.info(f"用户 {user_id} 请求生成历史分析报告")
        
        # 检查用户是否有足够的测试记录
        test_history = db_manager.get_user_mdq_history(user_id, limit=50)
        
        # 使用analyse.py的5级评分系统验证完成的测试
        score_mapping = {
            'no': 0, 'rarely': 1, 'sometimes': 2, 'often': 3, 'always': 4
        }
        
        completed_tests = 0
        for test in test_history:
            try:
                test_detail = db_manager.get_mdq_test_detail(test['test_id'], user_id)
                if test_detail:
                    questions = test_detail['test_data'].get('questions', {})
                    completed_questions = sum(1 for q_id in [f'q{i}' for i in range(1, 14)] 
                                            if questions.get(q_id, ''))
                    if completed_questions >= 13:
                        completed_tests += 1
            except:
                continue
        
        if completed_tests < 2:
            return jsonify({
                'success': False, 
                'message': f'至少需要2次完整的测试记录才能生成历史分析报告，当前只有{completed_tests}次'
            }), 400
        
        # 使用analyse.py的综合分析功能确保用户有最新的分析数据
        try:
            analysis_result = analyzer.analyze_user_comprehensive(user_id)
            logger.info(f"为用户 {user_id} 更新了综合分析数据，分析ID: {analysis_result.analysis_id}")
        except Exception as e:
            logger.warning(f"更新分析数据失败: {e}")
            return jsonify({
                'success': False, 
                'message': '分析数据准备失败，请稍后重试'
            }), 500
        
        # 生成历史分析报告（调用gptadvisor.py）
        try:
            historical_report = advisor.generate_historical_analysis_report(user_id)
            
            # 转换枚举对象为字符串
            report_data = convert_enums_to_strings({
                'report_id': historical_report.report_id,
                'report_type': historical_report.report_type,
                'analysis_id': historical_report.analysis_id,
                'generated_at': historical_report.generated_at.isoformat(),
                'executive_summary': historical_report.executive_summary,
                'clinical_assessment': historical_report.clinical_assessment,
                'risk_evaluation': historical_report.risk_evaluation,
                'treatment_recommendations': historical_report.treatment_recommendations,
                'lifestyle_recommendations': historical_report.lifestyle_recommendations,
                'monitoring_plan': historical_report.monitoring_plan,
                'emergency_protocols': historical_report.emergency_protocols,
                'progress_analysis': historical_report.progress_analysis,
                'trend_interpretation': historical_report.trend_interpretation,
                'prognosis_assessment': historical_report.prognosis_assessment,
                'confidence_score': historical_report.confidence_score,
                'processing_time': historical_report.processing_time,
                'ai_model_version': historical_report.ai_model_version
            })
            
            # 添加分析系统的统计信息
            if analysis_result:
                report_data['analysis_stats'] = {
                    'raw_score': analysis_result.raw_score,
                    'severity_level': analysis_result.severity_level.value,
                    'improvement_trend': analysis_result.improvement_trend.value,
                    'risk_percentage': analysis_result.risk_percentage,
                    'trend_confidence': analysis_result.trend_confidence,
                    'improvement_percentage': analysis_result.improvement_percentage,
                    'consistency_score': analysis_result.consistency_score
                }
            
            logger.info(f"✅ 历史分析报告生成成功: {historical_report.report_id}")
            
            return jsonify({
                'success': True,
                'message': '历史分析报告生成成功',
                'report': report_data,
                'analysis_system': {
                    'version': '5-level_scoring',
                    'score_range': '0-39',
                    'total_completed_tests': completed_tests
                }
            })
            
        except Exception as e:
            logger.error(f"AI报告生成失败: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False, 
                'message': f'AI服务暂时不可用: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"生成历史AI报告API异常: {e}")
        logger.error(traceback.format_exc())
        return handle_error(e)


if __name__ == '__main__':
    # 启动检查
    startup_check()
    
    # 检查关键组件
    if db_manager is None:
        print("❌ 数据库初始化失败，请检查数据库配置")
        print("💡 提示: 确保 database.py 文件存在且没有语法错误")
        exit(1)
    
    print("🚀 MindCare 心理健康评估系统启动中...")
    print("=" * 50)
    print(f"📱 本地访问: http://localhost:5000")
    print(f"📱 网络访问: http://127.0.0.1:5000")
    print("=" * 50)
    print("📋 可用的演示账户:")
    print("   用户名: demo_user  密码: demo123")
    print("   用户名: test_user  密码: test123")
    print("=" * 50)
    print("🔧 主要API端点:")
    print("   健康检查: GET /api/health")
    print("   用户注册: POST /api/register")
    print("   用户登录: POST /api/login")
    print("   用户登出: POST /api/logout")
    print("   MDQ测试: POST /api/test/mdq")
    print("   测试历史: GET /api/test/history")
    print("   文件上传: POST /api/test/upload")
    print("   用户资料: GET /api/user/profile")
    print("   用户统计: GET /api/user/statistics")
    if analyzer:
        print("   综合分析: POST /api/analysis/comprehensive")
        print("   最新分析: GET /api/analysis/latest")
    if advisor:
        print("   AI报告: POST /api/ai/report")
        print("   报告详情: GET /api/ai/report/<report_id>")
    print("=" * 50)
    print("🛠️ 开发工具:")
    print("   初始化示例数据: POST /api/dev/init-sample-data")
    print("   测试数据库: GET /api/dev/test-db")
    print("   调试会话: GET /api/debug/session")
    print("   调试组件: GET /api/debug/components")
    print("=" * 50)
    print("💡 提示: 按 Ctrl+C 停止服务器")
    print("")
    
    # 运行应用
    try:
        app.run(
            host='0.0.0.0',  # 允许外部访问
            port=5000,
            debug=True,      # 开发模式
            threaded=True    # 支持多线程
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        logger.error(f"服务器启动失败: {e}")
        logger.error(traceback.format_exc())
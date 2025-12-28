#!/usr/bin/env python3
"""
MindCare 心理健康评估系统启动脚本
"""
import os
import sys
import subprocess
import platform
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    return True

def install_requirements():
    """安装依赖包"""
    requirements = [
        'flask>=2.0.0',
        'flask-cors>=4.0.0',
        'numpy>=1.21.0',
        'openai>=1.0.0',
        'werkzeug>=2.0.0'
    ]
    
    print("📦 检查并安装依赖包...")
    for requirement in requirements:
        try:
            __import__(requirement.split('>=')[0].replace('-', '_'))
        except ImportError:
            print(f"安装 {requirement}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])

def check_files():
    """检查必要文件是否存在"""
    required_files = [
        'database.py',
        'analyse.py', 
        'gptadvisor.py',
        'index.html',
        'login.html',
        'dashboard.html',
        'test.html'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ 所有必要文件检查完成")
    return True

def create_directories():
    """创建必要目录"""
    directories = ['uploads', 'static', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ 目录结构创建完成")

def init_database():
    """初始化数据库"""
    try:
        print("🔍 初始化数据库...")
        from database import DatabaseManager
        db_manager = DatabaseManager()
        
        # 测试数据库连接
        conn = db_manager._get_connection()
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        users_table = cursor.fetchone()
        
        if users_table:
            print("✅ 数据库表已存在")
        else:
            print("⚠️  数据库表不存在，正在创建...")
        
        # 统计现有用户
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"✅ 现有用户数量: {user_count}")
        
        conn.close()
        
        # 如果没有用户，创建测试用户
        if user_count == 0:
            print("🔧 创建测试用户...")
            create_test_users(db_manager)
        
        print("✅ 数据库初始化完成")
        return True
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_users(db_manager):
    """创建测试用户"""
    test_users = [
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
    
    created_count = 0
    for user_data in test_users:
        try:
            result = db_manager.register_user(user_data)
            if result['success']:
                print(f"  ✅ 创建用户: {user_data['username']}")
                created_count += 1
            else:
                print(f"  ⚠️  用户可能已存在: {user_data['username']}")
        except Exception as e:
            print(f"  ❌ 创建用户失败: {user_data['username']} - {e}")
    
    print(f"✅ 测试用户准备完成 ({created_count} 个新用户)")

def check_api_key():
    """检查API密钥配置"""
    try:
        with open('gptadvisor.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'your_deepseek_api_key_here' in content:
                print("⚠️  警告: DeepSeek API密钥未配置")
                print("   AI功能将不可用，请在 gptadvisor.py 中设置正确的API密钥")
                return False
    except:
        pass
    return True

def get_local_ip():
    """获取本机IP地址"""
    import socket
    try:
        # 连接到外部地址来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def main():
    """主函数"""
    print("🧠 MindCare 心理健康评估系统")
    print("=" * 60)
    
    # 系统检查
    print("🔍 系统环境检查...")
    
    if not check_python_version():
        return False
    
    if not check_files():
        return False
    
    print(f"✅ Python版本: {sys.version}")
    print(f"✅ 操作系统: {platform.system()} {platform.release()}")
    
    # 安装依赖
    try:
        install_requirements()
        print("✅ 依赖包检查完成")
    except Exception as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False
    
    # 创建目录
    create_directories()
    
    # 初始化数据库
    if not init_database():
        return False
    
    # 检查API密钥
    api_configured = check_api_key()
    
    print("=" * 60)
    print("🚀 启动Web服务器...")
    
    # 获取访问地址
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("📱 系统访问地址:")
    print(f"   本地访问: http://localhost:5000")
    print(f"   本地访问: http://127.0.0.1:5000")
    if local_ip != "127.0.0.1":
        print(f"   局域网访问: http://{local_ip}:5000")
    print("=" * 60)
    print("👤 演示账户:")
    print("   用户名: demo_user  密码: demo123")
    print("   用户名: test_user  密码: test123")
    print("=" * 60)
    print("🔧 开发工具:")
    print("   数据库诊断: python database_fix.py")
    print("   初始化示例数据: POST /api/dev/init-sample-data")
    print("   数据库测试: GET /api/dev/test-db")
    print("=" * 60)
    
    if not api_configured:
        print("⚠️  AI功能提醒:")
        print("   当前AI分析功能不可用")
        print("   请在 gptadvisor.py 中配置 DeepSeek API 密钥")
        print("=" * 60)
    
    print("💡 提示:")
    print("   - 按 Ctrl+C 停止服务器")
    print("   - 服务器启动后会自动打开浏览器")
    print("   - 可以分享局域网地址给其他设备访问")
    print("=" * 60)
    
    # 启动Flask应用
    try:
        from app import app
        
        # 延迟打开浏览器
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # 启动应用
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # 生产模式
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 感谢使用 MindCare 系统！")
        return True
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        print("\n❌ 启动失败，请检查错误信息并重试")
        input("按回车键退出...")
        sys.exit(1)
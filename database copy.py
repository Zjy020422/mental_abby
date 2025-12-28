import sqlite3
from datetime import datetime
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
from enum import Enum
from dataclasses import dataclass
class SeverityLevel(Enum):
    """患病严重程度等级"""
    NORMAL = "normal"              # 正常状态 (0-3分)
    MILD_RISK = "mild_risk"        # 轻度风险 (4-8分)
    MODERATE_RISK = "moderate_risk" # 中度风险 (9-15分)
    HIGH_RISK = "high_risk"        # 高度风险 (16-25分)
    SEVERE_RISK = "severe_risk"    # 严重风险 (26-39分)

class ImprovementTrend(Enum):
    """改善趋势"""
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"      # 显著改善 (≥40%改善)
    MODERATE_IMPROVEMENT = "moderate_improvement"           # 中度改善 (20-40%改善)
    MILD_IMPROVEMENT = "mild_improvement"                   # 轻度改善 (10-20%改善)
    STABLE = "stable"                                       # 稳定 (±10%变化)
    MILD_DETERIORATION = "mild_deterioration"               # 轻度恶化 (10-20%恶化)
    MODERATE_DETERIORATION = "moderate_deterioration"       # 中度恶化 (20-40%恶化)
    SIGNIFICANT_DETERIORATION = "significant_deterioration" # 显著恶化 (≥40%恶化)

@dataclass
class AnalysisResult:
    """分析结果数据类"""
    # 基础信息
    user_id: str
    analysis_id: str
    analysis_date: datetime
    
    # 当前状态评估
    current_score: int
    raw_score: int  # 原始分数（0-39）
    severity_level: SeverityLevel
    risk_percentage: float
    bipolar_risk_indicators: Dict[str, bool]
    
    # 症状分析
    positive_symptoms: List[str]
    symptom_categories: Dict[str, int]
    symptom_severity_scores: Dict[str, float]  # 各症状类别的严重程度分数
    functional_impairment_level: str
    
    # 历史趋势分析
    improvement_trend: ImprovementTrend
    trend_confidence: float
    historical_baseline: float
    improvement_percentage: float
    consistency_score: float
    
    # 改善情况详细分析
    recovery_indicators: List[str]
    risk_factors: List[str]
    treatment_response_indicators: Dict[str, float]
    
    # AI分析准备数据
    ai_analysis_data: Dict
    
    # 建议和预测
    clinical_recommendations: List[str]
    monitoring_frequency: int
    emergency_flag: bool
    next_assessment_date: datetime
class DatabaseManager:
    """数据库管理类，负责创建和管理数据库表结构"""
    
    def __init__(self, db_path="mental_health_assessment.db"):
        self.db_path = db_path
        self.symptom_categories = {
            'elevated_mood': ['q1'],                 # 情绪高涨（问题1）
            'inflated_self_esteem': ['q2'],          # 自负/自信（问题2）
            'decreased_sleep': ['q3'],               # 睡眠减少（问题3）
            'increased_talkativeness': ['q4'],       # 话多（问题4）
            'racing_thoughts': ['q5'],               # 思维奔逸（问题5）
            'distractibility': ['q6'],               # 注意力分散（问题6）
            'increased_activity': ['q7'],            # 活动增加（问题7）
            'social_disinhibition': ['q8'],          # 社交去抑制（问题8）
            'hypersexuality': ['q9'],                # 性欲亢进（问题9）
            'poor_judgment': ['q10'],                # 判断力差/冲动决定（问题10）
            'reckless_spending': ['q11'],            # 冲动消费（问题11）
            'functional_impairment': ['q12'],        # 功能损害（问题12）
            'others_noticed': ['q13']                # 他人注意到（问题13）
        }
        
        # 5级评分系统的分值映射
        self.score_mapping = {
            'no': 0,        # 从未
            'rarely': 1,    # 很少
            'sometimes': 2, # 有时
            'often': 3,     # 经常
            'always': 4     # 总是
        }
        
        # 严重程度阈值（基于0-39分的总分）
        self.severity_thresholds = {
            SeverityLevel.NORMAL: (0, 3),
            SeverityLevel.MILD_RISK: (4, 8),
            SeverityLevel.MODERATE_RISK: (9, 15),
            SeverityLevel.HIGH_RISK: (16, 25),
            SeverityLevel.SEVERE_RISK: (26, 39)
        }
        
        # 症状权重 (临床重要性调整)
        self.symptom_weights = {
            'q1': 1.2,   'q2': 1.1,   'q3': 1.5,   'q4': 1.0,
            'q5': 1.8,   'q6': 1.2,   'q7': 1.1,   'q8': 1.4,
            'q9': 1.6,   'q10': 1.9,  'q11': 1.7,  'q12': 2.0,
            'q13': 1.8
        }
        
        # 双相障碍高危指标组合
        self.bipolar_indicators = {
            'core_manic_symptoms': ['q1', 'q2', 'q3', 'q5'],     # 核心躁狂症状
            'behavioral_symptoms': ['q8', 'q9', 'q10', 'q11'],   # 行为症状
            'social_impact': ['q12', 'q13'],                      # 社会功能影响
            'cognitive_symptoms': ['q5', 'q6']                    # 认知症状
        }
        
        # 症状描述映射（用于生成阳性症状列表）
        self.symptom_descriptions = {
            'q1': '情绪异常高涨或兴奋',
            'q2': '过度自信或自我感觉良好',
            'q3': '睡眠需求减少但仍感到精力充沛',
            'q4': '比平时更健谈或语速更快',
            'q5': '思维飞跃或感觉思维加速',
            'q6': '注意力容易分散',
            'q7': '精力充沛，异常活跃',
            'q8': '比平时更爱社交或更外向',
            'q9': '对性的兴趣比平时更强烈',
            'q10': '做出不寻常或冲动的决定',
            'q11': '花钱比平时更冲动或不理智',
            'q12': '行为变化对工作学习或人际关系造成困扰',
            'q13': '家人朋友或医生注意到行为变化'
        }
        self.init_database()
    
    # 在 DatabaseManager 的 init_database 方法中，更新 questionnaire_tests 表结构
    def init_database(self):
        """初始化数据库和表结构"""
        try:
            # 确保数据库文件目录存在
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建用户信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    email TEXT UNIQUE,
                    full_name TEXT,
                    gender TEXT CHECK(gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
                    age INTEGER CHECK(age >= 0 AND age <= 150),
                    phone TEXT,
                    occupation TEXT,
                    education_level TEXT,
                    emergency_contact TEXT,
                    registration_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    failed_login_attempts INTEGER DEFAULT 0,
                    last_failed_login DATETIME
                )
            ''')
            
            # **修复：更新问卷测试表，添加 ai_analysis_data 字段**
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS questionnaire_tests (
                    test_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    questionnaire_type TEXT NOT NULL DEFAULT 'MDQ',
                    test_data TEXT NOT NULL,
                    test_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    raw_score INTEGER,
                    interpretation TEXT,
                    severity_level TEXT,
                    completion_time INTEGER,
                    ai_analysis_data TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            ''')
            
            # **检查并添加 ai_analysis_data 字段（如果表已存在但缺少该字段）**
            try:
                cursor.execute("PRAGMA table_info(questionnaire_tests)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'ai_analysis_data' not in columns:
                    cursor.execute('ALTER TABLE questionnaire_tests ADD COLUMN ai_analysis_data TEXT')
                    print("✅ 已添加 ai_analysis_data 字段到 questionnaire_tests 表")
            except sqlite3.Error as e:
                print(f"⚠️ 添加字段时出错: {e}")
            
            # 其余表创建代码保持不变...
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    severity_level TEXT NOT NULL,
                    generated_by TEXT DEFAULT 'system',
                    generation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_read BOOLEAN DEFAULT 0,
                    follow_up_date DATE,
                    FOREIGN KEY (test_id) REFERENCES questionnaire_tests (test_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_tests ON questionnaire_tests(user_id, test_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_timestamp ON questionnaire_tests(test_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_recommendations ON test_recommendations(user_id, generation_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_email ON users(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions ON user_sessions(user_id, is_active)')
            
            conn.commit()
            print("数据库表结构初始化完成")
            
        except sqlite3.Error as e:
            print(f"数据库初始化失败: {e}")
            raise e
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """密码哈希函数"""
        if salt is None:
            salt = str(uuid.uuid4())
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 迭代次数
        ).hex()
        
        return password_hash, salt
    
    def _get_connection(self):
        """获取数据库连接"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')  # 启用WAL模式提高并发性能
            return conn
        except sqlite3.Error as e:
            print(f"数据库连接失败: {e}")
            raise e
    
    # ==================== 用户注册和登录功能 ====================
    
    def register_user(self, user_data: Dict) -> Dict:
        """注册新用户"""
        conn = None
        try:
            print(f"开始注册用户: {user_data.get('username')}")
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 检查用户名是否已存在
            cursor.execute('SELECT username FROM users WHERE username = ?', (user_data['username'],))
            existing_username = cursor.fetchone()
            
            if existing_username:
                print(f"用户名已存在: {user_data['username']}")
                return {'success': False, 'message': '用户名已存在'}
            
            # 检查邮箱是否已存在（如果提供了邮箱）
            if user_data.get('email'):
                cursor.execute('SELECT email FROM users WHERE email = ?', (user_data['email'],))
                existing_email = cursor.fetchone()
                
                if existing_email:
                    print(f"邮箱已存在: {user_data['email']}")
                    return {'success': False, 'message': '邮箱已被使用'}
            
            # 生成用户ID和密码哈希
            user_id = str(uuid.uuid4())
            password_hash, salt = self._hash_password(user_data['password'])
            
            print(f"生成用户ID: {user_id}")
            print(f"密码哈希完成")
            
            # 插入用户数据
            cursor.execute('''
                INSERT INTO users 
                (user_id, username, password_hash, salt, email, full_name, gender, age, 
                 phone, occupation, education_level, emergency_contact, registration_date, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1)
            ''', (
                user_id,
                user_data['username'],
                password_hash,
                salt,
                user_data.get('email'),
                user_data.get('full_name'),
                user_data.get('gender'),
                user_data.get('age'),
                user_data.get('phone'),
                user_data.get('occupation'),
                user_data.get('education_level'),
                user_data.get('emergency_contact')
            ))
            
            # 检查插入是否成功
            if cursor.rowcount == 0:
                print("插入用户失败: 没有行被插入")
                return {'success': False, 'message': '用户注册失败'}
            
            conn.commit()
            print(f"用户注册成功: {user_data['username']}, ID: {user_id}")
            
            # 验证插入结果
            cursor.execute('SELECT username FROM users WHERE user_id = ?', (user_id,))
            verify_result = cursor.fetchone()
            if not verify_result:
                print("验证失败: 用户未成功保存到数据库")
                return {'success': False, 'message': '用户注册失败，请重试'}
            
            return {'success': True, 'user_id': user_id, 'message': '注册成功'}
            
        except sqlite3.IntegrityError as e:
            print(f"数据库完整性错误: {e}")
            conn.rollback() if conn else None
            if 'username' in str(e).lower():
                return {'success': False, 'message': '用户名已存在'}
            elif 'email' in str(e).lower():
                return {'success': False, 'message': '邮箱已被使用'}
            else:
                return {'success': False, 'message': '注册失败，数据格式错误'}
        except sqlite3.Error as e:
            print(f"注册用户数据库错误: {e}")
            conn.rollback() if conn else None
            return {'success': False, 'message': '注册失败，数据库错误'}
        except Exception as e:
            print(f"注册用户未知错误: {e}")
            conn.rollback() if conn else None
            return {'success': False, 'message': f'注册失败: {str(e)}'}
        finally:
            if conn:
                conn.close()
    
    def login_user(self, username: str, password: str, ip_address: str = None, user_agent: str = None) -> Dict:
        """用户登录"""
        conn = None
        try:
            print(f"开始登录验证用户: {username}")
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 获取用户信息
            cursor.execute('''
                SELECT user_id, password_hash, salt, is_active, failed_login_attempts, last_failed_login
                FROM users 
                WHERE username = ? OR email = ?
            ''', (username, username))
            
            user_record = cursor.fetchone()
            
            if not user_record:
                print(f"用户不存在: {username}")
                return {'success': False, 'message': '用户不存在'}
            
            user_id, stored_hash, salt, is_active, failed_attempts, last_failed = user_record
            print(f"找到用户: {user_id}")
            
            # 检查账户是否被锁定
            if failed_attempts >= 5:
                if last_failed:
                    try:
                        last_failed_time = datetime.fromisoformat(last_failed.replace('Z', '+00:00'))
                        if (datetime.now() - last_failed_time).total_seconds() < 900:  # 15分钟锁定
                            return {'success': False, 'message': '账户已被锁定，请15分钟后重试'}
                    except:
                        pass  # 忽略日期解析错误
            
            if not is_active:
                return {'success': False, 'message': '账户已被禁用'}
            
            # 验证密码
            password_hash, _ = self._hash_password(password, salt)
            
            if password_hash != stored_hash:
                print(f"密码错误: {username}")
                # 更新失败登录次数
                cursor.execute('''
                    UPDATE users 
                    SET failed_login_attempts = failed_login_attempts + 1, 
                        last_failed_login = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (user_id,))
                conn.commit()
                return {'success': False, 'message': '密码错误'}
            
            print(f"密码验证成功: {username}")
            
            # 登录成功，创建会话
            session_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO user_sessions (session_id, user_id, ip_address, user_agent)
                VALUES (?, ?, ?, ?)
            ''', (session_id, user_id, ip_address, user_agent))
            
            # 更新用户登录信息
            cursor.execute('''
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP, failed_login_attempts = 0
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            print(f"登录成功: {username}, 会话ID: {session_id}")
            
            return {
                'success': True, 
                'user_id': user_id, 
                'session_id': session_id,
                'message': '登录成功'
            }
            
        except sqlite3.Error as e:
            print(f"用户登录数据库错误: {e}")
            return {'success': False, 'message': '登录失败，请稍后重试'}
        except Exception as e:
            print(f"用户登录未知错误: {e}")
            return {'success': False, 'message': f'登录失败: {str(e)}'}
        finally:
            if conn:
                conn.close()
    
    def logout_user(self, session_id: str) -> bool:
        """用户登出"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions 
                SET is_active = 0 
                WHERE session_id = ?
            ''', (session_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            print(f"用户登出失败: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def verify_session(self, session_id: str) -> Optional[str]:
        """验证会话有效性，返回用户ID"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM user_sessions 
                WHERE session_id = ? AND is_active = 1
            ''', (session_id,))
            
            result = cursor.fetchone()
            
            if result:
                # 更新最后活动时间
                cursor.execute('''
                    UPDATE user_sessions 
                    SET last_activity = CURRENT_TIMESTAMP 
                    WHERE session_id = ?
                ''', (session_id,))
                conn.commit()
                return result[0]
            
            return None
            
        except sqlite3.Error as e:
            print(f"验证会话失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    # ==================== 个人数据管理功能 ====================
    
    def update_user_profile(self, user_id: str, profile_data: Dict) -> Dict:
        """更新用户个人资料"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 构建动态更新SQL
            allowed_fields = ['email', 'full_name', 'gender', 'age', 'phone', 
                            'occupation', 'education_level', 'emergency_contact']
            
            update_fields = []
            update_values = []
            
            for field, value in profile_data.items():
                if field in allowed_fields and value is not None:
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            if not update_fields:
                return {'success': False, 'message': '没有有效的更新字段'}
            
            update_values.append(user_id)
            
            cursor.execute(f'''
                UPDATE users 
                SET {', '.join(update_fields)}
                WHERE user_id = ?
            ''', update_values)
            
            conn.commit()
            
            if cursor.rowcount > 0:
                return {'success': True, 'message': '个人资料更新成功'}
            else:
                return {'success': False, 'message': '用户不存在'}
                
        except sqlite3.Error as e:
            print(f"更新用户资料失败: {e}")
            return {'success': False, 'message': f'更新失败: {str(e)}'}
        finally:
            if conn:
                conn.close()
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """获取用户个人资料"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT username, email, full_name, gender, age, phone, 
                       occupation, education_level, emergency_contact,
                       registration_date, last_login
                FROM users 
                WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'username': result[0],
                    'email': result[1],
                    'full_name': result[2],
                    'gender': result[3],
                    'age': result[4],
                    'phone': result[5],
                    'occupation': result[6],
                    'education_level': result[7],
                    'emergency_contact': result[8],
                    'registration_date': result[9],
                    'last_login': result[10]
                }
            
            return None
            
        except sqlite3.Error as e:
            print(f"获取用户资料失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> Dict:
        """修改用户密码"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 验证旧密码
            cursor.execute('SELECT password_hash, salt FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            
            if not result:
                return {'success': False, 'message': '用户不存在'}
            
            stored_hash, salt = result
            old_password_hash, _ = self._hash_password(old_password, salt)
            
            if old_password_hash != stored_hash:
                return {'success': False, 'message': '原密码错误'}
            
            # 设置新密码
            new_password_hash, new_salt = self._hash_password(new_password)
            
            cursor.execute('''
                UPDATE users 
                SET password_hash = ?, salt = ?
                WHERE user_id = ?
            ''', (new_password_hash, new_salt, user_id))
            
            conn.commit()
            
            return {'success': True, 'message': '密码修改成功'}
            
        except sqlite3.Error as e:
            print(f"修改密码失败: {e}")
            return {'success': False, 'message': f'修改密码失败: {str(e)}'}
        finally:
            if conn:
                conn.close()
    
    # ==================== MDQ问卷数据管理功能 ====================
    def save_mdq_test(self, user_id: str, test_data: Dict, completion_time: int = None) -> Dict:
        """保存MDQ问卷测试结果"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            test_id = str(uuid.uuid4())
            
            # 计算得分和解释
            score_result = self._calculate_mdq_score(test_data)
            print(f"Score result: {score_result}")
            
            # 确保 severity_level 是字符串格式
            severity_level_str = score_result['severity_level'].value if hasattr(score_result['severity_level'], 'value') else str(score_result['severity_level'])
            
            # 确保 interpretation 是字符串格式
            interpretation_str = json.dumps(score_result['interpretation']) if isinstance(score_result['interpretation'], list) else str(score_result['interpretation'])
            
            # **关键修复：添加分析结果数据的准备**
            # 执行详细分析
            current_state = self._analyze_current_state(test_data)
            
            # 准备AI分析数据
            ai_analysis_data = {
                'current_score': current_state['raw_score'],
                'normalized_score': current_state['normalized_score'],
                'weighted_score': current_state['weighted_score'],
                'severity_level': severity_level_str,
                'positive_symptoms': current_state['positive_symptoms'],
                'symptom_categories': current_state['symptom_categories'],
                'symptom_severity_scores': current_state['symptom_severity_scores'],
                'bipolar_indicators': current_state['bipolar_indicators'],
                'co_occurrence': test_data.get('co_occurrence', 'no'),
                'severity_impact': test_data.get('severity', 'no'),
                'completion_time': completion_time,
                'test_timestamp': datetime.now().isoformat()
            }
            
            # 将AI分析数据序列化为JSON
            ai_analysis_json = json.dumps(ai_analysis_data)
            
            cursor.execute('''
                INSERT INTO questionnaire_tests 
                (test_id, user_id, questionnaire_type, test_data, raw_score, 
                interpretation, severity_level, completion_time, ai_analysis_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_id,
                user_id,
                'MDQ',
                json.dumps(test_data),
                score_result['raw_score'],
                interpretation_str,
                severity_level_str,
                completion_time,
                ai_analysis_json  # **新增：保存AI分析数据**
            ))
            
            conn.commit()
            
            # 生成建议
            self._generate_mdq_recommendations(test_id, user_id, score_result)
            
            # **返回更完整的结果，包含分析数据**
            return {
                'success': True, 
                'test_id': test_id,
                'score_result': score_result,
                'analysis_data': current_state,  # **新增：返回详细分析数据**
                'ai_analysis_data': ai_analysis_data,  # **新增：返回AI分析数据**
                'message': 'MDQ测试结果保存成功'
            }
            
        except sqlite3.Error as e:
            print(f"保存MDQ测试失败: {e}")
            if conn:
                conn.rollback()
            return {'success': False, 'message': f'保存失败: {str(e)}'}
        except Exception as e:
            print(f"保存MDQ测试异常: {e}")
            if conn:
                conn.rollback()
            return {'success': False, 'message': f'保存失败: {str(e)}'}
        finally:
            if conn:
                conn.close()
    def _calculate_severity_level(self, raw_score: int, weighted_score: float, 
                                bipolar_indicators: Dict[str, bool], test_data: Dict) -> SeverityLevel:
        """计算严重程度等级 - 基于新的评分系统"""
        
        # 基础等级（基于原始分数0-39）
        base_level = SeverityLevel.NORMAL
        for level, (min_score, max_score) in self.severity_thresholds.items():
            if min_score <= raw_score <= max_score:
                base_level = level
                break
        
        # 调整因子
        severity_boost = 0
        
        # 双相风险指标调整
        positive_indicators = sum(bipolar_indicators.values())
        if positive_indicators >= 3:
            severity_boost += 2
        elif positive_indicators >= 2:
            severity_boost += 1
        
        # 功能损害调整
        severity = test_data.get('severity', 'no')
        if severity == 'serious':
            severity_boost += 2
        elif severity == 'moderate':
            severity_boost += 1
        
        # 共现调整
        if test_data.get('co_occurrence') == 'yes':
            severity_boost += 1
        
        # 高危症状调整（核心躁狂症状和行为症状）
        high_risk_categories = ['core_manic_symptoms', 'behavioral_symptoms']
        high_risk_count = sum(1 for cat in high_risk_categories 
                            if bipolar_indicators.get(cat, False))
        if high_risk_count >= 2:
            severity_boost += 1
        
        # 应用调整
        severity_levels = list(SeverityLevel)
        base_index = severity_levels.index(base_level)
        adjusted_index = min(len(severity_levels) - 1, base_index + severity_boost)
        
        return severity_levels[adjusted_index]
    def _get_severity_text(self, score: int) -> str:
        """根据分数获取严重程度文本"""
        if score == 1:
            return "轻微"
        elif score == 2:
            return "中等"
        elif score == 3:
            return "严重"
        elif score == 4:
            return "非常严重"
        else:
            return "无"

    def _analyze_current_state(self, test_detail: Dict) -> Dict:
        """分析当前状态 - 适配5级评分系统"""
        test_data = test_detail
        questions = test_data.get('questions', {})
        
        # 计算原始分数（0-39分）
        raw_score = 0
        symptom_scores = {}
        
        for q_id, answer in questions.items():
            score = self.score_mapping.get(answer, 0)
            raw_score += score
            symptom_scores[q_id] = score
        
        # 计算标准化分数（0-13分，兼容原有系统）
        normalized_score = min(13, round(raw_score / 3))
        
        # 计算加权分数
        weighted_score = sum(
            symptom_scores.get(q_id, 0) * self.symptom_weights.get(q_id, 1)
            for q_id in symptom_scores
        )
        
        # 症状分类统计
        symptom_categories = {}
        symptom_severity_scores = {}
        
        for category, q_ids in self.symptom_categories.items():
            category_score = sum(symptom_scores.get(q_id, 0) for q_id in q_ids)
            max_possible = len(q_ids) * 4  # 每个问题最高4分
            
            symptom_categories[category] = category_score
            symptom_severity_scores[category] = round(
                (category_score / max_possible) * 100, 1
            ) if max_possible > 0 else 0
        
        # 阳性症状识别（分数≥2的症状）
        positive_symptoms = []
        for q_id, score in symptom_scores.items():
            if score >= 2 and q_id in self.symptom_descriptions:
                severity_text = self._get_severity_text(score)
                positive_symptoms.append(
                    f"{self.symptom_descriptions[q_id]} ({severity_text})"
                )
        
        # 双相障碍风险指标
        bipolar_indicators = {}
        print(self.bipolar_indicators.items())
        for indicator, q_ids in self.bipolar_indicators.items():
            # 计算该指标组的平均严重程度
            print('indicator',indicator)
            print('q_ids',q_ids)
            indicator_scores = [symptom_scores.get(q_id, 0) for q_id in q_ids]
            avg_score = np.mean(indicator_scores) if indicator_scores else 0
            print(avg_score)
            
            # 如果平均分数≥2，认为该指标阳性
            if avg_score >= 2.0:
                bipolar_indicators[indicator] = 1
            else:
                bipolar_indicators[indicator] = 0
            #bipolar_indicators[indicator] = avg_score >= 2.0
        print(bipolar_indicators)
        
        # 严重程度评估
        severity_level = self._calculate_severity_level(
            raw_score, weighted_score, bipolar_indicators, test_data
        )

        
        return {
            'raw_score': raw_score,
            'normalized_score': normalized_score,
            'weighted_score': weighted_score,
            'severity_level': severity_level,
            'positive_symptoms': positive_symptoms,
            'symptom_categories': symptom_categories,
            'symptom_severity_scores': symptom_severity_scores,
            'bipolar_indicators': bipolar_indicators,
            'symptom_scores': symptom_scores
        }
    def _calculate_mdq_score(self, test_data: Dict) -> Dict:
        """计算MDQ得分和解释"""
        # MDQ问卷包含13个是/否题目，统计"是"的数量
        
        current_state = self._analyze_current_state(test_data)
        print(f"当前状态: {current_state}")
        co_occurrence = test_data.get('co_occurrence', 'no')
        severity = test_data.get('severity', 'no')
        raw_score = current_state['raw_score']
        severity_level = current_state['severity_level']
        interpretation = current_state['positive_symptoms']
        return {
            'raw_score': raw_score,
            'severity_level': severity_level,
            'interpretation': interpretation,
            'co_occurrence': co_occurrence,
            'severity_impact': severity,
        }
    
    def _generate_mdq_recommendations(self, test_id: str, user_id: str, score_result: Dict):
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            recommendation_id = str(uuid.uuid4())
            
            # 获取严重程度字符串
            severity_level_str = score_result['severity_level'].value if hasattr(score_result['severity_level'], 'value') else str(score_result['severity_level'])
            
            # 根据结果生成不同的建议
            if severity_level_str in ['high_risk', 'severe_risk']:
                recommendations = {
                    'immediate_actions': [
                        '尽快联系心理健康专家进行专业评估',
                        '考虑预约精神科医生进行诊断',
                        '告知信任的家人或朋友您的情况'
                    ],
                    'self_care': [
                        '保持规律的作息时间',
                        '避免酒精和药物滥用',
                        '监控情绪变化并做记录'
                    ],
                    'resources': [
                        '国家心理健康热线',
                        '当地心理健康中心',
                        '双相情感障碍支持小组'
                    ],
                    'follow_up': '建议1周内寻求专业帮助'
                }
            elif severity_level_str == 'moderate_risk':
                recommendations = {
                    'immediate_actions': [
                        '考虑咨询心理健康专家',
                        '开始记录情绪日记',
                        '与信任的人分享您的感受'
                    ],
                    'self_care': [
                        '保持健康的生活方式',
                        '练习压力管理技巧',
                        '确保充足的睡眠'
                    ],
                    'resources': [
                        '心理健康应用程序',
                        '情绪管理书籍',
                        '在线支持社区'
                    ],
                    'follow_up': '建议2-4周内进行专业咨询'
                }
            else:
                recommendations = {
                    'immediate_actions': [
                        '继续关注自己的心理健康',
                        '保持良好的生活习惯'
                    ],
                    'self_care': [
                        '定期运动',
                        '保持社交联系',
                        '练习正念和放松技巧'
                    ],
                    'resources': [
                        '心理健康教育资源',
                        '压力管理工具'
                    ],
                    'follow_up': '如果症状加重，随时寻求帮助'
                }
            
            cursor.execute('''
                INSERT INTO test_recommendations 
                (recommendation_id, test_id, user_id, recommendations, severity_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                recommendation_id,
                test_id,
                user_id,
                json.dumps(recommendations),
                severity_level_str  # 使用字符串而不是枚举对象
            ))
            
            conn.commit()
            
        except sqlite3.Error as e:
            print(f"生成建议失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_user_mdq_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户的MDQ测试历史"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT test_id, test_timestamp, raw_score, interpretation, 
                        severity_level, completion_time
                FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
                ORDER BY test_timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            tests = cursor.fetchall()
            
            return [{
                'test_id': row[0],
                'test_timestamp': row[1],
                'raw_score': row[2],
                'interpretation': row[3],
                'severity_level': row[4],
                'completion_time': row[5]
            } for row in tests]
            
        except sqlite3.Error as e:
            print(f"获取MDQ历史失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_mdq_test_detail(self, test_id: str, user_id: str) -> Optional[Dict]:
        """获取MDQ测试详细信息"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # **修复：查询时包含 ai_analysis_data 字段**
            cursor.execute('''
                SELECT test_data, test_timestamp, raw_score, interpretation, severity_level, ai_analysis_data
                FROM questionnaire_tests 
                WHERE test_id = ? AND user_id = ? AND questionnaire_type = 'MDQ'
            ''', (test_id, user_id))
            
            result = cursor.fetchone()
            
            if result:
                detail = {
                    'test_data': json.loads(result[0]),
                    'test_timestamp': result[1],
                    'raw_score': result[2],
                    'interpretation': result[3],
                    'severity_level': result[4],
                }
                
                # **新增：包含AI分析数据**
                if result[5]:  # ai_analysis_data
                    try:
                        detail['ai_analysis_data'] = json.loads(result[5])
                    except json.JSONDecodeError as e:
                        print(f"解析AI分析数据失败: {e}")
                        detail['ai_analysis_data'] = None
                else:
                    detail['ai_analysis_data'] = None
                
                return detail
            
            return None
            
        except sqlite3.Error as e:
            print(f"获取MDQ测试详情失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_test_recommendations(self, test_id: str, user_id: str) -> List[Dict]:
        """获取测试建议"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT recommendation_id, recommendations, severity_level, 
                       generation_date, follow_up_date
                FROM test_recommendations
                WHERE test_id = ? AND user_id = ?
                ORDER BY generation_date DESC
            ''', (test_id, user_id))
            
            recommendations = cursor.fetchall()
            
            return [{
                'recommendation_id': row[0],
                'recommendations': json.loads(row[1]),
                'severity_level': row[2],
                'generation_date': row[3],
                'follow_up_date': row[4]
            } for row in recommendations]
            
        except sqlite3.Error as e:
            print(f"获取测试建议失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_user_statistics(self, user_id: str) -> Dict:
        """获取用户统计信息"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 获取测试总数
            cursor.execute('''
                SELECT COUNT(*) FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
            ''', (user_id,))
            total_tests = cursor.fetchone()[0]
            
            # 获取最近的测试结果
            cursor.execute('''
                SELECT raw_score, severity_level, test_timestamp
                FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
                ORDER BY test_timestamp DESC
                LIMIT 1
            ''', (user_id,))
            latest_test = cursor.fetchone()
            
            # 获取平均得分
            cursor.execute('''
                SELECT AVG(raw_score) FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
            ''', (user_id,))
            avg_score = cursor.fetchone()[0]
            
            return {
                'total_tests': total_tests,
                'latest_test': {
                    'score': latest_test[0] if latest_test else None,
                    'severity_level': latest_test[1] if latest_test else None,
                    'date': latest_test[2] if latest_test else None
                } if latest_test else None,
                'average_score': round(avg_score, 2) if avg_score else 0
            }
            
        except sqlite3.Error as e:
            print(f"获取用户统计失败: {e}")
            return {}
        finally:
            if conn:
                conn.close()
    def get_test_analysis_data(self, test_id: str, user_id: str) -> Optional[Dict]:
        """获取测试的分析数据"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ai_analysis_data, raw_score, severity_level, test_timestamp
                FROM questionnaire_tests 
                WHERE test_id = ? AND user_id = ? AND questionnaire_type = 'MDQ'
            ''', (test_id, user_id))
            
            result = cursor.fetchone()
            
            if result and result[0]:  # 如果有AI分析数据
                try:
                    analysis_data = json.loads(result[0])
                    analysis_data.update({
                        'raw_score': result[1],
                        'severity_level': result[2],
                        'test_timestamp': result[3]
                    })
                    return analysis_data
                except json.JSONDecodeError as e:
                    print(f"解析分析数据失败: {e}")
                    return None
            
            return None
            
        except sqlite3.Error as e:
            print(f"获取测试分析数据失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
# 在 database.py 中添加以下增强方法

    def get_user_profile_with_stats(self, user_id: str) -> Optional[Dict]:
        """获取包含统计信息的用户资料"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 获取用户基本信息
            cursor.execute('''
                SELECT username, email, full_name, gender, age, phone, 
                    occupation, education_level, emergency_contact,
                    registration_date, last_login, is_active
                FROM users 
                WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return None
            
            profile = {
                'user_id': user_id,
                'username': result[0],
                'email': result[1],
                'full_name': result[2],
                'gender': result[3],
                'age': result[4],
                'phone': result[5],
                'occupation': result[6],
                'education_level': result[7],
                'emergency_contact': result[8],
                'registration_date': result[9],
                'last_login': result[10],
                'is_active': result[11]
            }
            
            # 获取测试统计
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_tests,
                    COUNT(CASE WHEN raw_score IS NOT NULL THEN 1 END) as completed_tests,
                    AVG(CASE WHEN raw_score IS NOT NULL THEN raw_score END) as avg_score,
                    MAX(test_timestamp) as last_test_date
                FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
            ''', (user_id,))
            
            stats = cursor.fetchone()
            
            profile['statistics'] = {
                'total_tests': stats[0] if stats else 0,
                'completed_tests': stats[1] if stats else 0,
                'incomplete_tests': (stats[0] - stats[1]) if stats and stats[1] else 0,
                'average_score': round(stats[2], 1) if stats and stats[2] else 0,
                'last_test_date': stats[3] if stats else None
            }
            
            return profile
            
        except sqlite3.Error as e:
            print(f"获取用户资料（含统计）失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def update_user_last_activity(self, user_id: str) -> bool:
        """更新用户最后活动时间"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP 
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            print(f"更新用户活动时间失败: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def get_user_activity_log(self, user_id: str, limit: int = 20) -> List[Dict]:
        """获取用户活动日志"""
        conn = None
        conn = self._get_connection()
        cursor = conn.cursor()
            
        # 获取测试活动
        cursor.execute('''
                SELECT 
                    'test_completed' as activity_type,
                    test_timestamp as activity_date,
                    'MDQ测试' as activity_description,
                    raw_score,
                    severity_level
                FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
                ORDER BY test_timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
        activities = []
        for row in cursor.fetchall():
            activity = {
                    'type': row[0],
                    'date': row[1],
                    'description': row[2],
                    'details': {
                        'score': row[3],
                        'severity': row[4]
                    }
                }
            activities.append(activity)
            
        return activities

# 测试代码
if __name__ == "__main__":
    # 初始化数据库
    db_manager = DatabaseManager()
    conn = db_manager._get_connection()
    cursor = conn.cursor()
            
    mdq_data = {
                'questions': {
                    'q1': 'rarely', 'q2': 'always', 'q3': 'always', 'q4': 'always',
                    'q5': 'sometimes', 'q6': 'always', 'q7': 'always', 'q8': 'no',
                    'q9': 'always', 'q10': 'often', 'q11': 'always', 'q12': 'always', 'q13': 'always'
                },
                'co_occurrence': 'no',
                'severity': 'no'}

    
    score_result = db_manager._calculate_mdq_score(mdq_data)
    user_id = '43753170-b192-4f6a-8039-46aeaf5d5aeb'
    result = db_manager.save_mdq_test(user_id, mdq_data, '2023-04-05 12:00:00')
    print(result)
    
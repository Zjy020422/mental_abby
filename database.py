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

# 更新为标准MDQ严重程度等级
class SeverityLevel(Enum):
    """MDQ严重程度等级 - 基于标准MDQ评分"""
    NEGATIVE = "negative"              # 阴性结果 (0-6分 或 第二部分否定)
    MILD_POSITIVE = "mild_positive"    # 轻度阳性 (7-9分 + 轻度功能损害)
    MODERATE_POSITIVE = "moderate_positive"  # 中度阳性 (10-12分 + 中等功能损害)
    HIGH_POSITIVE = "high_positive"    # 高度阳性 (13分 + 严重功能损害)

class ImprovementTrend(Enum):
    """改善趋势"""
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    MILD_IMPROVEMENT = "mild_improvement"
    STABLE = "stable"
    MILD_DETERIORATION = "mild_deterioration"
    MODERATE_DETERIORATION = "moderate_deterioration"
    SIGNIFICANT_DETERIORATION = "significant_deterioration"

@dataclass
class AnalysisResult:
    """分析结果数据类 - 更新为标准MDQ格式"""
    # 基础信息
    user_id: str
    analysis_id: str
    analysis_date: datetime
    
    # MDQ标准评估结果
    mdq_part1_score: int  # 第一部分分数 (0-13)
    has_co_occurrence: bool  # 第二部分：症状是否同时出现
    functional_impact_level: str  # 第三部分：功能影响程度
    mdq_result: str  # MDQ最终结果
    severity_level: SeverityLevel
    risk_percentage: float
    
    # 症状分析
    positive_symptoms: List[str]
    symptom_profile: Dict[str, bool]  # 13个症状的是否阳性
    core_symptoms_count: int  # 核心症状数量
    
    # 历史趋势分析
    improvement_trend: ImprovementTrend
    trend_confidence: float
    historical_baseline: float
    improvement_percentage: float
    consistency_score: float
    
    # 详细分析
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
    """数据库管理类 - 适配标准MDQ评分"""
    
    def __init__(self, db_path="mental_health_assessment.db"):
        self.db_path = db_path
        
        # MDQ问题描述 (基于标准MDQ-13)
        self.symptom_descriptions = {
            'q1': '情绪异常高涨、兴奋或精力充沛',
            'q2': '过度自信或感觉自己有特殊才能',
            'q3': '睡眠需求减少（仍感到休息充分）',
            'q4': '比平时更健谈或语速更快',
            'q5': '思维跳跃或感觉思维加速',
            'q6': '注意力更容易分散',
            'q7': '精力充沛，异常活跃或做更多事情',
            'q8': '比平时更爱社交或更外向',
            'q9': '对性的兴趣比平时更强烈',
            'q10': '做事情时不顾后果或判断力差',
            'q11': '花钱比平时更冲动或不理智',
            'q12': '行为变化导致工作、学习困难',
            'q13': '他人觉得你与平时判若两人'
        }
        
        # 核心躁狂症状（MDQ诊断关键症状）
        self.core_symptoms = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
        
        # 功能损害级别映射
        self.functional_impact_mapping = {
            'no': 'no_problems',
            'minor': 'minor_problems', 
            'moderate': 'moderate_problems',
            'serious': 'serious_problems'
        }
        
        # MDQ评分阈值 (基于研究文献)
        self.mdq_thresholds = {
            'screening_cutoff': 7,  # 筛查阈值
            'high_sensitivity': 5,  # 高敏感性阈值
            'high_specificity': 9,  # 高特异性阈值
        }
        
        self.init_database()
    
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
            
            # 更新问卷测试表，支持标准MDQ分数
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS questionnaire_tests (
                    test_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    questionnaire_type TEXT NOT NULL DEFAULT 'MDQ',
                    test_data TEXT NOT NULL,
                    test_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    mdq_score INTEGER,  -- 标准MDQ分数 (0-13)
                    raw_score INTEGER,  -- 保留兼容性
                    interpretation TEXT,
                    severity_level TEXT,
                    completion_time INTEGER,
                    ai_analysis_data TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            ''')
            
            # 检查并添加 mdq_score 字段（如果表已存在但缺少该字段）
            try:
                cursor.execute("PRAGMA table_info(questionnaire_tests)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'mdq_score' not in columns:
                    cursor.execute('ALTER TABLE questionnaire_tests ADD COLUMN mdq_score INTEGER')
                    print("✅ 已添加 mdq_score 字段到 questionnaire_tests 表")
                if 'ai_analysis_data' not in columns:
                    cursor.execute('ALTER TABLE questionnaire_tests ADD COLUMN ai_analysis_data TEXT')
                    print("✅ 已添加 ai_analysis_data 字段到 questionnaire_tests 表")
            except sqlite3.Error as e:
                print(f"⚠️ 添加字段时出错: {e}")
            
            # 其余表保持不变
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
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_tests ON questionnaire_tests(user_id, test_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_timestamp ON questionnaire_tests(test_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_recommendations ON test_recommendations(user_id, generation_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_email ON users(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions ON user_sessions(user_id, is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_mdq_score ON questionnaire_tests(mdq_score)')
            
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
    
    # ==================== MDQ问卷数据管理功能 - 标准MDQ版本 ====================
    
    def save_mdq_test(self, user_id: str, test_data: Dict, completion_time: int = None) -> Dict:
        """保存MDQ问卷测试结果 - 标准MDQ评分"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            test_id = str(uuid.uuid4())
            
            # 计算标准MDQ得分和解释
            score_result = self._calculate_mdq_score_standard(test_data)
            print(f"标准MDQ评分结果: {score_result}")
            
            # 确保 severity_level 是字符串格式
            severity_level_str = score_result['severity_level'].value if hasattr(score_result['severity_level'], 'value') else str(score_result['severity_level'])
            
            # 确保 interpretation 是字符串格式
            interpretation_str = json.dumps(score_result['interpretation']) if isinstance(score_result['interpretation'], list) else str(score_result['interpretation'])
            
            # 执行详细分析
            current_state = self._analyze_current_state_standard(test_data)
            
            # 准备AI分析数据
            ai_analysis_data = {
                'mdq_part1_score': current_state['mdq_part1_score'],
                'has_co_occurrence': current_state['has_co_occurrence'],
                'functional_impact_level': current_state['functional_impact_level'],
                'mdq_result': current_state['mdq_result'],
                'severity_level': severity_level_str,
                'positive_symptoms': current_state['positive_symptoms'],
                'symptom_profile': current_state['symptom_profile'],
                'core_symptoms_count': current_state['core_symptoms_count'],
                'risk_percentage': current_state['risk_percentage'],
                'completion_time': completion_time,
                'test_timestamp': datetime.now().isoformat()
            }
            
            # 将AI分析数据序列化为JSON
            ai_analysis_json = json.dumps(ai_analysis_data)
            
            cursor.execute('''
                INSERT INTO questionnaire_tests 
                (test_id, user_id, questionnaire_type, test_data, mdq_score, raw_score,
                interpretation, severity_level, completion_time, ai_analysis_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_id,
                user_id,
                'MDQ',
                json.dumps(test_data),
                score_result['mdq_score'],  # 标准MDQ分数 (0-13)
                score_result['mdq_score'],  # 兼容性：raw_score = mdq_score
                interpretation_str,
                severity_level_str,
                completion_time,
                ai_analysis_json
            ))
            
            conn.commit()
            
            # 生成建议
            self._generate_mdq_recommendations_standard(test_id, user_id, score_result)
            
            # 返回更完整的结果，包含分析数据
            return {
                'success': True, 
                'test_id': test_id,
                'score_result': score_result,
                'analysis_data': current_state,
                'ai_analysis_data': ai_analysis_data,
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
    
    def _calculate_mdq_score_standard(self, test_data: Dict) -> Dict:
        """计算标准MDQ得分和解释"""
        # 第一部分：计算症状分数 (0-13分)
        questions = test_data.get('questions', {})
        mdq_score = 0
        
        # 标准MDQ评分：只有"no"为0分，其他都为1分
        for q_id in [f'q{i}' for i in range(1, 14)]:
            answer = questions.get(q_id, 'no')
            if answer != 'no':
                mdq_score += 1
        
        # 第二部分：症状共现性
        co_occurrence = test_data.get('co_occurrence', 'no')
        has_co_occurrence = co_occurrence == 'yes'
        
        # 第三部分：功能影响
        severity = test_data.get('severity', 'no')
        functional_impact_level = self.functional_impact_mapping.get(severity, 'no_problems')
        
        # MDQ结果判定
        mdq_result = self._determine_mdq_result_standard(mdq_score, has_co_occurrence, functional_impact_level)
        
        # 严重程度评估
        severity_level = self._determine_severity_level_standard(mdq_score, has_co_occurrence, functional_impact_level, mdq_result)
        
        # 风险百分比计算
        risk_percentage = self._calculate_risk_percentage_standard(mdq_score, has_co_occurrence, functional_impact_level)
        
        # 生成解释
        interpretation = self._generate_mdq_interpretation_standard(mdq_result, mdq_score, has_co_occurrence, functional_impact_level)
        
        return {
            'mdq_score': mdq_score,
            'has_co_occurrence': has_co_occurrence,
            'functional_impact_level': functional_impact_level,
            'mdq_result': mdq_result,
            'severity_level': severity_level,
            'risk_percentage': risk_percentage,
            'interpretation': interpretation
        }
    
    def _determine_mdq_result_standard(self, mdq_score: int, has_co_occurrence: bool, functional_impact_level: str) -> str:
        """确定MDQ结果 - 基于标准诊断算法"""
        
        # 标准MDQ阳性标准：
        # 1. 第一部分得分 ≥ 7分
        # 2. 第二部分为"是"（症状同时出现）
        # 3. 第三部分有功能损害（不是"no"）
        
        if mdq_score < self.mdq_thresholds['screening_cutoff']:
            return 'negative'
        
        if not has_co_occurrence:
            return 'negative'
        
        if functional_impact_level == 'no_problems':
            return 'positive_subclinical'  # 亚临床阳性
        
        # 根据功能损害程度确定阳性级别
        if functional_impact_level == 'serious_problems':
            return 'positive_high'
        elif functional_impact_level == 'moderate_problems':
            return 'positive_moderate'
        else:  # minor_problems
            return 'positive_mild'
    
    def _determine_severity_level_standard(self, mdq_score: int, has_co_occurrence: bool, 
                                         functional_impact_level: str, mdq_result: str) -> SeverityLevel:
        """确定严重程度等级"""
        
        if mdq_result == 'negative':
            return SeverityLevel.NEGATIVE
        
        if mdq_result == 'positive_mild' or mdq_result == 'positive_subclinical':
            return SeverityLevel.MILD_POSITIVE
        
        if mdq_result == 'positive_moderate':
            return SeverityLevel.MODERATE_POSITIVE
        
        if mdq_result == 'positive_high':
            return SeverityLevel.HIGH_POSITIVE
        
        # 基于分数的额外判断
        if mdq_score >= 10:
            return SeverityLevel.HIGH_POSITIVE
        elif mdq_score >= 8:
            return SeverityLevel.MODERATE_POSITIVE
        else:
            return SeverityLevel.MILD_POSITIVE
    
    def _calculate_risk_percentage_standard(self, mdq_score: int, has_co_occurrence: bool, 
                                          functional_impact_level: str) -> float:
        """计算双相障碍风险百分比"""
        
        # 基于研究文献的风险计算
        base_risk = 0.0
        
        # 第一部分分数贡献 (0-13分)
        score_risk = min(mdq_score / 13.0 * 60, 60)  # 最高60%
        
        # 共现性贡献
        co_occurrence_risk = 20 if has_co_occurrence else 0
        
        # 功能损害贡献
        functional_risk = {
            'no_problems': 0,
            'minor_problems': 5,
            'moderate_problems': 15,
            'serious_problems': 25
        }.get(functional_impact_level, 0)
        
        # 综合风险计算
        total_risk = base_risk + score_risk + co_occurrence_risk + functional_risk
        
        # 应用研究权重调整
        if mdq_score >= 9 and has_co_occurrence and functional_impact_level != 'no_problems':
            total_risk += 15  # 高风险组合奖励
        
        return round(min(100, max(0, total_risk)), 1)
    
    def _generate_mdq_interpretation_standard(self, mdq_result: str, mdq_score: int, 
                                            has_co_occurrence: bool, functional_impact_level: str) -> str:
        """生成MDQ解释"""
        
        interpretation_map = {
            'negative': f'MDQ筛查阴性 (分数: {mdq_score}/13)',
            'positive_subclinical': f'MDQ亚临床阳性 (分数: {mdq_score}/13，症状同时出现但功能损害轻微)',
            'positive_mild': f'MDQ轻度阳性 (分数: {mdq_score}/13，轻度功能损害)',
            'positive_moderate': f'MDQ中度阳性 (分数: {mdq_score}/13，中等功能损害)',
            'positive_high': f'MDQ高度阳性 (分数: {mdq_score}/13，严重功能损害)'
        }
        
        return interpretation_map.get(mdq_result, f'MDQ评估结果 (分数: {mdq_score}/13)')
    
    def _analyze_current_state_standard(self, test_data: Dict) -> Dict:
        """分析当前状态 - 标准MDQ版本"""
        questions = test_data.get('questions', {})
        
        # 第一部分：计算症状分数和症状档案
        mdq_score = 0
        symptom_profile = {}
        positive_symptoms = []
        
        for q_id in [f'q{i}' for i in range(1, 14)]:
            answer = questions.get(q_id, 'no')
            
            if answer != 'no':
                mdq_score += 1
                symptom_profile[q_id] = True
                positive_symptoms.append(self.symptom_descriptions[q_id])
            else:
                symptom_profile[q_id] = False
        
        # 核心症状计数
        core_symptoms_count = sum(1 for q_id in self.core_symptoms if symptom_profile.get(q_id, False))
        
        # 第二部分：症状共现性
        co_occurrence = test_data.get('co_occurrence', 'no')
        has_co_occurrence = co_occurrence == 'yes'
        
        # 第三部分：功能影响
        severity = test_data.get('severity', 'no')
        functional_impact_level = self.functional_impact_mapping.get(severity, 'no_problems')
        
        # MDQ结果判定
        mdq_result = self._determine_mdq_result_standard(mdq_score, has_co_occurrence, functional_impact_level)
        
        # 严重程度评估
        severity_level = self._determine_severity_level_standard(mdq_score, has_co_occurrence, functional_impact_level, mdq_result)
        
        # 风险百分比计算
        risk_percentage = self._calculate_risk_percentage_standard(mdq_score, has_co_occurrence, functional_impact_level)
        
        return {
            'mdq_part1_score': mdq_score,
            'has_co_occurrence': has_co_occurrence,
            'functional_impact_level': functional_impact_level,
            'mdq_result': mdq_result,
            'severity_level': severity_level,
            'risk_percentage': risk_percentage,
            'positive_symptoms': positive_symptoms,
            'symptom_profile': symptom_profile,
            'core_symptoms_count': core_symptoms_count
        }
    
    def _generate_mdq_recommendations_standard(self, test_id: str, user_id: str, score_result: Dict):
        """生成MDQ建议 - 标准版本"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            recommendation_id = str(uuid.uuid4())
            mdq_result = score_result['mdq_result']
            mdq_score = score_result['mdq_score']
            
            # 根据MDQ结果生成不同的建议
            if mdq_result == 'positive_high':
                recommendations = {
                    'immediate_actions': [
                        '立即进行精神科急诊评估',
                        '考虑住院治疗或危机干预',
                        'MDQ高度阳性，强烈提示双相障碍',
                        '联系紧急联系人'
                    ],
                    'self_care': [
                        '确保个人安全',
                        '避免做重大决定',
                        '监控情绪和行为变化'
                    ],
                    'resources': [
                        '精神科急诊服务',
                        '危机干预热线',
                        '双相障碍专科门诊'
                    ],
                    'follow_up': '需要立即专业干预'
                }
            elif mdq_result == 'positive_moderate':
                recommendations = {
                    'immediate_actions': [
                        '48-72小时内安排精神科专科评估',
                        'MDQ中度阳性，需要专业评估',
                        '详细的临床访谈和病史收集'
                    ],
                    'self_care': [
                        '保持规律的作息时间',
                        '避免过度刺激和压力',
                        '监控症状变化'
                    ],
                    'resources': [
                        '精神科专科门诊',
                        '心理健康中心',
                        '双相障碍教育资源'
                    ],
                    'follow_up': '建议72小时内寻求专业评估'
                }
            elif mdq_result == 'positive_mild':
                recommendations = {
                    'immediate_actions': [
                        '1-2周内安排专科咨询',
                        'MDQ轻度阳性，建议进一步评估',
                        '关注症状变化和发展'
                    ],
                    'self_care': [
                        '保持健康的生活方式',
                        '练习压力管理技巧',
                        '记录情绪变化'
                    ],
                    'resources': [
                        '心理健康专家咨询',
                        '情绪管理技巧',
                        '支持小组'
                    ],
                    'follow_up': '建议2周内专科咨询'
                }
            elif mdq_result == 'positive_subclinical':
                recommendations = {
                    'immediate_actions': [
                        '门诊随访观察',
                        '亚临床阳性，功能损害轻微',
                        '预防性心理干预'
                    ],
                    'self_care': [
                        '保持良好生活习惯',
                        '学习症状识别',
                        '定期自我评估'
                    ],
                    'resources': [
                        '心理健康教育',
                        '预防性咨询',
                        '生活方式指导'
                    ],
                    'follow_up': '定期监测，如症状加重及时就诊'
                }
            else:  # negative
                recommendations = {
                    'immediate_actions': [
                        'MDQ筛查阴性，继续关注心理健康',
                        '保持良好的生活习惯'
                    ],
                    'self_care': [
                        '定期运动',
                        '保持社交联系',
                        '练习正念和放松技巧'
                    ],
                    'resources': [
                        '心理健康教育资源',
                        '压力管理工具',
                        '健康生活指南'
                    ],
                    'follow_up': '如症状发生变化，及时寻求帮助'
                }
            
            # 获取严重程度字符串
            severity_level_str = score_result['severity_level'].value if hasattr(score_result['severity_level'], 'value') else str(score_result['severity_level'])
            
            cursor.execute('''
                INSERT INTO test_recommendations 
                (recommendation_id, test_id, user_id, recommendations, severity_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                recommendation_id,
                test_id,
                user_id,
                json.dumps(recommendations),
                severity_level_str
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
                SELECT test_id, test_timestamp, mdq_score, raw_score, interpretation, 
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
                'mdq_score': row[2] if row[2] is not None else row[3],  # 优先使用mdq_score
                'raw_score': row[2] if row[2] is not None else row[3],  # **修复：raw_score应该与mdq_score保持一致**
                'interpretation': row[4],
                'severity_level': row[5],
                'completion_time': row[6]
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
            
            cursor.execute('''
                SELECT test_data, test_timestamp, mdq_score, raw_score, interpretation, 
                    severity_level, ai_analysis_data
                FROM questionnaire_tests 
                WHERE test_id = ? AND user_id = ? AND questionnaire_type = 'MDQ'
            ''', (test_id, user_id))
            
            result = cursor.fetchone()
            
            if result:
                # **修复：确保分数字段的一致性**
                mdq_score = result[2] if result[2] is not None else result[3]
                
                detail = {
                    'test_data': json.loads(result[0]),
                    'test_timestamp': result[1],
                    'mdq_score': mdq_score,
                    'raw_score': mdq_score,  # **修复：raw_score应该与mdq_score保持一致**
                    'interpretation': result[4],
                    'severity_level': result[5],
                }
                
                # 包含AI分析数据
                if result[6]:  # ai_analysis_data
                    try:
                        ai_data = json.loads(result[6])
                        # **修复：确保AI分析数据中的分数字段一致**
                        ai_data['raw_score'] = mdq_score
                        detail['ai_analysis_data'] = ai_data
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
        """获取用户统计信息 - 标准MDQ版本"""
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
                SELECT mdq_score, raw_score, severity_level, test_timestamp
                FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
                ORDER BY test_timestamp DESC
                LIMIT 1
            ''', (user_id,))
            latest_test = cursor.fetchone()
            
            # 获取平均得分
            cursor.execute('''
                SELECT AVG(COALESCE(mdq_score, raw_score)) FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
            ''', (user_id,))
            avg_score = cursor.fetchone()[0]
            
            # 获取阳性结果统计
            cursor.execute('''
                SELECT COUNT(*) FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ' 
                AND COALESCE(mdq_score, raw_score) >= 7
            ''', (user_id,))
            positive_tests = cursor.fetchone()[0]
            
            # **修复：确保latest_test中的分数字段一致**
            latest_score = None
            if latest_test:
                latest_score = latest_test[0] if latest_test[0] is not None else latest_test[1]
            
            return {
                'total_tests': total_tests,
                'positive_tests': positive_tests,
                'negative_tests': total_tests - positive_tests,
                'positive_rate': round((positive_tests / total_tests) * 100, 1) if total_tests > 0 else 0,
                'latest_test': {
                    'mdq_score': latest_score,
                    'raw_score': latest_score,  # **修复：保持一致性**
                    'severity_level': latest_test[2] if latest_test else None,
                    'date': latest_test[3] if latest_test else None
                } if latest_test else None,
                'average_mdq_score': round(avg_score, 2) if avg_score else 0,
                'screening_threshold': 7,  # MDQ筛查阈值
                'max_possible_score': 13   # MDQ最高分
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
                SELECT ai_analysis_data, mdq_score, raw_score, severity_level, test_timestamp
                FROM questionnaire_tests 
                WHERE test_id = ? AND user_id = ? AND questionnaire_type = 'MDQ'
            ''', (test_id, user_id))
            
            result = cursor.fetchone()
            
            if result and result[0]:  # 如果有AI分析数据
                try:
                    analysis_data = json.loads(result[0])
                    analysis_data.update({
                        'mdq_score': result[1] if result[1] is not None else result[2],
                        'raw_score': result[2],  # 保留兼容性
                        'severity_level': result[3],
                        'test_timestamp': result[4]
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
    
    # ==================== 增强功能 ====================
    
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
                    COUNT(CASE WHEN COALESCE(mdq_score, raw_score) IS NOT NULL THEN 1 END) as completed_tests,
                    AVG(CASE WHEN COALESCE(mdq_score, raw_score) IS NOT NULL THEN COALESCE(mdq_score, raw_score) END) as avg_score,
                    COUNT(CASE WHEN COALESCE(mdq_score, raw_score) >= 7 THEN 1 END) as positive_tests,
                    MAX(test_timestamp) as last_test_date
                FROM questionnaire_tests 
                WHERE user_id = ? AND questionnaire_type = 'MDQ'
            ''', (user_id,))
            
            stats = cursor.fetchone()
            
            profile['statistics'] = {
                'total_tests': stats[0] if stats else 0,
                'completed_tests': stats[1] if stats else 0,
                'incomplete_tests': (stats[0] - stats[1]) if stats and stats[1] else 0,
                'average_mdq_score': round(stats[2], 1) if stats and stats[2] else 0,
                'positive_tests': stats[3] if stats else 0,
                'positive_rate': round((stats[3] / stats[1]) * 100, 1) if stats and stats[1] > 0 else 0,
                'last_test_date': stats[4] if stats else None,
                'screening_threshold': 7,
                'max_possible_score': 13
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
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
                
            # 获取测试活动
            cursor.execute('''
                    SELECT 
                        'test_completed' as activity_type,
                        test_timestamp as activity_date,
                        'MDQ测试' as activity_description,
                        COALESCE(mdq_score, raw_score) as score,
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
                            'mdq_score': row[3],
                            'severity': row[4]
                        }
                    }
                activities.append(activity)
                
            return activities
            
        except sqlite3.Error as e:
            print(f"获取用户活动日志失败: {e}")
            return []
        finally:
            if conn:
                conn.close()

    # ==================== 数据迁移和兼容性 ====================
    
    def migrate_old_scores_to_mdq_standard(self) -> Dict:
        """将旧的评分数据迁移到标准MDQ格式"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 查找需要迁移的记录（有raw_score但没有mdq_score的记录）
            cursor.execute('''
                SELECT test_id, user_id, test_data, raw_score
                FROM questionnaire_tests 
                WHERE questionnaire_type = 'MDQ' 
                AND raw_score IS NOT NULL 
                AND mdq_score IS NULL
            ''')
            
            records_to_migrate = cursor.fetchall()
            migrated_count = 0
            error_count = 0
            
            for test_id, user_id, test_data_str, raw_score in records_to_migrate:
                try:
                    # 解析测试数据
                    test_data = json.loads(test_data_str)
                    questions = test_data.get('questions', {})
                    
                    # 计算标准MDQ分数
                    mdq_score = 0
                    for q_id in [f'q{i}' for i in range(1, 14)]:
                        answer = questions.get(q_id, 'no')
                        if answer != 'no':  # 标准MDQ：只有'no'为0分
                            mdq_score += 1
                    
                    # 更新记录
                    cursor.execute('''
                        UPDATE questionnaire_tests 
                        SET mdq_score = ?
                        WHERE test_id = ?
                    ''', (mdq_score, test_id))
                    
                    migrated_count += 1
                    print(f"迁移记录: {test_id}, 原始分数: {raw_score}, 标准MDQ分数: {mdq_score}")
                    
                except Exception as e:
                    print(f"迁移记录失败 {test_id}: {e}")
                    error_count += 1
            
            conn.commit()
            
            return {
                'total_records': len(records_to_migrate),
                'migrated_count': migrated_count,
                'error_count': error_count,
                'success_rate': round((migrated_count / len(records_to_migrate)) * 100, 1) if records_to_migrate else 100
            }
            
        except sqlite3.Error as e:
            print(f"数据迁移失败: {e}")
            return {'error': str(e)}
        finally:
            if conn:
                conn.close()

# 测试代码
if __name__ == "__main__":
    # 初始化数据库
    db_manager = DatabaseManager()
    
    # 测试标准MDQ评分
    print("=== 测试标准MDQ评分系统 ===")
    
    # 测试数据：模拟不同严重程度的MDQ回答
    test_cases = [
        {
            'name': '阴性测试（分数低于阈值）',
            'test_data': {
                'questions': {f'q{i}': 'no' if i > 5 else 'yes' for i in range(1, 14)},  # 5个yes
                'co_occurrence': 'yes',
                'severity': 'moderate'
            },
            'expected_mdq_score': 5,
            'expected_result': 'negative'
        },
        {
            'name': '轻度阳性测试',
            'test_data': {
                'questions': {f'q{i}': 'yes' if i <= 8 else 'no' for i in range(1, 14)},  # 8个yes
                'co_occurrence': 'yes',
                'severity': 'minor'
            },
            'expected_mdq_score': 8,
            'expected_result': 'positive_mild'
        },
        {
            'name': '高度阳性测试',
            'test_data': {
                'questions': {f'q{i}': 'yes' if i <= 11 else 'no' for i in range(1, 14)},  # 11个yes
                'co_occurrence': 'yes',
                'severity': 'serious'
            },
            'expected_mdq_score': 11,
            'expected_result': 'positive_high'
        },
        {
            'name': '亚临床阳性测试（无功能损害）',
            'test_data': {
                'questions': {f'q{i}': 'yes' if i <= 9 else 'no' for i in range(1, 14)},  # 9个yes
                'co_occurrence': 'yes',
                'severity': 'no'
            },
            'expected_mdq_score': 9,
            'expected_result': 'positive_subclinical'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        # 计算分数
        result = db_manager._calculate_mdq_score_standard(test_case['test_data'])
        
        print(f"计算得分: {result['mdq_score']}")
        print(f"期望得分: {test_case['expected_mdq_score']}")
        print(f"MDQ结果: {result['mdq_result']}")
        print(f"期望结果: {test_case['expected_result']}")
        print(f"风险百分比: {result['risk_percentage']}%")
        print(f"严重程度: {result['severity_level'].value}")
        
        # 验证结果
        score_correct = result['mdq_score'] == test_case['expected_mdq_score']
        result_correct = result['mdq_result'] == test_case['expected_result']
        
        if score_correct and result_correct:
            print("✅ 测试通过")
        else:
            print("❌ 测试失败")
            if not score_correct:
                print(f"   分数不匹配: 计算={result['mdq_score']}, 期望={test_case['expected_mdq_score']}")
            if not result_correct:
                print(f"   结果不匹配: 计算={result['mdq_result']}, 期望={test_case['expected_result']}")
    
    print("\n=== 标准MDQ评分系统测试完成 ===")
    print("✅ 支持标准MDQ二分法评分 (0/1)")
    print("✅ 三部分完整评估")
    print("✅ 基于研究文献的风险评估")
    print("✅ 数据库兼容性支持")
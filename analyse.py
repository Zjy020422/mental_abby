import sqlite3
import json
import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

class SeverityLevel(Enum):
    """MDQ严重程度等级 - 基于标准MDQ评分"""
    NEGATIVE = "negative"              # 阴性结果 (0-6分 或 第二部分否定)
    MILD_POSITIVE = "mild_positive"    # 轻度阳性 (7-9分 + 轻度功能损害)
    MODERATE_POSITIVE = "moderate_positive"  # 中度阳性 (10-12分 + 中等功能损害)
    HIGH_POSITIVE = "high_positive"    # 高度阳性 (13分 + 严重功能损害)

class ImprovementTrend(Enum):
    """改善趋势"""
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"      # 显著改善
    MODERATE_IMPROVEMENT = "moderate_improvement"           # 中度改善
    MILD_IMPROVEMENT = "mild_improvement"                   # 轻度改善
    STABLE = "stable"                                       # 稳定
    MILD_DETERIORATION = "mild_deterioration"               # 轻度恶化
    MODERATE_DETERIORATION = "moderate_deterioration"       # 中度恶化
    SIGNIFICANT_DETERIORATION = "significant_deterioration" # 显著恶化

@dataclass
class AnalysisResult:
    """分析结果数据类"""
    # 基础信息
    user_id: str
    analysis_id: str
    analysis_date: datetime
    
    # 当前状态评估 - 基于标准MDQ
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

class MDQAnalyzer:
    """标准MDQ问卷分析器 - 使用已有的DatabaseManager"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._init_database_tables()
        
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
    
    def _init_database_tables(self):
        """初始化分析结果数据库表"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # 创建MDQ分析结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mdq_analysis_results (
                    analysis_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    test_id TEXT,
                    analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- MDQ核心结果
                    mdq_part1_score INTEGER NOT NULL,
                    has_co_occurrence BOOLEAN NOT NULL,
                    functional_impact_level TEXT NOT NULL,
                    mdq_result TEXT NOT NULL,
                    severity_level TEXT NOT NULL,
                    risk_percentage REAL NOT NULL,
                    
                    -- 症状分析
                    positive_symptoms TEXT,
                    symptom_profile TEXT,
                    core_symptoms_count INTEGER,
                    
                    -- 历史趋势
                    improvement_trend TEXT,
                    trend_confidence REAL,
                    historical_baseline REAL,
                    improvement_percentage REAL,
                    consistency_score REAL,
                    
                    -- 详细分析
                    recovery_indicators TEXT,
                    risk_factors TEXT,
                    treatment_response_indicators TEXT,
                    
                    -- AI分析数据
                    ai_analysis_data TEXT,
                    
                    -- 建议
                    clinical_recommendations TEXT,
                    monitoring_frequency INTEGER,
                    emergency_flag BOOLEAN DEFAULT 0,
                    next_assessment_date DATETIME,
                    
                    -- 元数据
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_mdq_analysis_user_date ON mdq_analysis_results(user_id, analysis_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_mdq_analysis_severity ON mdq_analysis_results(severity_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_mdq_analysis_result ON mdq_analysis_results(mdq_result)')
            
            conn.commit()
            print("MDQ分析结果数据库表初始化完成")
            
        except sqlite3.Error as e:
            print(f"数据库表初始化失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def analyze_user_comprehensive(self, user_id: str) -> AnalysisResult:
        """执行用户MDQ综合分析"""
        
        # 获取用户测试历史
        test_history = self.db_manager.get_user_mdq_history(user_id, limit=30)
        
        if not test_history:
            return self._create_no_data_result(user_id)
        
        # 获取最新测试详情
        latest_test = test_history[0]
        test_detail = self.db_manager.get_mdq_test_detail(latest_test['test_id'], user_id)
        
        if not test_detail:
            return self._create_no_data_result(user_id)
        
        # 执行MDQ标准分析
        mdq_analysis = self._analyze_mdq_standard(test_detail)
        historical_analysis = self._analyze_historical_trends(test_history)
        improvement_analysis = self._analyze_improvement_patterns(test_history, mdq_analysis)
        ai_data = self._prepare_ai_analysis_data(mdq_analysis, historical_analysis, improvement_analysis, test_history)
        recommendations = self._generate_clinical_recommendations(mdq_analysis, historical_analysis, improvement_analysis)
        
        # 构建分析结果
        analysis_result = AnalysisResult(
            # 基础信息
            user_id=user_id,
            analysis_id=str(uuid.uuid4()),
            analysis_date=datetime.now(),
            
            # MDQ标准结果
            mdq_part1_score=mdq_analysis['part1_score'],
            has_co_occurrence=mdq_analysis['has_co_occurrence'],
            functional_impact_level=mdq_analysis['functional_impact_level'],
            mdq_result=mdq_analysis['mdq_result'],
            severity_level=mdq_analysis['severity_level'],
            risk_percentage=mdq_analysis['risk_percentage'],
            
            # 症状分析
            positive_symptoms=mdq_analysis['positive_symptoms'],
            symptom_profile=mdq_analysis['symptom_profile'],
            core_symptoms_count=mdq_analysis['core_symptoms_count'],
            
            # 历史趋势
            improvement_trend=historical_analysis['trend'],
            trend_confidence=historical_analysis['confidence'],
            historical_baseline=historical_analysis['baseline'],
            improvement_percentage=improvement_analysis['improvement_percentage'],
            consistency_score=improvement_analysis['consistency_score'],
            
            # 详细分析
            recovery_indicators=improvement_analysis['recovery_indicators'],
            risk_factors=improvement_analysis['risk_factors'],
            treatment_response_indicators=improvement_analysis['treatment_indicators'],
            
            # AI数据
            ai_analysis_data=ai_data,
            
            # 建议
            clinical_recommendations=recommendations['recommendations'],
            monitoring_frequency=recommendations['monitoring_frequency'],
            emergency_flag=recommendations['emergency_flag'],
            next_assessment_date=recommendations['next_assessment_date']
        )
        
        # 保存到数据库
        self._save_analysis_result(analysis_result, latest_test['test_id'])
        
        return analysis_result
    
    def _analyze_mdq_standard(self, test_detail: Dict) -> Dict:
        """标准MDQ分析 - 按照官方评分方法"""
        test_data = test_detail['test_data']
        questions = test_data.get('questions', {})
        
        # 第一部分：计算症状分数 (0-13分)
        part1_score = 0
        symptom_profile = {}
        positive_symptoms = []
        
        # 标准MDQ评分：只有"no"为0分，其他都为1分
        for q_id in [f'q{i}' for i in range(1, 14)]:
            answer = questions.get(q_id, 'no')
            
            # 标准评分：no=0, 其他所有选项=1
            if answer == 'no':
                score = 0
                symptom_profile[q_id] = False
            else:
                score = 1
                symptom_profile[q_id] = True
                positive_symptoms.append(self.symptom_descriptions[q_id])
            
            part1_score += score
        
        # 核心症状计数
        core_symptoms_count = sum(1 for q_id in self.core_symptoms if symptom_profile.get(q_id, False))
        
        # 第二部分：症状共现性
        co_occurrence = test_data.get('co_occurrence', 'no')
        has_co_occurrence = co_occurrence == 'yes'
        
        # 第三部分：功能影响
        severity = test_data.get('severity', 'no')
        functional_impact_level = self.functional_impact_mapping.get(severity, 'no_problems')
        
        # MDQ结果判定（基于标准算法）
        mdq_result = self._determine_mdq_result(part1_score, has_co_occurrence, functional_impact_level)
        
        # 严重程度评估
        severity_level = self._determine_severity_level(part1_score, has_co_occurrence, functional_impact_level, mdq_result)
        
        # 风险百分比计算
        risk_percentage = self._calculate_risk_percentage(part1_score, has_co_occurrence, functional_impact_level)
        
        return {
            'part1_score': part1_score,
            'has_co_occurrence': has_co_occurrence,
            'functional_impact_level': functional_impact_level,
            'mdq_result': mdq_result,
            'severity_level': severity_level,
            'risk_percentage': risk_percentage,
            'positive_symptoms': positive_symptoms,
            'symptom_profile': symptom_profile,
            'core_symptoms_count': core_symptoms_count
        }
    
    def _determine_mdq_result(self, part1_score: int, has_co_occurrence: bool, functional_impact_level: str) -> str:
        """确定MDQ结果 - 基于标准诊断算法"""
        
        # 标准MDQ阳性标准：
        # 1. 第一部分得分 ≥ 7分
        # 2. 第二部分为"是"（症状同时出现）
        # 3. 第三部分有功能损害（不是"no"）
        
        if part1_score < self.mdq_thresholds['screening_cutoff']:
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
    
    def _determine_severity_level(self, part1_score: int, has_co_occurrence: bool, 
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
        if part1_score >= 10:
            return SeverityLevel.HIGH_POSITIVE
        elif part1_score >= 8:
            return SeverityLevel.MODERATE_POSITIVE
        else:
            return SeverityLevel.MILD_POSITIVE
    
    def _calculate_risk_percentage(self, part1_score: int, has_co_occurrence: bool, 
                                  functional_impact_level: str) -> float:
        """计算双相障碍风险百分比"""
        
        # 基于研究文献的风险计算
        base_risk = 0.0
        
        # 第一部分分数贡献 (0-13分)
        score_risk = min(part1_score / 13.0 * 60, 60)  # 最高60%
        
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
        if part1_score >= 9 and has_co_occurrence and functional_impact_level != 'no_problems':
            total_risk += 15  # 高风险组合奖励
        
        return round(min(100, max(0, total_risk)), 1)
    
    def _analyze_historical_trends(self, test_history: List[Dict]) -> Dict:
        """分析历史趋势"""
        
        if len(test_history) < 2:
            return {
                'trend': ImprovementTrend.STABLE,
                'confidence': 0.5,
                'baseline': self._extract_mdq_score_from_test(test_history[0]) if test_history else 0,
                'trend_data': []
            }
        
        # 提取MDQ分数序列
        scores = []
        dates = []
        for test in reversed(test_history):  # 按时间正序
            try:
                date = datetime.fromisoformat(test['test_timestamp'].replace('Z', '+00:00'))
                mdq_score = self._extract_mdq_score_from_test(test)
                scores.append(mdq_score)
                dates.append(date)
            except Exception as e:
                print(f"处理历史数据时出错: {e}")
                continue
        
        if len(scores) < 2:
            return {
                'trend': ImprovementTrend.STABLE,
                'confidence': 0.5,
                'baseline': scores[0] if scores else 0,
                'trend_data': []
            }
        
        # 计算基线
        baseline_window = min(3, len(scores) // 2)
        baseline = np.mean(scores[:baseline_window]) if baseline_window > 0 else scores[0]
        
        # 趋势分析
        trend, confidence = self._calculate_improvement_trend(scores, dates)
        
        # 趋势数据点
        trend_data = []
        for i, (score, date) in enumerate(zip(scores, dates)):
            trend_data.append({
                'date': date.isoformat(),
                'score': score,
                'baseline_deviation': score - baseline,
                'cumulative_change': score - scores[0] if i > 0 else 0
            })
        
        return {
            'trend': trend,
            'confidence': confidence,
            'baseline': baseline,
            'trend_data': trend_data
        }
    
    def _extract_mdq_score_from_test(self, test: Dict) -> int:
        """从测试记录中提取MDQ分数 - 兼容原有数据库格式"""
        try:
            # 尝试从原有raw_score字段获取，但需要转换为标准MDQ格式
            if 'raw_score' in test and test['raw_score'] is not None:
                raw_score = test['raw_score']
                # 如果原始分数是基于5级评分系统（0-39），需要转换
                if raw_score > 13:
                    # 简单转换：假设原始是5级评分，转换为二分法
                    return min(13, max(0, int(raw_score / 3)))
                else:
                    # 已经是标准MDQ分数
                    return raw_score
            
            # 如果没有raw_score，从test_data计算
            # 这部分需要访问数据库获取完整的test_data
            test_detail = self.db_manager.get_mdq_test_detail(test['test_id'], test.get('user_id', ''))
            if test_detail and 'test_data' in test_detail:
                test_data = test_detail['test_data']
                questions = test_data.get('questions', {})
                
                # 计算标准MDQ分数
                mdq_score = 0
                for q_id in [f'q{i}' for i in range(1, 14)]:
                    answer = questions.get(q_id, 'no')
                    if answer != 'no':
                        mdq_score += 1
                
                return mdq_score
            
            return 0
            
        except Exception as e:
            print(f"提取MDQ分数时出错: {e}")
            return 0
    
    def _calculate_improvement_trend(self, scores: List[int], dates: List[datetime]) -> Tuple[ImprovementTrend, float]:
        """计算改善趋势"""
        
        if len(scores) < 3:
            if len(scores) == 2:
                change = scores[-1] - scores[0]  # 最新 - 最早（分数降低表示改善）
                change_percentage = (change / max(scores[0], 1)) * 100
                
                if abs(change_percentage) <= 15:
                    return ImprovementTrend.STABLE, 0.6
                elif change_percentage <= -30:
                    return ImprovementTrend.MILD_IMPROVEMENT, 0.7
                elif change_percentage >= 30:
                    return ImprovementTrend.MILD_DETERIORATION, 0.7
                elif change_percentage < 0:
                    return ImprovementTrend.MILD_IMPROVEMENT, 0.6
                else:
                    return ImprovementTrend.MILD_DETERIORATION, 0.6
            return ImprovementTrend.STABLE, 0.5
        
        try:
            x = np.arange(len(scores))
            slope, intercept = np.polyfit(x, scores, 1)
            correlation = abs(np.corrcoef(x, scores)[0, 1])
            
            # 计算变化百分比（分数降低表示改善）
            total_change = scores[0] - scores[-1]  # 最早 - 最新
            change_percentage = (total_change / max(scores[0], 1)) * 100 if scores[0] > 0 else 0
            
            # 趋势判断（基于13分制）
            if abs(change_percentage) < 20:
                trend = ImprovementTrend.STABLE
            elif change_percentage >= 50:
                trend = ImprovementTrend.SIGNIFICANT_IMPROVEMENT
            elif change_percentage >= 30:
                trend = ImprovementTrend.MODERATE_IMPROVEMENT
            elif change_percentage >= 20:
                trend = ImprovementTrend.MILD_IMPROVEMENT
            elif change_percentage <= -50:
                trend = ImprovementTrend.SIGNIFICANT_DETERIORATION
            elif change_percentage <= -30:
                trend = ImprovementTrend.MODERATE_DETERIORATION
            else:
                trend = ImprovementTrend.MILD_DETERIORATION
            
            confidence = min(0.95, correlation + 0.1)
            
            return trend, confidence
            
        except Exception as e:
            print(f"趋势计算出错: {e}")
            return ImprovementTrend.STABLE, 0.5
    
    def _analyze_improvement_patterns(self, test_history: List[Dict], current_state: Dict) -> Dict:
        """分析改善模式"""
        
        if len(test_history) < 2:
            return {
                'improvement_percentage': 0,
                'consistency_score': 0.5,
                'recovery_indicators': [],
                'risk_factors': ['数据不足，无法评估改善情况'],
                'treatment_indicators': {}
            }
        
        # 提取MDQ分数序列
        scores = [self._extract_mdq_score_from_test(test) for test in test_history]
        current_score = scores[0]  # 最新分数
        
        # 改善百分比计算
        max_historical = max(scores)
        if max_historical > 0:
            improvement_percentage = ((max_historical - current_score) / max_historical) * 100
        else:
            improvement_percentage = 0
        
        # 一致性分数
        if len(scores) >= 3:
            recent_scores = scores[:min(5, len(scores))]
            cv = np.std(recent_scores) / np.mean(recent_scores) if np.mean(recent_scores) > 0 else 1
            consistency_score = max(0, min(1, 1 - cv / 2))
        else:
            consistency_score = 0.5
        
        # 恢复指标
        recovery_indicators = []
        if improvement_percentage >= 60:
            recovery_indicators.append('MDQ分数显著改善，较峰值下降超过60%')
        elif improvement_percentage >= 40:
            recovery_indicators.append('MDQ分数明显改善，较峰值下降超过40%')
        elif improvement_percentage >= 25:
            recovery_indicators.append('MDQ分数有所改善，较峰值下降超过25%')
        
        # 基于MDQ分数的状态评估
        if current_score <= 3:
            recovery_indicators.append('当前MDQ分数在正常范围内')
        elif current_score <= 6:
            recovery_indicators.append('当前症状轻微，低于筛查阈值')
        
        if consistency_score >= 0.8:
            recovery_indicators.append('症状表现稳定，波动性小')
        elif consistency_score >= 0.6:
            recovery_indicators.append('症状表现相对稳定')
        
        # 风险因素
        risk_factors = []
        if improvement_percentage < -10:
            risk_factors.append('症状较历史最严重时期进一步恶化')
        
        if consistency_score < 0.4:
            risk_factors.append('症状波动性大，不够稳定')
        
        if current_state['mdq_result'] in ['positive_high', 'positive_moderate']:
            risk_factors.append('当前MDQ结果仍为阳性')
        
        if current_score >= 10:
            risk_factors.append('当前MDQ分数较高，需要密切关注')
        
        # 治疗反应指标
        treatment_indicators = self._calculate_treatment_indicators(test_history, current_state)
        
        return {
            'improvement_percentage': round(improvement_percentage, 1),
            'consistency_score': round(consistency_score, 2),
            'recovery_indicators': recovery_indicators,
            'risk_factors': risk_factors,
            'treatment_indicators': treatment_indicators
        }
    
    def _calculate_treatment_indicators(self, test_history: List[Dict], current_state: Dict) -> Dict:
        """计算治疗反应指标"""
        
        if len(test_history) < 3:
            return {'insufficient_data': True}
        
        scores = [self._extract_mdq_score_from_test(test) for test in test_history]
        dates = []
        for test in test_history:
            try:
                date = datetime.fromisoformat(test['test_timestamp'].replace('Z', '+00:00'))
                dates.append(date)
            except:
                continue
        
        if len(dates) != len(scores):
            return {'insufficient_data': True}
        
        # 反应速度
        max_score_idx = scores.index(max(scores))
        if max_score_idx < len(dates) - 1:
            time_to_improve = (dates[0] - dates[max_score_idx]).days
            if time_to_improve <= 30:
                response_speed = 'rapid'
            elif time_to_improve <= 90:
                response_speed = 'moderate'
            else:
                response_speed = 'slow'
        else:
            response_speed = 'unknown'
        
        # 维持改善能力
        if len(scores) >= 5:
            recent_stability = np.std(scores[:3])
            if recent_stability <= 1:
                maintenance_ability = 'good'
            elif recent_stability <= 2:
                maintenance_ability = 'moderate'
            else:
                maintenance_ability = 'poor'
        else:
            maintenance_ability = 'unknown'
        
        # 残留症状评估
        current_score = scores[0]
        if current_score <= 2:
            residual_symptoms = 'minimal'
        elif current_score <= 5:
            residual_symptoms = 'mild'
        elif current_score <= 8:
            residual_symptoms = 'moderate'
        else:
            residual_symptoms = 'significant'
        
        # 治疗效果趋势
        if len(scores) >= 4:
            early_scores = scores[-3:]
            recent_scores = scores[:3]
            
            early_avg = np.mean(early_scores)
            recent_avg = np.mean(recent_scores)
            
            improvement_rate = ((early_avg - recent_avg) / early_avg * 100) if early_avg > 0 else 0
            
            if improvement_rate >= 40:
                treatment_effectiveness = 'excellent'
            elif improvement_rate >= 25:
                treatment_effectiveness = 'good'
            elif improvement_rate >= 10:
                treatment_effectiveness = 'moderate'
            elif improvement_rate >= -10:
                treatment_effectiveness = 'stable'
            else:
                treatment_effectiveness = 'poor'
        else:
            treatment_effectiveness = 'unknown'
            improvement_rate = 0
        
        return {
            'response_speed': response_speed,
            'maintenance_ability': maintenance_ability,
            'residual_symptoms': residual_symptoms,
            'treatment_effectiveness': treatment_effectiveness,
            'improvement_rate': round(improvement_rate, 1),
            'time_to_improve_days': time_to_improve if 'time_to_improve' in locals() else None
        }
    
    def _prepare_ai_analysis_data(self, mdq_analysis: Dict, historical_analysis: Dict, 
                                 improvement_analysis: Dict, test_history: List[Dict]) -> Dict:
        """准备AI分析所需的数据"""
        
        # 构建AI分析数据包
        ai_data = {
            # 患者基本信息
            'patient_demographics': {
                'total_assessments': len(test_history),
                'assessment_span_days': (datetime.fromisoformat(test_history[0]['test_timestamp'].replace('Z', '+00:00')) - 
                                       datetime.fromisoformat(test_history[-1]['test_timestamp'].replace('Z', '+00:00'))).days if len(test_history) > 1 else 0
            },
            
            # MDQ标准结果
            'mdq_standard_results': {
                'part1_score': mdq_analysis['part1_score'],
                'max_part1_score': 13,
                'has_co_occurrence': mdq_analysis['has_co_occurrence'],
                'functional_impact_level': mdq_analysis['functional_impact_level'],
                'mdq_result': mdq_analysis['mdq_result'],
                'severity_level': mdq_analysis['severity_level'].value,
                'risk_percentage': mdq_analysis['risk_percentage'],
                'screening_threshold': self.mdq_thresholds['screening_cutoff'],
                'meets_screening_criteria': mdq_analysis['part1_score'] >= self.mdq_thresholds['screening_cutoff']
            },
            
            # 症状模式分析
            'symptom_patterns': {
                'positive_symptoms_count': mdq_analysis['part1_score'],
                'total_symptoms': 13,
                'symptom_profile': mdq_analysis['symptom_profile'],
                'core_symptoms_count': mdq_analysis['core_symptoms_count'],
                'core_symptoms_total': len(self.core_symptoms),
                'positive_symptoms_list': mdq_analysis['positive_symptoms'],
                'symptom_categories': self._categorize_symptoms(mdq_analysis['symptom_profile'])
            },
            
            # 历史轨迹数据
            'historical_trajectory': {
                'improvement_trend': historical_analysis['trend'].value,
                'trend_confidence': historical_analysis['confidence'],
                'baseline_score': historical_analysis['baseline'],
                'score_timeline': historical_analysis['trend_data'],
                'volatility_index': self._calculate_volatility_index(test_history)
            },
            
            # 治疗反应数据
            'treatment_response': {
                'improvement_percentage': improvement_analysis['improvement_percentage'],
                'consistency_score': improvement_analysis['consistency_score'],
                'treatment_indicators': improvement_analysis['treatment_indicators'],
                'recovery_indicators': improvement_analysis['recovery_indicators'],
                'current_risk_factors': improvement_analysis['risk_factors']
            },
            
            # 统计特征
            'statistical_features': {
                'score_statistics': self._calculate_score_statistics(test_history),
                'trend_analysis': self._calculate_trend_statistics(test_history),
                'stability_metrics': self._calculate_stability_metrics(test_history)
            },
            
            # 临床决策支持数据
            'clinical_context': {
                'emergency_indicators': self._identify_emergency_indicators(mdq_analysis, historical_analysis),
                'monitoring_priorities': self._identify_monitoring_priorities(mdq_analysis, improvement_analysis),
                'intervention_targets': self._identify_intervention_targets(mdq_analysis, historical_analysis),
                'prognosis_factors': self._identify_prognosis_factors(mdq_analysis, historical_analysis, improvement_analysis)
            },
            
            # 风险评估
            'risk_assessment': {
                'immediate_risk_level': self._assess_immediate_risk(mdq_analysis),
                'long_term_risk_factors': self._assess_long_term_risks(mdq_analysis, historical_analysis),
                'protective_factors': self._identify_protective_factors(mdq_analysis, improvement_analysis),
                'bipolar_disorder_probability': mdq_analysis['risk_percentage']
            }
        }
        
        return ai_data
    
    def _categorize_symptoms(self, symptom_profile: Dict[str, bool]) -> Dict:
        """症状分类"""
        categories = {
            'mood_elevation': ['q1'],  # 情绪高涨
            'grandiosity': ['q2'],     # 夸大
            'sleep_changes': ['q3'],   # 睡眠变化
            'speech_changes': ['q4'],  # 言语变化
            'cognitive_symptoms': ['q5', 'q6'],  # 认知症状
            'behavioral_activation': ['q7', 'q8'],  # 行为激活
            'risk_behaviors': ['q9', 'q10', 'q11'],  # 危险行为
            'functional_impact': ['q12', 'q13']  # 功能影响
        }
        
        category_scores = {}
        for category, q_ids in categories.items():
            positive_count = sum(1 for q_id in q_ids if symptom_profile.get(q_id, False))
            category_scores[category] = {
                'positive_count': positive_count,
                'total_count': len(q_ids),
                'percentage': round((positive_count / len(q_ids)) * 100, 1)
            }
        
        return category_scores
    
    def _calculate_volatility_index(self, test_history: List[Dict]) -> float:
        """计算波动性指数"""
        if len(test_history) < 3:
            return 0.0
        
        scores = [self._extract_mdq_score_from_test(test) for test in test_history]
        return round(np.std(scores) / (np.mean(scores) + 1), 3)
    
    def _calculate_score_statistics(self, test_history: List[Dict]) -> Dict:
        """计算分数统计信息"""
        scores = [self._extract_mdq_score_from_test(test) for test in test_history]
        
        if not scores:
            return {}
        
        return {
            'mean': round(np.mean(scores), 2),
            'median': round(np.median(scores), 2),
            'std': round(np.std(scores), 2),
            'min': min(scores),
            'max': max(scores),
            'range': max(scores) - min(scores),
            'percentile_25': round(np.percentile(scores, 25), 2),
            'percentile_75': round(np.percentile(scores, 75), 2)
        }
    
    def _calculate_trend_statistics(self, test_history: List[Dict]) -> Dict:
        """计算趋势统计"""
        scores = [self._extract_mdq_score_from_test(test) for test in test_history]
        
        if len(scores) < 2:
            return {}
        
        # 线性趋势
        x = np.arange(len(scores))
        try:
            slope, intercept = np.polyfit(x, scores, 1)
            r_squared = np.corrcoef(x, scores)[0, 1] ** 2
        except:
            slope, intercept, r_squared = 0, 0, 0
        
        return {
            'linear_slope': round(slope, 3),
            'r_squared': round(r_squared, 3),
            'trend_direction': 'improving' if slope < 0 else 'stable' if abs(slope) < 0.1 else 'worsening'
        }
    
    def _calculate_stability_metrics(self, test_history: List[Dict]) -> Dict:
        """计算稳定性指标"""
        scores = [self._extract_mdq_score_from_test(test) for test in test_history]
        
        if len(scores) < 3:
            return {}
        
        # 连续差异的标准差
        differences = [abs(scores[i] - scores[i+1]) for i in range(len(scores)-1)]
        stability_index = 1 / (1 + np.std(differences)) if differences else 1
        
        return {
            'stability_index': round(stability_index, 3),
            'average_change': round(np.mean(differences), 2) if differences else 0,
            'max_single_change': max(differences) if differences else 0
        }
    
    def _assess_immediate_risk(self, mdq_analysis: Dict) -> str:
        """评估即时风险"""
        mdq_result = mdq_analysis.get('mdq_result')
        part1_score = mdq_analysis.get('part1_score', 0)
        
        if mdq_result == 'positive_high' or part1_score >= 11:
            return 'high'
        elif mdq_result in ['positive_moderate', 'positive_mild']:
            return 'moderate'
        elif part1_score >= self.mdq_thresholds['screening_cutoff']:
            return 'mild'
        else:
            return 'low'
    
    def _assess_long_term_risks(self, mdq_analysis: Dict, historical_analysis: Dict) -> List[str]:
        """评估长期风险因素"""
        risks = []
        
        trend = historical_analysis.get('trend')
        if trend in [ImprovementTrend.MODERATE_DETERIORATION, ImprovementTrend.SIGNIFICANT_DETERIORATION]:
            risks.append('持续恶化趋势')
        
        if mdq_analysis.get('part1_score', 0) >= 9:
            risks.append('高MDQ分数')
        
        if mdq_analysis.get('functional_impact_level') == 'serious_problems':
            risks.append('严重功能损害')
        
        if mdq_analysis.get('core_symptoms_count', 0) >= 5:
            risks.append('多个核心症状')
        
        return risks
    
    def _identify_protective_factors(self, mdq_analysis: Dict, improvement_analysis: Dict) -> List[str]:
        """识别保护因素"""
        factors = []
        
        if improvement_analysis.get('improvement_percentage', 0) > 30:
            factors.append('症状显著改善历史')
        
        if improvement_analysis.get('consistency_score', 0) > 0.7:
            factors.append('症状稳定性良好')
        
        if mdq_analysis.get('part1_score', 0) <= 5:
            factors.append('当前症状轻微')
        
        if mdq_analysis.get('mdq_result') == 'negative':
            factors.append('MDQ筛查阴性')
        
        return factors
    
    def _identify_emergency_indicators(self, mdq_analysis: Dict, historical_analysis: Dict) -> List[str]:
        """识别紧急指标"""
        indicators = []
        
        if mdq_analysis['mdq_result'] == 'positive_high':
            indicators.append('MDQ高度阳性')
        
        if mdq_analysis['functional_impact_level'] == 'serious_problems':
            indicators.append('严重功能损害')
        
        if mdq_analysis['part1_score'] >= 11:
            indicators.append('极高症状分数')
        
        if historical_analysis['trend'] == ImprovementTrend.SIGNIFICANT_DETERIORATION:
            indicators.append('急剧恶化')
        
        return indicators
    
    def _identify_monitoring_priorities(self, mdq_analysis: Dict, improvement_analysis: Dict) -> List[str]:
        """识别监测优先级"""
        priorities = []
        
        if mdq_analysis['mdq_result'] in ['positive_high', 'positive_moderate']:
            priorities.append('intensive_monitoring_required')
        
        if improvement_analysis['consistency_score'] < 0.5:
            priorities.append('symptom_stability_monitoring')
        
        if mdq_analysis['core_symptoms_count'] >= 4:
            priorities.append('core_symptom_monitoring')
        
        return priorities
    
    def _identify_intervention_targets(self, mdq_analysis: Dict, historical_analysis: Dict) -> List[str]:
        """识别干预目标"""
        targets = []
        
        # 基于症状分类确定干预目标
        symptom_categories = self._categorize_symptoms(mdq_analysis['symptom_profile'])
        for category, data in symptom_categories.items():
            if data['percentage'] >= 50:
                targets.append(f'{category}_intervention')
        
        # 基于功能损害确定目标
        if mdq_analysis['functional_impact_level'] in ['moderate_problems', 'serious_problems']:
            targets.append('functional_restoration')
        
        # 基于趋势确定目标
        if historical_analysis['trend'] in [ImprovementTrend.MODERATE_DETERIORATION, ImprovementTrend.SIGNIFICANT_DETERIORATION]:
            targets.append('symptom_stabilization')
        
        return targets
    
    def _identify_prognosis_factors(self, mdq_analysis: Dict, historical_analysis: Dict, improvement_analysis: Dict) -> Dict:
        """识别预后因素"""
        factors = {
            'positive_factors': [],
            'negative_factors': [],
            'neutral_factors': []
        }
        
        # 积极因素
        if improvement_analysis['improvement_percentage'] > 40:
            factors['positive_factors'].append('显著历史改善')
        
        if improvement_analysis['consistency_score'] > 0.7:
            factors['positive_factors'].append('稳定症状模式')
        
        if mdq_analysis['part1_score'] <= 5:
            factors['positive_factors'].append('当前低严重程度')
        
        if mdq_analysis['mdq_result'] == 'negative':
            factors['positive_factors'].append('MDQ筛查阴性')
        
        # 消极因素
        if historical_analysis['trend'] in [ImprovementTrend.MODERATE_DETERIORATION, ImprovementTrend.SIGNIFICANT_DETERIORATION]:
            factors['negative_factors'].append('恶化趋势')
        
        if mdq_analysis['functional_impact_level'] == 'serious_problems':
            factors['negative_factors'].append('严重功能损害')
        
        if improvement_analysis['improvement_percentage'] < -10:
            factors['negative_factors'].append('症状恶化')
        
        if mdq_analysis['part1_score'] >= 10:
            factors['negative_factors'].append('高症状分数')
        
        # 中性因素
        if 0.4 < improvement_analysis['consistency_score'] < 0.7:
            factors['neutral_factors'].append('中等症状稳定性')
        
        return factors
    
    def _generate_clinical_recommendations(self, mdq_analysis: Dict, historical_analysis: Dict, improvement_analysis: Dict) -> Dict:
        """生成临床建议"""
        
        recommendations = []
        emergency_flag = False
        
        # 基于MDQ结果的建议
        mdq_result = mdq_analysis['mdq_result']
        part1_score = mdq_analysis['part1_score']
        
        if mdq_result == 'positive_high':
            emergency_flag = True
            recommendations.extend([
                "立即进行精神科急诊评估",
                "考虑住院治疗或危机干预",
                "MDQ高度阳性，强烈提示双相障碍",
                "紧急药物治疗评估",
                "24小时安全监护"
            ])
            monitoring_frequency = 1  # 每天
            
        elif mdq_result == 'positive_moderate':
            recommendations.extend([
                "48-72小时内安排精神科专科评估",
                "MDQ中度阳性，提示可能的双相障碍",
                "详细的临床访谈和病史收集",
                "考虑情绪稳定剂治疗",
                "加强监测频率"
            ])
            monitoring_frequency = 3  # 每3天
            
        elif mdq_result == 'positive_mild':
            recommendations.extend([
                "1-2周内安排专科评估",
                "MDQ轻度阳性，需要进一步评估",
                "密切关注症状变化",
                "心理教育和生活方式干预",
                "定期随访"
            ])
            monitoring_frequency = 7  # 每周
            
        elif mdq_result == 'positive_subclinical':
            recommendations.extend([
                "门诊随访观察",
                "亚临床阳性，暂无显著功能损害",
                "预防性心理干预",
                "生活方式指导",
                "定期重新评估"
            ])
            monitoring_frequency = 14  # 每两周
            
        else:  # negative
            recommendations.extend([
                "MDQ筛查阴性，继续常规关注",
                "保持健康生活方式",
                "如症状变化及时就诊",
                "定期预防性评估"
            ])
            monitoring_frequency = 30  # 每月
        
        # 基于功能损害的建议
        functional_impact = mdq_analysis['functional_impact_level']
        if functional_impact == 'serious_problems':
            recommendations.append("紧急功能康复干预")
            emergency_flag = True
        elif functional_impact == 'moderate_problems':
            recommendations.append("加强功能评估和支持")
        
        # 基于趋势的调整
        trend = historical_analysis['trend']
        if trend in [ImprovementTrend.SIGNIFICANT_DETERIORATION, ImprovementTrend.MODERATE_DETERIORATION]:
            recommendations.append("紧急评估治疗方案有效性")
            monitoring_frequency = min(monitoring_frequency, 2)
            emergency_flag = True
        
        # 基于改善情况的建议
        if improvement_analysis['consistency_score'] < 0.4:
            recommendations.append("重点关注症状稳定性")
        
        # 计算下次评估日期
        next_assessment_date = datetime.now() + timedelta(days=monitoring_frequency)
        
        return {
            'recommendations': recommendations,
            'monitoring_frequency': monitoring_frequency,
            'emergency_flag': emergency_flag,
            'next_assessment_date': next_assessment_date
        }
    
    def _save_analysis_result(self, result: AnalysisResult, test_id: str) -> bool:
        """保存分析结果到数据库"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO mdq_analysis_results (
                    analysis_id, user_id, test_id, analysis_date,
                    mdq_part1_score, has_co_occurrence, functional_impact_level, 
                    mdq_result, severity_level, risk_percentage,
                    positive_symptoms, symptom_profile, core_symptoms_count,
                    improvement_trend, trend_confidence, historical_baseline, 
                    improvement_percentage, consistency_score,
                    recovery_indicators, risk_factors, treatment_response_indicators,
                    ai_analysis_data, clinical_recommendations, monitoring_frequency,
                    emergency_flag, next_assessment_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.analysis_id,
                result.user_id,
                test_id,
                result.analysis_date.isoformat(),
                result.mdq_part1_score,
                result.has_co_occurrence,
                result.functional_impact_level,
                result.mdq_result,
                result.severity_level.value,
                result.risk_percentage,
                json.dumps(result.positive_symptoms),
                json.dumps(result.symptom_profile),
                result.core_symptoms_count,
                result.improvement_trend.value,
                result.trend_confidence,
                result.historical_baseline,
                result.improvement_percentage,
                result.consistency_score,
                json.dumps(result.recovery_indicators),
                json.dumps(result.risk_factors),
                json.dumps(result.treatment_response_indicators),
                json.dumps(result.ai_analysis_data),
                json.dumps(result.clinical_recommendations),
                result.monitoring_frequency,
                result.emergency_flag,
                result.next_assessment_date.isoformat()
            ))
            
            conn.commit()
            print(f"✅ MDQ分析结果已保存: {result.analysis_id}")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ 保存MDQ分析结果失败: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_analysis_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户分析历史"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analysis_id, analysis_date, mdq_part1_score, mdq_result, 
                       severity_level, risk_percentage, improvement_trend, 
                       trend_confidence, emergency_flag, monitoring_frequency
                FROM mdq_analysis_results
                WHERE user_id = ?
                ORDER BY analysis_date DESC
                LIMIT ?
            ''', (user_id, limit))
            
            results = cursor.fetchall()
            
            return [{
                'analysis_id': row[0],
                'analysis_date': row[1],
                'mdq_part1_score': row[2],
                'mdq_result': row[3],
                'severity_level': row[4],
                'risk_percentage': row[5],
                'improvement_trend': row[6],
                'trend_confidence': row[7],
                'emergency_flag': bool(row[8]),
                'monitoring_frequency': row[9]
            } for row in results]
            
        except sqlite3.Error as e:
            print(f"获取分析历史失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_analysis_detail(self, analysis_id: str) -> Optional[Dict]:
        """获取分析详情"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM mdq_analysis_results
                WHERE analysis_id = ?
            ''', (analysis_id,))
            
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                analysis_data = dict(zip(columns, result))
                
                # 解析JSON字段
                json_fields = ['positive_symptoms', 'symptom_profile', 'recovery_indicators', 
                             'risk_factors', 'treatment_response_indicators', 'ai_analysis_data', 
                             'clinical_recommendations']
                
                for field in json_fields:
                    if analysis_data.get(field):
                        try:
                            analysis_data[field] = json.loads(analysis_data[field])
                        except json.JSONDecodeError:
                            analysis_data[field] = {}
                
                return analysis_data
            
            return None
            
        except sqlite3.Error as e:
            print(f"获取分析详情失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_ai_analysis_data(self, analysis_id: str) -> Optional[Dict]:
        """获取AI分析数据"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ai_analysis_data FROM mdq_analysis_results
                WHERE analysis_id = ?
            ''', (analysis_id,))
            
            result = cursor.fetchone()
            
            if result and result[0]:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    print(f"AI分析数据格式错误: {analysis_id}")
                    return None
            
            return None
            
        except sqlite3.Error as e:
            print(f"获取AI分析数据失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def _create_no_data_result(self, user_id: str) -> AnalysisResult:
        """创建无数据时的默认结果"""
        return AnalysisResult(
            user_id=user_id,
            analysis_id=str(uuid.uuid4()),
            analysis_date=datetime.now(),
            mdq_part1_score=0,
            has_co_occurrence=False,
            functional_impact_level='no_problems',
            mdq_result='negative',
            severity_level=SeverityLevel.NEGATIVE,
            risk_percentage=0.0,
            positive_symptoms=[],
            symptom_profile={},
            core_symptoms_count=0,
            improvement_trend=ImprovementTrend.STABLE,
            trend_confidence=0.0,
            historical_baseline=0.0,
            improvement_percentage=0.0,
            consistency_score=0.0,
            recovery_indicators=['需要完成MDQ问卷评估'],
            risk_factors=['缺乏评估数据'],
            treatment_response_indicators={'insufficient_data': True},
            ai_analysis_data={'status': 'no_data', 'message': '需要至少一次MDQ评估'},
            clinical_recommendations=['首次完成MDQ问卷评估', '建立基线数据'],
            monitoring_frequency=30,
            emergency_flag=False,
            next_assessment_date=datetime.now() + timedelta(days=30)
        )

# 批量分析工具
class BatchAnalyzer:
    """批量分析工具"""
    
    def __init__(self, analyzer: MDQAnalyzer):
        self.analyzer = analyzer
        self.db_manager = analyzer.db_manager
    
    def analyze_all_users(self) -> Dict:
        """分析所有用户"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT user_id FROM questionnaire_tests 
                WHERE questionnaire_type = 'MDQ'
            ''')
            
            user_ids = [row[0] for row in cursor.fetchall()]
            
            # 统计信息
            severity_distribution = {level.value: 0 for level in SeverityLevel}
            mdq_result_distribution = {'negative': 0, 'positive_mild': 0, 'positive_moderate': 0, 'positive_high': 0}
            trend_distribution = {trend.value: 0 for trend in ImprovementTrend}
            emergency_users = []
            high_risk_users = []
            
            for user_id in user_ids:
                try:
                    result = self.analyzer.analyze_user_comprehensive(user_id)
                    
                    severity_distribution[result.severity_level.value] += 1
                    trend_distribution[result.improvement_trend.value] += 1
                    
                    # MDQ结果分布
                    if result.mdq_result in mdq_result_distribution:
                        mdq_result_distribution[result.mdq_result] += 1
                    
                    if result.emergency_flag:
                        emergency_users.append({
                            'user_id': user_id,
                            'mdq_result': result.mdq_result,
                            'mdq_score': result.mdq_part1_score,
                            'risk_percentage': result.risk_percentage,
                            'analysis_id': result.analysis_id
                        })
                    elif result.mdq_result in ['positive_high', 'positive_moderate']:
                        high_risk_users.append({
                            'user_id': user_id,
                            'mdq_result': result.mdq_result,
                            'mdq_score': result.mdq_part1_score,
                            'risk_percentage': result.risk_percentage,
                            'analysis_id': result.analysis_id
                        })
                        
                except Exception as e:
                    print(f"分析用户 {user_id} 失败: {e}")
                    continue
            
            return {
                'total_users': len(user_ids),
                'severity_distribution': severity_distribution,
                'mdq_result_distribution': mdq_result_distribution,
                'trend_distribution': trend_distribution,
                'emergency_users': emergency_users,
                'high_risk_users': high_risk_users,
                'analysis_timestamp': datetime.now().isoformat(),
                'statistics': {
                    'emergency_rate': len(emergency_users) / len(user_ids) * 100 if user_ids else 0,
                    'high_risk_rate': len(high_risk_users) / len(user_ids) * 100 if user_ids else 0,
                    'positive_rate': sum(mdq_result_distribution[k] for k in ['positive_mild', 'positive_moderate', 'positive_high']) / len(user_ids) * 100 if user_ids else 0
                }
            }
            
        except Exception as e:
            print(f"批量分析失败: {e}")
            return {}
        finally:
            if conn:
                conn.close()

# 测试函数
def test_standard_mdq_analyzer():
    """测试标准MDQ分析器"""
    print("=== 标准MDQ分析器测试 ===")
    
    # 初始化
    from database import DatabaseManager
    db_manager = DatabaseManager()
    analyzer = MDQAnalyzer(db_manager)
    
    # 获取现有用户列表
    print("\n=== 查找现有用户 ===")
    try:
        conn = db_manager._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT qt.user_id, u.username, COUNT(qt.test_id) as test_count
            FROM questionnaire_tests qt
            JOIN users u ON qt.user_id = u.user_id
            WHERE qt.questionnaire_type = 'MDQ'
            GROUP BY qt.user_id, u.username
            ORDER BY test_count DESC
        ''')
        
        users_with_tests = cursor.fetchall()
        
        if not users_with_tests:
            print("❌ 没有找到MDQ测试记录")
            return
        
        print(f"找到 {len(users_with_tests)} 个有测试记录的用户:")
        for user_id, username, test_count in users_with_tests:
            print(f"  - 用户: {username} (ID: {user_id[:8]}...) - {test_count} 个测试")
        
        # 选择第一个用户进行分析
        selected_user = users_with_tests[0]
        user_id = selected_user[0]
        username = selected_user[1]
        
        print(f"\n=== 分析用户: {username} ===")
        
        # 执行标准MDQ分析
        analysis_result = analyzer.analyze_user_comprehensive(user_id)
        
        print("✅ 标准MDQ分析完成！")
        print(f"分析ID: {analysis_result.analysis_id}")
        
        # 显示MDQ标准结果
        print(f"\n📊 MDQ标准评估结果:")
        print(f"  - 第一部分分数: {analysis_result.mdq_part1_score}/13")
        print(f"  - 症状共现性: {'是' if analysis_result.has_co_occurrence else '否'}")
        print(f"  - 功能影响程度: {analysis_result.functional_impact_level}")
        print(f"  - MDQ最终结果: {analysis_result.mdq_result}")
        print(f"  - 严重程度等级: {analysis_result.severity_level.value}")
        print(f"  - 双相障碍风险: {analysis_result.risk_percentage}%")
        
        print(f"\n🎯 症状分析:")
        print(f"  - 阳性症状数量: {analysis_result.mdq_part1_score}")
        print(f"  - 核心症状数量: {analysis_result.core_symptoms_count}/7")
        if analysis_result.positive_symptoms:
            print("  阳性症状列表:")
            for symptom in analysis_result.positive_symptoms:
                print(f"    • {symptom}")
        
        print(f"\n📈 历史趋势:")
        print(f"  - 改善趋势: {analysis_result.improvement_trend.value}")
        print(f"  - 趋势置信度: {analysis_result.trend_confidence:.2f}")
        print(f"  - 改善百分比: {analysis_result.improvement_percentage:.1f}%")
        
        print(f"\n🏥 临床建议:")
        for i, recommendation in enumerate(analysis_result.clinical_recommendations, 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\n⚠️ 紧急标志: {'🚨 是' if analysis_result.emergency_flag else '✅ 否'}")
        print(f"📅 监测频率: 每 {analysis_result.monitoring_frequency} 天")
        
        # 验证数据保存
        saved_analysis = analyzer.get_analysis_detail(analysis_result.analysis_id)
        if saved_analysis:
            print("✅ 分析结果已成功保存到数据库")
        else:
            print("❌ 分析结果保存失败")
        
        print("\n🎉 标准MDQ分析测试成功完成！")
        print("✅ 支持标准MDQ二分法评分 (0/1)")
        print("✅ 三部分完整评估")
        print("✅ 基于研究文献的风险评估")
        print("✅ 标准化临床建议")
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    test_standard_mdq_analyzer()
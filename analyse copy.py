import sqlite3
import json
import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
from database import DatabaseManager

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

class MDQAnalyzer:
    """MDQ问卷分析器 - 适配5级评分系统"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._init_database_tables()
        
        # MDQ问题分类（基于test.html的13个问题）
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
            SeverityLevel.NORMAL: (0, 12),
            SeverityLevel.MILD_RISK: (13, 25),
            SeverityLevel.MODERATE_RISK: (26, 39),
            SeverityLevel.HIGH_RISK: (16, 39),
            SeverityLevel.SEVERE_RISK: (40, 52)
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
    
    def _init_database_tables(self):
        """初始化分析结果数据库表"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # 创建分析结果表（增加原始分数字段）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mdq_analysis_results (
                    analysis_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    test_id TEXT,
                    analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- 当前状态
                    current_score INTEGER NOT NULL,
                    raw_score INTEGER NOT NULL,
                    severity_level TEXT NOT NULL,
                    risk_percentage REAL NOT NULL,
                    bipolar_risk_indicators TEXT,
                    
                    -- 症状分析
                    positive_symptoms TEXT,
                    symptom_categories TEXT,
                    symptom_severity_scores TEXT,
                    functional_impairment_level TEXT,
                    
                    -- 历史趋势
                    improvement_trend TEXT,
                    trend_confidence REAL,
                    historical_baseline REAL,
                    improvement_percentage REAL,
                    consistency_score REAL,
                    
                    -- 改善情况
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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_user_date ON mdq_analysis_results(user_id, analysis_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_severity ON mdq_analysis_results(severity_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_trend ON mdq_analysis_results(improvement_trend)')
            
            conn.commit()
            print("分析结果数据库表初始化完成")
            
        except sqlite3.Error as e:
            print(f"数据库表初始化失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def analyze_user_comprehensive(self, user_id: str) -> AnalysisResult:
        """执行用户综合分析"""
        
        # 获取用户测试历史
        test_history = self.db_manager.get_user_mdq_history(user_id, limit=30)
        
        if not test_history:
            return self._create_no_data_result(user_id)
        
        # 获取最新测试详情
        latest_test = test_history[0]
        test_detail = self.db_manager.get_mdq_test_detail(latest_test['test_id'], user_id)
        
        if not test_detail:
            return self._create_no_data_result(user_id)
        # 执行分析
        current_state = self._analyze_current_state(test_detail)
        historical_analysis = self._analyze_historical_trends(test_history)
        improvement_analysis = self._analyze_improvement_patterns(test_history, current_state)
        ai_data = self._prepare_ai_analysis_data(current_state, historical_analysis, improvement_analysis, test_history)
        recommendations = self._generate_clinical_recommendations(current_state, historical_analysis, improvement_analysis)
        # 构建分析结果
        analysis_result = AnalysisResult(
            # 基础信息
            user_id=user_id,
            analysis_id=str(uuid.uuid4()),
            analysis_date=datetime.now(),
            
            # 当前状态
            current_score=current_state['normalized_score'],  # 兼容性分数
            raw_score=current_state['raw_score'],
            severity_level=current_state['severity_level'],
            risk_percentage=current_state['risk_percentage'],
            bipolar_risk_indicators=current_state['bipolar_indicators'],
    
            # 症状分析
            positive_symptoms=current_state['positive_symptoms'],
            symptom_categories=current_state['symptom_categories'],
            symptom_severity_scores=current_state['symptom_severity_scores'],
            functional_impairment_level=current_state['impairment_level'],
            
            # 历史趋势
            improvement_trend=historical_analysis['trend'],
            trend_confidence=historical_analysis['confidence'],
            historical_baseline=historical_analysis['baseline'],
            improvement_percentage=improvement_analysis['improvement_percentage'],
            consistency_score=improvement_analysis['consistency_score'],
            
            # 改善情况
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
    
    def _analyze_current_state(self, test_detail: Dict) -> Dict:
        """分析当前状态 - 适配5级评分系统"""
        test_data = test_detail['test_data']
        questions = test_data.get('questions', {})
        
        # 计算原始分数（0-39分）
        raw_score = 0
        symptom_scores = {}
        
        for q_id, answer in questions.items():
            score = self.score_mapping.get(answer, 0)
            raw_score += score
            symptom_scores[q_id] = score
        
        # 计算标准化分数（0-13分，兼容原有系统）
        normalized_score = min(13, round(raw_score / 5))
        
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
        for indicator, q_ids in self.bipolar_indicators.items():
            # 计算该指标组的平均严重程度
            print('indicator',indicator)
            print('q_ids',q_ids)
            indicator_scores = [symptom_scores.get(q_id, 0) for q_id in q_ids]
            avg_score = np.mean(indicator_scores) if indicator_scores else 0
            print(avg_score)
            
            # 如果平均分数≥2，认为该指标阳性
            if avg_score >= 3.0:
                bipolar_indicators[indicator] = 1
            else:
                bipolar_indicators[indicator] = 0
            #bipolar_indicators[indicator] = avg_score >= 2.0
        print(bipolar_indicators)
        
        # 严重程度评估
        severity_level = self._calculate_severity_level(
            raw_score, weighted_score, bipolar_indicators, test_data
        )
        
        # 风险百分比
        risk_percentage = self._calculate_risk_percentage(
            raw_score, weighted_score, bipolar_indicators, test_data
        )
        
        # 功能损害评估
        impairment_level = self._assess_functional_impairment(
            test_data, bipolar_indicators, symptom_scores
        )
        
        return {
            'raw_score': raw_score,
            'normalized_score': normalized_score,
            'weighted_score': weighted_score,
            'severity_level': severity_level,
            'risk_percentage': risk_percentage,
            'positive_symptoms': positive_symptoms,
            'symptom_categories': symptom_categories,
            'symptom_severity_scores': symptom_severity_scores,
            'bipolar_indicators': bipolar_indicators,
            'impairment_level': impairment_level,
            'symptom_scores': symptom_scores
        }
    
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
    
    def _calculate_risk_percentage(self, raw_score: int, weighted_score: float,
                                 bipolar_indicators: Dict[str, bool], test_data: Dict) -> float:
        """计算风险百分比 - 基于新的评分系统"""
        
        # 基础风险 (基于原始分数0-39)
        base_risk = min(100, (raw_score / 39) * 100)
        
        # 加权调整
        max_weighted_score = 39 * max(self.symptom_weights.values())
        weighted_adjustment = (weighted_score / max_weighted_score) * 100
        
        # 临床指标调整
        clinical_adjustments = 0
        
        if test_data.get('co_occurrence') == 'yes':
            clinical_adjustments += 15
        
        severity = test_data.get('severity', 'no')
        if severity == 'serious':
            clinical_adjustments += 25
        elif severity == 'moderate':
            clinical_adjustments += 15
        elif severity == 'minor':
            clinical_adjustments += 5
        
        # 双相指标调整
        positive_indicators = sum(bipolar_indicators.values())
        clinical_adjustments += positive_indicators * 8
        
        # 综合风险计算
        final_risk = (base_risk * 0.4 + weighted_adjustment * 0.4 + clinical_adjustments * 0.2)
        
        return round(min(100, max(0, final_risk)), 1)
    
    def _assess_functional_impairment(self, test_data: Dict, bipolar_indicators: Dict[str, bool], 
                                    symptom_scores: Dict[str, int]) -> str:
        """评估功能损害水平 - 基于新的评分系统"""
        
        severity = test_data.get('severity', 'no')
        co_occurrence = test_data.get('co_occurrence', 'no')
        
        # 获取功能损害相关的分数
        impairment_score = symptom_scores.get('q12', 0)  # 问题12：功能损害
        others_noticed_score = symptom_scores.get('q13', 0)  # 问题13：他人注意到
        
        # 行为症状严重程度
        behavioral_symptoms = bipolar_indicators.get('behavioral_symptoms', False)
        social_impact = bipolar_indicators.get('social_impact', False)
        
        # 综合评估
        if (severity == 'serious' and co_occurrence == 'yes') or impairment_score >= 3:
            return 'severe'
        elif (severity == 'serious' or 
              (severity == 'moderate' and behavioral_symptoms) or 
              impairment_score >= 2):
            return 'moderate'
        elif (severity == 'moderate' or 
              social_impact or 
              others_noticed_score >= 2 or 
              impairment_score >= 1):
            return 'mild'
        else:
            return 'minimal'
    
    def _analyze_historical_trends(self, test_history: List[Dict]) -> Dict:
        """分析历史趋势 - 适配新的评分系统"""
        
        if len(test_history) < 2:
            return {
                'trend': ImprovementTrend.STABLE,
                'confidence': 0.5,
                'baseline': self._extract_raw_score_from_test(test_history[0]) if test_history else 0,
                'trend_data': []
            }
        
        # 提取时间序列数据（使用原始分数）
        scores = []
        dates = []
        for test in reversed(test_history):  # 按时间正序
            try:
                date = datetime.fromisoformat(test['test_timestamp'].replace('Z', '+00:00'))
                # 尝试从测试数据中提取原始分数
                raw_score = self._extract_raw_score_from_test(test)
                scores.append(raw_score)
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
        
        # 计算基线 (使用早期数据的平均值)
        baseline_window = min(5, len(scores) // 2)
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
    
    def _extract_raw_score_from_test(self, test: Dict) -> int:
        """从测试记录中提取原始分数"""
        try:
            # 如果直接有raw_score字段
            if 'raw_score' in test:
                return test['raw_score']
            
            # 如果有test_data，计算原始分数
            if 'test_data' in test:
                test_data = test['test_data']
                if isinstance(test_data, str):
                    test_data = json.loads(test_data)
                
                questions = test_data.get('questions', {})
                raw_score = sum(self.score_mapping.get(answer, 0) 
                              for answer in questions.values())
                return raw_score
            
            # 兼容旧格式：使用normalized_score估算
            if 'raw_score' in test:
                return test['raw_score']
            
            # 最后的兼容性处理：从normalized_score估算
            normalized = test.get('normalized_score', test.get('score', 0))
            return normalized * 3  # 粗略估算
            
        except Exception as e:
            print(f"提取分数时出错: {e}")
            return 0
    
    def _calculate_improvement_trend(self, scores: List[int], dates: List[datetime]) -> Tuple[ImprovementTrend, float]:
        """计算改善趋势 - 基于新的评分系统"""
        
        if len(scores) < 3:
            if len(scores) == 2:
                change = scores[0] - scores[-1]  # 最新 - 最早（分数降低表示改善）
                change_percentage = (change / max(scores[-1], 1)) * 100
                
                if abs(change_percentage) <= 10:
                    return ImprovementTrend.STABLE, 0.6
                elif change_percentage >= 20:
                    return ImprovementTrend.MILD_IMPROVEMENT, 0.7
                elif change_percentage <= -20:
                    return ImprovementTrend.MILD_DETERIORATION, 0.7
                elif change_percentage > 0:
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
            change_percentage = (total_change / max(scores[-1], 1)) * 100 if scores[-1] > 0 else 0
            
            # 趋势判断（基于39分制）
            if abs(change_percentage) < 15:
                trend = ImprovementTrend.STABLE
            elif change_percentage >= 40:
                trend = ImprovementTrend.SIGNIFICANT_IMPROVEMENT
            elif change_percentage >= 25:
                trend = ImprovementTrend.MODERATE_IMPROVEMENT
            elif change_percentage >= 15:
                trend = ImprovementTrend.MILD_IMPROVEMENT
            elif change_percentage <= -40:
                trend = ImprovementTrend.SIGNIFICANT_DETERIORATION
            elif change_percentage <= -25:
                trend = ImprovementTrend.MODERATE_DETERIORATION
            else:
                trend = ImprovementTrend.MILD_DETERIORATION
            
            confidence = min(0.95, correlation + 0.1)
            
            return trend, confidence
            
        except Exception as e:
            print(f"趋势计算出错: {e}")
            # 备用方法：移动平均比较
            if len(scores) >= 6:
                recent_avg = np.mean(scores[-3:])
                earlier_avg = np.mean(scores[:3])
                change = (earlier_avg - recent_avg) / max(recent_avg, 1) * 100
                
                if abs(change) < 20:
                    return ImprovementTrend.STABLE, 0.7
                elif change >= 30:
                    return ImprovementTrend.MODERATE_IMPROVEMENT, 0.8
                elif change > 0:
                    return ImprovementTrend.MILD_IMPROVEMENT, 0.7
                elif change <= -30:
                    return ImprovementTrend.MODERATE_DETERIORATION, 0.8
                else:
                    return ImprovementTrend.MILD_DETERIORATION, 0.7
            
            return ImprovementTrend.STABLE, 0.5
    
    def _analyze_improvement_patterns(self, test_history: List[Dict], current_state: Dict) -> Dict:
        """分析改善模式 - 基于新的评分系统"""
        
        if len(test_history) < 2:
            return {
                'improvement_percentage': 0,
                'consistency_score': 0.5,
                'recovery_indicators': [],
                'risk_factors': ['数据不足，无法评估改善情况'],
                'treatment_indicators': {}
            }
        
        # 提取原始分数序列
        scores = [self._extract_raw_score_from_test(test) for test in test_history]
        current_score = scores[0]  # 最新分数
        
        # 改善百分比计算（基于39分制）
        max_historical = max(scores)
        if max_historical > 0:
            improvement_percentage = ((max_historical - current_score) / max_historical) * 100
        else:
            improvement_percentage = 0
        
        # 一致性分数 (低变异性 = 高一致性)
        if len(scores) >= 3:
            recent_scores = scores[:min(5, len(scores))]
            cv = np.std(recent_scores) / np.mean(recent_scores) if np.mean(recent_scores) > 0 else 1
            consistency_score = max(0, min(1, 1 - cv / 2))  # 调整CV的影响
        else:
            consistency_score = 0.5
        
        # 恢复指标
        recovery_indicators = []
        if improvement_percentage >= 60:
            recovery_indicators.append('症状显著改善，较峰值改善超过60%')
        elif improvement_percentage >= 40:
            recovery_indicators.append('症状明显改善，较峰值改善超过40%')
        elif improvement_percentage >= 25:
            recovery_indicators.append('症状有所改善，较峰值改善超过25%')
        
        # 基于39分制的状态评估
        if current_score <= 8:
            recovery_indicators.append('当前处于正常或低风险状态')
        elif current_score <= 15:
            recovery_indicators.append('当前症状轻微，风险可控')
        
        if consistency_score >= 0.8:
            recovery_indicators.append('症状表现稳定，波动性小')
        elif consistency_score >= 0.6:
            recovery_indicators.append('症状表现相对稳定')
        
        # 症状严重程度分析
        symptom_severity_scores = current_state.get('symptom_severity_scores', {})
        low_severity_categories = [cat for cat, score in symptom_severity_scores.items() if score < 25]
        if len(low_severity_categories) >= len(symptom_severity_scores) * 0.7:
            recovery_indicators.append('大部分症状类别处于轻微水平')
        
        # 风险因素
        risk_factors = []
        if improvement_percentage < -10:
            risk_factors.append('症状较历史最严重时期更加严重')
        
        if consistency_score < 0.4:
            risk_factors.append('症状波动性大，不够稳定')
        
        severity_level = current_state.get('severity_level')
        if severity_level in [SeverityLevel.HIGH_RISK, SeverityLevel.SEVERE_RISK]:
            risk_factors.append('当前仍处于高风险状态')
        
        # 检查高危症状
        high_severity_categories = [cat for cat, score in symptom_severity_scores.items() if score >= 75]
        if high_severity_categories:
            risk_factors.append(f'存在高严重程度症状类别: {", ".join(high_severity_categories)}')
        
        # 功能损害评估
        if current_state.get('impairment_level') in ['moderate', 'severe']:
            risk_factors.append('存在明显的功能损害')
        
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
        """计算治疗反应指标 - 基于新的评分系统"""
        
        if len(test_history) < 3:
            return {'insufficient_data': True}
        
        scores = [self._extract_raw_score_from_test(test) for test in test_history]
        dates = []
        for test in test_history:
            try:
                date = datetime.fromisoformat(test['test_timestamp'].replace('Z', '+00:00'))
                dates.append(date)
            except:
                continue
        
        if len(dates) != len(scores):
            return {'insufficient_data': True}
        
        # 反应速度 (从最高分到当前的时间)
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
            if recent_stability <= 3:  # 基于39分制调整
                maintenance_ability = 'good'
            elif recent_stability <= 6:
                maintenance_ability = 'moderate'
            else:
                maintenance_ability = 'poor'
        else:
            maintenance_ability = 'unknown'
        
        # 残留症状评估（基于39分制）
        current_score = scores[0]
        if current_score <= 4:
            residual_symptoms = 'minimal'
        elif current_score <= 12:
            residual_symptoms = 'mild'
        elif current_score <= 20:
            residual_symptoms = 'moderate'
        else:
            residual_symptoms = 'significant'
        
        # 治疗效果趋势
        if len(scores) >= 4:
            early_scores = scores[-3:]  # 早期3次
            recent_scores = scores[:3]  # 最近3次
            
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
    
    def _prepare_ai_analysis_data(self, current_state: Dict, historical_analysis: Dict, 
                                improvement_analysis: Dict, test_history: List[Dict]) -> Dict:
        """准备AI分析所需的数据 - 增强版"""
        
        # 获取用户基本信息
        user_profile = self.db_manager.get_user_profile(current_state.get('user_id', ''))
        
        # 构建AI分析数据包
        ai_data = {
            # 患者基本信息
            'patient_demographics': {
                'age': user_profile.get('age') if user_profile else None,
                'gender': user_profile.get('gender') if user_profile else None,
                'total_assessments': len(test_history),
                'assessment_span_days': (datetime.fromisoformat(test_history[0]['test_timestamp'].replace('Z', '+00:00')) - 
                                       datetime.fromisoformat(test_history[-1]['test_timestamp'].replace('Z', '+00:00'))).days if len(test_history) > 1 else 0
            },
            
            # 当前临床状态（增强版）
            'current_clinical_state': {
                'raw_score': current_state['raw_score'],
                'normalized_score': current_state['normalized_score'],
                'weighted_score': current_state['weighted_score'],
                'severity_level': current_state['severity_level'].value,
                'risk_percentage': current_state['risk_percentage'],
                'functional_impairment': current_state['impairment_level'],
                'positive_symptoms': current_state['positive_symptoms'],
                'symptom_distribution': current_state['symptom_categories'],
                'symptom_severity_scores': current_state['symptom_severity_scores'],
                'bipolar_risk_profile': current_state['bipolar_indicators'],
                'individual_question_scores': current_state.get('symptom_scores', {})
            },
            
            # 症状模式分析
            'symptom_patterns': {
                'core_symptoms': self._analyze_core_symptoms(current_state),
                'behavioral_indicators': self._analyze_behavioral_indicators(current_state),
                'cognitive_symptoms': self._analyze_cognitive_symptoms(current_state),
                'social_functional_impact': self._analyze_social_impact(current_state)
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
                'emergency_indicators': self._identify_emergency_indicators(current_state, historical_analysis),
                'monitoring_priorities': self._identify_monitoring_priorities(current_state, improvement_analysis),
                'intervention_targets': self._identify_intervention_targets(current_state, historical_analysis),
                'prognosis_factors': self._identify_prognosis_factors(current_state, historical_analysis, improvement_analysis)
            },
            
            # 风险评估
            'risk_assessment': {
                'immediate_risk_level': self._assess_immediate_risk(current_state),
                'long_term_risk_factors': self._assess_long_term_risks(current_state, historical_analysis),
                'protective_factors': self._identify_protective_factors(current_state, improvement_analysis)
            }
        }
        #print('ai_data',ai_data)
        
        return ai_data
    
    def _analyze_core_symptoms(self, current_state: Dict) -> Dict:
        """分析核心症状"""
        symptom_scores = current_state.get('symptom_scores', {})
        core_symptoms = {
            'mood_elevation': symptom_scores.get('q1', 0),
            'grandiosity': symptom_scores.get('q2', 0),
            'sleep_reduction': symptom_scores.get('q3', 0),
            'racing_thoughts': symptom_scores.get('q5', 0)
        }
        
        return {
            'individual_scores': core_symptoms,
            'total_score': sum(core_symptoms.values()),
            'average_severity': np.mean(list(core_symptoms.values())),
            'most_severe': max(core_symptoms.items(), key=lambda x: x[1])[0] if core_symptoms else None
        }
    
    def _analyze_behavioral_indicators(self, current_state: Dict) -> Dict:
        """分析行为指标"""
        symptom_scores = current_state.get('symptom_scores', {})
        behavioral_symptoms = {
            'social_disinhibition': symptom_scores.get('q8', 0),
            'hypersexuality': symptom_scores.get('q9', 0),
            'poor_judgment': symptom_scores.get('q10', 0),
            'reckless_spending': symptom_scores.get('q11', 0)
        }
        
        return {
            'individual_scores': behavioral_symptoms,
            'total_score': sum(behavioral_symptoms.values()),
            'risk_level': 'high' if sum(behavioral_symptoms.values()) >= 8 else 'moderate' if sum(behavioral_symptoms.values()) >= 4 else 'low'
        }
    
    def _analyze_cognitive_symptoms(self, current_state: Dict) -> Dict:
        """分析认知症状"""
        symptom_scores = current_state.get('symptom_scores', {})
        cognitive_symptoms = {
            'racing_thoughts': symptom_scores.get('q5', 0),
            'distractibility': symptom_scores.get('q6', 0),
            'talkativeness': symptom_scores.get('q4', 0)
        }
        
        return {
            'individual_scores': cognitive_symptoms,
            'total_score': sum(cognitive_symptoms.values()),
            'cognitive_impairment_level': 'severe' if sum(cognitive_symptoms.values()) >= 9 else 'moderate' if sum(cognitive_symptoms.values()) >= 6 else 'mild'
        }
    
    def _analyze_social_impact(self, current_state: Dict) -> Dict:
        """分析社会功能影响"""
        symptom_scores = current_state.get('symptom_scores', {})
        social_impact = {
            'functional_impairment': symptom_scores.get('q12', 0),
            'others_noticed': symptom_scores.get('q13', 0)
        }
        
        return {
            'individual_scores': social_impact,
            'total_score': sum(social_impact.values()),
            'social_functioning_level': 'severely_impaired' if sum(social_impact.values()) >= 6 else 'moderately_impaired' if sum(social_impact.values()) >= 4 else 'mildly_impaired' if sum(social_impact.values()) >= 2 else 'intact'
        }
    
    def _calculate_volatility_index(self, test_history: List[Dict]) -> float:
        """计算波动性指数"""
        if len(test_history) < 3:
            return 0.0
        
        scores = [self._extract_raw_score_from_test(test) for test in test_history]
        return round(np.std(scores) / (np.mean(scores) + 1), 3)
    
    def _calculate_score_statistics(self, test_history: List[Dict]) -> Dict:
        """计算分数统计信息"""
        scores = [self._extract_raw_score_from_test(test) for test in test_history]
        
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
        scores = [self._extract_raw_score_from_test(test) for test in test_history]
        
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
        scores = [self._extract_raw_score_from_test(test) for test in test_history]
        
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
    
    def _assess_immediate_risk(self, current_state: Dict) -> str:
        """评估即时风险"""
        raw_score = current_state.get('raw_score', 0)
        severity = current_state.get('severity_level')
        behavioral_symptoms = current_state.get('bipolar_indicators', {}).get('behavioral_symptoms', False)
        
        if severity == SeverityLevel.SEVERE_RISK or raw_score >= 30:
            return 'critical'
        elif severity == SeverityLevel.HIGH_RISK or behavioral_symptoms:
            return 'high'
        elif severity == SeverityLevel.MODERATE_RISK:
            return 'moderate'
        else:
            return 'low'
    
    def _assess_long_term_risks(self, current_state: Dict, historical_analysis: Dict) -> List[str]:
        """评估长期风险因素"""
        risks = []
        
        trend = historical_analysis.get('trend')
        if trend in [ImprovementTrend.MODERATE_DETERIORATION, ImprovementTrend.SIGNIFICANT_DETERIORATION]:
            risks.append('持续恶化趋势')
        
        bipolar_indicators = current_state.get('bipolar_indicators', {})
        if sum(bipolar_indicators.values()) >= 3:
            risks.append('多重双相障碍指标')
        
        if current_state.get('impairment_level') in ['moderate', 'severe']:
            risks.append('持续功能损害')
        
        return risks
    
    def _identify_protective_factors(self, current_state: Dict, improvement_analysis: Dict) -> List[str]:
        """识别保护因素"""
        factors = []
        
        if improvement_analysis.get('improvement_percentage', 0) > 20:
            factors.append('症状显著改善历史')
        
        if improvement_analysis.get('consistency_score', 0) > 0.7:
            factors.append('症状稳定性良好')
        
        if current_state.get('raw_score', 0) <= 8:
            factors.append('当前症状轻微')
        
        return factors
    
    def _identify_emergency_indicators(self, current_state: Dict, historical_analysis: Dict) -> List[str]:
        """识别紧急指标"""
        indicators = []
        
        if current_state['severity_level'] == SeverityLevel.SEVERE_RISK:
            indicators.append('severe_current_symptoms')
        
        if current_state['impairment_level'] == 'severe':
            indicators.append('severe_functional_impairment')
        
        if current_state['bipolar_indicators'].get('behavioral_symptoms', False):
            indicators.append('high_risk_behavioral_symptoms')
        
        if historical_analysis['trend'] in [ImprovementTrend.SIGNIFICANT_DETERIORATION, ImprovementTrend.MODERATE_DETERIORATION]:
            indicators.append('rapid_deterioration')
        
        # 检查高危单项症状
        symptom_scores = current_state.get('symptom_scores', {})
        high_risk_symptoms = ['q10', 'q11', 'q12']  # 冲动决定、冲动消费、功能损害
        for symptom in high_risk_symptoms:
            if symptom_scores.get(symptom, 0) >= 3:
                indicators.append(f'severe_{symptom}_symptoms')
        
        return indicators
    
    def _identify_monitoring_priorities(self, current_state: Dict, improvement_analysis: Dict) -> List[str]:
        """识别监测优先级"""
        priorities = []
        
        if current_state['severity_level'] in [SeverityLevel.HIGH_RISK, SeverityLevel.SEVERE_RISK]:
            priorities.append('intensive_monitoring_required')
        
        if improvement_analysis['consistency_score'] < 0.5:
            priorities.append('symptom_stability_monitoring')
        
        if current_state['bipolar_indicators'].get('behavioral_symptoms', False):
            priorities.append('behavioral_risk_monitoring')
        
        # 基于症状严重程度的监测
        symptom_severity_scores = current_state.get('symptom_severity_scores', {})
        high_severity_categories = [cat for cat, score in symptom_severity_scores.items() if score >= 60]
        if high_severity_categories:
            priorities.append('high_severity_symptom_monitoring')
        
        return priorities
    
    def _identify_intervention_targets(self, current_state: Dict, historical_analysis: Dict) -> List[str]:
        """识别干预目标"""
        targets = []
        
        # 基于症状分类确定干预目标
        symptom_severity_scores = current_state.get('symptom_severity_scores', {})
        for category, score in symptom_severity_scores.items():
            if score >= 50:  # 中等严重程度以上
                targets.append(f'{category}_intervention')
        
        # 基于功能损害确定目标
        if current_state['impairment_level'] in ['moderate', 'severe']:
            targets.append('functional_restoration')
        
        # 基于趋势确定目标
        if historical_analysis['trend'] in [ImprovementTrend.MODERATE_DETERIORATION, ImprovementTrend.SIGNIFICANT_DETERIORATION]:
            targets.append('symptom_stabilization')
        
        return targets
    
    def _identify_prognosis_factors(self, current_state: Dict, historical_analysis: Dict, improvement_analysis: Dict) -> Dict:
        """识别预后因素"""
        factors = {
            'positive_factors': [],
            'negative_factors': [],
            'neutral_factors': []
        }
        
        # 积极因素
        if improvement_analysis['improvement_percentage'] > 30:
            factors['positive_factors'].append('significant_historical_improvement')
        
        if improvement_analysis['consistency_score'] > 0.7:
            factors['positive_factors'].append('stable_symptom_pattern')
        
        if current_state['raw_score'] <= 8:
            factors['positive_factors'].append('current_low_severity')
        
        treatment_indicators = improvement_analysis.get('treatment_indicators', {})
        if treatment_indicators.get('treatment_effectiveness') in ['excellent', 'good']:
            factors['positive_factors'].append('good_treatment_response')
        
        # 消极因素
        if historical_analysis['trend'] in [ImprovementTrend.MODERATE_DETERIORATION, ImprovementTrend.SIGNIFICANT_DETERIORATION]:
            factors['negative_factors'].append('deteriorating_trend')
        
        if current_state['bipolar_indicators'].get('behavioral_symptoms', False):
            factors['negative_factors'].append('high_risk_behaviors')
        
        if improvement_analysis['improvement_percentage'] < -10:
            factors['negative_factors'].append('worsening_from_baseline')
        
        if current_state['impairment_level'] == 'severe':
            factors['negative_factors'].append('severe_functional_impairment')
        
        # 中性因素
        if 0.4<improvement_analysis['consistency_score'] < 0.7:
            factors['neutral_factors'].append('moderate_symptom_stability')
        
        return factors
    
    def _generate_clinical_recommendations(self, current_state: Dict, historical_analysis: Dict, improvement_analysis: Dict) -> Dict:
        """生成临床建议 - 增强版"""
        
        recommendations = []
        emergency_flag = False
        
        # 基于严重程度的建议
        severity = current_state['severity_level']
        raw_score = current_state.get('raw_score', 0)
        
        if severity == SeverityLevel.SEVERE_RISK or raw_score >= 30:
            emergency_flag = True
            recommendations.extend([
                "立即进行精神科急诊评估",
                "考虑住院治疗或危机干预",
                "24小时监护和支持",
                "紧急药物治疗调整",
                "联系紧急联系人"
            ])
            monitoring_frequency = 1  # 每天
            
        elif severity == SeverityLevel.HIGH_RISK:
            recommendations.extend([
                "48小时内安排精神科专科评估",
                "加强门诊监测频率",
                "评估药物治疗调整需求",
                "实施安全监护措施",
                "考虑增加治疗密度"
            ])
            monitoring_frequency = 2  # 每2天
            
        elif severity == SeverityLevel.MODERATE_RISK:
            recommendations.extend([
                "1-2周内安排专科随访",
                "继续当前治疗方案并评估效果",
                "加强心理社会支持",
                "定期症状监测",
                "评估生活方式干预"
            ])
            monitoring_frequency = 7  # 每周
            
        elif severity == SeverityLevel.MILD_RISK:
            recommendations.extend([
                "维持常规随访计划",
                "关注症状变化趋势",
                "加强生活方式管理",
                "预防性心理干预",
                "继续当前治疗"
            ])
            monitoring_frequency = 14  # 每两周
            
        else:  # NORMAL
            recommendations.extend([
                "维持当前稳定状态",
                "定期预防性评估",
                "继续健康生活方式",
                "保持治疗依从性"
            ])
            monitoring_frequency = 30  # 每月
        
        # 基于症状严重程度的具体建议
        symptom_severity_scores = current_state.get('symptom_severity_scores', {})
        high_severity_symptoms = [cat for cat, score in symptom_severity_scores.items() if score >= 75]
        
        for symptom_category in high_severity_symptoms:
            if symptom_category == 'behavioral_symptoms':
                recommendations.append("重点关注高危行为，加强行为管理")
            elif symptom_category == 'cognitive_symptoms':
                recommendations.append("考虑认知功能评估和干预")
            elif symptom_category == 'functional_impairment':
                recommendations.append("加强功能康复训练")
        
        # 基于趋势的调整
        trend = historical_analysis['trend']
        
        if trend in [ImprovementTrend.SIGNIFICANT_DETERIORATION, ImprovementTrend.MODERATE_DETERIORATION]:
            recommendations.append("紧急评估治疗方案有效性")
            recommendations.append("考虑更换或调整治疗策略")
            monitoring_frequency = min(monitoring_frequency, 3)
            
        elif trend in [ImprovementTrend.SIGNIFICANT_IMPROVEMENT, ImprovementTrend.MODERATE_IMPROVEMENT]:
            recommendations.append("维持当前有效治疗策略")
            recommendations.append("考虑逐步减少监测频率")
            
        # 基于改善情况的建议
        if improvement_analysis['consistency_score'] < 0.4:
            recommendations.append("重点关注症状稳定性，识别触发因素")
            recommendations.append("考虑情绪稳定剂治疗")
        
        if improvement_analysis['improvement_percentage'] < -15:
            recommendations.append("全面重新评估治疗计划")
            recommendations.append("考虑多学科会诊")
            emergency_flag = True
        
        # 基于双相风险的建议
        bipolar_indicators = current_state.get('bipolar_indicators', {})
        positive_indicators = sum(bipolar_indicators.values())
        
        if positive_indicators >= 3:
            recommendations.append("进行双相情感障碍专项评估")
            recommendations.append("考虑情绪稳定剂治疗")
        elif positive_indicators >= 2:
            recommendations.append("监测双相情感障碍风险")
        
        # 基于治疗反应的建议
        treatment_indicators = improvement_analysis.get('treatment_indicators', {})
        if treatment_indicators.get('treatment_effectiveness') == 'poor':
            recommendations.append("评估治疗依从性和药物浓度")
            recommendations.append("考虑治疗方案调整")
        elif treatment_indicators.get('response_speed') == 'slow':
            recommendations.append("评估治疗剂量是否充分")
        
        # 功能损害相关建议
        impairment_level = current_state.get('impairment_level')
        if impairment_level == 'severe':
            recommendations.append("紧急功能评估和康复干预")
            recommendations.append("考虑住院或日间治疗")
        elif impairment_level == 'moderate':
            recommendations.append("加强功能康复训练")
            recommendations.append("评估工作/学习能力")
        
        # 计算下次评估日期
        next_assessment_date = datetime.now() + timedelta(days=monitoring_frequency)
        
        return {
            'recommendations': recommendations,
            'monitoring_frequency': monitoring_frequency,
            'emergency_flag': emergency_flag,
            'next_assessment_date': next_assessment_date
        }
    
    '''def _save_analysis_result(self, result: AnalysisResult, test_id: str) -> bool:
        """保存分析结果到数据库"""
        print('result.bipolar_risk_indicators:',result.bipolar_risk_indicators)
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute''''''(
                INSERT INTO mdq_analysis_results (
                    analysis_id, user_id, test_id, analysis_date,
                    current_score, raw_score, severity_level, risk_percentage, bipolar_risk_indicators,
                    positive_symptoms, symptom_categories, symptom_severity_scores, functional_impairment_level,
                    improvement_trend, trend_confidence, historical_baseline, 
                    improvement_percentage, consistency_score,
                    recovery_indicators, risk_factors, treatment_response_indicators,
                    ai_analysis_data, clinical_recommendations, monitoring_frequency,
                    emergency_flag, next_assessment_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''''', (
                result.analysis_id,
                result.user_id,
                test_id,
                result.analysis_date.isoformat(),
                result.current_score,
                result.raw_score,
                result.severity_level.value,
                result.risk_percentage,
                json.dumps(result.bipolar_risk_indicators),
                json.dumps(result.positive_symptoms),
                json.dumps(result.symptom_categories),
                json.dumps(result.symptom_severity_scores),
                result.functional_impairment_level,
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
            print(f"分析结果已保存: {result.analysis_id}")
            return True
            
        except sqlite3.Error as e:
            print(f"保存分析结果失败: {e}")
            return False
        finally:
            if conn:
                conn.close()'''
    def _save_analysis_result(self, result: AnalysisResult, test_id: str) -> bool:
        """保存分析结果到数据库"""
        print('尝试保存分析结果...')
        print(f'分析ID: {result.analysis_id}')
        print(f'用户ID: {result.user_id}')
        print(f'测试ID: {test_id}')
        
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # 准备数据
            bipolar_risk_json = json.dumps(result.bipolar_risk_indicators)
            positive_symptoms_json = json.dumps(result.positive_symptoms)
            symptom_categories_json = json.dumps(result.symptom_categories)
            symptom_severity_json = json.dumps(result.symptom_severity_scores)
            recovery_json = json.dumps(result.recovery_indicators)
            risk_factors_json = json.dumps(result.risk_factors)
            treatment_json = json.dumps(result.treatment_response_indicators)
            ai_json = json.dumps(result.ai_analysis_data)
            recommendations_json = json.dumps(result.clinical_recommendations)
            
            print('数据准备完成，准备执行SQL...')
            
            cursor.execute('''
                INSERT INTO mdq_analysis_results (
                    analysis_id, user_id, test_id, analysis_date,
                    current_score, raw_score, severity_level, risk_percentage, bipolar_risk_indicators,
                    positive_symptoms, symptom_categories, symptom_severity_scores, functional_impairment_level,
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
                result.current_score,
                result.raw_score,
                result.severity_level.value,
                result.risk_percentage,
                bipolar_risk_json,
                positive_symptoms_json,
                symptom_categories_json,
                symptom_severity_json,
                result.functional_impairment_level,
                result.improvement_trend.value,
                result.trend_confidence,
                result.historical_baseline,
                result.improvement_percentage,
                result.consistency_score,
                recovery_json,
                risk_factors_json,
                treatment_json,
                ai_json,
                recommendations_json,
                result.monitoring_frequency,
                result.emergency_flag,
                result.next_assessment_date.isoformat()
            ))
            
            conn.commit()
            print(f"✅ 分析结果已保存: {result.analysis_id}")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ 保存分析结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        except json.JSONDecodeError as e:
            print(f"❌ JSON转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            import traceback
            traceback.print_exc()
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
                SELECT analysis_id, analysis_date, current_score, raw_score, severity_level,
                       risk_percentage, improvement_trend, trend_confidence,
                       emergency_flag, monitoring_frequency
                FROM mdq_analysis_results
                WHERE user_id = ?
                ORDER BY analysis_date DESC
                LIMIT ?
            ''', (user_id, limit))
            
            results = cursor.fetchall()
            
            return [{
                'analysis_id': row[0],
                'analysis_date': row[1],
                'current_score': row[2],
                'raw_score': row[3],
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
                json_fields = ['bipolar_risk_indicators', 'positive_symptoms', 'symptom_categories',
                             'symptom_severity_scores', 'recovery_indicators', 'risk_factors', 
                             'treatment_response_indicators', 'ai_analysis_data', 'clinical_recommendations']
                
                for field in json_fields:
                    if analysis_data.get(field):
                        try:
                            analysis_data[field] = json.loads(analysis_data[field])
                        except json.JSONDecodeError:
                            analysis_data[field] = {}
                print(analysis_data)
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
            print(result)
            if result and result[0]:
                try:
                    print(json.loads(result[0]))
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
            current_score=0,
            raw_score=0,
            severity_level=SeverityLevel.NORMAL,
            risk_percentage=0.0,
            bipolar_risk_indicators={},
            positive_symptoms=[],
            symptom_categories={},
            symptom_severity_scores={},
            functional_impairment_level='minimal',
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

# 兼容性函数：处理between运算符
def between(value, min_val, max_val):
    """Python中没有between运算符，这里实现一个"""
    return min_val <= value <= max_val

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
            trend_distribution = {trend.value: 0 for trend in ImprovementTrend}
            emergency_users = []
            high_risk_users = []
            raw_score_distribution = {'0-8': 0, '9-15': 0, '16-25': 0, '26-39': 0}
            
            for user_id in user_ids:
                try:
                    result = self.analyzer.analyze_user_comprehensive(user_id)
                    
                    severity_distribution[result.severity_level.value] += 1
                    trend_distribution[result.improvement_trend.value] += 1
                    
                    # 原始分数分布统计
                    raw_score = result.raw_score
                    if raw_score <= 8:
                        raw_score_distribution['0-8'] += 1
                    elif raw_score <= 15:
                        raw_score_distribution['9-15'] += 1
                    elif raw_score <= 25:
                        raw_score_distribution['16-25'] += 1
                    else:
                        raw_score_distribution['26-39'] += 1
                    
                    if result.emergency_flag:
                        emergency_users.append({
                            'user_id': user_id,
                            'severity_level': result.severity_level.value,
                            'raw_score': result.raw_score,
                            'risk_percentage': result.risk_percentage,
                            'analysis_id': result.analysis_id
                        })
                    elif result.severity_level in [SeverityLevel.HIGH_RISK, SeverityLevel.SEVERE_RISK]:
                        high_risk_users.append({
                            'user_id': user_id,
                            'severity_level': result.severity_level.value,
                            'raw_score': result.raw_score,
                            'risk_percentage': result.risk_percentage,
                            'analysis_id': result.analysis_id
                        })
                        
                except Exception as e:
                    print(f"分析用户 {user_id} 失败: {e}")
                    continue
            
            return {
                'total_users': len(user_ids),
                'severity_distribution': severity_distribution,
                'trend_distribution': trend_distribution,
                'raw_score_distribution': raw_score_distribution,
                'emergency_users': emergency_users,
                'high_risk_users': high_risk_users,
                'analysis_timestamp': datetime.now().isoformat(),
                'statistics': {
                    'emergency_rate': len(emergency_users) / len(user_ids) * 100 if user_ids else 0,
                    'high_risk_rate': len(high_risk_users) / len(user_ids) * 100 if user_ids else 0
                }
            }
            
        except Exception as e:
            print(f"批量分析失败: {e}")
            return {}
        finally:
            if conn:
                conn.close()

# 数据迁移工具
class DataMigrationTool:
    """数据迁移工具：帮助迁移旧格式数据到新格式"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def migrate_old_test_data(self) -> Dict:
        """迁移旧的测试数据格式"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # 查找旧格式的测试数据
            cursor.execute('''
                SELECT test_id, user_id, test_data, test_timestamp
                FROM questionnaire_tests 
                WHERE questionnaire_type = 'MDQ'
            ''')
            
            tests = cursor.fetchall()
            migration_count = 0
            error_count = 0
            
            for test_id, user_id, test_data_str, timestamp in tests:
                try:
                    test_data = json.loads(test_data_str) if isinstance(test_data_str, str) else test_data_str
                    
                    # 检查是否需要迁移
                    if self._needs_migration(test_data):
                        migrated_data = self._migrate_test_data(test_data)
                        
                        # 更新数据库
                        cursor.execute('''
                            UPDATE questionnaire_tests 
                            SET test_data = ?
                            WHERE test_id = ?
                        ''', (json.dumps(migrated_data), test_id))
                        
                        migration_count += 1
                        print(f"迁移测试数据: {test_id}")
                        
                except Exception as e:
                    print(f"迁移测试 {test_id} 失败: {e}")
                    error_count += 1
            
            conn.commit()
            
            return {
                'total_tests': len(tests),
                'migrated_count': migration_count,
                'error_count': error_count,
                'success_rate': migration_count / len(tests) * 100 if tests else 0
            }
            
        except Exception as e:
            print(f"数据迁移失败: {e}")
            return {'error': str(e)}
        finally:
            if conn:
                conn.close()
    
    def _needs_migration(self, test_data: Dict) -> bool:
        """检查数据是否需要迁移"""
        questions = test_data.get('questions', {})
        
        # 检查是否包含旧格式的yes/no答案
        for answer in questions.values():
            if answer in ['yes', 'no'] and not any(answer in ['no', 'rarely', 'sometimes', 'often', 'always'] for answer in questions.values()):
                return True
        
        return False
    
    def _migrate_test_data(self, test_data: Dict) -> Dict:
        """迁移测试数据格式"""
        migrated_data = test_data.copy()
        questions = migrated_data.get('questions', {})
        
        # 将yes/no格式转换为5级评分
        # 这是一个简化的转换，实际应用中可能需要更复杂的逻辑
        for q_id, answer in questions.items():
            if answer == 'yes':
                # 将yes转换为sometimes（中等频率）
                questions[q_id] = 'sometimes'
            elif answer == 'no':
                # 保持no不变
                questions[q_id] = 'no'
        
        migrated_data['questions'] = questions
        migrated_data['migrated'] = True
        migrated_data['migration_timestamp'] = datetime.now().isoformat()
        
        return migrated_data

# 测试函数 - 使用现有数据版本（更新）
def test_analyzer_with_existing_data():
    """使用现有数据库数据测试分析器功能 - 更新版"""
    print("=== MDQ分析系统测试 (5级评分系统) ===")
    
    # 初始化
    db_manager = DatabaseManager()
    analyzer = MDQAnalyzer(db_manager)
    
    # 获取现有用户列表
    print("\n=== 查找现有用户 ===")
    try:
        conn = db_manager._get_connection()
        cursor = conn.cursor()
        
        # 查找有MDQ测试记录的用户
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
            print("❌ 数据库中没有找到MDQ测试记录")
            print("请先使用test.html创建一些测试数据")
            return
        
        print(f"找到 {len(users_with_tests)} 个有测试记录的用户:")
        for user_id, username, test_count in users_with_tests:
            print(f"  - 用户: {username} (ID: {user_id[:8]}...) - {test_count} 个测试")
        
        # 选择第一个用户进行分析
        selected_user = users_with_tests[0]
        user_id = selected_user[0]
        username = selected_user[1]
        test_count = selected_user[2]
        
        print(f"\n=== 分析用户: {username} ===")
        print(f"用户ID: {user_id}")
        print(f"测试记录数: {test_count}")
        
        # 查看用户的测试历史
        test_history = db_manager.get_user_mdq_history(user_id)
        print(f"获取到历史记录: {len(test_history)} 条")
        
        if test_history:
            print("最近的测试记录:")
            for i, test in enumerate(test_history[:3]):
                print(f"  {i+1}. 时间: {test['test_timestamp'][:19]} - 分数: {test.get('raw_score', 'N/A')}")
        
        # 执行综合分析
        print(f"\n=== 执行分析 ===")
        try:
            analysis_result = analyzer.analyze_user_comprehensive(user_id)
            
            print("✅ 分析完成！")
            print(f"分析ID: {analysis_result.analysis_id}")
            
            # 显示分析结果
            print(f"\n📊 当前状态评估:")
            print(f"  - 原始分数: {analysis_result.raw_score}/39")
            print(f"  - 标准化分数: {analysis_result.current_score}/13")
            print(f"  - 严重程度: {analysis_result.severity_level.value}")
            print(f"  - 风险百分比: {analysis_result.risk_percentage}%")
            print(f"  - 功能损害: {analysis_result.functional_impairment_level}")
            print(f"  - 紧急标志: {'🚨 是' if analysis_result.emergency_flag else '✅ 否'}")
            
            print(f"\n📈 历史趋势分析:")
            print(f"  - 改善趋势: {analysis_result.improvement_trend.value}")
            print(f"  - 趋势置信度: {analysis_result.trend_confidence:.2f}")
            print(f"  - 历史基线: {analysis_result.historical_baseline:.1f}")
            print(f"  - 改善百分比: {analysis_result.improvement_percentage:.1f}%")
            print(f"  - 一致性评分: {analysis_result.consistency_score:.2f}")
            
            print(f"\n🎯 症状分析:")
            if analysis_result.positive_symptoms:
                print("  阳性症状:")
                for symptom in analysis_result.positive_symptoms:
                    print(f"    • {symptom}")
            else:
                print("  无明显阳性症状")
            
            print(f"\n📊 症状严重程度评分:")
            for category, score in analysis_result.symptom_severity_scores.items():
                print(f"  - {category}: {score}%")
            
            print(f"\n📋 双相风险指标:")
            for indicator, value in analysis_result.bipolar_risk_indicators.items():
                status = "✓" if value else "✗"
                print(f"  {status} {indicator}: {value}")
            
            print(f"\n💡 恢复指标:")
            for indicator in analysis_result.recovery_indicators:
                print(f"  ✅ {indicator}")
            
            print(f"\n⚠️ 风险因素:")
            for factor in analysis_result.risk_factors:
                print(f"  ⚠️ {factor}")
            
            print(f"\n🏥 临床建议:")
            for i, recommendation in enumerate(analysis_result.clinical_recommendations, 1):
                print(f"  {i}. {recommendation}")
            
            print(f"\n📅 监测计划:")
            print(f"  - 监测频率: 每 {analysis_result.monitoring_frequency} 天")
            print(f"  - 下次评估: {analysis_result.next_assessment_date.strftime('%Y-%m-%d %H:%M')}")
            
            # 验证数据保存
            print(f"\n=== 验证数据保存 ===")
            
            # 检查分析结果是否保存到数据库
            print(analysis_result.analysis_id)
            saved_analysis = analyzer.get_analysis_detail(analysis_result.analysis_id)
            if saved_analysis:
                print("✅ 分析结果已成功保存到数据库")
                print(f"  - 保存时间: {saved_analysis['created_at']}")
                print(f"  - 原始分数: {saved_analysis['raw_score']}")
            else:
                print("❌ 分析结果保存失败")
            
            # 获取AI分析数据
            ai_data = analyzer.get_ai_analysis_data(analysis_result.analysis_id)
            if ai_data:
                print("✅ AI分析数据准备完成")
                print("  包含的数据模块:")
                for key in ai_data.keys():
                    print(f"    • {key}")
                
                # 显示症状模式分析
                if 'symptom_patterns' in ai_data:
                    patterns = ai_data['symptom_patterns']
                    print(f"\n🔍 症状模式分析:")
                    if 'core_symptoms' in patterns:
                        core = patterns['core_symptoms']
                        print(f"  - 核心症状总分: {core.get('total_score', 0)}")
                        print(f"  - 平均严重程度: {core.get('average_severity', 0):.1f}")
            else:
                print("❌ AI分析数据准备失败")
            
            print("\n🎉 5级评分系统分析测试成功完成！")
            print(f"✅ 支持0-39分评分范围")
            print(f"✅ 症状严重程度百分比评估")
            print(f"✅ 增强的临床建议系统")
            print(f"✅ 完整的AI分析数据准备")
            
        except Exception as e:
            print(f"❌ 分析过程出错: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 数据库查询出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if conn:
            conn.close()

# 数据迁移测试
def test_data_migration():
    """测试数据迁移功能"""
    print("=== 数据迁移测试 ===")
    
    db_manager = DatabaseManager()
    migration_tool = DataMigrationTool(db_manager)
    
    try:
        result = migration_tool.migrate_old_test_data()
        
        print(f"迁移结果:")
        print(f"  - 总测试数: {result.get('total_tests', 0)}")
        print(f"  - 迁移成功: {result.get('migrated_count', 0)}")
        print(f"  - 迁移失败: {result.get('error_count', 0)}")
        print(f"  - 成功率: {result.get('success_rate', 0):.1f}%")
        
    except Exception as e:
        print(f"迁移测试失败: {e}")

if __name__ == "__main__":
    # 默认使用现有数据测试
    test_analyzer_with_existing_data()
    
    # 可以取消注释下面的行来运行其他测试
    # test_data_migration()  # 测试数据迁移
    # test_all_existing_users()  # 测试所有用户
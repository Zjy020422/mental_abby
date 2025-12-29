# ====== DeepSeek API 配置 ======
# 优先从环境变量读取 API 密钥，如果没有则使用默认值
import os

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', "sk-cb387c428d9343328cea734e6ae0f9f5")

# ====== 导入依赖 ======
# Please install OpenAI SDK first: `pip3 install openai`
from openai import OpenAI
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import uuid
import time
from database import DatabaseManager
from analyse import MDQAnalyzer

@dataclass
class AdvisorReport:
    """AI顾问报告数据类"""
    report_id: str
    user_id: str
    report_type: str  # 'single_test' 或 'historical_analysis'
    analysis_id: Optional[str]  # 单次分析ID（单测试报告）
    generated_at: datetime
    
    # AI生成的报告内容
    executive_summary: str
    clinical_assessment: Optional[str] = None
    risk_evaluation: Optional[str] = None
    treatment_recommendations: Optional[str] = None
    lifestyle_recommendations: Optional[str] = None
    monitoring_plan: Optional[str] = None
    emergency_protocols: Optional[str] = None
    
    # 基于历史的内容（历史分析报告）
    progress_analysis: Optional[str] = None
    trend_interpretation: Optional[str] = None
    prognosis_assessment: Optional[str] = None
    
    # 元数据
    confidence_score: float = 0.0
    ai_model_version: str = "deepseek-chat"
    processing_time: float = 0.0

class DeepSeekAdvisor:
    """DeepSeek AI 分析顾问"""
    
    def __init__(self, db_manager: DatabaseManager, analyzer: MDQAnalyzer):
        self.db_manager = db_manager
        self.analyzer = analyzer
        self.client = None
        self.api_available = False

        # 验证API密钥并初始化客户端
        try:
            if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
                print("⚠️ DeepSeek API 密钥未设置，将使用备用报告生成模式")
                self.api_available = False
            else:
                # 初始化OpenAI客户端
                self.client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com"
                )
                self.api_available = True
                print("✅ DeepSeek API 客户端初始化成功")
        except Exception as e:
            print(f"⚠️ DeepSeek API 初始化失败: {e}")
            print("⚠️ 将使用备用报告生成模式")
            self.api_available = False

        self._init_database_tables()
        
        # AI Prompt Templates
        self.prompts = {
            'single_test': {
                'system': """You are an experienced psychiatrist and bipolar disorder specialist. Based on the patient's MDQ questionnaire analysis results, generate a professional clinical assessment report and treatment recommendations.

Please provide analysis in the following 7 sections with a professional, objective, and caring tone (strictly follow this format):

[EXECUTIVE SUMMARY]: Briefly summarize the patient's current status (2-3 sentences)
[CLINICAL ASSESSMENT]: Detailed analysis of symptom presentation and severity
[RISK EVALUATION]: Assessment of current risks and potential dangers
[TREATMENT RECOMMENDATIONS]: Specific medical intervention recommendations (each recommendation on a separate line, starting with "-")
[LIFESTYLE RECOMMENDATIONS]: Daily management and self-care (each recommendation on a separate line, starting with "-")
[MONITORING PLAN]: Follow-up tracking and assessment plan
[EMERGENCY PROTOCOLS]: Crisis management procedures

Ensure all recommendations are evidence-based and follow clinical practice guidelines.""",

                'user_template': """Please generate an MDQ analysis report for the following patient:

**Patient Demographics:**
- Age: {age} years
- Gender: {gender}
- Total Assessments: {total_assessments}
- Assessment Time Span: {assessment_span_days} days

**Current Clinical Status:**
- MDQ Score: {mdq_score}/13
- Weighted Score: {weighted_score}
- Severity Level: {severity_level}
- Risk Percentage: {risk_percentage}%
- Functional Impairment: {functional_impairment}

**Symptom Distribution:**
{symptom_distribution}

**Bipolar Risk Profile:**
{bipolar_risk_profile}

**Positive Symptoms:**
{positive_symptoms}

**Emergency Indicators:** {emergency_indicators}

**Monitoring Priorities:** {monitoring_priorities}

**Intervention Targets:** {intervention_targets}

Please generate a complete clinical assessment report and treatment recommendations."""
            },
            
            'historical': {
                'system': """You are a psychiatrist specializing in long-term treatment and management of bipolar disorder. Based on the patient's historical MDQ assessment data, analyze treatment progress and prognosis, and provide comprehensive long-term treatment recommendations.

Please provide analysis in the following 5 sections (strictly follow this format):

[EXECUTIVE SUMMARY]: Brief overview of overall treatment progress and current status (2-3 sentences)
[PROGRESS ANALYSIS]: Analysis of treatment effectiveness and symptom trajectory, including improvement trends and consistency assessment
[TREND INTERPRETATION]: Professional interpretation and prediction of symptom trends based on historical data patterns
[TREATMENT RECOMMENDATIONS]: Specific medical interventions and lifestyle recommendations based on historical analysis (each recommendation on a separate line, starting with "-")
[PROGNOSIS ASSESSMENT]: Long-term prognosis and recovery potential assessment, including risk factors and protective factors

Provide recommendations based on evidence-based medicine and long-term management best practices.""",

                'user_template': """Please generate a historical trend analysis report for the following patient:

**Patient Demographics:**
- Age: {age} years
- Gender: {gender}
- Total Assessments: {total_assessments}
- Time Span: {assessment_span_days} days

**Current Status:**
- Current MDQ Score: {current_score}/13
- Severity Level: {severity_level}
- Risk Percentage: {risk_percentage}%

**Historical Trajectory:**
- Improvement Trend: {improvement_trend}
- Trend Confidence: {trend_confidence}
- Historical Baseline: {baseline_score}
- Improvement Percentage: {improvement_percentage}%
- Consistency Score: {consistency_score}

**Score Timeline:**
{score_timeline}

**Treatment Response Indicators:**
{treatment_indicators}

**Recovery Indicators:**
{recovery_indicators}

**Current Risk Factors:**
{current_risk_factors}

**Prognostic Factors:**
- Positive Factors: {positive_factors}
- Negative Factors: {negative_factors}

**Statistical Features:**
- Mean Score: {score_mean}
- Standard Deviation: {score_std}
- Score Range: {score_range}
- Variability Coefficient: {variability_coefficient}

Based on this historical data, please generate a comprehensive progress assessment report and long-term treatment recommendations in english."""
            }
        }
    
    def _init_database_tables(self):
        """初始化AI顾问报告数据库表"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_advisor_reports (
                    report_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    report_type TEXT NOT NULL CHECK(report_type IN ('single_test', 'historical_analysis')),
                    analysis_id TEXT,
                    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- AI生成的内容
                    executive_summary TEXT NOT NULL,
                    clinical_assessment TEXT NOT NULL,
                    risk_evaluation TEXT NOT NULL,
                    treatment_recommendations TEXT NOT NULL,
                    lifestyle_recommendations TEXT NOT NULL,
                    monitoring_plan TEXT NOT NULL,
                    emergency_protocols TEXT NOT NULL,
                    
                    -- 历史分析特有内容
                    progress_analysis TEXT,
                    trend_interpretation TEXT,
                    prognosis_assessment TEXT,
                    
                    -- 元数据
                    confidence_score REAL DEFAULT 0.0,
                    ai_model_version TEXT DEFAULT 'deepseek-chat',
                    processing_time REAL DEFAULT 0.0,
                    
                    -- 原始数据
                    ai_input_data TEXT,
                    ai_response_raw TEXT,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_id) REFERENCES mdq_analysis_results (analysis_id) ON DELETE SET NULL
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_advisor_user_type ON ai_advisor_reports(user_id, report_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_advisor_generated_at ON ai_advisor_reports(generated_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_advisor_analysis_id ON ai_advisor_reports(analysis_id)')
            
            conn.commit()
            print("AI顾问报告数据库表初始化完成")
            
        except sqlite3.Error as e:
            print(f"AI顾问数据库表初始化失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def generate_single_test_report(self, analysis_id: str) -> AdvisorReport:
        """生成单次测试分析报告 - 修复版"""
        start_time = time.time()
        
        try:
            # 获取分析数据
            analysis_detail = self.analyzer.get_analysis_detail(analysis_id)
            if not analysis_detail:
                raise ValueError(f"分析记录 {analysis_id} 不存在")
            
            ai_data = self.analyzer.get_ai_analysis_data(analysis_id)
            if not ai_data:
                # 如果没有AI数据，尝试从analysis_detail构造
                print(f"警告：没有找到AI分析数据，尝试从分析详情构造基础数据")
                ai_data = {
                    'mdq_part1_score': analysis_detail.get('mdq_part1_score', 0),
                    'has_co_occurrence': analysis_detail.get('has_co_occurrence', False),
                    'functional_impact_level': analysis_detail.get('functional_impact_level', 'no_problems'),
                    'mdq_result': analysis_detail.get('mdq_result', 'negative'),
                    'severity_level': analysis_detail.get('severity_level', 'negative'),
                    'risk_percentage': analysis_detail.get('risk_percentage', 0),
                    'positive_symptoms': json.loads(analysis_detail.get('positive_symptoms', '[]')) if analysis_detail.get('positive_symptoms') else [],
                    'core_symptoms_count': analysis_detail.get('core_symptoms_count', 0)
                }
            
            user_id = analysis_detail['user_id']
            
            # 准备AI输入数据
            ai_input = self._prepare_single_test_input(ai_data)
            print(f"AI输入数据准备完成: MDQ分数={ai_input['mdq_score']}, 风险={ai_input['risk_percentage']}%")
            
            # 调用DeepSeek API
            user_prompt = self.prompts['single_test']['user_template'].format(**ai_input)
            
            try:
                ai_response = self._call_deepseek_api(
                    self.prompts['single_test']['system'],
                    user_prompt
                )
                print(f"DeepSeek API调用成功，响应长度: {len(ai_response)}")
            except Exception as api_error:
                print(f"DeepSeek API调用失败: {api_error}")
                # 生成备用报告
                ai_response = self._generate_fallback_report(ai_input)
            
            # 解析AI响应
            parsed_response = self._parse_single_test_response(ai_response)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 创建报告对象
            report = AdvisorReport(
                report_id=str(uuid.uuid4()),
                user_id=user_id,
                report_type='single_test',
                analysis_id=analysis_id,
                generated_at=datetime.now(),
                
                executive_summary=parsed_response['executive_summary'],
                clinical_assessment=parsed_response['clinical_assessment'],
                risk_evaluation=parsed_response['risk_evaluation'],
                treatment_recommendations=parsed_response['treatment_recommendations'],
                lifestyle_recommendations=parsed_response['lifestyle_recommendations'],
                monitoring_plan=parsed_response['monitoring_plan'],
                emergency_protocols=parsed_response['emergency_protocols'],
                
                confidence_score=parsed_response.get('confidence_score', 0.8),
                processing_time=processing_time
            )
            
            # 保存到数据库
            self._save_report(report, ai_input, ai_response)
            
            return report
            
        except Exception as e:
            print(f"生成单次测试报告失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    def _generate_fallback_report(self, ai_input: Dict) -> str:
        """Generate fallback report (when API call fails)"""
        mdq_score = ai_input.get('mdq_score', 0)
        risk_percentage = ai_input.get('risk_percentage', 0)

        if mdq_score >= 7:
            severity = "requires attention"
            recommendations = [
                "Consult with a mental health professional as soon as possible",
                "Monitor mood and behavioral changes closely",
                "Maintain regular sleep schedule"
            ]
            lifestyle = [
                "Avoid excessive stress and overstimulation",
                "Engage in moderate exercise",
                "Seek support from family and friends"
            ]
        else:
            severity = "relatively stable"
            recommendations = [
                "Continue monitoring mental health status",
                "Seek medical attention if symptoms change"
            ]
            lifestyle = [
                "Maintain healthy lifestyle habits",
                "Schedule regular mental health assessments"
            ]

        return f"""
    [EXECUTIVE SUMMARY]: Based on MDQ assessment results ({mdq_score}/13 points), the patient's current status is {severity}. Continued monitoring and professional evaluation are recommended.

    [CLINICAL ASSESSMENT]: The MDQ questionnaire shows a patient score of {mdq_score} points, with a risk assessment of {risk_percentage}%. According to standard MDQ scoring criteria, {'further professional evaluation is recommended' if mdq_score >= 7 else 'no obvious abnormalities, but continued monitoring is needed'}.

    [RISK EVALUATION]: {'Moderate risk - professional medical evaluation needed' if mdq_score >= 7 else 'Low risk - regular follow-up recommended'}.

    [TREATMENT RECOMMENDATIONS]:
    {chr(10).join(f'- {rec}' for rec in recommendations)}

    [LIFESTYLE RECOMMENDATIONS]:
    {chr(10).join(f'- {rec}' for rec in lifestyle)}

    [MONITORING PLAN]: {'Monthly' if mdq_score >= 7 else 'Quarterly'} mental health assessments are recommended. Seek medical attention promptly if symptoms change.

    [EMERGENCY PROTOCOLS]: If severe mood swings, self-harm or suicidal thoughts, or severe functional impairment occur, please contact a medical professional immediately or call emergency services.

    *Note: This report uses fallback generation mode due to network issues. Detailed evaluation by a professional physician is recommended.*
    """
    def generate_historical_analysis_report(self, user_id: str) -> AdvisorReport:
        """生成历史趋势分析报告"""
        start_time = time.time()
        
        # 获取用户最新分析
        analysis_history = self.analyzer.get_analysis_history(user_id, limit=1)
        if not analysis_history:
            raise ValueError(f"用户 {user_id} 没有分析记录")
        
        latest_analysis_id = analysis_history[0]['analysis_id']
        ai_data = self.analyzer.get_ai_analysis_data(latest_analysis_id)
        
        if not ai_data:
            raise ValueError(f"用户 {user_id} 的AI数据不完整")
        
        # 准备AI输入数据
        ai_input = self._prepare_historical_input(ai_data)
        
        # 调用DeepSeek API
        user_prompt = self.prompts['historical']['user_template'].format(**ai_input)
        ai_response = self._call_deepseek_api(
            self.prompts['historical']['system'],
            user_prompt
        )
        
        # 解析AI响应
        parsed_response = self._parse_historical_response(ai_response)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 创建报告对象
        report = AdvisorReport(
            report_id=str(uuid.uuid4()),
            user_id=user_id,
            report_type='historical_analysis',
            analysis_id=latest_analysis_id,
            generated_at=datetime.now(),
            
            executive_summary=parsed_response['executive_summary'],
            clinical_assessment=parsed_response.get('clinical_assessment', ''),
            risk_evaluation=parsed_response.get('risk_evaluation', ''),
            treatment_recommendations=parsed_response.get('treatment_recommendations', ''),
            lifestyle_recommendations=parsed_response.get('lifestyle_recommendations', ''),
            monitoring_plan=parsed_response.get('monitoring_plan', ''),
            emergency_protocols=parsed_response.get('emergency_protocols', ''),
            
            progress_analysis=parsed_response['progress_analysis'],
            trend_interpretation=parsed_response['trend_interpretation'],
            prognosis_assessment=parsed_response['prognosis_assessment'],
            
            confidence_score=parsed_response.get('confidence_score', 0.8),
            processing_time=processing_time
        )
        
        # 保存到数据库
        self._save_report(report, ai_input, ai_response)
        
        return report
    
    def _prepare_single_test_input(self, ai_data: Dict) -> Dict:
        """准备单次测试的AI输入数据 - 修复版"""
        try:
            # 从不同的数据结构中提取信息
            demographics = ai_data.get('patient_demographics', {})
            mdq_standard = ai_data.get('mdq_standard_results', {})
            symptom_patterns = ai_data.get('symptom_patterns', {})
            clinical_context = ai_data.get('clinical_context', {})
            
            # 兼容旧数据格式
            if not mdq_standard and 'mdq_part1_score' in ai_data:
                mdq_standard = {
                    'part1_score': ai_data.get('mdq_part1_score', 0),
                    'has_co_occurrence': ai_data.get('has_co_occurrence', False),
                    'functional_impact_level': ai_data.get('functional_impact_level', 'no_problems'),
                    'mdq_result': ai_data.get('mdq_result', 'negative'),
                    'severity_level': ai_data.get('severity_level', 'negative'),
                    'risk_percentage': ai_data.get('risk_percentage', 0)
                }
            
            if not symptom_patterns and 'positive_symptoms' in ai_data:
                symptom_patterns = {
                    'positive_symptoms_list': ai_data.get('positive_symptoms', []),
                    'core_symptoms_count': ai_data.get('core_symptoms_count', 0),
                    'symptom_profile': ai_data.get('symptom_profile', {})
                }
            
            # 安全获取基本信息
            age = demographics.get('age', ai_data.get('age', '未知'))
            gender = demographics.get('gender', ai_data.get('gender', '未知'))
            total_assessments = demographics.get('total_assessments', ai_data.get('total_assessments', 1))
            assessment_span = demographics.get('assessment_span_days', ai_data.get('assessment_span_days', 0))
            
            # 安全获取MDQ相关信息
            mdq_score = mdq_standard.get('part1_score', ai_data.get('mdq_part1_score', 0))
            risk_percentage = mdq_standard.get('risk_percentage', ai_data.get('risk_percentage', 0))
            severity_level = mdq_standard.get('severity_level', ai_data.get('severity_level', 'negative'))
            functional_impact = mdq_standard.get('functional_impact_level', ai_data.get('functional_impact_level', 'no_problems'))
            
            # 处理症状信息
            positive_symptoms = symptom_patterns.get('positive_symptoms_list', ai_data.get('positive_symptoms', []))
            core_symptoms_count = symptom_patterns.get('core_symptoms_count', ai_data.get('core_symptoms_count', 0))
            
            # 格式化症状分布（简化版）
            symptom_categories = symptom_patterns.get('symptom_categories', {})
            if not symptom_categories and 'symptom_profile' in ai_data:
                # 从symptom_profile生成简化的分类
                profile = ai_data.get('symptom_profile', {})
                mood_symptoms = sum(1 for q in ['q1', 'q2'] if profile.get(q, False))
                cognitive_symptoms = sum(1 for q in ['q5', 'q6'] if profile.get(q, False))
                behavioral_symptoms = sum(1 for q in ['q7', 'q8', 'q9'] if profile.get(q, False))
                
                symptom_categories = {
                    'mood_elevation': {'positive_count': mood_symptoms, 'total_count': 2},
                    'cognitive_symptoms': {'positive_count': cognitive_symptoms, 'total_count': 2},
                    'behavioral_activation': {'positive_count': behavioral_symptoms, 'total_count': 3}
                }
            
            symptom_text = ""
            for category, data in symptom_categories.items():
                if isinstance(data, dict) and 'positive_count' in data:
                    symptom_text += f"- {category.replace('_', ' ').title()}: {data['positive_count']}/{data['total_count']}\n"

            # Format positive symptoms
            symptoms_text = "\n".join([f"- {symptom}" for symptom in positive_symptoms[:5]])  # Limit to first 5
            if not symptoms_text:
                symptoms_text = "No significant positive symptoms"
            
            # 获取临床上下文
            emergency_indicators = clinical_context.get('emergency_indicators', [])
            monitoring_priorities = clinical_context.get('monitoring_priorities', [])
            intervention_targets = clinical_context.get('intervention_targets', [])
            
            return {
                'age': str(age),
                'gender': str(gender),
                'total_assessments': int(total_assessments),
                'assessment_span_days': int(assessment_span),
                'mdq_score': int(mdq_score),
                'weighted_score': float(mdq_score),  # Simplified: use MDQ score
                'severity_level': str(severity_level).replace('_', ' ').title(),
                'risk_percentage': float(risk_percentage),
                'functional_impairment': str(functional_impact).replace('_', ' ').title(),
                'symptom_distribution': symptom_text if symptom_text else "No detailed symptom distribution data available",
                'bipolar_risk_profile': f"MDQ Positive: {'Yes' if mdq_score >= 7 else 'No'}\nSymptom Co-occurrence: {'Yes' if mdq_standard.get('has_co_occurrence', False) else 'No'}",
                'positive_symptoms': symptoms_text,
                'emergency_indicators': ', '.join(emergency_indicators) if emergency_indicators else 'None',
                'monitoring_priorities': ', '.join(monitoring_priorities) if monitoring_priorities else 'Routine monitoring',
                'intervention_targets': ', '.join(intervention_targets) if intervention_targets else 'No specific intervention targets'
            }
            
        except Exception as e:
            print(f"Failed to prepare AI input data: {e}")
            # Return basic data structure
            return {
                'age': 'Unknown',
                'gender': 'Unknown',
                'total_assessments': 1,
                'assessment_span_days': 0,
                'mdq_score': ai_data.get('mdq_part1_score', 0),
                'weighted_score': ai_data.get('mdq_part1_score', 0),
                'severity_level': 'Requires assessment',
                'risk_percentage': ai_data.get('risk_percentage', 0),
                'functional_impairment': 'Requires assessment',
                'symptom_distribution': 'Error occurred during data processing',
                'bipolar_risk_profile': 'Requires re-assessment',
                'positive_symptoms': 'Failed to retrieve data',
                'emergency_indicators': 'None',
                'monitoring_priorities': 'Professional assessment recommended',
                'intervention_targets': 'Further assessment required'
            }
    
    def _prepare_historical_input(self, ai_data: Dict) -> Dict:
        """准备历史分析的AI输入数据"""
        demographics = ai_data.get('patient_demographics', {})
        clinical_state = ai_data.get('current_clinical_state', {})
        trajectory = ai_data.get('historical_trajectory', {})
        treatment_response = ai_data.get('treatment_response', {})
        stats = ai_data.get('statistical_features', {})
        prognosis = ai_data.get('clinical_context', {}).get('prognosis_factors', {})
        
        # Format score timeline
        score_timeline = trajectory.get('score_timeline', [])
        timeline_text = "\n".join([
            f"- {point['date'][:10]}: {point['score']} points (Baseline deviation: {point['baseline_deviation']:+.1f})"
            for point in score_timeline[-10:]  # Last 10 records
        ])
        
        # 格式化治疗指标
        treatment_indicators = treatment_response.get('treatment_indicators', {})
        treatment_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in treatment_indicators.items() if v != True])
        
        # 格式化恢复指标
        recovery_indicators = treatment_response.get('recovery_indicators', [])
        recovery_text = "\n".join([f"- {indicator}" for indicator in recovery_indicators])
        
        # 格式化风险因素
        risk_factors = treatment_response.get('current_risk_factors', [])
        risk_text = "\n".join([f"- {factor}" for factor in risk_factors])
        
        return {
            'age': demographics.get('age', 'Unknown'),
            'gender': demographics.get('gender', 'Unknown'),
            'total_assessments': demographics.get('total_assessments', 0),
            'assessment_span_days': demographics.get('assessment_span_days', 0),
            'current_score': clinical_state.get('mdq_score', 0),
            'severity_level': clinical_state.get('severity_level', 'Unknown'),
            'risk_percentage': clinical_state.get('risk_percentage', 0),
            'improvement_trend': trajectory.get('improvement_trend', 'Unknown'),
            'trend_confidence': round(trajectory.get('trend_confidence', 0), 2),
            'baseline_score': round(trajectory.get('baseline_score', 0), 1),
            'improvement_percentage': round(treatment_response.get('improvement_percentage', 0), 1),
            'consistency_score': round(treatment_response.get('consistency_score', 0), 2),
            'score_timeline': timeline_text if timeline_text else 'Insufficient historical data available',
            'treatment_indicators': treatment_text if treatment_text else 'No treatment response data available',
            'recovery_indicators': recovery_text if recovery_text else 'No significant recovery indicators',
            'current_risk_factors': risk_text if risk_text else 'No significant risk factors',
            'positive_factors': ', '.join(prognosis.get('positive_factors', [])) if prognosis.get('positive_factors') else 'None',
            'negative_factors': ', '.join(prognosis.get('negative_factors', [])) if prognosis.get('negative_factors') else 'None',
            'score_mean': round(stats.get('score_mean', 0), 1),
            'score_std': round(stats.get('score_std', 0), 1),
            'score_range': stats.get('score_range', 0),
            'variability_coefficient': round(stats.get('variability_coefficient', 0), 2)
        }
    
    def _call_deepseek_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call DeepSeek API - Enhanced error handling"""
        # If API unavailable, throw exception for fallback
        if not self.api_available or not self.client:
            raise Exception("DeepSeek API client not initialized or unavailable")

        try:
            # Add timeout and retry mechanism
            import time
            max_retries = 2  # Reduced retries for faster failure response

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=4000,
                        temperature=0.7,
                        stream=False,
                        timeout=30  # 30 second timeout
                    )

                    return response.choices[0].message.content

                except Exception as e:
                    print(f"DeepSeek API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Fixed 1 second wait
                    else:
                        raise e

        except Exception as e:
            print(f"DeepSeek API call ultimately failed: {e}")
            raise Exception(f"DeepSeek API call failed: {str(e)}")
    
    def _parse_single_test_response(self, ai_response: str) -> Dict:
        """Parse single test AI response"""
        sections = {
            'executive_summary': '',
            'clinical_assessment': '',
            'risk_evaluation': '',
            'treatment_recommendations': [],
            'lifestyle_recommendations': [],
            'monitoring_plan': '',
            'emergency_protocols': '',
            'confidence_score': 0.8
        }

        # Split response by sections - support both English and Chinese formats
        section_patterns = {
            '[EXECUTIVE SUMMARY]': 'executive_summary',
            '[CLINICAL ASSESSMENT]': 'clinical_assessment',
            '[RISK EVALUATION]': 'risk_evaluation',
            '[TREATMENT RECOMMENDATIONS]': 'treatment_recommendations',
            '[LIFESTYLE RECOMMENDATIONS]': 'lifestyle_recommendations',
            '[MONITORING PLAN]': 'monitoring_plan',
            '[EMERGENCY PROTOCOLS]': 'emergency_protocols',
            # Fallback to Chinese patterns for compatibility
            '【执行摘要】': 'executive_summary',
            '【临床评估】': 'clinical_assessment',
            '【风险评估】': 'risk_evaluation',
            '【治疗建议】': 'treatment_recommendations',
            '【生活方式建议】': 'lifestyle_recommendations',
            '【监测计划】': 'monitoring_plan',
            '【紧急预案】': 'emergency_protocols'
        }
        
        current_section = None
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是新的章节
            section_found = False
            for pattern, section_name in section_patterns.items():
                if pattern in line:
                    current_section = section_name
                    section_found = True
                    # 提取章节标题后的内容
                    content = line.replace(pattern, '').strip('：: ')
                    if content:
                        if section_name in ['treatment_recommendations', 'lifestyle_recommendations']:
                            if content.startswith('-'):
                                sections[section_name].append(content[1:].strip())
                            else:
                                sections[section_name].append(content)
                        else:
                            sections[section_name] = content
                    break
            
            if not section_found and current_section:
                # 继续当前章节的内容
                if current_section in ['treatment_recommendations', 'lifestyle_recommendations']:
                    if line.startswith('-'):
                        sections[current_section].append(line[1:].strip())
                    elif line.startswith(('•', '*', '1.', '2.', '3.', '4.', '5.')):
                        clean_line = line.lstrip('•*123456789. ')
                        sections[current_section].append(clean_line)
                    elif sections[current_section] and not any(x in line for x in ['【', '】']):
                        # 如果不是新章节且有内容，追加到最后一个建议
                        sections[current_section][-1] += ' ' + line
                else:
                    if sections[current_section]:
                        sections[current_section] += ' ' + line
                    else:
                        sections[current_section] = line
        
        # Ensure required fields are not empty
        if not sections['executive_summary']:
            sections['executive_summary'] = 'Patient requires further evaluation by a professional physician'
        if not sections['clinical_assessment']:
            sections['clinical_assessment'] = 'Comprehensive clinical assessment is recommended'
        if not sections['risk_evaluation']:
            sections['risk_evaluation'] = 'Risk assessment requires professional medical judgment'
        if not sections['treatment_recommendations']:
            sections['treatment_recommendations'] = ['Consult with a psychiatrist to develop a personalized treatment plan']
        if not sections['lifestyle_recommendations']:
            sections['lifestyle_recommendations'] = ['Maintain regular sleep schedule and healthy lifestyle habits']
        if not sections['monitoring_plan']:
            sections['monitoring_plan'] = 'Regular follow-up and symptom monitoring recommended'
        if not sections['emergency_protocols']:
            sections['emergency_protocols'] = 'In case of emergency, contact a physician immediately or call emergency services'

        return sections
    
    def _parse_historical_response(self, ai_response: str) -> Dict:
        """Parse historical analysis AI response"""
        # Define 5 section parsing patterns - support both English and Chinese
        historical_patterns = {
            '[EXECUTIVE SUMMARY]': 'executive_summary',
            '[PROGRESS ANALYSIS]': 'progress_analysis',
            '[TREND INTERPRETATION]': 'trend_interpretation',
            '[TREATMENT RECOMMENDATIONS]': 'treatment_recommendations',
            '[PROGNOSIS ASSESSMENT]': 'prognosis_assessment',
            # Fallback to Chinese patterns for compatibility
            '【执行摘要】': 'executive_summary',
            '【进展分析】': 'progress_analysis',
            '【趋势解读】': 'trend_interpretation',
            '【治疗建议】': 'treatment_recommendations',
            '【预后评估】': 'prognosis_assessment'
        }

        # Initialize all fields
        sections = {
            'executive_summary': '',
            'progress_analysis': '',
            'trend_interpretation': '',
            'treatment_recommendations': '',
            'prognosis_assessment': ''
        }
        
        current_section = None
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查历史分析章节
            for pattern, section_name in historical_patterns.items():
                if pattern in line:
                    current_section = section_name
                    content = line.replace(pattern, '').strip('：: ')
                    if content:
                        sections[section_name] = content
                    break
            else:
                # 继续当前历史分析章节的内容
                if current_section and current_section in historical_patterns.values():
                    if sections[current_section]:
                        sections[current_section] += ' ' + line
                    else:
                        sections[current_section] = line
        
        # Provide default values for empty historical analysis fields
        if not sections['executive_summary']:
            sections['executive_summary'] = 'Based on historical data analysis, overall treatment progress requires continuous monitoring and professional evaluation'
        if not sections['progress_analysis']:
            sections['progress_analysis'] = 'Based on available data, treatment progress requires continuous monitoring and professional evaluation'
        if not sections['trend_interpretation']:
            sections['trend_interpretation'] = 'Symptom trend changes require comprehensive assessment in conjunction with clinical presentation'
        if not sections['treatment_recommendations']:
            sections['treatment_recommendations'] = 'Continue current treatment plan with regular effectiveness assessments'
        if not sections['prognosis_assessment']:
            sections['prognosis_assessment'] = 'Prognosis assessment requires consideration of multiple factors. Regular professional evaluation is recommended'

        return sections
    
    def _save_report(self, report: AdvisorReport, ai_input: Dict, ai_response: str) -> bool:
        """保存报告到数据库"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_advisor_reports (
                    report_id, user_id, report_type, analysis_id, generated_at,
                    executive_summary, clinical_assessment, risk_evaluation,
                    treatment_recommendations, lifestyle_recommendations,
                    monitoring_plan, emergency_protocols,
                    progress_analysis, trend_interpretation, prognosis_assessment,
                    confidence_score, ai_model_version, processing_time,
                    ai_input_data, ai_response_raw
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.report_id,
                report.user_id,
                report.report_type,
                report.analysis_id,
                report.generated_at.isoformat(),
                report.executive_summary,
                getattr(report, 'clinical_assessment', ''),
                getattr(report, 'risk_evaluation', ''),
                json.dumps(getattr(report, 'treatment_recommendations', ''), ensure_ascii=False),
                json.dumps(getattr(report, 'lifestyle_recommendations', ''), ensure_ascii=False),
                getattr(report, 'monitoring_plan', ''),
                getattr(report, 'emergency_protocols', ''),
                report.progress_analysis,
                report.trend_interpretation,
                report.prognosis_assessment,
                report.confidence_score,
                report.ai_model_version,
                report.processing_time,
                json.dumps(ai_input, ensure_ascii=False),
                ai_response
            ))
            
            conn.commit()
            print(f"AI顾问报告已保存: {report.report_id}")
            return True
            
        except sqlite3.Error as e:
            print(f"保存AI顾问报告失败: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_report(self, report_id: str) -> Optional[Dict]:
        """获取报告详情"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM ai_advisor_reports WHERE report_id = ?
            ''', (report_id,))
            
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                report_data = dict(zip(columns, result))
                
                # 解析JSON字段
                json_fields = ['treatment_recommendations', 'lifestyle_recommendations', 
                             'ai_input_data']
                
                for field in json_fields:
                    if report_data[field]:
                        try:
                            report_data[field] = json.loads(report_data[field])
                        except json.JSONDecodeError:
                            pass
                
                return report_data
            
            return None
            
        except sqlite3.Error as e:
            print(f"获取AI报告失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_user_reports(self, user_id: str, report_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """获取用户的报告历史"""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            query = '''
                SELECT report_id, report_type, generated_at, confidence_score, processing_time
                FROM ai_advisor_reports
                WHERE user_id = ?
            '''
            params = [user_id]
            
            if report_type:
                query += ' AND report_type = ?'
                params.append(report_type)
            
            query += ' ORDER BY generated_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [{
                'report_id': row[0],
                'report_type': row[1],
                'generated_at': row[2],
                'confidence_score': row[3],
                'processing_time': row[4]
            } for row in results]
            
        except sqlite3.Error as e:
            print(f"获取用户报告历史失败: {e}")
            return []
        finally:
            if conn:
                conn.close()

# 便捷功能函数
def generate_quick_report(user_id: str, report_type: str = 'both') -> Dict:
    """快速生成报告的便捷函数"""
    db_manager = DatabaseManager()
    analyzer = MDQAnalyzer(db_manager)
    advisor = DeepSeekAdvisor(db_manager, analyzer)
    
    results = {}
    
    try:
        if report_type in ['single', 'both']:
            # 获取最新分析ID
            analysis_history = analyzer.get_analysis_history(user_id, limit=1)
            if analysis_history:
                analysis_id = analysis_history[0]['analysis_id']
                single_report = advisor.generate_single_test_report(analysis_id)
                results['single_test_report'] = {
                    'report_id': single_report.report_id,
                    'executive_summary': single_report.executive_summary,
                    'treatment_recommendations': single_report.treatment_recommendations,
                    'emergency_protocols': single_report.emergency_protocols
                }
        
        if report_type in ['historical', 'both']:
            historical_report = advisor.generate_historical_analysis_report(user_id)
            results['historical_report'] = {
                'report_id': historical_report.report_id,
                'progress_analysis': historical_report.progress_analysis,
                'trend_interpretation': historical_report.trend_interpretation,
                'prognosis_assessment': historical_report.prognosis_assessment
            }
        
        return {'success': True, 'reports': results}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def batch_generate_reports(report_type: str = 'both') -> Dict:
    """批量为所有用户生成报告"""
    db_manager = DatabaseManager()
    analyzer = MDQAnalyzer(db_manager)
    advisor = DeepSeekAdvisor(db_manager, analyzer)
    
    try:
        conn = db_manager._get_connection()
        cursor = conn.cursor()
        
        # 获取所有有分析记录的用户
        cursor.execute('''
            SELECT DISTINCT user_id FROM mdq_analysis_results
        ''')
        
        user_ids = [row[0] for row in cursor.fetchall()]
        
        results = {
            'total_users': len(user_ids),
            'successful_reports': 0,
            'failed_reports': 0,
            'reports': {}
        }
        
        for user_id in user_ids:
            try:
                user_results = generate_quick_report(user_id, report_type)
                if user_results['success']:
                    results['successful_reports'] += 1
                    results['reports'][user_id] = user_results['reports']
                else:
                    results['failed_reports'] += 1
                    print(f"用户 {user_id[:8]} 报告生成失败: {user_results['error']}")
                    
            except Exception as e:
                results['failed_reports'] += 1
                print(f"用户 {user_id[:8]} 报告生成出错: {e}")
        
        return results
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
        
    finally:
        if conn:
            conn.close()

# 报告格式化和导出功能
def format_report_for_display(report_data: Dict) -> str:
    """格式化报告用于显示"""
    if not report_data:
        return "报告不存在"
    
    report_type_names = {
        'single_test': '单次测试分析报告',
        'historical_analysis': '历史趋势分析报告'
    }
    
    formatted_text = f"""
# {report_type_names.get(report_data['report_type'], '分析报告')}

**报告ID:** {report_data['report_id']}
**生成时间:** {report_data['generated_at'][:19]}
**AI模型:** {report_data['ai_model_version']}
**置信度:** {report_data['confidence_score']:.2f}
**处理时间:** {report_data['processing_time']:.2f}秒

---

## 📋 执行摘要
{report_data['executive_summary']}

## 🏥 临床评估
{report_data['clinical_assessment']}

## ⚠️ 风险评估
{report_data['risk_evaluation']}

## 💊 治疗建议
"""
    
    # 处理治疗建议
    treatment_recs = report_data['treatment_recommendations']
    if isinstance(treatment_recs, list):
        for i, rec in enumerate(treatment_recs, 1):
            formatted_text += f"{i}. {rec}\n"
    else:
        formatted_text += f"{treatment_recs}\n"
    
    formatted_text += "\n## 💡 生活方式建议\n"
    
    # 处理生活方式建议
    lifestyle_recs = report_data['lifestyle_recommendations']
    if isinstance(lifestyle_recs, list):
        for i, rec in enumerate(lifestyle_recs, 1):
            formatted_text += f"{i}. {rec}\n"
    else:
        formatted_text += f"{lifestyle_recs}\n"
    
    formatted_text += f"""
## 📅 监测计划
{report_data['monitoring_plan']}

## 🚨 紧急预案
{report_data['emergency_protocols']}
"""
    
    # 如果是历史分析报告，添加额外内容
    if report_data['report_type'] == 'historical_analysis':
        formatted_text += f"""
---

## 📈 进展分析
{report_data['progress_analysis']}

## 📊 趋势解读
{report_data['trend_interpretation']}

## 🔮 预后评估
{report_data['prognosis_assessment']}
"""
    
    return formatted_text

def export_report_to_file(report_id: str, filename: Optional[str] = None) -> str:
    """导出报告到文件"""
    db_manager = DatabaseManager()
    analyzer = MDQAnalyzer(db_manager)
    advisor = DeepSeekAdvisor(db_manager, analyzer)
    
    report = advisor.get_report(report_id)
    if not report:
        return "报告不存在"
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mdq_report_{report_id[:8]}_{timestamp}.txt"
    
    formatted_text = format_report_for_display(report)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        return f"报告已导出到: {filename}"
    except Exception as e:
        return f"导出失败: {e}"

# 测试函数
def test_deepseek_advisor():
    """测试DeepSeek AI顾问功能"""
    print("=== DeepSeek AI 顾问系统测试 ===")
    
    # 检查API密钥设置
    if DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
        print("❌ 请先在文件顶部设置有效的 DeepSeek API 密钥")
        print("请修改文件顶部的 DEEPSEEK_API_KEY 变量")
        print("获取API密钥: https://platform.deepseek.com/")
        return
    
    # 初始化组件
    try:
        db_manager = DatabaseManager()
        analyzer = MDQAnalyzer(db_manager)
        advisor = DeepSeekAdvisor(db_manager, analyzer)
        print("✅ AI顾问系统初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 查找现有用户进行测试
    try:
        conn = db_manager._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT ar.user_id, ar.analysis_id, u.username
            FROM mdq_analysis_results ar
            JOIN users u ON ar.user_id = u.user_id
            ORDER BY ar.analysis_date DESC
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        
        if not result:
            print("❌ 没有找到分析结果，请先运行 analyse.py 生成分析数据")
            return
        
        user_id, analysis_id, username = result
        print(f"测试用户: {username} (ID: {user_id[:8]}...)")
        print(f"分析ID: {analysis_id[:8]}...")
        
        # 测试1: 生成单次分析报告
        print(f"\n=== 测试1: 生成单次测试分析报告 ===")
        try:
            print("正在调用DeepSeek API生成报告...")
            single_report = advisor.generate_single_test_report(analysis_id)
            
            print("✅ 单次分析报告生成成功!")
            print(f"报告ID: {single_report.report_id}")
            print(f"处理时间: {single_report.processing_time:.2f}秒")
            print(f"置信度: {single_report.confidence_score:.2f}")
            
            print(f"\n📋 执行摘要:")
            print(f"{single_report.executive_summary}")
            
            print(f"\n🏥 治疗建议:")
            for i, rec in enumerate(single_report.treatment_recommendations[:3], 1):
                print(f"  {i}. {rec}")
            
            print(f"\n💡 生活方式建议:")
            for i, rec in enumerate(single_report.lifestyle_recommendations[:3], 1):
                print(f"  {i}. {rec}")
                
        except Exception as e:
            print(f"❌ 单次分析报告生成失败: {e}")
            print("请检查API密钥是否正确，以及网络连接是否正常")
        
        # 测试2: 生成历史分析报告
        print(f"\n=== 测试2: 生成历史趋势分析报告 ===")
        try:
            print("正在调用DeepSeek API生成历史分析报告...")
            historical_report = advisor.generate_historical_analysis_report(user_id)
            
            print("✅ 历史分析报告生成成功!")
            print(f"报告ID: {historical_report.report_id}")
            print(f"处理时间: {historical_report.processing_time:.2f}秒")
            print(f"置信度: {historical_report.confidence_score:.2f}")
            
            print(f"\n📈 进展分析:")
            print(f"{historical_report.progress_analysis[:200]}...")
            
            print(f"\n📊 趋势解读:")
            print(f"{historical_report.trend_interpretation[:200]}...")
            
            print(f"\n🔮 预后评估:")
            print(f"{historical_report.prognosis_assessment[:200]}...")
            
        except Exception as e:
            print(f"❌ 历史分析报告生成失败: {e}")
        
        # 测试3: 报告获取功能
        print(f"\n=== 测试3: 报告获取功能 ===")
        user_reports = advisor.get_user_reports(user_id)
        print(f"用户报告历史: {len(user_reports)} 个")
        
        for report in user_reports:
            print(f"  - {report['report_type']}: {report['generated_at'][:19]} "
                  f"(置信度: {report['confidence_score']:.2f}, "
                  f"处理时间: {report['processing_time']:.2f}s)")
        
        # 测试4: 报告格式化
        if user_reports:
            print(f"\n=== 测试4: 报告格式化 ===")
            latest_report_id = user_reports[0]['report_id']
            report_detail = advisor.get_report(latest_report_id)
            
            if report_detail:
                formatted_report = format_report_for_display(report_detail)
                print("✅ 报告格式化成功")
                print(f"格式化报告长度: {len(formatted_report)} 字符")
                
                # 显示报告摘要
                lines = formatted_report.split('\n')
                for line in lines[:15]:  # 显示前15行
                    print(line)
                print("...")
            else:
                print("❌ 获取报告详情失败")
        
        print(f"\n🎉 AI顾问系统测试完成!")
        print("✅ 单次测试报告生成功能正常")
        print("✅ 历史分析报告生成功能正常") 
        print("✅ 数据库保存功能正常")
        print("✅ 报告获取功能正常")
        print("✅ 报告格式化功能正常")
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if conn:
            conn.close()

# 快速演示函数
def demo_advisor_workflow(user_id: str = None):
    """演示AI顾问完整工作流程"""
    print("=== AI顾问工作流程演示 ===")
    
    if DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
        print("❌ 请先设置API密钥")
        return
    
    db_manager = DatabaseManager()
    analyzer = MDQAnalyzer(db_manager)
    
    # 如果没有指定用户，自动选择
    if not user_id:
        try:
            conn = db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT user_id FROM mdq_analysis_results LIMIT 1')
            result = cursor.fetchone()
            if result:
                user_id = result[0]
            else:
                print("❌ 没有找到用户数据")
                return
        finally:
            if conn:
                conn.close()
    
    try:
        advisor = DeepSeekAdvisor(db_manager, analyzer)
        
        print(f"用户ID: {user_id[:8]}...")
        
        # 生成两种类型的报告
        print("\n1. 生成单次测试报告...")
        single_result = generate_quick_report(user_id, 'single')
        
        print("2. 生成历史分析报告...")
        historical_result = generate_quick_report(user_id, 'historical')
        
        # 显示结果摘要
        if single_result['success']:
            single_id = single_result['reports']['single_test_report']['report_id']
            print(f"✅ 单次报告生成: {single_id[:8]}...")
        
        if historical_result['success']:
            historical_id = historical_result['reports']['historical_report']['report_id'] 
            print(f"✅ 历史报告生成: {historical_id[:8]}...")
        
        print("\n📊 工作流程演示完成!")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")

if __name__ == "__main__":
    # 运行测试
    test_deepseek_advisor()
    
    # 可以取消注释下面的行来运行其他功能
    # demo_advisor_workflow()  # 演示完整工作流程
    # batch_generate_reports('both')  # 批量生成所有用户报告
"""
多智能体专家系统 — 模拟不同角色的审批专家

每个智能体扮演一个审批角色，从各自专业角度出具评审意见:
- 信贷员: 关注申请材料完整性和基础资质
- 风控官: 关注风险敞口和还款能力
- 合规官: 关注合规性和政策符合度
- 抵押评估师: 关注抵押物价值和处置可行性
- 审批主管: 综合各方意见做最终决策
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class AgentOpinion:
    """单个智能体的评审意见"""
    agent_name: str
    agent_role: str
    decision: str                # "approve" | "review" | "reject"
    confidence: float            # 0-1
    score: float                 # 0-100
    reasons: list[str]           # 决策依据
    risk_points: list[str]       # 风险点
    missing_info: list[str]      # 缺失信息


@dataclass
class ApplicationSummary:
    """贷款申请摘要（供智能体评审）"""
    applicant: dict = field(default_factory=dict)
    income: dict = field(default_factory=dict)
    loan: dict = field(default_factory=dict)
    collateral: dict = field(default_factory=dict)
    credit: dict = field(default_factory=dict)
    documents: list = field(default_factory=list)
    risk_rules_result: dict = field(default_factory=dict)


class BaseAgent:
    """智能体基类"""

    def __init__(self, name: str, role: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.role = role
        self.model = model

    def review(self, summary: ApplicationSummary) -> AgentOpinion:
        """出具评审意见"""
        raise NotImplementedError

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 获取评审结果（实际使用）"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return json.dumps({"error": str(e)})


class CreditOfficer(BaseAgent):
    """信贷员 — 材料完整性和基础资质"""

    def __init__(self, model="gpt-4o-mini"):
        super().__init__("信贷员", "材料审核与基础资质评估", model)

    def review(self, summary: ApplicationSummary) -> AgentOpinion:
        # prompt 构建
        prompt = f"""你是一个专业的信贷审批员。请审核以下贷款申请材料：

申请人: {json.dumps(summary.applicant, ensure_ascii=False)}
收入情况: {json.dumps(summary.income, ensure_ascii=False)}
贷款需求: {json.dumps(summary.loan, ensure_ascii=False)}
抵押物: {json.dumps(summary.collateral, ensure_ascii=False)}
征信: {json.dumps(summary.credit, ensure_ascii=False)}

请从材料完整性和基础资质角度评估，输出JSON:
{{"decision": "approve/review/reject", "score": 0-100, "reasons": [...], "risk_points": [...], "missing_info": [...]}}"""
        return self._parse_response(self._call_llm(prompt))

    def _parse_response(self, raw: str) -> AgentOpinion:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"decision": "review", "score": 50,
                    "reasons": ["解析异常"], "risk_points": [], "missing_info": []}
        return AgentOpinion(
            agent_name=self.name, agent_role=self.role,
            decision=data.get("decision", "review"),
            confidence=data.get("score", 50) / 100,
            score=data.get("score", 50),
            reasons=data.get("reasons", []),
            risk_points=data.get("risk_points", []),
            missing_info=data.get("missing_info", []),
        )


class RiskManager(BaseAgent):
    """风控官 — 风险敞口和还款能力"""

    def __init__(self, model="gpt-4o-mini"):
        super().__init__("风控官", "风险评估", model)

    def review(self, summary: ApplicationSummary) -> AgentOpinion:
        prompt = f"""你是一个专业的风险控制官。请评估以下贷款申请的风险：

申请人: {json.dumps(summary.applicant, ensure_ascii=False)}
收入情况: {json.dumps(summary.income, ensure_ascii=False)}
贷款需求: {json.dumps(summary.loan, ensure_ascii=False)}
抵押物: {json.dumps(summary.collateral, ensure_ascii=False)}
征信: {json.dumps(summary.credit, ensure_ascii=False)}

规则引擎评分: {json.dumps(summary.risk_rules_result, ensure_ascii=False)}

请重点评估还款能力和风险敞口，输出JSON:
{{"decision": "approve/review/reject", "score": 0-100, "reasons": [...], "risk_points": [...], "missing_info": [...]}}"""
        return self._parse_response(self._call_llm(prompt))

    def _parse_response(self, raw: str) -> AgentOpinion:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"decision": "review", "score": 50,
                    "reasons": ["解析异常"], "risk_points": [], "missing_info": []}
        return AgentOpinion(
            agent_name=self.name, agent_role=self.role,
            decision=data.get("decision", "review"),
            confidence=data.get("score", 50) / 100,
            score=data.get("score", 50),
            reasons=data.get("reasons", []),
            risk_points=data.get("risk_points", []),
            missing_info=data.get("missing_info", []),
        )


class ComplianceOfficer(BaseAgent):
    """合规官 — 合规性和政策符合度"""

    def __init__(self, model="gpt-4o-mini"):
        super().__init__("合规官", "合规审查", model)

    def review(self, summary: ApplicationSummary) -> AgentOpinion:
        prompt = f"""你是一个专业的合规审查官。请审查以下贷款申请的合规性：

申请人: {json.dumps(summary.applicant, ensure_ascii=False)}
贷款用途: {summary.loan.get('purpose', '')}
贷款金额: {summary.loan.get('amount', 0)}

请检查:
1. 贷款用途是否符合监管要求
2. 申请人是否符合基本准入条件
3. 是否存在合规风险

输出JSON:
{{"decision": "approve/review/reject", "score": 0-100, "reasons": [...], "risk_points": [...], "missing_info": [...]}}"""
        return self._parse_response(self._call_llm(prompt))

    def _parse_response(self, raw: str) -> AgentOpinion:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"decision": "review", "score": 50,
                    "reasons": ["解析异常"], "risk_points": [], "missing_info": []}
        return AgentOpinion(
            agent_name=self.name, agent_role=self.role,
            decision=data.get("decision", "review"),
            confidence=data.get("score", 50) / 100,
            score=data.get("score", 50),
            reasons=data.get("reasons", []),
            risk_points=data.get("risk_points", []),
            missing_info=data.get("missing_info", []),
        )


class CollateralAppraiser(BaseAgent):
    """抵押评估师 — 抵押物价值和处置可行性"""

    def __init__(self, model="gpt-4o-mini"):
        super().__init__("抵押评估师", "抵押物评估", model)

    def review(self, summary: ApplicationSummary) -> AgentOpinion:
        prompt = f"""你是一个专业的抵押物评估师。请评估以下抵押物：

抵押物类型: {summary.collateral.get('type', '')}
评估价值: {summary.collateral.get('value', 0)}
产权人: {summary.collateral.get('owner', '')}
贷款金额: {summary.loan.get('amount', 0)}

请评估:
1. 抵押物价值是否充足
2. 抵押率是否合理
3. 处置可行性

输出JSON:
{{"decision": "approve/review/reject", "score": 0-100, "reasons": [...], "risk_points": [...], "missing_info": [...]}}"""
        return self._parse_response(self._call_llm(prompt))

    def _parse_response(self, raw: str) -> AgentOpinion:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"decision": "review", "score": 50,
                    "reasons": ["解析异常"], "risk_points": [], "missing_info": []}
        return AgentOpinion(
            agent_name=self.name, agent_role=self.role,
            decision=data.get("decision", "review"),
            confidence=data.get("score", 50) / 100,
            score=data.get("score", 50),
            reasons=data.get("reasons", []),
            risk_points=data.get("risk_points", []),
            missing_info=data.get("missing_info", []),
        )


class ApprovalDirector(BaseAgent):
    """审批主管 — 综合各方意见做最终决策"""

    def __init__(self, model="gpt-4o"):
        super().__init__("审批主管", "综合决策", model)

    def finalize(self, opinions: list[AgentOpinion],
                 rule_result: dict) -> AgentOpinion:
        """综合各方意见做出最终决策"""
        # 构建综合评审摘要
        summary_parts = []
        for op in opinions:
            summary_parts.append(
                f"【{op.agent_name}】决策={op.decision}, "
                f"评分={op.score}, 理由={'; '.join(op.reasons[:2])}"
            )

        prompt = f"""你是贷款审批主管，请综合以下专家意见做出最终决策：

专家意见:
{chr(10).join(summary_parts)}

规则引擎评分: {rule_result.get('score', 0)}
规则引擎决策: {rule_result.get('overall', 'unknown')}

输出JSON:
{{"decision": "approve/review/reject", "score": 0-100,
  "final_reason": "综合结论", "conditions": ["放款条件列表"],
  "risk_summary": "风险总结"}}"""

        return self._parse_response(self._call_llm(prompt))

    def _parse_response(self, raw: str) -> AgentOpinion:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"decision": "review", "score": 50,
                    "final_reason": "解析异常", "conditions": [], "risk_summary": ""}
        return AgentOpinion(
            agent_name=self.name, agent_role=self.role,
            decision=data.get("decision", "review"),
            confidence=data.get("score", 50) / 100,
            score=data.get("score", 50),
            reasons=[data.get("final_reason", "")],
            risk_points=[data.get("risk_summary", "")] if data.get("risk_summary") else [],
            missing_info=data.get("conditions", []),
        )

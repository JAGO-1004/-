"""
多智能体决策编排器 — 风险规则引擎 + 多智能体综合研判

流程:
  导入 Step 2 提取的字段 → 组装 ApplicationData
    │
    ├── 规则引擎硬性过滤 ──→ reject → 直接拒绝
    │       │
    │       通过
    │       │
    ├── 多智能体并行评审 ──→ 信贷员 / 风控官 / 合规官 / 抵押评估师
    │       │                (可配置使用 LLM API 或本地模型)
    │       │
    ├── 审批主管综合 ──────→ 加权融合各方意见
    │       │
    └── 最终决策输出 ──────→ {decision, score, conditions, report}
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_rules import RiskRuleEngine, ApplicationData
from agents import (
    CreditOfficer, RiskManager, ComplianceOfficer,
    CollateralAppraiser, ApprovalDirector,
    ApplicationSummary, AgentOpinion,
)


class DecisionOrchestrator:
    """多智能体决策编排器"""

    # 决策权重配置
    DECISION_WEIGHTS = {
        "approve": 1.0,
        "review": 0.5,
        "reject": 0.0,
    }

    def __init__(self, use_llm_agents: bool = False):
        """
        Args:
            use_llm_agents: 是否使用 LLM API 驱动智能体
                             False 时使用基于规则的模拟评审
        """
        self.rule_engine = RiskRuleEngine()
        self.use_llm = use_llm_agents

        # 初始化智能体池
        if self.use_llm:
            self.agents = [
                CreditOfficer(),
                RiskManager(),
                ComplianceOfficer(),
                CollateralAppraiser(),
            ]
        else:
            self.agents = []  # 使用规则模拟
        self.director = ApprovalDirector()

    def process(self, extracted_data: dict) -> dict:
        """
        执行完整审批决策

        Args:
            extracted_data: Step 2 提取的字段 + Step 1 分类结果
                            {
                                "doc_type": str,
                                "fields": {field: value},
                                "classification": {...},
                            }

        Returns:
            {
                "decision": "approve" | "review" | "reject",
                "score": float,
                "risk_engine": {...},
                "agent_opinions": [AgentOpinion, ...],
                "final_report": {...},
                "processing_time": float,
            }
        """
        start_time = time.time()
        fields = extracted_data.get("fields", {})

        print(f"\n{'=' * 60}")
        print(f"Step 4: 多智能体决策")
        print(f"  文档: {extracted_data.get('doc_type', 'unknown')}")
        print(f"{'=' * 60}")

        # Step 4-1: 组装申请数据
        print(f"\n[4-1] 组装申请数据...")
        app = self._build_application_data(fields)
        summary = self._build_summary(app, fields)

        # Step 4-2: 规则引擎硬性过滤
        print(f"[4-2] 规则引擎评估...")
        rule_result = self.rule_engine.evaluate(app)
        print(f"  规则评分: {rule_result['score']}/100")
        print(f"  规则决策: {rule_result['overall']}")
        if rule_result["reject_reasons"]:
            for r in rule_result["reject_reasons"]:
                print(f"    [拒绝] {r}")

        # 硬性拒绝直接终止
        if rule_result["overall"] == "reject":
            elapsed = time.time() - start_time
            result = {
                "decision": "reject",
                "score": rule_result["score"],
                "reason": "规则引擎拒绝",
                "risk_engine": rule_result,
                "agent_opinions": [],
                "final_report": {
                    "conclusion": "规则引擎硬性拒绝，未进入智能体评审",
                    "reject_reasons": rule_result["reject_reasons"],
                },
                "processing_time": round(elapsed, 2),
            }
            print(f"\n最终决策: 拒绝 ❌ (规则引擎)")
            print(f"处理时间: {result['processing_time']}s")
            return result

        # Step 4-3: 多智能体并行评审
        print(f"\n[4-3] 多智能体评审...")
        opinions = self._run_agents(app, summary, rule_result)
        for op in opinions:
            tag = {"approve": "✓", "review": "△", "reject": "✗"}.get(op.decision, "?")
            print(f"  [{tag}] {op.agent_name}: {op.decision} (评分: {op.score})")

        # Step 4-4: 审批主管综合决策
        print(f"\n[4-4] 审批主管综合决策...")
        if self.use_llm and opinions:
            final = self.director.finalize(opinions, rule_result)
        else:
            final = self._weighted_decision(opinions, rule_result)
        tag = {"approve": "✓", "review": "△", "reject": "✗"}.get(final.decision, "?")
        print(f"  [{tag}] 最终决策: {final.decision} (评分: {final.score})")
        if final.reasons:
            print(f"  结论: {final.reasons[0][:100]}")

        elapsed = time.time() - start_time
        return {
            "decision": final.decision,
            "score": final.score,
            "risk_engine": rule_result,
            "agent_opinions": [
                {"agent": op.agent_name, "role": op.agent_role,
                 "decision": op.decision, "score": op.score,
                 "reasons": op.reasons, "risk_points": op.risk_points}
                for op in opinions
            ],
            "final_report": {
                "conclusion": final.reasons[0] if final.reasons else "",
                "risk_summary": final.risk_points[0] if final.risk_points else "",
                "conditions": final.missing_info if final.decision == "approve" else [],
            },
            "processing_time": round(elapsed, 2),
        }

    def _build_application_data(self, fields: dict) -> ApplicationData:
        """从提取字段组装 ApplicationData"""
        return ApplicationData(
            applicant_name=fields.get("姓名") or fields.get("申请人姓名") or "",
            applicant_id=fields.get("身份证号") or fields.get("借款人身份证号") or "",
            applicant_age=self._extract_age(fields),
            monthly_income=self._safe_float(fields.get("月均收入")),
            loan_amount=self._safe_float(fields.get("贷款金额") or fields.get("申请金额")),
            loan_term_months=self._safe_int(fields.get("贷款期限")),
            loan_purpose=fields.get("贷款用途", ""),
            collateral_value=self._safe_float(fields.get("抵押物评估价值") or
                                               fields.get("合同总价")),
            collateral_owner=fields.get("抵押人") or fields.get("权利人", ""),
            credit_overdue_count=self._safe_int(fields.get("逾期笔数")),
            existing_loans=self._safe_int(fields.get("未结清贷款笔数")),
            existing_loan_balance=self._safe_float(
                fields.get("剩余本金") or fields.get("现有贷款余额")),
            social_security_months=self._safe_int(fields.get("累计缴纳月数")),
            doc_valid_until=fields.get("有效期至", ""),
        )

    def _build_summary(self, app: ApplicationData, fields: dict) -> ApplicationSummary:
        return ApplicationSummary(
            applicant={
                "name": app.applicant_name,
                "id": app.applicant_id,
                "age": app.applicant_age,
                "marriage": app.applicant_marriage,
            },
            income={
                "monthly": app.monthly_income,
                "employer": app.employer,
                "social_security_months": app.social_security_months,
            },
            loan={
                "amount": app.loan_amount,
                "term_months": app.loan_term_months,
                "purpose": app.loan_purpose,
            },
            collateral={
                "type": fields.get("抵押物类型", "住宅"),
                "value": app.collateral_value,
                "owner": app.collateral_owner,
            },
            credit={
                "overdue_count": app.credit_overdue_count,
                "existing_loans": app.existing_loans,
                "existing_balance": app.existing_loan_balance,
            },
            documents=[fields.get("doc_type", "")],
            risk_rules_result={},
        )

    def _run_agents(self, app: ApplicationData, summary: ApplicationSummary,
                    rule_result: dict) -> list[AgentOpinion]:
        """执行多智能体评审"""
        # 更新规则结果到 summary
        summary.risk_rules_result = rule_result

        if self.use_llm and self.agents:
            # LLM 驱动
            opinions = []
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_map = {
                    executor.submit(a.review, summary): a for a in self.agents
                }
                for future in as_completed(future_map):
                    try:
                        opinions.append(future.result(timeout=30))
                    except Exception as e:
                        agent = future_map[future]
                        opinions.append(AgentOpinion(
                            agent_name=agent.name, agent_role=agent.role,
                            decision="review", confidence=0.3, score=30,
                            reasons=[f"评审异常: {e}"],
                            risk_points=[], missing_info=[],
                        ))
            return opinions
        else:
            # 基于规则的模拟评审
            return self._rule_based_agents(app, rule_result)

    def _rule_based_agents(self, app: ApplicationData, rule_result: dict) -> list[AgentOpinion]:
        """基于规则引擎结果模拟多智能体评审"""
        cat_scores = rule_result.get("category_scores", {})
        score = rule_result["score"]

        return [
            AgentOpinion("信贷员", "材料审核",
                         "approve" if score >= 70 else "review",
                         score / 100, max(50, score),
                         [f"申请材料基本完整，综合评分{score}"],
                         [], []),
            AgentOpinion("风控官", "风险评估",
                         "approve" if cat_scores.get("还款能力", 0) >= 0.7 else "review",
                         cat_scores.get("还款能力", 0.5), 60,
                         [f"还款能力评分{cat_scores.get('还款能力', 0):.0%}"],
                         [], []),
            AgentOpinion("合规官", "合规审查",
                         "approve", 0.9, 85,
                         ["贷款用途合规，申请人资质符合要求"],
                         [], []),
            AgentOpinion("抵押评估师", "抵押物评估",
                         "approve" if cat_scores.get("抵押担保", 0) >= 0.6 else "review",
                         cat_scores.get("抵押担保", 0.5), 70,
                         [f"抵押物评估评分{cat_scores.get('抵押担保', 0):.0%}"],
                         [], []),
        ]

    def _weighted_decision(self, opinions: list[AgentOpinion],
                           rule_result: dict) -> AgentOpinion:
        """加权融合决策"""
        if not opinions:
            score = rule_result.get("score", 0)
            decision = "approve" if score >= 75 else "review" if score >= 50 else "reject"
            return AgentOpinion("审批主管", "综合决策", decision, score / 100, score,
                                [f"规则引擎评分{score}"], [], [])

        total_score = 0
        total_weight = 0

        for op in opinions:
            weight = {"approve": 1.0, "review": 0.6, "reject": 0.3}.get(op.decision, 0.5)
            total_score += op.score * weight
            total_weight += weight

        # 加上规则引擎的权重
        rule_weight = 1.5
        total_score += rule_result.get("score", 0) * rule_weight
        total_weight += rule_weight

        final_score = total_score / total_weight if total_weight else 0

        # 统计决策分布
        decisions = [op.decision for op in opinions]
        approves = decisions.count("approve")
        reviews = decisions.count("review")
        rejects = decisions.count("reject")

        if rejects >= 2:
            decision = "reject"
        elif approves >= 3 and final_score >= 75:
            decision = "approve"
        else:
            decision = "review"

        return AgentOpinion(
            "审批主管", "综合决策",
            decision, final_score / 100, round(final_score, 1),
            [f"综合{len(opinions)}位专家意见，最终评分{final_score:.1f}，建议{decision}"],
            [f"规则引擎评分{rule_result.get('score', 0)}"],
            [],
        )

    def _extract_age(self, fields: dict) -> int:
        """从身份证号或出生日期提取年龄"""
        id_num = fields.get("身份证号", "")
        if len(id_num) >= 14:
            try:
                birth_year = int(id_num[6:10])
                return 2026 - birth_year
            except ValueError:
                pass
        birth = fields.get("出生日期", "")
        if birth and len(birth) >= 4:
            try:
                birth_year = int(birth[:4])
                return 2026 - birth_year
            except ValueError:
                pass
        return 30  # 默认

    def _safe_float(self, val) -> float:
        try:
            return float(val) if val else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(self, val) -> int:
        try:
            return int(val) if val else 0
        except (ValueError, TypeError):
            return 0


if __name__ == "__main__":
    # 演示
    orchestrator = DecisionOrchestrator(use_llm_agents=False)

    test_data = {
        "doc_type": "贷款申请表",
        "fields": {
            "姓名": "张三",
            "身份证号": "110101199001010011",
            "出生日期": "1990-01-01",
            "月均收入": "30000",
            "贷款金额": "1000000",
            "贷款期限": "360",
            "贷款用途": "房屋装修",
            "抵押物评估价值": "2000000",
            "逾期笔数": "0",
            "未结清贷款笔数": "1",
            "累计缴纳月数": "36",
        },
    }

    result = orchestrator.process(test_data)

    print(f"\n{'=' * 60}")
    print(f"最终决策: {result['decision'].upper()}")
    print(f"综合评分: {result['score']}/100")
    print(f"处理时间: {result['processing_time']}s")

    report = result["final_report"]
    if report.get("conclusion"):
        print(f"结论: {report['conclusion']}")
    if report.get("conditions"):
        print(f"放款条件:")
        for c in report["conditions"]:
            print(f"  - {c}")

"""
审批报告生成器 — 格式化输出完整的审批视图
"""
import json
from datetime import datetime


class ApprovalReport:
    """审批报告 — 最终输出给审批师的可视化报告"""

    @staticmethod
    def generate(result: dict) -> str:
        """生成格式化审批报告"""
        lines = []
        lines.append("=" * 66)
        lines.append("  零售宅抵贷 — 智能审批报告")
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 66)

        # 决策摘要
        decision = result.get("decision", "unknown")
        score = result.get("score", 0)
        dt = {"approve": "✓ 通过", "review": "△ 需人工复核", "reject": "✗ 拒绝"}
        lines.append(f"\n  决策: {dt.get(decision, decision)}")
        lines.append(f"  评分: {score}/100")

        # 规则引擎
        risk = result.get("risk_engine", {})
        lines.append(f"\n  ┌─ 规则引擎评分")
        for cat, s in risk.get("category_scores", {}).items():
            bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
            lines.append(f"  │  {cat}: {bar} {s:.0%}")
        lines.append(f"  └─ 总分: {risk.get('score', 0)}")

        if risk.get("reject_reasons"):
            lines.append(f"\n  [!] 拒绝原因:")
            for r in risk["reject_reasons"]:
                lines.append(f"      • {r}")

        # 智能体意见
        lines.append(f"\n  ┌─ 多智能体评审")
        for op in result.get("agent_opinions", []):
            icons = {"approve": "✓", "review": "△", "reject": "✗"}
            lines.append(f"  │  {icons.get(op['decision'], '?')} {op['agent']} "
                        f"({op['role']}): {op['decision']} [{op['score']}分]")
            for r in op.get("reasons", [])[:1]:
                lines.append(f"  │    └ {r}")
        lines.append(f"  └─")

        # 最终结论
        report = result.get("final_report", {})
        lines.append(f"\n  ┌─ 审批结论")
        lines.append(f"  │  {report.get('conclusion', '')}")
        if report.get("risk_summary"):
            lines.append(f"  │  风险: {report['risk_summary']}")
        if report.get("conditions"):
            lines.append(f"  │  放款条件:")
            for c in report["conditions"]:
                lines.append(f"  │    • {c}")
        lines.append(f"  └─")

        lines.append(f"\n  处理耗时: {result.get('processing_time', 0)}s")
        lines.append("=" * 66)

        return "\n".join(lines)


if __name__ == "__main__":
    # 演示
    demo_result = {
        "decision": "approve",
        "score": 82.5,
        "risk_engine": {
            "score": 85,
            "category_scores": {
                "身份核验": 1.0, "还款能力": 0.8,
                "征信状况": 0.9, "抵押担保": 0.85, "合规审查": 1.0,
            },
            "reject_reasons": [],
        },
        "agent_opinions": [
            {"agent": "信贷员", "role": "材料审核", "decision": "approve",
             "score": 85, "reasons": ["申请材料完整，资质良好"]},
            {"agent": "风控官", "role": "风险评估", "decision": "approve",
             "score": 80, "reasons": ["还款能力充足，风险可控"]},
        ],
        "final_report": {
            "conclusion": "综合评审通过，建议批准该贷款申请",
            "risk_summary": "整体风险可控",
            "conditions": ["抵押登记办妥后放款", "确认首付款来源"],
        },
        "processing_time": 3.2,
    }
    print(ApprovalReport.generate(demo_result))

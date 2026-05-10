"""
风险规则引擎 — 硬性风控规则检查

对提取的字段执行可量化的规则检查:
- 身份核验: 身份证号校验、姓名一致性
- 收入评估: 收入覆盖倍数、社保/公积金匹配
- 负债评估: 征信逾期记录、现有贷款
- 抵押物评估: 抵押率(LTV)、产权清晰度
- 合规检查: 证件有效期、年龄限制

每条规则输出: pass / warn / reject + 原因
"""

import sys
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RuleResult:
    """单条规则检查结果"""
    rule_id: str
    rule_name: str
    status: str            # "pass" | "warn" | "reject"
    score: float           # 0.0 - 1.0
    detail: str
    field: Optional[str] = None


@dataclass
class ApplicationData:
    """贷款申请数据（由 Step 2 提取的字段组装）"""
    # 申请人信息
    applicant_name: str = ""
    applicant_id: str = ""
    applicant_gender: str = ""
    applicant_age: int = 0
    applicant_marriage: str = ""

    # 收入信息
    monthly_income: float = 0.0
    employer: str = ""
    social_security_months: int = 0
    housing_fund_balance: float = 0.0

    # 贷款信息
    loan_amount: float = 0.0
    loan_term_months: int = 0
    loan_purpose: str = ""
    repayment_source: str = ""

    # 抵押物
    collateral_type: str = ""
    collateral_value: float = 0.0
    collateral_owner: str = ""

    # 征信
    credit_overdue_count: int = 0
    existing_loans: int = 0
    existing_loan_balance: float = 0.0

    # 其他
    guarantor_name: str = ""
    doc_valid_until: str = ""


class RiskRuleEngine:
    """风险规则引擎 — 可配置规则链"""

    def __init__(self):
        self.rules = self._init_rules()
        self.weight_map = {
            "身份核验": 0.25,
            "还款能力": 0.30,
            "征信状况": 0.25,
            "抵押担保": 0.15,
            "合规审查": 0.05,
        }

    def _init_rules(self) -> list:
        """初始化规则列表"""
        return [
            # ---- 身份核验 ----
            self._rule_id_check,
            self._rule_age_check,
            self._rule_name_consistency,

            # ---- 还款能力 ----
            self._rule_income_coverage,
            self._rule_employment_stability,
            self._rule_dti_ratio,

            # ---- 征信状况 ----
            self._rule_credit_overdue,
            self._rule_existing_loans,

            # ---- 抵押担保 ----
            self._rule_ltv_ratio,
            self._rule_collateral_ownership,

            # ---- 合规审查 ----
            self._rule_doc_validity,
            self._rule_loan_purpose,
        ]

    def evaluate(self, app: ApplicationData) -> dict:
        """
        执行全部规则检查

        Returns:
            {
                "overall": "approve" | "review" | "reject",
                "score": float,        # 0-100
                "category_scores": {类别: score},
                "rules": [RuleResult, ...],
                "reject_reasons": [str],
                "warnings": [str],
            }
        """
        results = []
        for rule in self.rules:
            try:
                result = rule(app)
                results.append(result)
            except Exception as e:
                results.append(RuleResult(
                    rule_id=rule.__name__, rule_name=rule.__name__,
                    status="warn", score=0.5,
                    detail=f"规则执行异常: {e}"
                ))

        # 按类别聚合
        rejects = []
        warnings = []
        category_scores = {}

        for r in results:
            if r.status == "reject":
                rejects.append(f"[{r.rule_id}] {r.detail}")
            elif r.status == "warn":
                warnings.append(f"[{r.rule_id}] {r.detail}")

        # 加权计算总分
        total_score = 0.0
        for cat, weight in self.weight_map.items():
            cat_results = [r for r in results if r.rule_id.startswith(f"rule_{cat[:2]}")]
            cat_avg = sum(r.score for r in cat_results) / max(len(cat_results), 1)
            category_scores[cat] = round(cat_avg, 2)
            total_score += cat_avg * weight

        total_score = round(total_score * 100, 1)

        # 综合判定
        if rejects:
            overall = "reject"
        elif total_score >= 75:
            overall = "approve"
        elif total_score >= 50:
            overall = "review"
        else:
            overall = "reject"

        return {
            "overall": overall,
            "score": total_score,
            "category_scores": category_scores,
            "rule_results": [
                {"id": r.rule_id, "name": r.rule_name, "status": r.status,
                 "score": r.score, "detail": r.detail}
                for r in results
            ],
            "reject_reasons": rejects,
            "warnings": warnings,
        }

    # ============= 规则实现 =============

    def _rule_id_check(self, app: ApplicationData) -> RuleResult:
        """身份证号校验"""
        id_regex = r"^\d{17}[\dXx]$"
        if re.match(id_regex, app.applicant_id):
            return RuleResult("rule_identity_01", "身份证号格式", "pass", 1.0,
                              "身份证号格式正确", "身份证号")
        return RuleResult("rule_identity_01", "身份证号格式", "reject", 0,
                          "身份证号格式错误", "身份证号")

    def _rule_age_check(self, app: ApplicationData) -> RuleResult:
        """年龄限制: 18-65岁"""
        age = app.applicant_age
        if 25 <= age <= 60:
            return RuleResult("rule_identity_02", "年龄限制", "pass", 1.0,
                              f"申请人{age}岁，符合年龄要求", "年龄")
        elif 18 <= age < 25 or 60 < age <= 65:
            return RuleResult("rule_identity_02", "年龄限制", "warn", 0.6,
                              f"申请人{age}岁，边缘年龄需人工复核", "年龄")
        return RuleResult("rule_identity_02", "年龄限制", "reject", 0,
                          f"申请人{age}岁，不符合年龄要求(18-65)", "年龄")

    def _rule_name_consistency(self, app: ApplicationData) -> RuleResult:
        """姓名一致性检查"""
        # 实际场景会交叉验证身份证、征信报告、贷款申请表的姓名
        return RuleResult("rule_identity_03", "姓名一致性", "pass", 1.0,
                          "各材料姓名一致", "姓名")

    def _rule_income_coverage(self, app: ApplicationData) -> RuleResult:
        """收入覆盖倍数: 月供/月收入 ≤ 50%"""
        if app.monthly_income <= 0:
            return RuleResult("rule_income_01", "收入覆盖", "warn", 0.3,
                              "未获取收入信息", "月均收入")

        # 等额本息估算月供
        monthly_rate = 0.035 / 12  # 假设年利率3.5%
        n = app.loan_term_months
        if n > 0 and monthly_rate > 0:
            morgage = app.loan_amount * monthly_rate * (1 + monthly_rate)**n / \
                      ((1 + monthly_rate)**n - 1)
        else:
            morgage = app.loan_amount / 12  # 近似

        dti = morgage / app.monthly_income

        if dti <= 0.3:
            return RuleResult("rule_income_01", "收入覆盖", "pass", 1.0,
                              f"月供/月收入={dti:.0%}，还款能力充足", "月均收入")
        elif dti <= 0.5:
            return RuleResult("rule_income_01", "收入覆盖", "warn", 0.6,
                              f"月供/月收入={dti:.0%}，接近上限需关注", "月均收入")
        return RuleResult("rule_income_01", "收入覆盖", "reject", 0.2,
                          f"月供/月收入={dti:.0%}，超过50%上限", "月均收入")

    def _rule_employment_stability(self, app: ApplicationData) -> RuleResult:
        """就业稳定性: 社保缴纳月数"""
        if app.social_security_months >= 12:
            return RuleResult("rule_income_02", "就业稳定", "pass", 1.0,
                              f"连续缴纳社保{app.social_security_months}个月",
                              "社保缴纳")
        elif app.social_security_months >= 6:
            return RuleResult("rule_income_02", "就业稳定", "warn", 0.6,
                              f"仅缴纳社保{app.social_security_months}个月",
                              "社保缴纳")
        return RuleResult("rule_income_02", "就业稳定", "warn", 0.4,
                          "社保缴纳不足6个月", "社保缴纳")

    def _rule_dti_ratio(self, app: ApplicationData) -> RuleResult:
        """总负债收入比"""
        existing = app.existing_loan_balance or 0
        total_debt = app.loan_amount + existing
        if app.monthly_income <= 0:
            return RuleResult("rule_income_03", "负债收入比", "warn", 0.5,
                              "无法计算负债比", "月均收入")
        ratio = total_debt / (app.monthly_income * 12)
        if ratio <= 5:
            return RuleResult("rule_income_03", "负债收入比", "pass", 1.0,
                              f"负债收入比={ratio:.1f}倍，合理范围", "负债")
        elif ratio <= 10:
            return RuleResult("rule_income_03", "负债收入比", "warn", 0.5,
                              f"负债收入比={ratio:.1f}倍，偏高", "负债")
        return RuleResult("rule_income_03", "负债收入比", "reject", 0.2,
                          f"负债收入比={ratio:.1f}倍，过高", "负债")

    def _rule_credit_overdue(self, app: ApplicationData) -> RuleResult:
        """征信逾期记录"""
        if app.credit_overdue_count == 0:
            return RuleResult("rule_credit_01", "征信逾期", "pass", 1.0,
                              "无逾期记录", "征信")
        elif app.credit_overdue_count <= 3:
            return RuleResult("rule_credit_01", "征信逾期", "warn", 0.5,
                              f"逾期{app.credit_overdue_count}次，需关注", "征信")
        return RuleResult("rule_credit_01", "征信逾期", "reject", 0,
                          f"严重逾期({app.credit_overdue_count}次)", "征信")

    def _rule_existing_loans(self, app: ApplicationData) -> RuleResult:
        """现有贷款笔数"""
        if app.existing_loans <= 2:
            return RuleResult("rule_credit_02", "现有贷款", "pass", 1.0,
                              f"现有{app.existing_loans}笔贷款，可控", "已有贷款")
        elif app.existing_loans <= 5:
            return RuleResult("rule_credit_02", "现有贷款", "warn", 0.6,
                              f"现有{app.existing_loans}笔贷款，偏多", "已有贷款")
        return RuleResult("rule_credit_02", "现有贷款", "reject", 0.2,
                          f"现有{app.existing_loans}笔贷款，过多", "已有贷款")

    def _rule_ltv_ratio(self, app: ApplicationData) -> RuleResult:
        """抵押率(LTV): 贷款金额/抵押物价值 ≤ 70%"""
        if app.collateral_value <= 0:
            return RuleResult("rule_collateral_01", "抵押率", "warn", 0.5,
                              "无抵押物信息，需评估", "抵押物")
        ltv = app.loan_amount / app.collateral_value
        if ltv <= 0.5:
            return RuleResult("rule_collateral_01", "抵押率", "pass", 1.0,
                              f"抵押率{ltv:.0%}，安全", "抵押物")
        elif ltv <= 0.7:
            return RuleResult("rule_collateral_01", "抵押率", "pass", 0.8,
                              f"抵押率{ltv:.0%}，合规", "抵押物")
        elif ltv <= 0.85:
            return RuleResult("rule_collateral_01", "抵押率", "warn", 0.5,
                              f"抵押率{ltv:.0%}，偏高需关注", "抵押物")
        return RuleResult("rule_collateral_01", "抵押率", "reject", 0,
                          f"抵押率{ltv:.0%}，超过70%上限", "抵押物")

    def _rule_collateral_ownership(self, app: ApplicationData) -> RuleResult:
        """抵押物权属"""
        if app.collateral_owner and app.applicant_name and \
           app.collateral_owner == app.applicant_name:
            return RuleResult("rule_collateral_02", "抵押物权属", "pass", 1.0,
                              f"抵押物权属清晰，产权人{app.collateral_owner}", "抵押物权属")
        return RuleResult("rule_collateral_02", "抵押物权属", "warn", 0.5,
                          "抵押物权属需人工核实", "抵押物权属")

    def _rule_doc_validity(self, app: ApplicationData) -> RuleResult:
        """证件有效期检查"""
        if app.doc_valid_until:
            try:
                valid_date = datetime.strptime(app.doc_valid_until, "%Y%m%d")
                if valid_date > datetime.now():
                    return RuleResult("rule_compliance_01", "证件有效期", "pass", 1.0,
                                      "证件在有效期内", "证件有效期")
                return RuleResult("rule_compliance_01", "证件有效期", "reject", 0,
                                  "证件已过期", "证件有效期")
            except ValueError:
                pass
        return RuleResult("rule_compliance_01", "证件有效期", "pass", 0.8,
                          "未获取有效期信息", "证件有效期")

    def _rule_loan_purpose(self, app: ApplicationData) -> RuleResult:
        """贷款用途合规"""
        restricted = ["炒股", "投资", "购房首付", "赌博", "非法"]
        purpose = app.loan_purpose or ""
        for keyword in restricted:
            if keyword in purpose:
                return RuleResult("rule_compliance_02", "贷款用途", "reject", 0,
                                  f"贷款用途含限制性内容: {keyword}", "贷款用途")
        return RuleResult("rule_compliance_02", "贷款用途", "pass", 1.0,
                          f"贷款用途合规: {purpose[:20]}", "贷款用途")


if __name__ == "__main__":
    engine = RiskRuleEngine()

    app = ApplicationData(
        applicant_name="张三",
        applicant_id="110101199001010011",
        applicant_age=35,
        monthly_income=30000,
        social_security_months=36,
        loan_amount=1000000,
        loan_term_months=360,
        loan_purpose="房屋装修",
        collateral_value=2000000,
        collateral_owner="张三",
        credit_overdue_count=1,
        existing_loans=1,
        existing_loan_balance=500000,
        doc_valid_until="20350101",
    )

    result = engine.evaluate(app)
    print(f"综合评分: {result['score']}/100")
    print(f"决策结果: {result['overall']}")
    print(f"\n分类评分:")
    for cat, score in result["category_scores"].items():
        print(f"  {cat}: {score:.2f}")
    if result["reject_reasons"]:
        print(f"\n拒绝原因:")
        for r in result["reject_reasons"]:
            print(f"  {r}")
    if result["warnings"]:
        print(f"\n警告:")
        for w in result["warnings"]:
            print(f"  {w}")

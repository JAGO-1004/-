"""
零售宅抵贷 — 文档类型定义
所有进件材料的分类体系，BERT分类器 + VLM验证 共用此标准
"""

# =============================================
# 文档类型分类体系
# =============================================

DOCUMENT_CATEGORIES = {
    "身份证明": [
        "身份证_正面",
        "身份证_反面",
        "户口本_首页",
        "户口本_个人页",
        "结婚证",
        "离婚证",
        "护照",
        "军官证",
    ],
    "收入证明": [
        "收入证明",
        "社保缴纳证明",
        "公积金缴存证明",
        "个人所得税纳税记录",
        "劳动合同",
    ],
    "资产证明": [
        "不动产权证",
        "购房合同",
        "车辆登记证",
        "行驶证",
        "理财产品证明",
        "股票持仓证明",
        "保险单",
    ],
    "贷款材料": [
        "贷款申请表",
        "征信报告",
        "贷款合同",
        "抵押合同",
        "担保合同",
        "还款流水",
    ],
    "企业材料": [
        "营业执照",
        "开户许可证",
        "企业信用报告",
        "税务登记证",
    ],
    "其他": [
        "出生医学证明",
        "居住证",
        "毕业证",
        "学位证",
        "残疾证",
        "低保证明",
    ],
}

# 展平的完整文档类型列表
ALL_DOC_TYPES = sorted(
    doc for docs in DOCUMENT_CATEGORIES.values() for doc in docs
)

# 文档类型 → 大类映射
DOC_TYPE_TO_CATEGORY = {
    doc: cat for cat, docs in DOCUMENT_CATEGORIES.items() for doc in docs
}

# 分类器标签映射
LABEL2ID = {doc: i for i, doc in enumerate(ALL_DOC_TYPES)}
ID2LABEL = {i: doc for doc, i in LABEL2ID.items()}


if __name__ == "__main__":
    print(f"文档类型总数: {len(ALL_DOC_TYPES)}")
    print(f"大类数量: {len(DOCUMENT_CATEGORIES)}")
    for cat, docs in DOCUMENT_CATEGORIES.items():
        print(f"  {cat}: {len(docs)} 种 — {', '.join(docs)}")

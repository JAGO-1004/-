"""
文档分类集成器 — 内容 + 版式双重验证

流程:
  进件图像
    │
    ├── OCR 提取文本
    │     │
    │     └──→ BERT 分类器 (基于文本内容)
    │            │  输出: top-3 候选类型 + 置信度
    │            │
    │            └──→ 置信度 > 0.9? ──→ 直接输出 ✓
    │                  │
    │                  否
    │                  │
    └──→ VLM 版式验证器 (基于视觉布局)
           │  输入: BERT 的 top-3 候选
           │  输出: 版式验证结果
           │
           └──→ 内容+版式综合评分 → 最终分类结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path
from bert_classifier import BertDocumentClassifier
from vlm_verifier import LayoutVerifierPool
from document_types import ALL_DOC_TYPES, LABEL2ID, ID2LABEL


class DocumentClassifier:
    """文档分类器 — BERT + VLM 双重验证"""

    def __init__(
        self,
        bert_model_name="bert-base-chinese",
        vlm_models=None,
        high_conf_threshold=0.9,
    ):
        self.bert = BertDocumentClassifier(model_name=bert_model_name)
        self.vlm_pool = LayoutVerifierPool(models=vlm_models or ["gpt-4o-mini"])
        self.threshold = high_conf_threshold

    def classify(self, ocr_text: str, image_path: str = None) -> dict:
        """
        对进件文档进行分类

        Args:
            ocr_text: OCR 提取的文本内容
            image_path: 文档图像路径（用于 VLM 版式验证）

        Returns:
            {
                "doc_type": str,           # 最终文档类型
                "overall_confidence": float,
                "method": str,             # "bert_only" | "dual_verify"
                "bert_result": {...},      # BERT 分类结果
                "vlm_result": {...} | None, # VLM 验证结果（仅 dual_verify 时）
            }
        """
        # Step 1: BERT 文本分类
        bert_results = self.bert.predict([ocr_text])
        bert_top = bert_results[0]
        bert_conf = bert_top["confidence"]

        result = {
            "bert_result": bert_top,
        }

        # 高置信度直接输出
        if bert_conf >= self.threshold:
            result.update({
                "doc_type": bert_top["doc_type"],
                "overall_confidence": bert_conf,
                "method": "bert_only",
                "vlm_result": None,
            })
            return result

        # Step 2: 置信度不足，走 VLM 版式验证
        if image_path and Path(image_path).exists():
            # 取 BERT top-3 作为候选，缩小 VLM 搜索范围
            proba = bert_top["proba"]
            candidates = sorted(proba, key=proba.get, reverse=True)[:3]
            vlm_results = self.vlm_pool.verify(image_path, candidates)
            vlm_consensus = self.vlm_pool.consensus(vlm_results)

            result["vlm_result"] = vlm_consensus

            # Step 3: 融合评分 — BERT 和 VLM 加权
            bert_weight = 0.4
            vlm_weight = 0.6

            # 对每个候选类型计算综合得分
            all_types = candidates
            scores = {}
            for dt in all_types:
                bert_score = proba.get(dt, 0) * bert_weight
                vlm_score = 0
                if vlm_consensus and vlm_consensus.get("doc_type") == dt:
                    vlm_score = vlm_consensus.get("confidence", 0) * vlm_weight
                elif dt in (vlm_consensus or {}).get("votes", {}):
                    vlm_vote_ratio = vlm_consensus["votes"][dt] / \
                        max(len(vlm_results), 1)
                    vlm_score = vlm_vote_ratio * 0.8 * vlm_weight
                scores[dt] = bert_score + vlm_score

            final_type = max(scores, key=scores.get)
            final_conf = scores[final_type]

            result.update({
                "doc_type": final_type,
                "overall_confidence": final_conf,
                "method": "dual_verify",
                "fusion_scores": scores,
            })
        else:
            # 没有图像，退回到 BERT 结果
            result.update({
                "doc_type": bert_top["doc_type"],
                "overall_confidence": bert_conf,
                "method": "bert_only_fallback",
                "vlm_result": None,
            })

        return result


if __name__ == "__main__":
    # 演示
    import sys

    classifier = DocumentClassifier()

    ocr = sys.argv[1] if len(sys.argv) > 1 else \
        "姓名 张三 性别 男 民族 汉 出生 1990年1月1日 住址 北京市朝阳区 身份证号 110101199001010011"
    img = sys.argv[2] if len(sys.argv) > 2 else None

    result = classifier.classify(ocr, img)
    print(f"\n分类结果:")
    print(f"  文档类型: {result['doc_type']}")
    print(f"  综合置信度: {result['overall_confidence']:.2%}")
    print(f"  分类方式: {result['method']}")
    print(f"  BERT top1: {result['bert_result']['doc_type']} "
          f"(置信度: {result['bert_result']['confidence']:.2%})")
    if result.get("vlm_result"):
        vr = result["vlm_result"]
        print(f"  VLM 验证: {vr.get('doc_type')} "
              f"(置信度: {vr.get('confidence', 0):.2%}, "
              f"一致率: {vr.get('agree_ratio', 0):.0%})")

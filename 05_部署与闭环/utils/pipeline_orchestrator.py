"""
全流程编排器 — 串联 Step 1-5，一键完成审批

将五步架构整合为一个端到端管线:
  进件图像 → 分类 → 提取 → 决策 → 报告

同时管理:
- 人工标注闭环的数据流
- 每步的中间结果缓存
- 各步骤的成本和耗时统计
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 懒加载各步骤模块（避免启动时加载所有依赖）
_step_modules = {}

def _lazy_import(step: str):
    if step not in _step_modules:
        if step == "classify":
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01_文档分类"))
            from ensemble import DocumentClassifier
            _step_modules["classify"] = DocumentClassifier()
        elif step == "extract":
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "02_关键信息抽取"))
            from extractor import FieldExtractor, OpenAIVLM
            _step_modules["extract"] = FieldExtractor(models=[
                OpenAIVLM("gpt-4o-mini", "openai-gpt-4o-mini"),
            ])
        elif step == "review":
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "02_关键信息抽取"))
            from review_closed_loop import ReviewClosedLoop
            _step_modules["review"] = ReviewClosedLoop()
        elif step == "decision":
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "04_多智能体决策"))
            from orchestrator import DecisionOrchestrator
            _step_modules["decision"] = DecisionOrchestrator(use_llm_agents=False)
    return _step_modules[step]


class PipelineOrchestrator:
    """全流程编排器"""

    def __init__(self):
        self.records_dir = Path("./data/records")
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def step1_classify(self, image_path: str, ocr_text: str,
                       doc_type_hint: str = None) -> dict:
        """Step 1: 文档分类"""
        classifier = _lazy_import("classify")
        result = classifier.classify(ocr_text=ocr_text, image_path=image_path)

        return {
            "doc_type": result["doc_type"],
            "confidence": result["overall_confidence"],
            "method": result["method"],
            "bert_top1": {
                "doc_type": result["bert_result"]["doc_type"],
                "confidence": result["bert_result"]["confidence"],
            },
        }

    def step2_extract(self, image_path: str, doc_type: str,
                      auto_approve: bool = True) -> dict:
        """Step 2: 关键信息抽取"""
        extractor = _lazy_import("extract")
        review_loop = _lazy_import("review")

        result = extractor.extract(image_path=image_path, doc_type=doc_type)

        # 加入人工标注闭环
        record_id = review_loop.add_to_review_pool(result)

        # 高置信度自动通过
        if auto_approve and result["overall_confidence"] >= 0.8:
            review_loop.approve_fields(record_id, {}, reviewer="auto")

        return {
            "record_id": record_id,
            "doc_type": result["doc_type"],
            "fields": result["extracted_fields"],
            "confidence": result["overall_confidence"],
            "needs_review": result["needs_review"],
            "cost": result["cost"],
            "disagreements": [
                {"field": d["field"], "suggested": d["suggested"], "votes": d["votes"]}
                for d in result["disagreements"]
            ],
        }

    def step4_decision(self, doc_type: str, fields: dict) -> dict:
        """Step 4: 多智能体决策"""
        decision_engine = _lazy_import("decision")

        result = decision_engine.process({
            "doc_type": doc_type,
            "fields": fields,
        })

        return {
            "decision": result["decision"],
            "score": result["score"],
            "rule_engine": {
                "score": result["risk_engine"]["score"],
                "reject_reasons": result["risk_engine"]["reject_reasons"],
                "warnings": result["risk_engine"]["warnings"],
            },
            "agents": [
                {"name": op["agent"], "decision": op["decision"], "score": op["score"]}
                for op in result["agent_opinions"]
            ],
            "final_report": result["final_report"],
            "processing_time": result["processing_time"],
        }

    def run_full_pipeline(self, image_path: str, doc_type: str = None,
                          ocr_text: str = "") -> dict:
        """
        一键全流程审批

        1. 分类 → 2. 提取 → 4. 决策 → 生成报告
        """
        start = time.time()
        pipeline_id = uuid.uuid4().hex[:12]
        print(f"\n{'=' * 60}")
        print(f"全流程审批 [{pipeline_id}]")
        print(f"{'=' * 60}")

        # Step 1: 文档分类
        t1 = time.time()
        classify_result = self.step1_classify(image_path, ocr_text, doc_type)
        doc_type = doc_type or classify_result["doc_type"]
        print(f"\n[Step 1] 文档分类: {doc_type} "
              f"(置信度: {classify_result['confidence']:.2%}) — "
              f"{time.time()-t1:.1f}s")

        # Step 2: 关键信息抽取
        t2 = time.time()
        extract_result = self.step2_extract(image_path, doc_type)
        print(f"[Step 2] 字段提取: {len(extract_result['fields'])} 个字段 "
              f"(成本: ${extract_result['cost']}) — "
              f"{time.time()-t2:.1f}s")

        if extract_result["needs_review"]:
            print(f"[Step 2] ⚠ 部分字段需人工审核: "
                  f"{[d['field'] for d in extract_result['disagreements']]}")

        # Step 4: 多智能体决策
        t4 = time.time()
        decision_result = self.step4_decision(doc_type, extract_result["fields"])
        print(f"[Step 4] 审批决策: {decision_result['decision']} "
              f"(评分: {decision_result['score']}/100) — "
              f"{time.time()-t4:.1f}s")

        elapsed = time.time() - start
        print(f"\n全流程完成 — 总耗时: {elapsed:.1f}s")

        # 组装完整结果
        full_result = {
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(elapsed, 2),
            "step1_classification": classify_result,
            "step2_extraction": extract_result,
            "step4_decision": decision_result,
        }

        # 保存记录
        record_path = self.records_dir / f"{pipeline_id}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(full_result, f, ensure_ascii=False, indent=2)

        return full_result

    def get_pending_reviews(self) -> list[dict]:
        """获取待审核记录"""
        review_loop = _lazy_import("review")
        pending = review_loop.get_review_pool(status="pending")
        return [
            {
                "id": rec["id"],
                "doc_type": rec["doc_type"],
                "image": rec.get("image", ""),
                "fields": rec.get("llm_fields", {}),
                "disagreements": rec.get("disagreements", []),
                "created_at": rec.get("reviewed_at", ""),
            }
            for rec in pending
        ]

    def submit_review(self, record_id: str, corrections: dict,
                      reviewer: str = "审批师") -> dict:
        """提交人工审核"""
        review_loop = _lazy_import("review")
        record = review_loop.approve_fields(record_id, corrections, reviewer)
        return {
            "id": record["id"],
            "status": "approved",
            "final_fields": record.get("final_fields", {}),
            "corrections": record.get("human_corrections", {}),
        }

    def get_stats(self) -> dict:
        """获取系统统计"""
        review_loop = _lazy_import("review")
        review_stats = review_loop.stats()

        records = list(self.records_dir.glob("*.json"))
        decisions = {"approve": 0, "review": 0, "reject": 0}
        total_time = 0

        for f in records:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                dec = data.get("step4_decision", {}).get("decision", "")
                if dec in decisions:
                    decisions[dec] += 1
                total_time += data.get("processing_time", 0)
            except Exception:
                pass

        return {
            "total_pipelines": len(records),
            "decisions": decisions,
            "avg_processing_time": round(total_time / len(records), 2) if records else 0,
            "review": review_stats,
        }

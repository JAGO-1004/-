"""
模型批量评测 — 对 Step 2 产出的训练数据做批测，识别薄弱字段

流程:
  训练数据 → 用当前模型批量预测 → 与标准答案对比
  → 按字段统计准确率 → 标记薄弱字段 → 针对性优化
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FieldEvaluator:
    """字段级模型评测器"""

    def __init__(self, data_dir: str = "../02_关键信息抽取/data"):
        self.data_dir = Path(data_dir)
        self.training_file = self.data_dir / "training_data.jsonl"

    def load_ground_truth(self, doc_type: str = None) -> list[dict]:
        """
        加载标准答案（来自 Step 2 人工审核后的数据）

        Returns:
            [{"id": str, "doc_type": str, "image": str, "fields": {...}}, ...]
        """
        if not self.training_file.exists():
            # 也检查 approved 目录
            approved_dir = self.data_dir / "approved"
            if approved_dir.exists():
                return self._load_approved(approved_dir, doc_type)
            print(f"训练数据不存在: {self.training_file}")
            return []

        data = []
        with open(self.training_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if doc_type is None or entry.get("doc_type") == doc_type:
                        data.append(entry)
        return data

    def _load_approved(self, approved_dir: Path, doc_type: str = None) -> list[dict]:
        data = []
        for f in approved_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                record = json.load(fp)
            if doc_type is None or record.get("doc_type") == doc_type:
                entry = {
                    "id": record["id"],
                    "doc_type": record["doc_type"],
                    "image": record["image"],
                    "fields": record.get("final_fields", record.get("llm_fields", {})),
                }
                data.append(entry)
        return data

    def evaluate(self, predictions: list[dict], ground_truth: list[dict]) -> dict:
        """
        评测预测结果 vs 标准答案

        Args:
            predictions: [{"id": str, "fields": {field: value, ...}}, ...]
            ground_truth: [{"id": str, "fields": {field: value, ...}}, ...]

        Returns:
            {
                "overall_accuracy": float,
                "field_accuracy": {field: {"correct": N, "total": N, "accuracy": float}},
                "weak_fields": [{"field": str, "accuracy": float, "error_samples": [...]}],
                "by_doc_type": {doc_type: {...}},
            }
        """
        # 构建 lookup
        truth_map = {g["id"]: g for g in ground_truth}

        # 按字段统计
        field_stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
        doc_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for pred in predictions:
            pid = pred["id"]
            truth = truth_map.get(pid)
            if not truth:
                continue

            for field, pred_val in pred.get("fields", {}).items():
                truth_val = truth.get("fields", {}).get(field)

                field_stats[field]["total"] += 1
                doc_stats[truth.get("doc_type", "unknown")]["total"] += 1

                if self._match(pred_val, truth_val):
                    field_stats[field]["correct"] += 1
                    doc_stats[truth.get("doc_type", "unknown")]["correct"] += 1
                else:
                    field_stats[field]["errors"].append({
                        "id": pid,
                        "predicted": pred_val,
                        "expected": truth_val,
                    })

        # 计算指标
        field_accuracy = {}
        weak_fields = []
        total_correct = 0
        total_all = 0

        for field, stats in field_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            field_accuracy[field] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": round(acc, 4),
            }
            total_correct += stats["correct"]
            total_all += stats["total"]

            # 准确率低于 0.9 标记为薄弱字段
            if acc < 0.9 and stats["total"] >= 5:
                weak_fields.append({
                    "field": field,
                    "accuracy": round(acc, 4),
                    "error_samples": stats["errors"][:5],  # 最多取 5 个错误样本
                })

        doc_type_accuracy = {}
        for dt, stats in doc_stats.items():
            doc_type_accuracy[dt] = {
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] else 0,
                "total": stats["total"],
            }

        weak_fields.sort(key=lambda x: x["accuracy"])

        return {
            "overall_accuracy": round(total_correct / total_all, 4) if total_all else 0,
            "total_fields": total_all,
            "total_correct": total_correct,
            "field_accuracy": field_accuracy,
            "weak_fields": weak_fields,
            "by_doc_type": doc_type_accuracy,
        }

    def _match(self, pred, truth) -> bool:
        """字段值匹配（支持容错）"""
        if pred is None and truth is None:
            return True
        if pred is None or truth is None:
            return False
        return str(pred).strip() == str(truth).strip()

    def generate_training_plan(self, eval_result: dict) -> dict:
        """
        根据评测结果生成训练计划

        Returns:
            {
                "focus_fields": [field, ...],    # 需要优化的字段
                "priority": "high"|"medium",      # 优先级
                "suggested_method": "lora"|"rl",  # 建议方法
                "data_needed": int,               # 需要的额外数据量
            }
        """
        weak = eval_result.get("weak_fields", [])
        if not weak:
            return {"focus_fields": [], "priority": "low", "message": "无需优化"}

        focus_fields = [w["field"] for w in weak if w["accuracy"] < 0.85]
        very_weak = [w["field"] for w in weak if w["accuracy"] < 0.7]

        plan = {
            "focus_fields": focus_fields,
            "priority": "high" if very_weak else "medium",
            "suggested_method": "rl" if very_weak else "lora",
            "weak_field_details": weak,
            "data_needed": len(focus_fields) * 50,  # 每个薄弱字段建议增加 50 条
        }
        return plan


if __name__ == "__main__":
    # 演示
    evaluator = FieldEvaluator()

    # 模拟评测数据
    predictions = [
        {"id": "001", "fields": {"姓名": "张三", "身份证号": "110101199001010011"}},
        {"id": "002", "fields": {"姓名": "李四", "身份证号": "11010119900101001X"}},
    ]
    ground_truth = [
        {"id": "001", "doc_type": "身份证_正面", "fields": {"姓名": "张三", "身份证号": "110101199001010011"}},
        {"id": "002", "doc_type": "身份证_正面", "fields": {"姓名": "李四", "身份证号": "110101199001010011"}},
    ]

    result = evaluator.evaluate(predictions, ground_truth)
    print(f"总体准确率: {result['overall_accuracy']:.2%}")
    for field, stats in result["field_accuracy"].items():
        print(f"  {field}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    plan = evaluator.generate_training_plan(result)
    if plan["focus_fields"]:
        print(f"\n薄弱字段: {plan['focus_fields']}")
        print(f"建议方法: {plan['suggested_method']}")

"""
人工标注闭环 — LLM 初步标注 → 人工校验 → 高质量训练数据

核心逻辑:
1. LLM 对争议字段标注多个候选值，人工只审核争议字段
2. 人工修正后的数据自动进入"标准答案库"
3. 标准答案库可用于后续 Few-shot 提示词注入或模型训练
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_templates import TEMPLATES


class ReviewClosedLoop:
    """
    人工标注闭环管理器

    数据流:
    LLM 提取结果  →  review_pool/  (待审核)
                        ↓  人工审核
                  approved/  (已确认)
                        ↓
                  training_data.jsonl  (高质量训练数据)
                        ↓
                  Few-shot 示例 / 模型微调数据
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.review_dir = self.data_dir / "review_pool"
        self.approved_dir = self.data_dir / "approved"
        self.training_file = self.data_dir / "training_data.jsonl"

        for d in [self.review_dir, self.approved_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def add_to_review_pool(self, extract_result: dict):
        """
        将 LLM 提取结果加入审核池

        Args:
            extract_result: extractor.py 返回的结果 dict
        """
        doc_type = extract_result["doc_type"]
        image_path = extract_result["image"]
        fields = extract_result["extracted_fields"]
        disagreements = extract_result.get("disagreements", [])

        record = {
            "id": f"{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "doc_type": doc_type,
            "image": image_path,
            "llm_fields": fields,
            "disagreements": disagreements,
            "summary": extract_result.get("summary", {}),
            "status": "pending",  # pending | approved | rejected
            "human_corrections": {},
            "reviewer": None,
            "reviewed_at": None,
        }

        out_path = self.review_dir / f"{record['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return record["id"]

    def get_review_pool(self, status: str = "pending") -> list[dict]:
        """获取待审核列表"""
        pending = []
        for f in self.review_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                record = json.load(fp)
            if record.get("status") == status:
                pending.append(record)
        return sorted(pending, key=lambda x: x["id"])

    def approve_fields(self, record_id: str, corrected_fields: dict[str, str],
                       reviewer: str = "system") -> dict:
        """
        人工审核通过（可修正字段值）

        Args:
            record_id: 审核记录 ID
            corrected_fields: 人工修正后的字段值（只填有修改的字段，其余用 LLM 结果）
            reviewer: 审核人

        Returns:
            更新后的记录
        """
        record_path = self.review_dir / f"{record_id}.json"
        if not record_path.exists():
            raise FileNotFoundError(f"记录不存在: {record_id}")

        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        # 合并：LLM 结果 + 人工修正
        final_fields = dict(record["llm_fields"])
        for field, value in corrected_fields.items():
            if value is not None and str(value).strip():
                final_fields[field] = str(value).strip()

        record["human_corrections"] = corrected_fields
        record["final_fields"] = final_fields
        record["status"] = "approved"
        record["reviewer"] = reviewer
        record["reviewed_at"] = datetime.now().isoformat()

        # 保存审核结果
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # 同步到 approved 目录
        approved_path = self.approved_dir / f"{record_id}.json"
        with open(approved_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # 追加到训练数据文件
        self._append_training_data(record)

        return record

    def _append_training_data(self, record: dict):
        """将已审核数据追加到训练数据集"""
        entry = {
            "id": record["id"],
            "doc_type": record["doc_type"],
            "image": record["image"],
            "fields": record.get("final_fields", record["llm_fields"]),
            "has_corrections": len(record.get("human_corrections", {})) > 0,
            "corrected_fields": list(record.get("human_corrections", {}).keys()),
        }
        with open(self.training_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_training_data(self, doc_type: str = None) -> list[dict]:
        """导出训练数据"""
        if not self.training_file.exists():
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

    def get_high_confidence_data(self) -> list[dict]:
        """获取高置信度的自动通过数据（对模型微调特别有用）"""
        approved = []
        for f in self.approved_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                record = json.load(fp)
            if record["status"] == "approved":
                approved.append(record)
        return approved

    def stats(self) -> dict:
        """统计当前标注进度"""
        total = len(list(self.review_dir.glob("*.json")))
        approved = sum(1 for f in self.review_dir.glob("*.json")
                       if json.load(open(f, "r", encoding="utf-8")).get("status") == "approved")
        pending = total - approved

        # 按类型统计
        by_type = defaultdict(lambda: {"total": 0, "approved": 0})
        for f in self.review_dir.glob("*.json"):
            rec = json.load(open(f, "r", encoding="utf-8"))
            dt = rec.get("doc_type", "unknown")
            by_type[dt]["total"] += 1
            if rec.get("status") == "approved":
                by_type[dt]["approved"] += 1

        return {
            "total": total,
            "approved": approved,
            "pending": pending,
            "completion_rate": round(approved / total * 100, 1) if total else 0,
            "by_type": dict(by_type),
            "training_samples": len(self.get_training_data()),
        }


if __name__ == "__main__":
    loop = ReviewClosedLoop()
    stats = loop.stats()
    print(f"标注池总计: {stats['total']}")
    print(f"已完成: {stats['approved']}")
    print(f"待审核: {stats['pending']}")
    print(f"完成率: {stats['completion_rate']}%")
    print(f"训练数据: {stats['training_samples']} 条")

    if stats["pending"] > 0:
        print(f"\n待审核记录: {stats['pending']} 条")
        pending = loop.get_review_pool()
        for rec in pending[:3]:
            print(f"  [{rec['id']}] {rec['doc_type']} — 争议字段: "
                  f"{[d['field'] for d in rec.get('disagreements', [])]}")

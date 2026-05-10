"""
关键信息抽取 — 完整管线

LLM 初步标注 → 人工标注闭环 → 数据伪造增强

流程:
  进件图像
    │
    ├── 1. LLM 多模型标注 ──→ 提取结果
    │        │                     │
    │        ├── 一致字段 ────────→ 自动通过
    │        └── 分歧字段 ────────→ 标记待审
    │
    ├── 2. 人工标注闭环 ──→ 审核争议字段
    │        │                     │
    │        ├── 确认/修正 ──────→ 标准答案库
    │        └── 反馈注入 ────────→ Few-shot 提升后续准确率
    │
    ├── 3. 数据增强 ──→ 针对薄弱字段生成多样化训练数据
    │
    └── 4. 训练数据导出 ──→ JSONL 格式，供 Step 3 微调使用
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor import FieldExtractor, OpenAIVLM
from review_closed_loop import ReviewClosedLoop
from data_augmentation import DataAugmentation


class KIE_Pipeline:
    """
    关键信息抽取 (Key Information Extraction) 完整管线
    """

    def __init__(self, models: list = None):
        self.extractor = FieldExtractor(models=models or [
            OpenAIVLM("gpt-4o-mini", "openai-gpt-4o-mini"),
            OpenAIVLM("gpt-4o", "openai-gpt-4o"),
        ])
        self.review_loop = ReviewClosedLoop()
        self.augmentation = DataAugmentation()

    def run(self, image_path: str, doc_type: str, auto_approve_threshold: float = 0.8):
        """
        执行完整抽取管线

        Args:
            image_path: 证件图像路径
            doc_type: 证件类型
            auto_approve_threshold: 自动通过阈值

        Returns:
            {"extracted": dict, "needs_review": bool, "record_id": str}
        """
        print(f"=" * 60)
        print(f"Step 2: 关键信息抽取")
        print(f"  文档: {doc_type}")
        print(f"  图像: {image_path}")
        print(f"=" * 60)

        # Step 2-1: LLM 多模型提取
        print(f"\n[2-1] LLM 多模型提取...")
        result = self.extractor.extract(image_path, doc_type)
        print(f"  综合置信度: {result['overall_confidence']:.0%}")
        print(f"  API 成本: ${result['cost']}")

        for field, value in result["extracted_fields"].items():
            print(f"    {field}: {value or '(空)'}")

        # Step 2-2: 加入审核池
        print(f"\n[2-2] 加入人工标注闭环...")
        record_id = self.review_loop.add_to_review_pool(result)
        print(f"  记录 ID: {record_id}")

        if result["needs_review"]:
            print(f"  争议字段: {len(result['disagreements'])} 个")
            for d in result["disagreements"]:
                print(f"    [{d['field']}] 建议: {d['suggested']}")
        else:
            print(f"  无争议字段，可直接自动通过")
            # 高置信度自动通过
            self.review_loop.approve_fields(record_id, {}, reviewer="auto")

        return {
            "extracted": result,
            "needs_review": result["needs_review"],
            "record_id": record_id,
        }

    def batch_run(self, image_dir: str, doc_type: str) -> list[dict]:
        """批量运行抽取管线"""
        import glob
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images.extend(glob.glob(os.path.join(image_dir, ext)))
        images.sort()

        if not images:
            print(f"在 {image_dir} 中未找到图像")
            return []

        print(f"找到 {len(images)} 张图像")
        results = []
        for img in images:
            try:
                res = self.run(img, doc_type)
                results.append(res)
                print(f"  → {'需审核' if res['needs_review'] else '自动通过'}")
            except Exception as e:
                print(f"  ✗ {os.path.basename(img)}: {e}")
        return results

    def generate_training_data(self, target_dir: str = "../03_微调训练/data"):
        """导出训练数据供 Step 3 使用"""
        data = self.review_loop.get_training_data()
        if not data:
            print("暂无训练数据")
            return

        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, "training_data.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"已导出 {len(data)} 条训练数据到 {output_path}")

        # 生成薄弱字段分析
        self.augmentation.generate_report(data)

        # 对薄弱字段生成增强数据
        corrections = self.augmentation.generate_report(data) or {}
        for field in corrections:
            if corrections[field] >= 3:  # 被修正 3 次以上的字段
                print(f"\n  自动增强薄弱字段: {field}")
                hard_cases = self.augmentation.generate_hard_cases(
                    doc_type="*", field_name=field, count=30
                )
                self.augmentation.save_samples(hard_cases,
                                                f"hard_{field}.jsonl")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python pipeline.py <图像路径> <证件类型>")
        print(f"\n证件类型示例: 身份证_正面, 结婚证, 不动产权证")
        sys.exit(1)

    pipeline = KIE_Pipeline()
    result = pipeline.run(sys.argv[1], sys.argv[2])

    print(f"\n{'=' * 60}")
    if result["needs_review"]:
        print(f"结果: 需人工审核 → 请运行 review_closed_loop.py 审核记录 {result['record_id']}")
    else:
        print(f"结果: 自动通过 ✓")

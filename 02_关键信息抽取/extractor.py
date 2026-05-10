"""
LLM 字段提取器 — 多模型并行提取 + 共识投票

对一张证件图像，调用多个 VLM 分别提取字段，
一致的自动通过，分歧的标记待人工审核。
"""

import base64
import json
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_templates import TEMPLATES


# ============================================================
# 图像处理
# ============================================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ============================================================
# JSON 解析
# ============================================================

def safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    pass
    return {"_parse_error": raw[:200]}


# ============================================================
# VLM 后端
# ============================================================

class VLMBackend:
    def __init__(self, name: str):
        self.name = name

    def extract(self, image_b64: str, media_type: str, prompt: str) -> dict:
        raise NotImplementedError


class OpenAIVLM(VLMBackend):
    def __init__(self, model="gpt-4o", name=None):
        super().__init__(name or f"openai-{model}")
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.input_price = {"gpt-4o": 2.5, "gpt-4o-mini": 0.15}.get(model, 2.5)
        self.output_price = {"gpt-4o": 10, "gpt-4o-mini": 0.6}.get(model, 10)

    def extract(self, image_b64, media_type, prompt):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}"
                        }},
                    ],
                }],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            usage = resp.usage
            cost = (
                usage.prompt_tokens * self.input_price
                + usage.completion_tokens * self.output_price
            ) / 1_000_000 if usage else 0
            return {
                "raw": raw, "parsed": safe_parse_json(raw), "success": True,
                "usage": {"input": usage.prompt_tokens, "output": usage.completion_tokens,
                          "cost": round(cost, 4)} if usage else None,
            }
        except Exception as e:
            return {"raw": f"ERROR: {e}", "parsed": {}, "success": False, "usage": None}


class AnthropicVLM(VLMBackend):
    def __init__(self, model="claude-sonnet-4-20250514", name=None):
        super().__init__(name or f"anthropic-{model}")
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.input_price = 3
        self.output_price = 15

    def extract(self, image_b64, media_type, prompt):
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {
                            "type": "base64", "media_type": media_type, "data": image_b64,
                        }},
                    ],
                }],
            )
            raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
            usage = resp.usage
            cost = (
                usage.input_tokens * self.input_price
                + usage.output_tokens * self.output_price
            ) / 1_000_000 if usage else 0
            return {
                "raw": raw, "parsed": safe_parse_json(raw), "success": True,
                "usage": {"input": usage.input_tokens, "output": usage.output_tokens,
                          "cost": round(cost, 4)} if usage else None,
            }
        except Exception as e:
            return {"raw": f"ERROR: {e}", "parsed": {}, "success": False, "usage": None}


# ============================================================
# 共识引擎
# ============================================================

def normalize(v: str) -> str:
    return re.sub(r"[\s\-_—,，、]", "", v).lower()


def build_consensus(results: list[dict], field_names: list[str]) -> dict:
    """
    多模型共识判断
    一致 → 高置信度自动通过
    分歧 → 标记待人工审核
    """
    n = len(results)
    annotations = {}
    disagreements = []

    for field in field_names:
        votes = {}
        for r in results:
            mn = r["model"]
            val = r.get("fields", {}).get(field)
            if val is not None and str(val).strip():
                votes[mn] = str(val).strip()

        if not votes:
            annotations[field] = {"value": None, "confidence": "无输出", "votes": {}}
            continue

        # 归一化投票
        norm_to_raw = {}
        for mn, val in votes.items():
            nv = normalize(val)
            if nv not in norm_to_raw or len(val) > len(norm_to_raw[nv]):
                norm_to_raw[nv] = val

        value_counts = {}
        for val in votes.values():
            nv = normalize(val)
            value_counts[nv] = value_counts.get(nv, 0) + 1

        best_norm, best_count = max(value_counts.items(), key=lambda x: x[1])
        ratio = best_count / n

        confidence = "高" if ratio >= 0.8 else "中" if ratio >= 0.6 else "低"
        majority_value = norm_to_raw.get(best_norm)

        annotations[field] = {"value": majority_value, "confidence": confidence, "votes": votes}
        if confidence == "低":
            disagreements.append({"field": field, "votes": votes, "suggested": majority_value})

    auto = sum(1 for a in annotations.values() if a["confidence"] in ("高", "中"))
    need = sum(1 for a in annotations.values() if a["confidence"] in ("低", "无输出"))

    return {
        "annotations": annotations,
        "summary": {
            "total": len(field_names),
            "auto_accepted": auto,
            "needs_review": need,
            "accept_rate": round(auto / len(field_names) * 100, 1) if field_names else 0,
        },
        "disagreements": disagreements,
    }


# ============================================================
# 主管线
# ============================================================

class FieldExtractor:
    """多模型字段提取器"""

    def __init__(self, models: list[VLMBackend] = None):
        self.models = models or [
            OpenAIVLM("gpt-4o-mini", "openai-gpt-4o-mini"),
            OpenAIVLM("gpt-4o", "openai-gpt-4o"),
        ]

    def extract(self, image_path: str, doc_type: str) -> dict:
        """
        对一张证件图像提取字段

        Args:
            image_path: 证件图像路径
            doc_type: 证件类型（需在 TEMPLATES 中定义）

        Returns:
            {
                "doc_type": str,
                "extracted_fields": {字段: 值, ...},
                "confidence": 总体置信度,
                "needs_review": bool,
                "disagreements": [...],
                "cost": float,
                "per_model": {...},
            }
        """
        if doc_type not in TEMPLATES:
            raise ValueError(f"未知证件类型: {doc_type}")

        tpl = TEMPLATES[doc_type]
        field_names = tpl["fields"]
        prompt = tpl["prompt"]

        image_b64 = encode_image(image_path)
        media_type = get_media_type(image_path)

        # 并行调用
        raw_results = {}
        total_cost = 0.0

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_map = {
                executor.submit(m.extract, image_b64, media_type, prompt): m
                for m in self.models
            }
            for future in as_completed(future_map):
                model = future_map[future]
                try:
                    result = future.result(timeout=60)
                    raw_results[model.name] = result
                    if result.get("usage"):
                        total_cost += result["usage"].get("cost", 0)
                except Exception as e:
                    raw_results[model.name] = {
                        "raw": f"ERROR: {e}", "parsed": {}, "success": False, "usage": None,
                    }

        # 共识
        consensus_input = [
            {"model": name, "fields": res.get("parsed", {}), "success": res["success"]}
            for name, res in raw_results.items()
        ]
        consensus = build_consensus(consensus_input, field_names)

        # 组装结果
        extracted = {}
        overall_conf = 0
        for f, info in consensus["annotations"].items():
            extracted[f] = info["value"]
            if info["confidence"] == "高":
                overall_conf += 1

        return {
            "doc_type": doc_type,
            "image": image_path,
            "extracted_fields": extracted,
            "overall_confidence": round(overall_conf / len(field_names), 2) if field_names else 0,
            "needs_review": len(consensus["disagreements"]) > 0,
            "disagreements": consensus["disagreements"],
            "summary": consensus["summary"],
            "cost": round(total_cost, 4),
            "per_model": {
                name: {
                    "fields": res.get("parsed", {}),
                    "cost": (res.get("usage") or {}).get("cost", 0),
                    "success": res["success"],
                }
                for name, res in raw_results.items()
            },
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python extractor.py <图像路径> <证件类型>")
        print(f"证件类型: {list(TEMPLATES.keys())[:5]}...")
        sys.exit(1)

    extractor = FieldExtractor()
    result = extractor.extract(sys.argv[1], sys.argv[2])

    print(f"\n文档: {result['doc_type']}")
    print(f"图像: {result['image']}")
    print(f"综合置信度: {result['overall_confidence']:.0%}")
    print(f"需人工复审: {'是' if result['needs_review'] else '否'}")
    print(f"成本: ${result['cost']}")
    print(f"\n提取字段:")
    for field, value in result["extracted_fields"].items():
        print(f"  {field}: {value or '(空)'}")
    if result["disagreements"]:
        print(f"\n--- 需复审的字段 ---")
        for d in result["disagreements"]:
            print(f"  [{d['field']}] 建议: {d['suggested']}")
            for model, val in d["votes"].items():
                print(f"    {model}: {val}")

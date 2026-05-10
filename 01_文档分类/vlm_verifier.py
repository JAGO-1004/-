"""
VLM 图文版式验证器
通过视觉特征（布局、印章、格式等）验证 BERT 的分类结果
作为第二道验证关卡，实现内容+版式双重确认
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64
import json
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from document_types import ALL_DOC_TYPES


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")


VERIFICATION_PROMPT = """你是一个专业的贷款审批文档分类专家。
请识别这张图像属于哪种贷款审批材料。

可选文档类型（只输出其中之一）：
{doc_types}

判断依据：
1. **版式布局**：表格形式、证件格式、合同样式等
2. **视觉元素**：印章、照片、条码、水印等
3. **文本内容**：标题、关键词等

请只输出JSON格式（不要多余文字）：
{{"doc_type": "文档类型", "confidence": 0-1之间的置信度, "reason": "简要判别依据"}}"""


class VLMDocumentVerifier:
    """VLM 图文版式验证器"""

    def __init__(self, model="gpt-4o", api_key=None):
        self.model = model
        if model.startswith("gpt"):
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        elif model.startswith("claude"):
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

    def verify(self, image_path: str, candidate_types: list[str] = None) -> dict:
        """
        验证/识别文档类型

        Args:
            image_path: 文档图像路径
            candidate_types: BERT 分类的候选类型（缩小范围）
                            为 None 则在全部类型中判断

        Returns:
            {"doc_type": str, "confidence": float, "reason": str}
        """
        types_to_check = candidate_types or ALL_DOC_TYPES
        doc_list = "\n".join(f"  - {t}" for t in types_to_check)
        prompt = VERIFICATION_PROMPT.format(doc_types=doc_list)

        image_b64 = encode_image(image_path)
        media_type = get_media_type(image_path)

        if self.model.startswith("gpt"):
            return self._call_gpt(image_b64, media_type, prompt)
        elif self.model.startswith("claude"):
            return self._call_claude(image_b64, media_type, prompt)
        else:
            raise ValueError(f"不支持的模型: {self.model}")

    def _call_gpt(self, image_b64, media_type, prompt) -> dict:
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
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        return self._parse_response(raw)

    def _call_claude(self, image_b64, media_type, prompt) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                ],
            }],
        )
        raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"doc_type": None, "confidence": 0.0, "reason": f"解析失败: {raw[:100]}"}


class LayoutVerifierPool:
    """多 VLM 版式验证池 — 多个模型投票确认"""

    def __init__(self, models: list[str] = None):
        self.models = models or ["gpt-4o", "gpt-4o-mini"]
        self.verifiers = [VLMDocumentVerifier(m) for m in self.models]

    def verify(self, image_path: str, candidate_types: list[str] = None) -> list[dict]:
        """所有模型并行验证"""
        results = []
        with ThreadPoolExecutor(max_workers=len(self.verifiers)) as executor:
            future_map = {
                executor.submit(v.verify, image_path, candidate_types): v
                for v in self.verifiers
            }
            for future in as_completed(future_map):
                verifier = future_map[future]
                try:
                    result = future.result(timeout=30)
                    result["model"] = verifier.model
                    results.append(result)
                except Exception as e:
                    results.append({
                        "model": verifier.model,
                        "doc_type": None,
                        "confidence": 0.0,
                        "reason": f"错误: {e}",
                    })
        return results

    def consensus(self, results: list[dict]) -> dict:
        """多模型投票形成共识"""
        votes = {}
        for r in results:
            dt = r.get("doc_type")
            if dt:
                votes[dt] = votes.get(dt, 0) + 1

        if not votes:
            return {"doc_type": None, "confidence": 0.0, "vote_detail": results}

        best_type = max(votes, key=votes.get)
        agree_ratio = votes[best_type] / len([r for r in results if r.get("doc_type")])

        # 取各模型置信度均值
        avg_conf = 0.0
        for r in results:
            if r.get("doc_type") == best_type:
                avg_conf += (r.get("confidence") or 0)
        avg_conf /= max(votes.get(best_type, 1), 1)

        return {
            "doc_type": best_type,
            "confidence": avg_conf,
            "agree_ratio": agree_ratio,
            "votes": votes,
            "vote_detail": results,
        }


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        pool = LayoutVerifierPool(["gpt-4o-mini"])
        results = pool.verify(image_path)
        consensus = pool.consensus(results)
        print(f"\n验证结果:")
        print(f"  文档类型: {consensus['doc_type']}")
        print(f"  置信度: {consensus['confidence']:.2%}")
        print(f"  一致率: {consensus['agree_ratio']:.0%}")
    else:
        print("用法: python vlm_verifier.py <证件图像路径>")

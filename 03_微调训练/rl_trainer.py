"""
强化学习优化 — GRPO 风格奖励训练

参考 HunyuanOCR 的 GRPO 方法，对 OCR 字段提取结果
设计可验证的奖励函数，通过强化学习优化薄弱字段。

奖励设计:
- 字段存在奖励: 输出包含了目标字段 +1
- 字段格式奖励: 身份证号18位、日期格式正确等 +0.5
- 字段内容奖励: 与标准答案一致 +1
- 结构化奖励: 输出为合法 JSON +0.5
"""

import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class OCRRewardFunction:
    """
    OCR 字段提取的奖励函数

    对模型的输出，从多个维度评估质量并给出奖励分
    """

    # 字段格式校验规则
    FORMAT_RULES = {
        "身份证号": (r"^\d{17}[\dXx]$", "18位身份证号，末位可为X"),
        "手机号": (r"^1[3-9]\d{9}$", "11位手机号"),
        "出生日期": (r"^\d{4}[\d\-/]\d{1,2}[\d\-/]\d{1,2}$", "日期格式"),
        "金额": (r"^\d+(\.\d{1,2})?$", "数字金额"),
    }

    def __init__(self, field_definitions: dict = None):
        """
        Args:
            field_definitions: {field_name: {"format": str, "required": bool}}
        """
        self.field_defs = field_definitions or {}

    def compute(self, output: dict, target: dict = None) -> dict:
        """
        计算多维奖励

        Args:
            output: 模型输出的字段 dict
            target: 标准答案字段 dict（可选，有则计算内容奖励）

        Returns:
            {
                "total": float,           # 总奖励
                "breakdown": {
                    "json_validity": float,
                    "field_existence": float,
                    "format_correctness": float,
                    "field_accuracy": float,  # 仅在有 target 时
                },
                "details": {field: {reason: score}},
            }
        """
        rewards = {
            "json_validity": 0.0,
            "field_existence": 0.0,
            "format_correctness": 0.0,
            "field_accuracy": 0.0,
        }
        details = {}

        # 1. JSON 合法性奖励
        if isinstance(output, dict):
            rewards["json_validity"] = 1.0

        if not isinstance(output, dict):
            return {"total": rewards["json_validity"], "breakdown": rewards, "details": details}

        # 2. 字段存在奖励 — 要求的字段是否都输出了
        if self.field_defs:
            required = [f for f, defs in self.field_defs.items()
                        if defs.get("required", True)]
            if required:
                exist_ratio = sum(1 for f in required if f in output and output[f]) / len(required)
                rewards["field_existence"] = exist_ratio

        # 3. 格式正确性奖励
        format_hits = 0
        format_total = 0
        for field, value in output.items():
            if not value or not isinstance(value, str):
                continue
            rule = self.FORMAT_RULES.get(field)
            if rule:
                format_total += 1
                if re.match(rule[0], value.strip()):
                    format_hits += 1
                    details[field] = details.get(field, {})
                    details[field]["format"] = 1.0

        if format_total > 0:
            rewards["format_correctness"] = format_hits / format_total

        # 4. 字段准确率奖励（与标准答案对比）
        if target and isinstance(target, dict):
            total = 0
            correct = 0
            for field, target_val in target.items():
                if field in output:
                    total += 1
                    if str(output.get(field, "")).strip() == str(target_val).strip():
                        correct += 1
                        details[field] = details.get(field, {})
                        details[field]["accuracy"] = 1.0
                    else:
                        details[field] = details.get(field, {})
                        details[field]["accuracy"] = 0.0

            if total > 0:
                rewards["field_accuracy"] = correct / total

        # 总奖励 = 加权和
        weights = {
            "json_validity": 0.1,
            "field_existence": 0.2,
            "format_correctness": 0.3,
            "field_accuracy": 0.4,
        }

        total_reward = sum(rewards[k] * weights.get(k, 0) for k in rewards)

        return {
            "total": round(total_reward, 4),
            "breakdown": rewards,
            "details": details,
        }


class GRPOTrainer:
    """
    GRPO 风格强化学习训练器

    对批量采样结果计算优势，用组内相对奖励做策略优化。
    参考: HunyuanOCR GRPO + olmOCR 2 RLVR
    """

    def __init__(self, reward_fn: OCRRewardFunction,
                 learning_rate: float = 8e-7,  # 同 HunyuanOCR 论文
                 output_dir: str = "./models/rl"):
        self.reward_fn = reward_fn
        self.lr = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_group_rewards(self, samples: list[dict],
                               ground_truth: dict = None) -> list[float]:
        """
        对一组采样结果计算奖励，并做组内归一化

        Args:
            samples: [{"output": dict, "target": dict}, ...]
                     同一 prompt 的 8 个采样结果（参考 GRPO 的 8 responses）
            ground_truth: 标准答案

        Returns:
            [advantage_1, ..., advantage_8]  归一化后的优势值
        """
        rewards = []
        for sample in samples:
            output = sample.get("output", {})
            target = sample.get("target") or ground_truth
            reward = self.reward_fn.compute(output, target)
            rewards.append(reward["total"])

        # 组内归一化（GRPO 核心）
        if len(rewards) > 1:
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std_r = max(std_r, 1e-6)  # 避免除零
            advantages = [(r - mean_r) / std_r for r in rewards]
        else:
            advantages = rewards

        return advantages

    def train_step(self, model_outputs: list[dict],
                   ground_truth: dict = None) -> dict:
        """
        单步 GRPO 训练

        Args:
            model_outputs: 模型的 8 个采样输出
                           [{"output": {field: val}, "target": {field: val}}, ...]
            ground_truth: 标准答案

        Returns:
            {"mean_reward": float, "advantages": list, "best_output": dict}
        """
        advantages = self.compute_group_rewards(model_outputs, ground_truth)

        # 找出最佳输出（优势最高的）
        best_idx = advantages.index(max(advantages))
        best_output = model_outputs[best_idx].get("output", {})

        # 实际训练中，这里用 advantages 作为权重更新模型参数
        # loss = -E[advantage * log_prob(output)]

        result = {
            "mean_reward": sum(advantages) / len(advantages),
            "advantages": advantages,
            "best_output": best_output,
            "num_samples": len(model_outputs),
        }
        return result

    def train_loop(self, dataset: list[dict], num_steps: int = 100,
                   samples_per_step: int = 8):
        """
        完整 GRPO 训练循环

        Args:
            dataset: 训练数据集
            num_steps: 训练步数
            samples_per_step: 每步采样数（GRPO 建议 8）
        """
        print(f"=" * 60)
        print(f"GRPO 强化学习训练")
        print(f"  数据集: {len(dataset)} 条")
        print(f"  步数: {num_steps}, 每步采样: {samples_per_step}")
        print(f"  学习率: {self.lr}")
        print(f"=" * 60)

        if not dataset:
            print("[!] 数据集为空，请先运行 Step 2 生成训练数据")
            return {"steps_completed": 0}

        # 模拟训练过程
        rewards_history = []
        best_reward = 0

        for step in range(num_steps):
            # 从数据集中采样
            batch = random.choices(dataset, k=min(samples_per_step, len(dataset)))

            # 模拟模型输出（实际使用时这里调用 model.generate）
            model_outputs = []
            for item in batch:
                sample_output = self._simulate_output(item.get("fields", {}))
                model_outputs.append({
                    "output": sample_output,
                    "target": item.get("fields", {}),
                })

            # GRPO 训练
            result = self.train_step(model_outputs)
            rewards_history.append(result["mean_reward"])

            if result["mean_reward"] > best_reward:
                best_reward = result["mean_reward"]

            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{num_steps} — "
                      f"平均奖励: {result['mean_reward']:.4f} — "
                      f"最佳奖励: {best_reward:.4f}")

        print(f"\n[完成] GRPO 训练结束")
        print(f"  最终平均奖励: {sum(rewards_history[-10:])/10:.4f}")
        print(f"  最佳奖励: {best_reward:.4f}")

        return {
            "steps_completed": num_steps,
            "final_reward": sum(rewards_history[-10:]) / 10,
            "best_reward": best_reward,
            "rewards_history": rewards_history,
        }

    def _simulate_output(self, target_fields: dict) -> dict:
        """模拟模型输出（带随机扰动）— 用于演示"""
        output = {}
        for field, value in target_fields.items():
            if random.random() < 0.85:  # 85% 概率正确
                output[field] = value
            else:
                output[field] = f"错误值_{field}"
        return output


if __name__ == "__main__":
    # 演示奖励函数
    reward_fn = OCRRewardFunction({
        "姓名": {"required": True},
        "身份证号": {"required": True},
        "性别": {"required": False},
    })

    # 模拟输出测试
    output = {"姓名": "张三", "身份证号": "110101199001010011", "性别": "男"}
    target = {"姓名": "张三", "身份证号": "110101199001010011", "性别": "男"}

    reward = reward_fn.compute(output, target)
    print(f"奖励测试:")
    print(f"  总奖励: {reward['total']}")
    print(f"  细分: {reward['breakdown']}")

    # 演示 GRPO 训练
    trainer = GRPOTrainer(reward_fn)
    dataset = [
        {"id": f"demo_{i}", "fields": {"姓名": f"姓名{i}", "身份证号": f"110101{i:09d}"}}
        for i in range(20)
    ]
    result = trainer.train_loop(dataset, num_steps=40, samples_per_step=8)
    print(f"\n训练完成: {result['steps_completed']} 步")

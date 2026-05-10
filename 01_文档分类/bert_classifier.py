"""
轻量级 BERT 快分类器
基于 OCR 文本内容，快速初步判断文档类型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from document_types import ALL_DOC_TYPES, LABEL2ID, ID2LABEL


class OCRTextDataset(Dataset):
    """OCR 文本 → 文档类型 训练数据集"""

    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BertDocumentClassifier:
    """轻量级 BERT 文档分类器"""

    def __init__(
        self,
        model_name="bert-base-chinese",
        num_labels=None,
        device=None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels or len(ALL_DOC_TYPES)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._build_model()

    def _build_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        ).to(self.device)

    def predict(self, texts: list[str], return_proba=False) -> list[dict]:
        """
        批量预测文档类型

        Args:
            texts: OCR 文本列表
            return_proba: 是否返回概率分布

        Returns:
            [{"doc_type": str, "confidence": float, "proba": {...}}]
        """
        self.model.eval()
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        results = []
        for i, pred in enumerate(preds):
            proba = {ID2LABEL[j]: float(probs[i, j]) for j in range(self.num_labels)}
            top_label = ID2LABEL[int(pred)]
            top_conf = float(probs[i, pred])
            results.append({
                "doc_type": top_label,
                "confidence": top_conf,
                "proba": proba,
            })
        return results

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] = None,
        val_labels: list[int] = None,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
    ):
        """微调训练"""
        train_dataset = OCRTextDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    def save(self, path: str):
        """保存模型"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"  模型已保存到: {save_path}")

    def load(self, path: str):
        """加载模型"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=self.num_labels
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"  模型已加载: {path}")


if __name__ == "__main__":
    # 快速测试
    classifier = BertDocumentClassifier()
    test_texts = [
        "姓名 张三 性别 男 民族 汉 出生 1990年1月1日 住址 北京市朝阳区 身份证号 110101199001010011",
        "房屋坐落在北京市海淀区 建筑面积 120平方米 合同总价 800万元",
        "申请人张三 申请贷款金额 500万元 贷款期限 360个月",
    ]
    results = classifier.predict(test_texts)
    for text, result in zip(test_texts, results):
        print(f"\n输入: {text[:30]}...")
        print(f"  预测: {result['doc_type']} (置信度: {result['confidence']:.2%})")

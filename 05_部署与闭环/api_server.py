"""
模型服务化 — FastAPI 后端服务

提供完整的 REST API，覆盖五步架构的全部能力:
  POST /classify        — 文档分类 (Step 1)
  POST /extract         — 关键信息抽取 (Step 2)
  POST /train           — 微调训练 (Step 3)
  POST /decision        — 多智能体决策 (Step 4)
  GET  /review/{id}     — 获取待审核记录
  POST /review/{id}     — 提交人工审核结果
  GET  /pipeline        — 全流程审批（一键走完五步）
"""

import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pipeline_orchestrator import PipelineOrchestrator

app = FastAPI(
    title="零售宅抵贷智能审批 API",
    description="AI-native 零售宅抵贷审批系统 — 五步架构全流程服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局编排器
orchestrator = PipelineOrchestrator()


# ============================================================
# 数据模型
# ============================================================

class ClassifyRequest(BaseModel):
    ocr_text: str
    doc_type_hint: Optional[str] = None


class ExtractRequest(BaseModel):
    doc_type: str


class DecisionRequest(BaseModel):
    doc_type: str
    fields: dict


class ReviewSubmit(BaseModel):
    corrections: dict[str, str]
    reviewer: str = "审批师"


class PipelineRequest(BaseModel):
    doc_type: Optional[str] = None


# ============================================================
# API 端点
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    ocr_text: str = Form(""),
    doc_type_hint: str = Form(""),
):
    """Step 1: 文档分类 — BERT + VLM 双重验证"""
    image_path = await save_upload(file)
    try:
        result = orchestrator.step1_classify(
            image_path=image_path,
            ocr_text=ocr_text,
            doc_type_hint=doc_type_hint or None,
        )
        return {"code": 0, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(image_path)


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
):
    """Step 2: 关键信息抽取 — LLM 多模型字段提取"""
    image_path = await save_upload(file)
    try:
        result = orchestrator.step2_extract(image_path=image_path, doc_type=doc_type)
        return {"code": 0, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(image_path)


@app.post("/decision")
async def decision(req: DecisionRequest):
    """Step 4: 多智能体决策 — 风控规则 + 多智能体研判"""
    try:
        result = orchestrator.step4_decision(
            doc_type=req.doc_type,
            fields=req.fields,
        )
        return {"code": 0, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    doc_type: str = Form(""),
    ocr_text: str = Form(""),
):
    """全流程审批 — 一键走完五步"""
    image_path = await save_upload(file)
    try:
        result = orchestrator.run_full_pipeline(
            image_path=image_path,
            doc_type=doc_type or None,
            ocr_text=ocr_text,
        )
        return {"code": 0, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(image_path)


@app.get("/review/pending")
def get_pending_reviews():
    """获取待人工审核的标注记录"""
    records = orchestrator.get_pending_reviews()
    return {"code": 0, "data": records}


@app.post("/review/{record_id}")
def submit_review(record_id: str, req: ReviewSubmit):
    """提交人工审核结果"""
    try:
        result = orchestrator.submit_review(
            record_id=record_id,
            corrections=req.corrections,
            reviewer=req.reviewer,
        )
        return {"code": 0, "data": result}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/stats")
def get_stats():
    """获取系统统计信息"""
    stats = orchestrator.get_stats()
    return {"code": 0, "data": stats}


# ============================================================
# 辅助函数
# ============================================================

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_upload(file: UploadFile) -> str:
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / filename
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    return str(path)


def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

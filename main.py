#!/usr/bin/env python3
"""
FastAPI application exposing AI endpoints (classification + cascade prediction).
Minimal copy stripped of documentation/test utilities.
"""

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import importlib.util
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime

# Load config from local file (deployment_minimal/config.py)
DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DEPLOYMENT_DIR, "config.py")
SPEC = importlib.util.spec_from_file_location("deployment_config", CONFIG_PATH)
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG_MODULE)
config = CONFIG_MODULE.config

# Add deployment_minimal directory (bundled modules) and project root to import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mechanism_classifier import (  # type: ignore
    DrugTargetMechanismClassifier,
    MechanismClassification
)
from cascade_predictor import (     # type: ignore
    BiologicalCascadePredictor,
    CascadePrediction,
    CascadeEffect
)

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger("fastapi-ai")

app = FastAPI(title="Drug-Target Graph AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)):
    if not config.API_KEY:
        return
    if not api_key or api_key != config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


# Pydantic model version of MechanismClassification for FastAPI responses
class MechanismClassificationModel(BaseModel):
    relationship_type: str
    target_class: str
    target_subclass: str
    mechanism: str
    confidence: float
    reasoning: str
    source: str
    timestamp: str


class ClassificationRequest(BaseModel):
    drug_name: str
    target_name: str
    additional_context: Optional[str] = None
    force_reclassify: bool = False


class ClassificationResponse(BaseModel):
    success: bool
    classification: Optional[MechanismClassificationModel] = None
    error: Optional[str] = None


class BatchClassificationRequest(BaseModel):
    drug_name: str
    targets: List[str] = Field(..., min_items=1)
    additional_context: Optional[str] = None
    force_reclassify: bool = False


class BatchClassificationResult(BaseModel):
    target: str
    success: bool
    classification: Optional[MechanismClassificationModel] = None
    error: Optional[str] = None


class BatchClassificationResponse(BaseModel):
    results: List[BatchClassificationResult]


class AutoClassificationRequest(BaseModel):
    drug_name: str
    limit: Optional[int] = Field(default=None, ge=1, le=500)
    additional_context: Optional[str] = None
    force_reclassify: bool = False
    only_unclassified: bool = True


class CascadePredictionRequest(BaseModel):
    drug_name: str
    target_name: str
    depth: int = Field(default=2, ge=1, le=3)
    additional_context: Optional[str] = None
    force_repredict: bool = False


class CascadeEffectModel(BaseModel):
    entity_name: str
    entity_type: str
    effect_type: str
    confidence: float
    reasoning: str
    depth: int


class CascadeData(BaseModel):
    drug_name: str
    target_name: str
    direct_effects: List[CascadeEffectModel]
    secondary_effects: List[CascadeEffectModel]
    tertiary_effects: List[CascadeEffectModel]
    stored_in_db: bool
    timestamp: Optional[str]
    warning: Optional[str]


class CascadeResponse(BaseModel):
    success: bool
    data: Optional[CascadeData] = None
    error: Optional[str] = None


classifier: Optional[DrugTargetMechanismClassifier] = None
cascade_predictor: Optional[BiologicalCascadePredictor] = None


@app.on_event("startup")
async def startup_event():
    global classifier, cascade_predictor
    try:
        classifier = DrugTargetMechanismClassifier(
            gemini_api_key=config.GEMINI_API_KEY,
            neo4j_uri=config.NEO4J_URI,
            neo4j_user=config.NEO4J_USER,
            neo4j_password=config.NEO4J_PASSWORD,
            neo4j_database=config.NEO4J_DATABASE
        )
        logger.info("Mechanism classifier initialized")
    except Exception as exc:
        logger.error(f"Failed to initialize classifier: {exc}")
        classifier = None

    try:
        cascade_predictor = BiologicalCascadePredictor(
            gemini_api_key=config.GEMINI_API_KEY,
            neo4j_uri=config.NEO4J_URI,
            neo4j_user=config.NEO4J_USER,
            neo4j_password=config.NEO4J_PASSWORD,
            neo4j_database=config.NEO4J_DATABASE
        )
        logger.info("Cascade predictor initialized")
    except Exception as exc:
        logger.error(f"Failed to initialize cascade predictor: {exc}")
        cascade_predictor = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="drug-target-ai", environment=config.ENVIRONMENT)


@app.post("/classification/classify", response_model=ClassificationResponse)
async def classify_drug_target(payload: ClassificationRequest, api_key=Depends(verify_api_key)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier unavailable")
    try:
        classification = classifier.classify_drug_target(
            drug_name=payload.drug_name,
            target_name=payload.target_name,
            additional_context=payload.additional_context,
            force_reclassify=payload.force_reclassify
        )
        if not classification:
            return ClassificationResponse(success=False, error="Classification failed")
        # Convert dataclass or dict to Pydantic model
        if is_dataclass(classification):
            classification = MechanismClassificationModel(**asdict(classification))
        elif isinstance(classification, dict):
            classification = MechanismClassificationModel(**classification)
        else:
            classification = MechanismClassificationModel(**asdict(classification))
        return ClassificationResponse(success=True, classification=classification)
    except Exception as exc:
        logger.error(f"Classification error: {exc}")
        return ClassificationResponse(success=False, error=str(exc))


@app.post("/classification/batch", response_model=BatchClassificationResponse)
async def batch_classify(payload: BatchClassificationRequest, api_key=Depends(verify_api_key)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier unavailable")
    results: List[BatchClassificationResult] = []
    for target in payload.targets:
        try:
            classification = classifier.classify_drug_target(
                drug_name=payload.drug_name,
                target_name=target,
                additional_context=payload.additional_context,
                force_reclassify=payload.force_reclassify
            )
            if classification:
                # Convert dataclass or dict to Pydantic model
                if is_dataclass(classification):
                    classification = MechanismClassificationModel(**asdict(classification))
                elif isinstance(classification, dict):
                    classification = MechanismClassificationModel(**classification)
                else:
                    classification = MechanismClassificationModel(**asdict(classification))
                results.append(BatchClassificationResult(target=target, success=True, classification=classification))
            else:
                results.append(BatchClassificationResult(target=target, success=False, error="Classification failed"))
        except Exception as exc:
            logger.error(f"Batch classification error ({payload.drug_name}->{target}): {exc}")
            results.append(BatchClassificationResult(target=target, success=False, error=str(exc)))
    return BatchClassificationResponse(results=results)


@app.get("/classification/status/{drug_name}/{target_name}", response_model=ClassificationResponse)
async def classification_status(drug_name: str, target_name: str, api_key=Depends(verify_api_key)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier unavailable")
    try:
        classification = classifier.get_existing_classification(drug_name, target_name)
        classification_obj = None
        if classification:
            # Convert dict to Pydantic model
            classification_obj = MechanismClassificationModel(**classification)
        return ClassificationResponse(success=classification_obj is not None, classification=classification_obj)
    except Exception as exc:
        logger.error(f"Status check error: {exc}")
        return ClassificationResponse(success=False, error=str(exc))


@app.post("/classification/auto", response_model=BatchClassificationResponse)
async def auto_classify_targets(payload: AutoClassificationRequest, api_key=Depends(verify_api_key)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier unavailable")
    try:
        raw_results = classifier.batch_classify_drug_targets(
            drug_name=payload.drug_name,
            limit=payload.limit,
            additional_context=payload.additional_context or "",
            force_reclassify=payload.force_reclassify,
            only_unclassified=payload.only_unclassified
        )
        results: List[BatchClassificationResult] = []
        if not raw_results:
            logger.info(f"No targets processed during auto classification for {payload.drug_name}")
        for item in raw_results:
            try:
                normalized_item = {str(k).strip().strip('"').strip(): v for k, v in item.items()}
                logger.debug(f"Auto classification normalized keys: {list(normalized_item.keys())}")
                target_name_guess = str(
                    normalized_item.get("target_name")
                    or normalized_item.get("target")
                    or item.get("target_name")
                    or item.get("target")
                    or "unknown"
                ).strip()

                # Create classification Pydantic model directly from normalized data
                classification = MechanismClassificationModel(
                    relationship_type=str(normalized_item.get("relationship_type", "Unknown")).strip(),
                    target_class=str(normalized_item.get("target_class", "Unknown")).strip(),
                    target_subclass=str(normalized_item.get("target_subclass", "Unknown")).strip(),
                    mechanism=str(normalized_item.get("mechanism", "Unknown")).strip(),
                    confidence=float(normalized_item.get("confidence", 0.0)),
                    reasoning=str(normalized_item.get("reasoning", "")).strip(),
                    source=str(normalized_item.get("source", "Gemini_API")).strip(),
                    timestamp=str(normalized_item.get("timestamp", datetime.now().isoformat()))
                )

                results.append(
                    BatchClassificationResult(
                        target=target_name_guess,
                        success=True,
                        classification=classification
                    )
                )
            except Exception as exc:
                target_name_guess = str(
                    item.get("target_name")
                    or item.get("target")
                    or "unknown"
                ).strip()
                logger.exception(f"Failed to construct classification object for {payload.drug_name}->{target_name_guess}: {exc}")
                results.append(
                    BatchClassificationResult(
                        target=target_name_guess,
                        success=False,
                        error=f"Failed to serialize classification: {exc}"
                    )
                )
        return BatchClassificationResponse(results=results)
    except Exception as exc:
        logger.error(f"Auto classification error for {payload.drug_name}: {exc}")
        return BatchClassificationResponse(
            results=[
                BatchClassificationResult(
                    target=payload.drug_name,
                    success=False,
                    error=str(exc)
                )
            ]
        )


@app.post("/cascade/predict", response_model=CascadeResponse)
async def predict_cascade(payload: CascadePredictionRequest, api_key=Depends(verify_api_key)):
    if not cascade_predictor:
        raise HTTPException(status_code=503, detail="Cascade predictor unavailable")
    try:
        cascade, warning = cascade_predictor.predict_and_store(
            drug_name=payload.drug_name,
            target_name=payload.target_name,
            depth=payload.depth,
            force_repredict=payload.force_repredict,
            additional_context=payload.additional_context or ""
        )
        if not cascade:
            return CascadeResponse(success=False, error="Cascade prediction failed or not found")
        data = CascadeData(
            drug_name=cascade.drug_name,
            target_name=cascade.target_name,
            direct_effects=[CascadeEffectModel(**asdict(e)) for e in cascade.direct_effects],
            secondary_effects=[CascadeEffectModel(**asdict(e)) for e in cascade.secondary_effects],
            tertiary_effects=[CascadeEffectModel(**asdict(e)) for e in cascade.tertiary_effects],
            stored_in_db=True,
            timestamp=cascade.prediction_timestamp,
            warning=warning
        )
        return CascadeResponse(success=True, data=data)
    except Exception as exc:
        logger.error(f"Cascade prediction error: {exc}")
        return CascadeResponse(success=False, error=str(exc))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

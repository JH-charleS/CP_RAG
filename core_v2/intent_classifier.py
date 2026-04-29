from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config import get_settings
from core_v2.models import IntentLabel

INTENT_LABELS: list[IntentLabel] = [
    "hypothesis_retrieval",
    "subproblem_retrieval",
    "constraint_focus",
    "implementation_debug",
    "concept_explanation",
]


@lru_cache(maxsize=1)
def _load_classifier() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    model_path = get_settings().v2_bert_classifier_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def classify_query_intent(text: str) -> IntentLabel:
    tokenizer, model = _load_classifier()
    with torch.no_grad():
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        logits = model(**encoded).logits
        index = int(torch.argmax(logits, dim=-1).item())
    if index < 0 or index >= len(INTENT_LABELS):
        return "concept_explanation"
    return INTENT_LABELS[index]

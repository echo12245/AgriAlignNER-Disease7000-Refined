from __future__ import annotations
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
def decode_word_level(paths, labels, mask, word_start_mask, id2label):
    y_true_all, y_pred_all = [], []
    for b, path in enumerate(paths):
        y_true, y_pred = [], []
        pidx = 0
        for t in range(mask.shape[1]):
            if not bool(mask[b, t]):
                break
            pred_id = path[pidx]
            pidx += 1
            if bool(word_start_mask[b, t]):
                y_true.append(id2label[int(labels[b, t])])
                y_pred.append(id2label[int(pred_id)])
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
    return y_true_all, y_pred_all
def extract_entities(sequence):
    entities = set()
    current_type = None
    start = None
    for i, label in enumerate(sequence):
        if label == "O" or "-" not in label:
            if current_type is not None:
                entities.add((current_type, start, i - 1))
                current_type = None
                start = None
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            if current_type is not None:
                entities.add((current_type, start, i - 1))
            current_type = entity_type
            start = i
        elif prefix == "I":
            if current_type == entity_type:
                continue
            if current_type is not None:
                entities.add((current_type, start, i - 1))
            current_type = entity_type
            start = i
        else:
            if current_type is not None:
                entities.add((current_type, start, i - 1))
                current_type = None
                start = None
    if current_type is not None:
        entities.add((current_type, start, len(sequence) - 1))
    return entities
def calculate_macro_f1(y_true, y_pred):
    true_by_type = {}
    pred_by_type = {}
    for sample_id, sequence in enumerate(y_true):
        entities = extract_entities(sequence)
        for entity_type, start, end in entities:
            true_by_type.setdefault(entity_type, set()).add(
                (sample_id, start, end)
            )
    for sample_id, sequence in enumerate(y_pred):
        entities = extract_entities(sequence)

        for entity_type, start, end in entities:
            pred_by_type.setdefault(entity_type, set()).add(
                (sample_id, start, end)
            )
    entity_types = sorted(
        set(true_by_type.keys()) | set(pred_by_type.keys())
    )
    report_dict = {}
    f1_values = []
    for entity_type in entity_types:
        true_entities = true_by_type.get(entity_type, set())
        pred_entities = pred_by_type.get(entity_type, set())
        tp = len(true_entities & pred_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )
        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        report_dict[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": len(true_entities),
        }
        f1_values.append(f1)
    macro_f1 = (
        sum(f1_values) / len(f1_values)
        if len(f1_values) > 0
        else 0.0
    )
    report_dict["macro avg"] = {
        "f1-score": macro_f1
    }
    return macro_f1, report_dict
def metrics(y_true, y_pred):
    precision = precision_score(
        y_true,
        y_pred
    )
    recall = recall_score(
        y_true,
        y_pred
    )
    f1 = f1_score(
        y_true,
        y_pred
    )
    macro_f1, report_dict = calculate_macro_f1(
        y_true,
        y_pred
    )
    report_text = classification_report(
        y_true,
        y_pred,
        digits=4
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "report_dict": report_dict,
        "report_text": report_text,
    }
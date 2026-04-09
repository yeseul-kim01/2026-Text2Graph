"""
============================================================
평가 스크립트 (scripts/evaluate.py)
============================================================
역할:
  학습된 모델을 dev/test set에서 평가하고,
  Micro F1 / Ign F1 / Evidence F1 / Inter/Intra F1을 계산하며,
  CodaLab 제출용 result.json도 생성한다.

사용법:
  python scripts/evaluate.py \
      --config configs/stage1.yaml \
      --checkpoint checkpoints/stage1/best_model.pt

  python scripts/evaluate.py \
      --config configs/stage2.yaml \
      --checkpoint checkpoints/stage2/best_model.pt \
      --split test \
      --output_file outputs/stage2_test_predictions.json

입력:
  - config YAML
  - checkpoint 경로
  - split (dev / test)

출력:
  - Micro F1
  - Ign F1
  - Evidence F1 (가능한 경우)
  - Inter / Intra F1 (가능한 경우)
  - prediction json
  - CodaLab submission json

기반 논문:
  - DocRED (Wang et al., 2019)
  - ATLOP (Zhou et al., 2021)
  - DREEAM (Ma et al., 2023)

TODO (김예슬)

[완료]
  - train triple 기반 Ign F1 계산
  - prediction JSON 저장
  - CodaLab 제출용 result.json 생성
  - Evidence F1 계산
  - Inter/Intra F1 분리 계산
  - batch 평가 및 전체 metric 출력

[추가 예정]
  - official evaluation script와 수치 직접 비교
  - relation별 F1 분석
  - confusion matrix / error case dump
============================================================
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DocREModel
from src.postprocessing import postprocess_predictions, to_docred_submission_batch
from src.preprocessing import load_rel2id, create_dataloader
from src.utils import load_checkpoint, load_config, set_seed


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------
def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _normalize_evidence(evidence: Any) -> Tuple[int, ...]:
    if evidence is None:
        return tuple()
    if isinstance(evidence, list):
        return tuple(sorted(int(x) for x in evidence))
    if isinstance(evidence, tuple):
        return tuple(sorted(int(x) for x in evidence))
    return tuple()


# -------------------------------------------------------------
# Gold / Train Fact Loaders
# -------------------------------------------------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_train_facts(train_path: str) -> Set[Tuple[str, str, str]]:
    """
    Ign F1용 train facts 수집.
    (head_name, tail_name, relation) 기준으로 저장
    """
    if not os.path.exists(train_path):
        return set()

    data = _load_json(train_path)
    train_facts = set()

    for doc in data:
        title = doc.get("title", "")
        vertex_set = doc.get("vertexSet", [])
        labels = doc.get("labels", [])

        for label in labels:
            h = label["h"]
            t = label["t"]
            r = label["r"]

            if h >= len(vertex_set) or t >= len(vertex_set):
                continue

            head_names = set(m["name"] for m in vertex_set[h] if "name" in m)
            tail_names = set(m["name"] for m in vertex_set[t] if "name" in m)

            for hn in head_names:
                for tn in tail_names:
                    train_facts.add((hn, tn, r))

    return train_facts


def _build_gold_triples(doc: Dict) -> List[Dict]:
    """
    DocRED gold labels를 내부 triple 포맷으로 변환.
    """
    vertex_set = doc.get("vertexSet", [])
    labels = doc.get("labels", [])

    gold_triples = []

    for label in labels:
        h = label["h"]
        t = label["t"]
        r = label["r"]
        evidence = sorted(label.get("evidence", []))

        triple = {
            "h": h,
            "t": t,
            "r": r,
            "evidence": evidence,
        }

        if h < len(vertex_set):
            head_names = [m.get("name", "") for m in vertex_set[h]]
            if len(head_names) > 0:
                triple["head_name"] = head_names[0]

        if t < len(vertex_set):
            tail_names = [m.get("name", "") for m in vertex_set[t]]
            if len(tail_names) > 0:
                triple["tail_name"] = tail_names[0]

        gold_triples.append(triple)

    return gold_triples


def _group_gold_by_title(data: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = {}
    for doc in data:
        title = doc.get("title", "")
        grouped[title] = _build_gold_triples(doc)
    return grouped


def _build_doc_entity_names(doc: Dict) -> List[str]:
    """
    vertexSet에서 entity canonical names 추출.
    첫 mention name을 대표 이름으로 사용.
    """
    names = []
    for mentions in doc.get("vertexSet", []):
        if len(mentions) > 0:
            names.append(mentions[0].get("name", ""))
        else:
            names.append("")
    return names


def _build_doc_entity_types(doc: Dict) -> List[str]:
    """
    vertexSet에서 entity type 추출.
    첫 mention type을 대표 type으로 사용.
    """
    types = []
    for mentions in doc.get("vertexSet", []):
        if len(mentions) > 0:
            types.append(mentions[0].get("type", ""))
        else:
            types.append("")
    return types


# -------------------------------------------------------------
# Metric Computation
# -------------------------------------------------------------
def compute_re_metrics(
    all_preds: List[Dict],
    gold_by_title: Dict[str, List[Dict]],
    train_facts: Set[Tuple[str, str, str]],
) -> Dict[str, float]:
    """
    Micro F1 / Ign F1 계산
    """
    pred_set = []
    gold_set = []
    pred_ign_set = []
    gold_ign_set = []

    for doc_pred in all_preds:
        title = doc_pred["title"]
        preds = doc_pred["predictions"]
        golds = gold_by_title.get(title, [])

        for g in golds:
            gold_set.append((title, g["h"], g["t"], g["r"]))

            hn = g.get("head_name", "")
            tn = g.get("tail_name", "")
            if (hn, tn, g["r"]) not in train_facts:
                gold_ign_set.append((title, g["h"], g["t"], g["r"]))

        for p in preds:
            pred_set.append((title, p["h"], p["t"], p["r"]))

            hn = p.get("head_name", "")
            tn = p.get("tail_name", "")
            if (hn, tn, p["r"]) not in train_facts:
                pred_ign_set.append((title, p["h"], p["t"], p["r"]))

    pred_set = set(pred_set)
    gold_set = set(gold_set)
    pred_ign_set = set(pred_ign_set)
    gold_ign_set = set(gold_ign_set)

    correct = len(pred_set & gold_set)
    correct_ign = len(pred_ign_set & gold_ign_set)

    precision = _safe_div(correct, len(pred_set))
    recall = _safe_div(correct, len(gold_set))
    f1 = _f1(precision, recall)

    ign_precision = _safe_div(correct_ign, len(pred_ign_set))
    ign_recall = _safe_div(correct_ign, len(gold_ign_set))
    ign_f1 = _f1(ign_precision, ign_recall)

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "ign_precision": ign_precision,
        "ign_recall": ign_recall,
        "ign_f1": ign_f1,
        "num_pred": len(pred_set),
        "num_gold": len(gold_set),
        "num_correct": correct,
        "num_pred_ign": len(pred_ign_set),
        "num_gold_ign": len(gold_ign_set),
        "num_correct_ign": correct_ign,
    }


def compute_evidence_f1(
    all_preds: List[Dict],
    gold_by_title: Dict[str, List[Dict]],
) -> Dict[str, float]:
    """
    Evidence F1 계산.
    (title, h, t, r)가 일치하는 triple에 대해 evidence index 비교
    """
    pred_evidence = set()
    gold_evidence = set()

    for doc_pred in all_preds:
        title = doc_pred["title"]
        preds = doc_pred["predictions"]
        golds = gold_by_title.get(title, [])

        gold_map = {(g["h"], g["t"], g["r"]): g for g in golds}

        for g in golds:
            ev = _normalize_evidence(g.get("evidence", []))
            for e in ev:
                gold_evidence.add((title, g["h"], g["t"], g["r"], e))

        for p in preds:
            key = (p["h"], p["t"], p["r"])
            if key not in gold_map:
                continue
            ev = _normalize_evidence(p.get("evidence", []))
            for e in ev:
                pred_evidence.add((title, p["h"], p["t"], p["r"], e))

    correct = len(pred_evidence & gold_evidence)
    precision = _safe_div(correct, len(pred_evidence))
    recall = _safe_div(correct, len(gold_evidence))
    f1 = _f1(precision, recall)

    return {
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "num_pred_evidence": len(pred_evidence),
        "num_gold_evidence": len(gold_evidence),
        "num_correct_evidence": correct,
    }


def compute_inter_intra_f1(
    all_preds: List[Dict],
    raw_data_by_title: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Stage 3 분석용:
    relation triple을 intra-sentence / inter-sentence로 나눠 F1 계산
    """
    pred_inter = set()
    pred_intra = set()
    gold_inter = set()
    gold_intra = set()

    for doc_pred in all_preds:
        title = doc_pred["title"]
        preds = doc_pred["predictions"]
        raw_doc = raw_data_by_title.get(title, None)

        if raw_doc is None:
            continue

        vertex_set = raw_doc.get("vertexSet", [])
        labels = raw_doc.get("labels", [])

        gold_map = []
        for g in labels:
            h, t, r = g["h"], g["t"], g["r"]

            h_sents = set(m["sent_id"] for m in vertex_set[h] if "sent_id" in m)
            t_sents = set(m["sent_id"] for m in vertex_set[t] if "sent_id" in m)

            is_intra = len(h_sents & t_sents) > 0
            if is_intra:
                gold_intra.add((title, h, t, r))
            else:
                gold_inter.add((title, h, t, r))

        for p in preds:
            h, t, r = p["h"], p["t"], p["r"]

            if h >= len(vertex_set) or t >= len(vertex_set):
                continue

            h_sents = set(m["sent_id"] for m in vertex_set[h] if "sent_id" in m)
            t_sents = set(m["sent_id"] for m in vertex_set[t] if "sent_id" in m)

            is_intra = len(h_sents & t_sents) > 0
            if is_intra:
                pred_intra.add((title, h, t, r))
            else:
                pred_inter.add((title, h, t, r))

    correct_intra = len(pred_intra & gold_intra)
    intra_p = _safe_div(correct_intra, len(pred_intra))
    intra_r = _safe_div(correct_intra, len(gold_intra))
    intra_f1 = _f1(intra_p, intra_r)

    correct_inter = len(pred_inter & gold_inter)
    inter_p = _safe_div(correct_inter, len(pred_inter))
    inter_r = _safe_div(correct_inter, len(gold_inter))
    inter_f1 = _f1(inter_p, inter_r)

    return {
        "intra_precision": intra_p,
        "intra_recall": intra_r,
        "intra_f1": intra_f1,
        "inter_precision": inter_p,
        "inter_recall": inter_r,
        "inter_f1": inter_f1,
        "num_pred_intra": len(pred_intra),
        "num_gold_intra": len(gold_intra),
        "num_correct_intra": correct_intra,
        "num_pred_inter": len(pred_inter),
        "num_gold_inter": len(gold_inter),
        "num_correct_inter": correct_inter,
    }


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DocRE Evaluation Script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="예측 결과 JSON 저장 경로",
    )
    parser.add_argument(
        "--submission_file",
        type=str,
        default=None,
        help="CodaLab 제출용 result.json 저장 경로",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Config / Seed / Device
    # ---------------------------------------------------------
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    device = torch.device(
        config["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model_name"])
    rel2id = load_rel2id(config["data"]["meta_dir"])
    id2rel = {v: k for k, v in rel2id.items()}

    data_dir = config["data"]["data_dir"]
    train_file = os.path.join(data_dir, config["data"]["train_file"])
    eval_file = os.path.join(
        data_dir,
        config["data"]["dev_file"] if args.split == "dev" else config["data"]["test_file"],
    )

    # ---------------------------------------------------------
    # Raw data for metrics / names
    # ---------------------------------------------------------
    raw_eval_data = _load_json(eval_file)
    raw_data_by_title = {doc["title"]: doc for doc in raw_eval_data}
    gold_by_title = _group_gold_by_title(raw_eval_data)
    train_facts = _collect_train_facts(train_file)

    # ---------------------------------------------------------
    # Dataloader
    # ---------------------------------------------------------
    dataloader = create_dataloader(
        data_dir=config["data"]["data_dir"],
        data_file=config["data"]["dev_file"] if args.split == "dev" else config["data"]["test_file"],
        tokenizer=tokenizer,
        rel2id=rel2id,
        max_seq_len=config["data"]["max_seq_length"],
        batch_size=config["training"]["finetune_batch_size"],
        shuffle=False,
        stage=config["experiment"]["stage"],
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = DocREModel(config).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    all_preds = []
    submission_batch = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({args.split})"):
            batch = _move_batch_to_device(batch, device)
            all_outputs = model(batch)

            batch_titles = batch["title"]
            batch_triples = []

            for b, outputs in enumerate(all_outputs):
                title = batch_titles[b]
                raw_doc = raw_data_by_title.get(title, {})

                entity_names = _build_doc_entity_names(raw_doc)
                entity_types = _build_doc_entity_types(raw_doc)

                preds = postprocess_predictions(
                    outputs=outputs,
                    entity_pairs=batch["entity_pairs"][b],
                    id2rel=id2rel,
                    entity_names=entity_names,
                    entity_types=entity_types,
                    threshold_type=config["relation_head"].get("threshold_type", "fixed"),
                    fixed_threshold=config["relation_head"].get("fixed_threshold", 0.5),
                    evidence_threshold=0.5,
                    evidence_top_k=None,
                    remove_duplicates=True,
                    skip_no_relation=True,
                )

                doc_pred = {
                    "title": title,
                    "predictions": preds,
                }
                all_preds.append(doc_pred)
                batch_triples.append(preds)

            submission_batch.extend(
                to_docred_submission_batch(batch_triples, batch_titles)
            )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    metrics = compute_re_metrics(
        all_preds=all_preds,
        gold_by_title=gold_by_title,
        train_facts=train_facts,
    )

    print(f"\n[Eval] Split: {args.split}")
    print(f"[Eval] Documents: {len(all_preds)}")
    print(f"[Eval] Pred Triples: {metrics['num_pred']}")
    print(f"[Eval] Gold Triples: {metrics['num_gold']}")
    print(f"[Eval] Correct: {metrics['num_correct']}")
    print(
        f"[Eval] Micro F1: {metrics['micro_f1'] * 100:.2f}% "
        f"(P={metrics['micro_precision'] * 100:.2f}% / "
        f"R={metrics['micro_recall'] * 100:.2f}%)"
    )
    print(
        f"[Eval] Ign F1: {metrics['ign_f1'] * 100:.2f}% "
        f"(P={metrics['ign_precision'] * 100:.2f}% / "
        f"R={metrics['ign_recall'] * 100:.2f}%)"
    )

    # Evidence F1
    evidence_metrics = compute_evidence_f1(all_preds, gold_by_title)
    if evidence_metrics["num_gold_evidence"] > 0:
        print(
            f"[Eval] Evidence F1: {evidence_metrics['evidence_f1'] * 100:.2f}% "
            f"(P={evidence_metrics['evidence_precision'] * 100:.2f}% / "
            f"R={evidence_metrics['evidence_recall'] * 100:.2f}%)"
        )

    # Inter / Intra F1
    if config["experiment"]["stage"] in ["stage3", "stage4"]:
        ii_metrics = compute_inter_intra_f1(all_preds, raw_data_by_title)
        print(
            f"[Eval] Intra F1: {ii_metrics['intra_f1'] * 100:.2f}% "
            f"(P={ii_metrics['intra_precision'] * 100:.2f}% / "
            f"R={ii_metrics['intra_recall'] * 100:.2f}%)"
        )
        print(
            f"[Eval] Inter F1: {ii_metrics['inter_f1'] * 100:.2f}% "
            f"(P={ii_metrics['inter_precision'] * 100:.2f}% / "
            f"R={ii_metrics['inter_recall'] * 100:.2f}%)"
        )

    # ---------------------------------------------------------
    # Save prediction JSON
    # ---------------------------------------------------------
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_preds, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Predictions saved to {args.output_file}")

    # ---------------------------------------------------------
    # Save CodaLab submission
    # ---------------------------------------------------------
    submission_file = args.submission_file
    if submission_file is None and args.output_file is not None:
        base_dir = os.path.dirname(args.output_file) or "."
        submission_file = os.path.join(base_dir, "result.json")

    if submission_file is not None:
        os.makedirs(os.path.dirname(submission_file) or ".", exist_ok=True)
        with open(submission_file, "w", encoding="utf-8") as f:
            json.dump(submission_batch, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Submission saved to {submission_file}")


if __name__ == "__main__":
    main()
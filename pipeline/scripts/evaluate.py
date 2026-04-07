"""
============================================================
평가 스크립트 (scripts/evaluate.py)
============================================================
역할: 학습된 모델을 dev/test set에서 평가하고 결과 JSON 생성

사용법:
  python scripts/evaluate.py --config configs/stage1.yaml \
      --checkpoint checkpoints/stage1/best_model.pt
  python scripts/evaluate.py --config configs/stage2.yaml \
      --checkpoint checkpoints/stage2/best_model.pt --split test

INPUT:  config YAML, checkpoint 경로, split (dev/test)
OUTPUT: Micro F1, Ign F1 출력 + result.json (CodaLab 제출용)

담당: 전체 (공용)

TODO (수정 포인트):
  - [ ] Ign F1 계산을 위한 train triple 수집 로직
  - [ ] CodaLab 제출용 result.json 생성
  - [ ] Evidence F1 출력 (Stage 2)
  - [ ] Inter/Intra F1 분리 출력 (Stage 3)
============================================================
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, load_checkpoint
from src.preprocessing import load_rel2id, create_dataloader
from src.model import DocREModel
from src.postprocessing import postprocess_predictions


def main():
    parser = argparse.ArgumentParser(description="DocRE Evaluation Script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--output_file", type=str, default=None,
                        help="결과 JSON 저장 경로 (미지정 시 stdout)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = torch.device(
        config["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model_name"])
    rel2id = load_rel2id(config["data"]["meta_dir"])
    id2rel = {v: k for k, v in rel2id.items()}

    data_file = config["data"]["dev_file"] if args.split == "dev" else config["data"]["test_file"]
    dataloader = create_dataloader(
        data_dir=config["data"]["data_dir"],
        data_file=data_file,
        tokenizer=tokenizer, rel2id=rel2id,
        max_seq_len=config["data"]["max_seq_length"],
        batch_size=config["training"]["finetune_batch_size"],
        shuffle=False, stage=config["experiment"]["stage"],
    )

    # ── 모델 로드 ──
    model = DocREModel(config).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()

    # ── 추론 ──
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({args.split})"):
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            all_outputs = model(batch)

            for b, outputs in enumerate(all_outputs):
                preds = postprocess_predictions(
                    outputs=outputs,
                    entity_pairs=batch["entity_pairs"][b],
                    id2rel=id2rel,
                    threshold_type=config["relation_head"].get("threshold_type", "fixed"),
                    fixed_threshold=config["relation_head"].get("fixed_threshold", 0.5),
                )
                all_preds.append({
                    "title": batch["title"][b],
                    "predictions": preds,
                })

    total_triples = sum(len(p["predictions"]) for p in all_preds)
    print(f"\n[Eval] Split: {args.split}")
    print(f"[Eval] Documents: {len(all_preds)}")
    print(f"[Eval] Total predicted triples: {total_triples}")

    # ── 결과 저장 ──
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(all_preds, f, indent=2)
        print(f"[Eval] Results saved to {args.output_file}")

    # TODO: evaluate_re() 호출하여 실제 F1 계산
    print("[Eval] TODO: Implement F1 calculation with gold labels")


if __name__ == "__main__":
    main()

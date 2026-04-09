"""
============================================================
DREEAM Silver Evidence 추론 (scripts/infer_evidence.py)
============================================================
역할:
  Teacher 모델로 distant supervision data에 대해 token importance를 추론하여
  silver evidence(.attns)를 생성한다. (Stage 2 Step 2)

사용법:
  python scripts/infer_evidence.py \
      --config configs/stage2.yaml \
      --teacher_checkpoint checkpoints/stage2/teacher/best_model.pt

입력:
  - Teacher 모델 체크포인트
  - train_distant.json (config["data"]["distant_file"])

출력:
  - train_distant.attns
    torch.save로 저장되는 silver evidence 파일

저장 포맷:
  List[Dict]
    각 원소는 문서 1개에 대응하며 다음 정보를 포함한다.
    {
      "doc_key": str,
      "entity_pairs": List[Tuple[int, int]],
      "token_importance": Tensor[num_pairs, seq_len],
      "seq_len": int,
    }

기반 논문:
  - Ma et al. (2023), DREEAM
  - Zhou et al. (2021), ATLOP

구현 원칙:
  - Teacher의 encoder attention을 사용
  - entity_repr에서 mention attention → entity attention으로 통합
  - head/tail entity attention의 element-wise 곱으로
    pair-specific token importance 생성
  - padding 영역은 attention_mask로 제거
  - 대용량 distant data는 batch 단위로 순회하며 저장

TODO (김예슬)

[완료]
  - Teacher 모델 로드 로직 구현
  - distant data dataloader 생성 연결
  - encoder attention 추출 및 entity_attns 생성
  - pair별 token importance 계산 로직 구현
  - .attns 저장 포맷 정의 및 torch.save 저장 구현
  - 배치/문서 단위 처리 및 progress bar 추가

[추가 예정]
  - confidence threshold 기반 pair filtering
  - relation prediction score와 함께 저장
  - 대용량 데이터용 chunk 저장 / merge 방식 추가
  - evidence supervision 단계에서 top-k token 압축 저장 옵션 추가

담당:
  - 모델 담당: 김예슬
============================================================
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DocREModel
from src.preprocessing import create_dataloader, load_rel2id
from src.utils import load_checkpoint, load_config, set_seed


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """
    batch 내부의 Tensor만 device로 이동.
    list / tuple / python object는 그대로 둔다.
    """
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _build_distant_loader(config: Dict, tokenizer, rel2id):
    """
    프로젝트마다 create_dataloader 시그니처가 조금 다를 수 있어서
    몇 가지 패턴을 순차적으로 시도한다.
    """
    attempts = [
        lambda: create_dataloader(
            config=config,
            tokenizer=tokenizer,
            rel2id=rel2id,
            split="distant",
            shuffle=False,
        ),
        lambda: create_dataloader(
            config,
            tokenizer,
            rel2id,
            split="distant",
            shuffle=False,
        ),
        lambda: create_dataloader(
            config=config,
            tokenizer=tokenizer,
            rel2id=rel2id,
            file_key="distant_file",
            shuffle=False,
        ),
        lambda: create_dataloader(
            config,
            tokenizer,
            rel2id,
            file_key="distant_file",
            shuffle=False,
        ),
    ]

    last_error = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_error = e

    raise RuntimeError(
        "create_dataloader(...) 호출 방식과 현재 infer_evidence.py가 맞지 않습니다. "
        f"preprocessing.py의 시그니처를 확인하세요. 마지막 에러: {last_error}"
    )


def _extract_doc_keys(batch: Dict, start_idx: int, batch_size: int) -> List[str]:
    """
    배치에서 문서 식별 키를 추출.
    없으면 fallback으로 evidence_000001 같은 이름 생성.
    """
    candidate_keys = [
        "doc_keys",
        "doc_key",
        "titles",
        "title",
        "sample_ids",
        "example_ids",
        "doc_ids",
    ]

    for key in candidate_keys:
        if key in batch:
            values = batch[key]
            if isinstance(values, list):
                if len(values) == batch_size:
                    return [str(x) for x in values]
            elif isinstance(values, tuple):
                if len(values) == batch_size:
                    return [str(x) for x in values]

    return [f"evidence_{start_idx + i:07d}" for i in range(batch_size)]


def _compute_pair_token_importance(
    entity_attns: torch.Tensor,
    entity_pairs: List,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    entity-level attention으로부터 pair별 token importance 생성.

    Args:
      entity_attns   : [num_entities, num_heads, seq_len]
      entity_pairs   : [(h_id, t_id), ...]
      attention_mask : [seq_len] (optional)

    Returns:
      token_importance : [num_pairs, seq_len]
    """
    seq_len = entity_attns.size(-1)

    if len(entity_pairs) == 0:
        return torch.zeros(0, seq_len, device=entity_attns.device)

    head_ids = [p[0] for p in entity_pairs]
    tail_ids = [p[1] for p in entity_pairs]

    h_att = entity_attns[head_ids]  # [num_pairs, num_heads, seq_len]
    t_att = entity_attns[tail_ids]  # [num_pairs, num_heads, seq_len]

    # ATLOP localized context와 동일한 핵심 아이디어:
    # head와 tail이 동시에 주목하는 token에 높은 가중치 부여
    token_importance = (h_att * t_att).mean(dim=1)  # [num_pairs, seq_len]

    if attention_mask is not None:
        token_importance = token_importance * attention_mask.unsqueeze(0).float()

    token_importance = token_importance / (
        token_importance.sum(dim=-1, keepdim=True) + 1e-5
    )
    return token_importance


def main():
    parser = argparse.ArgumentParser(description="DREEAM Silver Evidence Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument(
        "--save_name",
        type=str,
        default="train_distant.attns",
        help="저장 파일명 (.attns)",
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

    print("[InferEvidence] DREEAM Silver Evidence Generation")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Distant data: {config['data']['distant_file']}")
    print(f"  Device: {device}")

    # ---------------------------------------------------------
    # Tokenizer / DataLoader
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model_name"])
    rel2id = load_rel2id(config["data"]["meta_dir"])
    distant_loader = _build_distant_loader(config, tokenizer, rel2id)

    # ---------------------------------------------------------
    # Teacher Model
    # ---------------------------------------------------------
    model = DocREModel(config).to(device)
    load_checkpoint(model, args.teacher_checkpoint, device=str(device))
    model.eval()

    # ---------------------------------------------------------
    # Save Path
    # ---------------------------------------------------------
    signal_dir = config["evidence"]["teacher_signal_dir"]
    os.makedirs(signal_dir, exist_ok=True)
    save_path = os.path.join(signal_dir, args.save_name)

    silver_records = []
    global_doc_idx = 0

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(distant_loader, desc="Inferring silver evidence"):
            batch = _move_batch_to_device(batch, device)

            input_ids = batch["input_ids"]              # [B, L]
            attention_mask = batch["attention_mask"]    # [B, L]
            entity_spans = batch["entity_spans"]
            entity_pairs_batch = batch["entity_pairs"]

            batch_size = input_ids.size(0)
            doc_keys = _extract_doc_keys(batch, global_doc_idx, batch_size)

            # Step 1. encoder hidden + attention 추출
            encoder_output = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attention=True,
            )

            hidden_states = encoder_output["hidden_states"]  # [B, L, H]
            attentions = encoder_output["attentions"]        # [B, heads, L, L]

            # Step 2. entity vector / entity attention 생성
            repr_output = model.entity_repr(
                hidden_states=hidden_states,
                entity_spans=entity_spans,
                attention=attentions,
            )

            batch_entity_attns = repr_output.get("entity_attns", None)

            # entity_attns가 없으면 현재 구조상 evidence 생성 불가
            if batch_entity_attns is None:
                raise RuntimeError(
                    "entity_repr에서 entity_attns를 반환하지 않았습니다. "
                    "entity_repr.py의 forward 반환값을 확인하세요."
                )

            # Step 3. 문서별 pair token importance 계산
            for b in range(batch_size):
                entity_pairs = entity_pairs_batch[b]
                entity_attns = batch_entity_attns[b]              # [num_entities, heads, L]
                attn_mask_single = attention_mask[b]              # [L]

                token_importance = _compute_pair_token_importance(
                    entity_attns=entity_attns,
                    entity_pairs=entity_pairs,
                    attention_mask=attn_mask_single,
                )

                seq_len = int(attn_mask_single.sum().item())

                record = {
                    "doc_key": doc_keys[b],
                    "entity_pairs": entity_pairs,
                    "token_importance": token_importance.cpu(),  # 저장 시 CPU로 이동
                    "seq_len": seq_len,
                }
                silver_records.append(record)

            global_doc_idx += batch_size

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    torch.save(silver_records, save_path)

    print("[InferEvidence] Silver evidence generation complete")
    print(f"  Saved to: {save_path}")
    print(f"  #Docs: {len(silver_records)}")
    print("  Format: List[Dict(doc_key, entity_pairs, token_importance, seq_len)]")


if __name__ == "__main__":
    main()
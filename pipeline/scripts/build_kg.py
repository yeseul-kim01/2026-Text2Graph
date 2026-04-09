
"""
============================================================
Knowledge Graph 구축 (scripts/build_kg.py)
============================================================
역할: 모델 예측 결과(triples)를 Neo4j Knowledge Graph로 변환 저장

사용법:
  # evaluate.py로 predictions.json 생성 후:
  python scripts/build_kg.py \
      --predictions outputs/predictions.json \
      --neo4j_uri neo4j+s://xxxxx.databases.neo4j.io \
      --neo4j_user neo4j \
      --neo4j_password YOUR_PASSWORD

  # 또는 .env 파일에 설정:
  # NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
  # NEO4J_USER=neo4j
  # NEO4J_PASSWORD=YOUR_PASSWORD
  python scripts/build_kg.py --predictions outputs/predictions.json

  # Multi-hop 쿼리 테스트:
  python scripts/build_kg.py \
      --predictions outputs/predictions.json \
      --query "Steve Jobs" --max_hops 3

INPUT:  predictions JSON (evaluate.py 출력물)
OUTPUT: Neo4j 그래프 DB에 Node/Edge 생성

담당: 후처리 + 그래프 담당

수정 이력:
  [DONE] v1 — 김예슬: 기본 skeleton
  [DONE] v2 — 김예슬: 실제 Entity/Edge batch insert 구현
    - Entity 추출 + Node 생성
    - Triple → Edge 생성
    - Multi-hop 쿼리 시연
    - 통계 출력 (노드 수, 엣지 수, top relation 등)
============================================================
"""

import os
import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg_builder import KnowledgeGraphBuilder

# .env 파일 로드 (있으면)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 없어도 CLI 인자로 동작


def extract_entities_and_triples(all_docs):
    """
    predictions.json에서 Entity 목록과 Triple 목록을 추출.

    INPUT:
      all_docs — evaluate.py가 생성한 predictions JSON
      구조: [{"title": str, "predictions": [{"h": int, "t": int, "r": str,
              "head_name": str, "tail_name": str, "head_type": str, ...}]}]

    OUTPUT:
      entities — [{"name": str, "type": str, "aliases": []}]
      triples  — [{"head_name": str, "tail_name": str, "r": str, "score": float}]
    """
    entity_info = {}   # name → {"type": str}
    triples = []

    for doc in all_docs:
        preds = doc.get("predictions", [])
        for triple in preds:
            head_name = triple.get("head_name", "")
            tail_name = triple.get("tail_name", "")
            if not head_name or not tail_name:
                continue

            # Entity 수집
            if head_name not in entity_info:
                entity_info[head_name] = {
                    "name": head_name,
                    "type": triple.get("head_type", "UNK"),
                    "aliases": [],
                }
            if tail_name not in entity_info:
                entity_info[tail_name] = {
                    "name": tail_name,
                    "type": triple.get("tail_type", "UNK"),
                    "aliases": [],
                }

            # Triple 수집
            triples.append({
                "head_name": head_name,
                "tail_name": tail_name,
                "r": triple.get("r", "UNK"),
                "score": triple.get("score", 0.0),
                "evidence": triple.get("evidence", []),
            })

    entities = list(entity_info.values())
    return entities, triples


def print_statistics(entities, triples):
    """KG 통계 출력"""
    print(f"\n{'=' * 50}")
    print(f"  Knowledge Graph 통계")
    print(f"{'=' * 50}")
    print(f"  Entity (Node) 수: {len(entities)}")
    print(f"  Relation (Edge) 수: {len(triples)}")

    # Entity type 분포
    type_counter = Counter(e["type"] for e in entities)
    print(f"\n  Entity Type 분포:")
    for t, c in type_counter.most_common():
        print(f"    {t}: {c}")

    # Top relation 분포
    rel_counter = Counter(t["r"] for t in triples)
    print(f"\n  Top 10 Relation:")
    for r, c in rel_counter.most_common(10):
        print(f"    {r}: {c}")

    # Score 통계
    if triples:
        scores = [t["score"] for t in triples]
        print(f"\n  Score 통계:")
        print(f"    Mean: {sum(scores)/len(scores):.4f}")
        print(f"    Min:  {min(scores):.4f}")
        print(f"    Max:  {max(scores):.4f}")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph in Neo4j")
    parser.add_argument("--predictions", type=str, required=True,
                        help="예측 결과 JSON 파일 경로 (evaluate.py 출력)")
    parser.add_argument("--neo4j_uri", type=str, default=None)
    parser.add_argument("--neo4j_user", type=str, default=None)
    parser.add_argument("--neo4j_password", type=str, default=None)
    parser.add_argument("--query", type=str, default=None,
                        help="Multi-hop 쿼리 시작 entity (예: 'Steve Jobs')")
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--stats_only", action="store_true",
                        help="통계만 출력하고 Neo4j 업로드는 안 함")
    args = parser.parse_args()

    # ── 예측 결과 로드 ──
    with open(args.predictions, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
    print(f"[BuildKG] Loaded {len(all_docs)} documents")

    # ── Entity / Triple 추출 ──
    entities, triples = extract_entities_and_triples(all_docs)
    print_statistics(entities, triples)

    if args.stats_only:
        print("[BuildKG] --stats_only: 통계만 출력하고 종료.")
        return

    # ── Neo4j 연결 ──
    neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI")
    neo4j_user = args.neo4j_user or os.getenv("NEO4J_USER")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")

    if not neo4j_uri or not neo4j_user or not neo4j_password:
        print("[BuildKG] Neo4j 접속 정보가 없습니다.")
        print("  방법 1: --neo4j_uri, --neo4j_user, --neo4j_password 인자 사용")
        print("  방법 2: .env 파일에 NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 설정")
        return

    kg = KnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    kg.connect()

    # ── Constraint 설정 ──
    print("[BuildKG] Setting up constraints...")
    kg.setup_constraints()

    # ── Node 생성 ──
    print(f"[BuildKG] Inserting {len(entities)} entities...")
    kg.create_nodes_batch(entities)

    # ── Edge 생성 ──
    print(f"[BuildKG] Inserting {len(triples)} triples...")
    kg.create_edges_batch(triples)

    # ── Multi-hop 쿼리 시연 ──
    if args.query:
        print(f"\n[BuildKG] Multi-hop query: '{args.query}' (max {args.max_hops} hops)")
        results = kg.multi_hop_query(args.query, max_hops=args.max_hops)
        print(f"  Found {len(results)} paths")
        for i, path in enumerate(results[:5]):
            print(f"  Path {i+1}: {path}")

    kg.close()
    print("\n[BuildKG] Knowledge Graph 구축 완료!")


if __name__ == "__main__":
    main()
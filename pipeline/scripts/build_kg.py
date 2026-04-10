"""
============================================================
Knowledge Graph 구축 (scripts/build_kg.py)
============================================================
역할: 모델 예측 결과(triples)를 Neo4j Knowledge Graph로 변환 저장

변경사항:
  - --stage 인자 추가
  - Edge 저장 시 stage 속성 함께 저장
  - stage별 비교를 위해 relation merge key에 stage 포함
============================================================
"""

import os
import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg_builder import KnowledgeGraphBuilder

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def extract_entities_and_triples(all_docs):
    """
    predictions.json에서 Entity 목록과 Triple 목록을 추출.
    """
    entity_info = {}
    triples = []

    for doc in all_docs:
        preds = doc.get("predictions", [])
        for triple in preds:
            head_name = triple.get("head_name", "")
            tail_name = triple.get("tail_name", "")
            if not head_name or not tail_name:
                continue

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

            triples.append({
                "head_name": head_name,
                "tail_name": tail_name,
                "r": triple.get("r", "UNK"),
                "score": triple.get("score", 0.0),
                "evidence": triple.get("evidence", []),
            })

    entities = list(entity_info.values())
    return entities, triples


def print_statistics(entities, triples, stage):
    print(f"\n{'=' * 50}")
    print("  Knowledge Graph 통계")
    print(f"{'=' * 50}")
    print(f"  Stage: {stage}")
    print(f"  Entity (Node) 수: {len(entities)}")
    print(f"  Relation (Edge) 수: {len(triples)}")

    type_counter = Counter(e["type"] for e in entities)
    print("\n  Entity Type 분포:")
    for t, c in type_counter.most_common():
        print(f"    {t}: {c}")

    rel_counter = Counter(t["r"] for t in triples)
    print("\n  Top 10 Relation:")
    for r, c in rel_counter.most_common(10):
        print(f"    {r}: {c}")

    if triples:
        scores = [t["score"] for t in triples]
        print("\n  Score 통계:")
        print(f"    Mean: {sum(scores)/len(scores):.4f}")
        print(f"    Min:  {min(scores):.4f}")
        print(f"    Max:  {max(scores):.4f}")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph in Neo4j")
    parser.add_argument("--predictions", type=str, required=True,
                        help="예측 결과 JSON 파일 경로 (evaluate.py 출력)")
    parser.add_argument("--stage", type=str, required=True,
                        help="저장할 stage 태그 (예: stage2, stage3, stage4)")
    parser.add_argument("--neo4j_uri", type=str, default=None)
    parser.add_argument("--neo4j_user", type=str, default=None)
    parser.add_argument("--neo4j_password", type=str, default=None)
    parser.add_argument("--query", type=str, default=None,
                        help="Multi-hop 쿼리 시작 entity (예: 'Steve Jobs')")
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--stats_only", action="store_true",
                        help="통계만 출력하고 Neo4j 업로드는 안 함")
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
    print(f"[BuildKG] Loaded {len(all_docs)} documents")

    entities, triples = extract_entities_and_triples(all_docs)
    print_statistics(entities, triples, args.stage)

    if args.stats_only:
        print("[BuildKG] --stats_only: 통계만 출력하고 종료.")
        return

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

    print("[BuildKG] Setting up constraints...")
    kg.setup_constraints()

    print(f"[BuildKG] Inserting {len(entities)} entities...")
    kg.create_nodes_batch(entities)

    print(f"[BuildKG] Inserting {len(triples)} triples with stage={args.stage}...")
    kg.create_edges_batch(triples, stage=args.stage)

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

"""
============================================================
Knowledge Graph 구축 (scripts/build_kg.py)
============================================================
역할: 모델 예측 결과(triples)를 Neo4j Knowledge Graph로 변환 저장

사용법:
  python scripts/build_kg.py --predictions results/predictions.json \
      --neo4j_uri bolt://localhost:7687

INPUT:  predictions JSON (evaluate.py 출력물)
OUTPUT: Neo4j 그래프 DB에 Node/Edge 생성

담당: 후처리 + 그래프 담당

TODO (수정 포인트):
  - [ ] Neo4j 연결 설정 (URI, 인증)
  - [ ] Entity canonical name 정규화 로직
  - [ ] Batch insert 최적화 (대량 데이터 처리)
  - [ ] Multi-hop query 시연 함수
============================================================
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg_builder import KnowledgeGraphBuilder


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph in Neo4j")
    parser.add_argument("--predictions", type=str, required=True,
                        help="예측 결과 JSON 파일 경로")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default="password")
    args = parser.parse_args()

    # ── 예측 결과 로드 ──
    with open(args.predictions, "r") as f:
        all_docs = json.load(f)
    print(f"[BuildKG] Loaded {len(all_docs)} documents")

    # ── KG Builder 초기화 ──
    kg = KnowledgeGraphBuilder(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    kg.connect()

    # ── Entity Node 생성 + Relation Edge 생성 ──
    total_nodes = 0
    total_edges = 0

    for doc in all_docs:
        preds = doc.get("predictions", [])
        for triple in preds:
            if "head_name" in triple and "tail_name" in triple:
                # TODO: entity type, aliases 등 추가 정보 포함
                total_edges += 1

    print(f"[BuildKG] Total triples to insert: {total_edges}")
    print("[BuildKG] TODO: Implement batch insert to Neo4j")

    # ── Multi-hop 시연 ──
    # results = kg.multi_hop_query("Steve Jobs", max_hops=3)
    # print(f"  Multi-hop results: {len(results)} paths found")

    kg.close()
    print("[BuildKG] Done.")


if __name__ == "__main__":
    main()

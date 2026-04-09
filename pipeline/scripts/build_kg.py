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

TODO (김예슬):
  - [완] Neo4j 연결 설정 (URI, 인증)
  - [완] Entity canonical name 정규화 로직
  - [완] Batch insert 최적화 (대량 데이터 처리)
  - [완] Multi-hop query 시연 함수
============================================================
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg_builder import KnowledgeGraphBuilder
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph in Neo4j")
    parser.add_argument("--predictions", type=str, required=True,
                        help="예측 결과 JSON 파일 경로")
    parser.add_argument("--neo4j_uri", type=str, default=None)
    parser.add_argument("--neo4j_user", type=str, default=None)
    parser.add_argument("--neo4j_password", type=str, default=None)
    args = parser.parse_args()

    # CLI 우선, 없으면 .env 사용
    neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI")
    neo4j_user = args.neo4j_user or os.getenv("NEO4J_USER")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")

    if not neo4j_uri or not neo4j_user or not neo4j_password:
        raise ValueError(
            "Neo4j 접속 정보가 없습니다. "
            ".env의 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD를 확인하세요."
        )

    # ── 예측 결과 로드 ──
    with open(args.predictions, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
    print(f"[BuildKG] Loaded {len(all_docs)} documents")

    # ── KG Builder 초기화 ──
    kg = KnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    kg.connect()

    # ── Entity Node 생성 + Relation Edge 생성 ──
    total_edges = 0
    for doc in all_docs:
        preds = doc.get("predictions", [])
        for triple in preds:
            if "head_name" in triple and "tail_name" in triple:
                total_edges += 1

    print(f"[BuildKG] Total triples to insert: {total_edges}")
    print("[BuildKG] TODO: Implement batch insert to Neo4j")

    kg.close()
    print("[BuildKG] Done.")


if __name__ == "__main__":
    main()
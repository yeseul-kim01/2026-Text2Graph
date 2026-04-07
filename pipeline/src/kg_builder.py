"""
============================================================
Knowledge Graph Layer (kg_builder.py)
============================================================
역할: 후처리된 triple을 Neo4j 그래프 DB에 저장

INPUT:
  - triples: List of {head, tail, relation, score, evidence}
  - entity_info: entity별 type, aliases 정보

OUTPUT:
  - Neo4j 그래프 (nodes + edges)

담당: 후처리 + 그래프 담당

TODO ( 수정 포인트):
  - [ ] Neo4j 연결 설정 (URI, 인증 정보)
  - [ ] Batch insert 최적화
  - [ ] Multi-hop query 함수 추가
============================================================
"""

from typing import Dict, List, Optional


class KnowledgeGraphBuilder:
    """
    Neo4j 기반 Knowledge Graph 구축기.

    Args:
        uri      : Neo4j bolt URI (e.g., 'bolt://localhost:7687')
        user     : Neo4j 사용자명
        password : Neo4j 비밀번호
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        """Neo4j 연결"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            print("[KG] Connected to Neo4j")
        except ImportError:
            print("[KG] neo4j package not installed. pip install neo4j")
        except Exception as e:
            print(f"[KG] Connection failed: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def create_nodes(self, entities: List[Dict]):
        """
        Entity를 Node로 생성.
        INPUT: [{'name': str, 'type': str, 'aliases': List[str]}, ...]
        """
        if not self.driver:
            print("[KG] Not connected. Call connect() first.")
            return

        with self.driver.session() as session:
            for ent in entities:
                session.run(
                    "MERGE (e:Entity {name: $name}) "
                    "SET e.type = $type, e.aliases = $aliases",
                    name=ent["name"],
                    type=ent.get("type", "UNK"),
                    aliases=ent.get("aliases", []),
                )

    def create_edges(self, triples: List[Dict]):
        """
        Relation triple을 Edge로 생성.
        INPUT: [{'head_name': str, 'tail_name': str, 'r': str, 'score': float}, ...]
        """
        if not self.driver:
            print("[KG] Not connected.")
            return

        with self.driver.session() as session:
            for t in triples:
                session.run(
                    "MATCH (h:Entity {name: $head}), (t:Entity {name: $tail}) "
                    "CREATE (h)-[:REL {type: $rel, score: $score}]->(t)",
                    head=t["head_name"],
                    tail=t["tail_name"],
                    rel=t["r"],
                    score=t.get("score", 0.0),
                )

    def multi_hop_query(self, entity_name: str, max_hops: int = 3) -> List:
        """
        Multi-hop 경로 탐색.
        INPUT:  시작 entity명, 최대 hop 수
        OUTPUT: 경로 리스트
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run(
                f"MATCH p=(a:Entity)-[*1..{max_hops}]->(b:Entity) "
                "WHERE a.name = $name RETURN p LIMIT 100",
                name=entity_name,
            )
            return [record["p"] for record in result]

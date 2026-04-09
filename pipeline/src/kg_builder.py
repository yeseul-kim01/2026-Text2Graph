"""
============================================================
Knowledge Graph Layer (kg_builder.py)
============================================================
역할: 후처리된 triple을 Neo4j 그래프 DB에 저장

INPUT:
  - triples: List of {head_name, tail_name, r, score, evidence}
  - entities: List of {name, type, aliases}

OUTPUT:
  - Neo4j 그래프 (nodes + edges)
============================================================
"""

from typing import Dict, List, Any, Optional


class KnowledgeGraphBuilder:
    """
    Neo4j 기반 Knowledge Graph 구축기.

    Args:
        uri      : Neo4j URI (e.g., 'bolt://localhost:7687', 'neo4j+s://xxxx.databases.neo4j.io')
        user     : Neo4j 사용자명
        password : Neo4j 비밀번호
        database : Neo4j database name (Aura는 보통 'neo4j')
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect(self):
        """Neo4j 연결"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self.driver.verify_connectivity()
            print(f"[KG] Connected to Neo4j: {self.uri}")
        except ImportError:
            raise ImportError("[KG] neo4j package not installed. Run: pip install neo4j")
        except Exception as e:
            raise RuntimeError(f"[KG] Connection failed: {e}")

    def close(self):
        """Neo4j 연결 종료"""
        if self.driver:
            self.driver.close()
            print("[KG] Connection closed")

    def setup_constraints(self):
        """
        Entity name 유니크 제약 생성
        """
        if not self.driver:
            raise RuntimeError("[KG] Not connected. Call connect() first.")

        query = """
        CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.name IS UNIQUE
        """

        with self.driver.session(database=self.database) as session:
            session.run(query)

        print("[KG] Constraint ensured: Entity.name UNIQUE")

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        """
        Entity canonical name 최소 정규화
        - 앞뒤 공백 제거
        - 중복 공백 제거
        """
        if not name:
            return ""
        return " ".join(str(name).strip().split())

    def create_nodes_batch(self, entities: List[Dict[str, Any]], batch_size: int = 1000):
        """
        Entity를 batch로 Node 생성
        INPUT:
          [{'name': str, 'type': str, 'aliases': List[str]}, ...]
        """
        if not self.driver:
            raise RuntimeError("[KG] Not connected. Call connect() first.")

        cleaned_entities = []
        seen = set()

        for ent in entities:
            name = self.normalize_entity_name(ent.get("name", ""))
            if not name:
                continue

            if name in seen:
                continue
            seen.add(name)

            cleaned_entities.append({
                "name": name,
                "type": ent.get("type", "UNK"),
                "aliases": ent.get("aliases", []),
            })

        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {name: row.name})
        SET e.type = row.type,
            e.aliases = row.aliases
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(cleaned_entities), batch_size):
                batch = cleaned_entities[i:i + batch_size]
                session.run(query, rows=batch)

        print(f"[KG] Inserted/Merged nodes: {len(cleaned_entities)}")

    def create_edges_batch(self, triples: List[Dict[str, Any]], batch_size: int = 1000):
        """
        Relation triple을 batch로 Edge 생성
        INPUT:
          [{'head_name': str, 'tail_name': str, 'r': str, 'score': float, 'evidence': ...}, ...]
        """
        if not self.driver:
            raise RuntimeError("[KG] Not connected. Call connect() first.")

        cleaned_triples = []
        seen = set()

        for t in triples:
            head = self.normalize_entity_name(t.get("head_name", ""))
            tail = self.normalize_entity_name(t.get("tail_name", ""))
            rel = str(t.get("r", "")).strip()

            if not head or not tail or not rel:
                continue

            key = (head, tail, rel)
            if key in seen:
                continue
            seen.add(key)

            cleaned_triples.append({
                "head": head,
                "tail": tail,
                "rel": rel,
                "score": float(t.get("score", 0.0)),
                "evidence": t.get("evidence", []),
            })

        query = """
        UNWIND $rows AS row
        MATCH (h:Entity {name: row.head})
        MATCH (t:Entity {name: row.tail})
        MERGE (h)-[r:REL {relation: row.rel}]->(t)
        SET r.score = row.score,
            r.evidence = row.evidence
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(cleaned_triples), batch_size):
                batch = cleaned_triples[i:i + batch_size]
                session.run(query, rows=batch)

        print(f"[KG] Inserted/Merged edges: {len(cleaned_triples)}")

    def multi_hop_query(self, entity_name: str, max_hops: int = 3, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Multi-hop 경로 탐색.
        INPUT:
          - entity_name: 시작 entity명
          - max_hops: 최대 hop 수
          - limit: 반환 경로 수 제한
        OUTPUT:
          - 경로 정보 리스트
        """
        if not self.driver:
            raise RuntimeError("[KG] Not connected. Call connect() first.")

        entity_name = self.normalize_entity_name(entity_name)

        query = f"""
        MATCH p=(a:Entity)-[:REL*1..{max_hops}]->(b:Entity)
        WHERE a.name = $name
        RETURN p
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.database) as session:
            records = session.run(query, name=entity_name, limit=limit)
            for record in records:
                path = record["p"]
                results.append(path)

        return results
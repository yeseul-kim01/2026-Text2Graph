"""
============================================================
Knowledge Graph Layer (kg_builder.py)
============================================================
역할: 후처리된 triple을 Neo4j 그래프 DB에 저장

변경사항:
  - Edge에 stage 속성 저장
  - relation merge key를 (head, tail, relation, stage)로 분리
============================================================
"""

from typing import Dict, List, Any


class KnowledgeGraphBuilder:
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
        if self.driver:
            self.driver.close()
            print("[KG] Connection closed")

    def setup_constraints(self):
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
        if not name:
            return ""
        return " ".join(str(name).strip().split())

    def create_nodes_batch(self, entities: List[Dict[str, Any]], batch_size: int = 1000):
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

    def create_edges_batch(self, triples: List[Dict[str, Any]], stage: str, batch_size: int = 1000):
        if not self.driver:
            raise RuntimeError("[KG] Not connected. Call connect() first.")

        cleaned_triples = []
        seen = set()

        for t in triples:
            head = self.normalize_entity_name(t.get("head_name", ""))
            tail = self.normalize_entity_name(t.get("tail_name", ""))
            rel = str(t.get("r", "")).strip()
            stage_value = str(stage).strip()

            if not head or not tail or not rel or not stage_value:
                continue

            key = (head, tail, rel, stage_value)
            if key in seen:
                continue
            seen.add(key)

            cleaned_triples.append({
                "head": head,
                "tail": tail,
                "rel": rel,
                "stage": stage_value,
                "score": float(t.get("score", 0.0)),
                "evidence": t.get("evidence", []),
            })

        query = """
        UNWIND $rows AS row
        MATCH (h:Entity {name: row.head})
        MATCH (t:Entity {name: row.tail})
        MERGE (h)-[r:REL {relation: row.rel, stage: row.stage}]->(t)
        SET r.score = row.score,
            r.evidence = row.evidence
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(cleaned_triples), batch_size):
                batch = cleaned_triples[i:i + batch_size]
                session.run(query, rows=batch)

        print(f"[KG] Inserted/Merged edges: {len(cleaned_triples)} (stage={stage})")

    def multi_hop_query(self, entity_name: str, max_hops: int = 3, limit: int = 20):
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
                results.append(record["p"])

        return results

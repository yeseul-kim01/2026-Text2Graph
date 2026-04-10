"""
============================================================
DocRE & KG 시연 데모 (scripts/demo_app.py)
============================================================
역할: 학습된 모델로 문서를 입력받아 관계를 추출하고
      Knowledge Graph를 시각적으로 보여주는 웹 데모

사용법 (Colab):
  !pip install gradio networkx matplotlib -q
  !python scripts/demo_app.py \
      --config configs/stage2.yaml \
      --checkpoint checkpoints/stage2/finetune/best_model.pt

  → 자동으로 공유 가능한 URL 생성 (예: https://xxxxx.gradio.live)
  → 이 URL을 사람들에게 공유하면 됨!

============================================================
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from src.utils import load_config, set_seed, load_checkpoint
from src.preprocessing import load_rel2id, DocREDDataset, docred_collate_fn
from src.model import DocREModel
from src.postprocessing import postprocess_predictions


# ══════════════════════════════════════════════════════════════
# 전역 변수 (Gradio에서 접근)
# ══════════════════════════════════════════════════════════════
MODEL = None
TOKENIZER = None
REL2ID = None
ID2REL = None
CONFIG = None
DEVICE = None

# Wikidata Property → 사람이 읽을 수 있는 이름
REL_DISPLAY = {
    "P6": "head of government", "P17": "country", "P19": "place of birth",
    "P20": "place of death", "P22": "father", "P25": "mother",
    "P26": "spouse", "P27": "country of citizenship", "P30": "continent",
    "P31": "instance of", "P36": "capital", "P37": "official language",
    "P39": "position held", "P40": "child", "P50": "author",
    "P54": "member of sports team", "P57": "director", "P58": "screenwriter",
    "P69": "educated at", "P86": "composer", "P102": "member of political party",
    "P108": "employer", "P112": "founded by", "P118": "league",
    "P131": "located in", "P136": "genre", "P137": "operator",
    "P150": "contains", "P155": "follows", "P156": "followed by",
    "P159": "headquarters location", "P161": "cast member",
    "P162": "producer", "P166": "award received", "P170": "creator",
    "P171": "parent taxon", "P172": "ethnic group", "P175": "performer",
    "P176": "manufacturer", "P178": "developer", "P179": "series",
    "P190": "sister city", "P194": "legislative body", "P205": "basin country",
    "P206": "located on body of water", "P241": "military branch",
    "P264": "record label", "P272": "production company",
    "P276": "location", "P279": "subclass of", "P355": "subsidiary",
    "P361": "part of", "P400": "platform", "P403": "mouth of watercourse",
    "P449": "original network", "P463": "member of",
    "P495": "country of origin", "P509": "cause of death",
    "P527": "has part", "P551": "residence", "P569": "date of birth",
    "P570": "date of death", "P571": "inception", "P577": "publication date",
    "P580": "start time", "P582": "end time", "P607": "conflict",
    "P674": "characters", "P706": "located on terrain feature",
    "P710": "participant", "P737": "influenced by",
    "P740": "location of formation", "P749": "parent organization",
    "P800": "notable work", "P807": "separated from",
    "P840": "narrative location", "P937": "work location",
    "P1001": "applies to jurisdiction", "P1056": "product",
    "P1198": "unemployment rate",
}

TYPE_COLORS = {
    "PER": "#3B82F6", "LOC": "#10B981", "ORG": "#F59E0B",
    "TIME": "#8B5CF6", "NUM": "#EC4899", "MISC": "#6B7280",
}


def load_model(config_path, checkpoint_path):
    """모델 로드"""
    global MODEL, TOKENIZER, REL2ID, ID2REL, CONFIG, DEVICE

    CONFIG = load_config(config_path)
    set_seed(CONFIG["experiment"]["seed"])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TOKENIZER = AutoTokenizer.from_pretrained(CONFIG["encoder"]["model_name"])
    REL2ID = load_rel2id(CONFIG["data"]["meta_dir"], CONFIG["data"]["rel2id_file"])
    ID2REL = {v: k for k, v in REL2ID.items()}

    MODEL = DocREModel(CONFIG).to(DEVICE)
    load_checkpoint(MODEL, checkpoint_path, device=str(DEVICE))
    MODEL.eval()
    print(f"[Demo] Model loaded: {checkpoint_path}")
    print(f"[Demo] Device: {DEVICE}")


def process_docred_json(json_text):
    """
    DocRED 형식 JSON 입력 → 관계 추출 → 결과 반환

    INPUT: DocRED JSON 문자열 (단일 문서 또는 리스트)
    OUTPUT: (결과 테이블, 그래프 이미지, JSON 결과)
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return "❌ JSON 파싱 실패. DocRED 형식을 확인하세요.", None, "{}"

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or len(data) == 0:
        return "❌ 빈 데이터입니다.", None, "{}"

    all_results = []

    for doc in data:
        # 전처리
        dataset = DocREDDataset.__new__(DocREDDataset)
        dataset.tokenizer = TOKENIZER
        dataset.rel2id = REL2ID
        dataset.num_relations = len(REL2ID)
        dataset.max_seq_len = CONFIG["data"]["max_seq_length"]
        dataset.stage = CONFIG["experiment"]["stage"]
        dataset.teacher_attns = None

        feature = dataset._process_document(doc, 0)
        if feature is None:
            continue

        # 배치 구성
        batch = docred_collate_fn([feature])
        batch["input_ids"] = batch["input_ids"].to(DEVICE)
        batch["attention_mask"] = batch["attention_mask"].to(DEVICE)

        # 추론
        with torch.no_grad():
            outputs_list = MODEL(batch)

        # 후처리
        for b, outputs in enumerate(outputs_list):
            # Entity names 추출
            vertex_set = doc.get("vertexSet", [])
            entity_names = [v[0].get("name", f"E{i}") for i, v in enumerate(vertex_set)]
            entity_types = [v[0].get("type", "UNK") for v in vertex_set]

            triples = postprocess_predictions(
                outputs=outputs,
                entity_pairs=batch["entity_pairs"][b],
                id2rel=ID2REL,
                entity_names=entity_names,
                entity_types=entity_types,
                threshold_type=CONFIG["relation_head"].get("threshold_type", "fixed"),
                fixed_threshold=CONFIG["relation_head"].get("fixed_threshold", 0.3),
            )

            all_results.append({
                "title": doc.get("title", "Unknown"),
                "predictions": triples,
                "entity_names": entity_names,
                "entity_types": entity_types,
            })

    if len(all_results) == 0:
        return "❌ 유효한 문서가 없습니다.", None, "{}"

    # ── 결과 테이블 생성 ──
    table_rows = []
    for doc_result in all_results:
        for t in doc_result["predictions"]:
            rel_id = t.get("r", "")
            rel_name = REL_DISPLAY.get(rel_id, rel_id)
            table_rows.append([
                t.get("head_name", "?"),
                rel_name,
                t.get("tail_name", "?"),
                f"{t.get('score', 0) * 100:.1f}%",
            ])

    if len(table_rows) == 0:
        table_rows = [["(관계 없음)", "", "", ""]]

    # ── 그래프 시각화 ──
    graph_img = create_graph_visualization(all_results)

    # ── JSON 결과 ──
    result_json = json.dumps(all_results, indent=2, ensure_ascii=False)

    return table_rows, graph_img, result_json


def create_graph_visualization(all_results):
    """NetworkX + Matplotlib로 KG 시각화 이미지 생성"""
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        return None

    G = nx.DiGraph()

    # 노드/엣지 추가
    node_types = {}
    for doc_result in all_results:
        for t in doc_result["predictions"]:
            head = t.get("head_name", "?")
            tail = t.get("tail_name", "?")
            rel = t.get("r", "?")
            score = t.get("score", 0)

            G.add_node(head)
            G.add_node(tail)
            node_types[head] = t.get("head_type", "UNK")
            node_types[tail] = t.get("tail_type", "UNK")

            rel_name = REL_DISPLAY.get(rel, rel)
            G.add_edge(head, tail, label=rel_name, weight=score)

    if len(G.nodes()) == 0:
        return None

    # 레이아웃
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")

    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # 엣지
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#475569",
                           arrows=True, arrowsize=15, width=1.5,
                           connectionstyle="arc3,rad=0.1", alpha=0.7)

    # 엣지 라벨
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                  font_size=7, font_color="#94A3B8",
                                  bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="#1E293B", edgecolor="none",
                                           alpha=0.8))

    # 노드 색상
    colors = [TYPE_COLORS.get(node_types.get(n, "UNK"), "#6B7280") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=800, alpha=0.9, edgecolors="white", linewidths=2)

    # 노드 라벨
    labels = {n: (n[:10] + "…" if len(n) > 10 else n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8,
                            font_weight="bold", font_color="white")

    # 범례
    legend_elements = []
    for t, c in TYPE_COLORS.items():
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=c, label=t))
    ax.legend(handles=legend_elements, loc="upper left",
              facecolor="#1E293B", edgecolor="#475569",
              labelcolor="white", fontsize=8)

    ax.set_title(f"Knowledge Graph ({len(G.nodes())} entities, {len(G.edges())} relations)",
                 color="white", fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()

    # 이미지로 저장
    save_path = "/tmp/kg_graph.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    return save_path


def create_demo():
    """Gradio 데모 생성"""
    import gradio as gr

    # 샘플 DocRED JSON
    sample_json = json.dumps({
        "title": "Steve Jobs",
        "sents": [
            ["Steve", "Jobs", "was", "the", "co-founder", "of", "Apple", "Inc", "."],
            ["He", "was", "born", "in", "San", "Francisco", ",", "California", "."],
            ["Jobs", "also", "founded", "Pixar", "Animation", "Studios", "."],
            ["Apple", "is", "headquartered", "in", "Cupertino", ",", "California", "."],
        ],
        "vertexSet": [
            [{"name": "Steve Jobs", "sent_id": 0, "pos": [0, 2], "type": "PER"},
             {"name": "Jobs", "sent_id": 2, "pos": [0, 1], "type": "PER"}],
            [{"name": "Apple Inc", "sent_id": 0, "pos": [6, 8], "type": "ORG"},
             {"name": "Apple", "sent_id": 3, "pos": [0, 1], "type": "ORG"}],
            [{"name": "San Francisco", "sent_id": 1, "pos": [4, 6], "type": "LOC"}],
            [{"name": "California", "sent_id": 1, "pos": [7, 8], "type": "LOC"},
             {"name": "California", "sent_id": 3, "pos": [6, 7], "type": "LOC"}],
            [{"name": "Pixar Animation Studios", "sent_id": 2, "pos": [3, 6], "type": "ORG"}],
            [{"name": "Cupertino", "sent_id": 3, "pos": [4, 5], "type": "LOC"}],
        ],
        "labels": [
            {"h": 0, "t": 1, "r": "P112", "evidence": [0]},
            {"h": 0, "t": 2, "r": "P19", "evidence": [1]},
            {"h": 0, "t": 4, "r": "P112", "evidence": [2]},
            {"h": 1, "t": 5, "r": "P159", "evidence": [3]},
        ],
    }, indent=2, ensure_ascii=False)

    with gr.Blocks(
        title="DocRE & KG Explorer",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.Markdown("""
        # 📊 Document-level Relation Extraction & Knowledge Graph
        
        **DocRED 기반 문서 수준 관계 추출 시스템**
        
        ATLOP + DREEAM + GAIN 구조 기반 3단계 Incremental Stacking
        
        ---
        
        📌 **사용법**: DocRED 형식 JSON을 입력하면 관계를 추출하고 Knowledge Graph로 시각화합니다.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_json = gr.Textbox(
                    label="📄 DocRED JSON 입력",
                    placeholder="DocRED 형식 JSON을 붙여넣으세요...",
                    lines=15,
                    value=sample_json,
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 관계 추출", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 초기화", size="lg")

            with gr.Column(scale=1):
                result_table = gr.Dataframe(
                    headers=["Head Entity", "Relation", "Tail Entity", "Score"],
                    label="📋 추출된 관계 (Triples)",
                    wrap=True,
                )

        with gr.Row():
            graph_output = gr.Image(label="🕸️ Knowledge Graph", type="filepath")

        with gr.Accordion("📝 Raw JSON 결과", open=False):
            json_output = gr.Textbox(label="결과 JSON", lines=10)

        gr.Markdown("""
        ---
        **모델 정보**
        - Encoder: BERT-base-uncased
        - Entity Repr: LogSumExp Pooling (ATLOP)
        - Classifier: Adaptive Threshold (ATLOP) + Evidence Head (DREEAM)
        - Graph: GAIN-lite Heterogeneous GNN / U-Net
        
        **참고 논문**: ATLOP (Zhou 2021), DREEAM (Ma 2023), GAIN (Zeng 2020)
        """)

        submit_btn.click(
            fn=process_docred_json,
            inputs=[input_json],
            outputs=[result_table, graph_output, json_output],
        )
        clear_btn.click(
            fn=lambda: ("", None, ""),
            outputs=[result_table, graph_output, json_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="DocRE Demo App")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True,
                        help="Gradio 공유 링크 생성 (기본: True)")
    args = parser.parse_args()

    # 모델 로드
    load_model(args.config, args.checkpoint)

    # Gradio 데모 실행
    demo = create_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,  # 공유 URL 생성!
        show_error=True,
    )


if __name__ == "__main__":
    main()
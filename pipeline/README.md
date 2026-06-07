# pipeline/

DocRED 기반 Document-level Relation Extraction → Knowledge Graph 구축 파이프라인의 실행 코드입니다.

전체 설계·단계별 상세·평가 결과·실행 방법은 저장소 루트의 **[README](../README.md)** 를 참고하세요.

## 구조 요약

```
pipeline/
├── configs/      # Stage별 YAML 설정 (stage1~4)
├── data/         # DocRED 원본 JSON + relation 매핑
├── src/          # 핵심 모듈 (전처리 → 인코더 → 개체표현 → 그래프 → 관계분류 → 후처리 → KG)
├── scripts/      # train / evaluate / infer_evidence / build_kg
├── notebooks/    # Colab 실행 노트북
└── requirements.txt
```

## 빠른 실행

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/stage1.yaml
```

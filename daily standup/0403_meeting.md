# 데이터 구조
- json 구조 파악
- input과 output 설정을 어떻게 할 것인가?
- dev.json / train_annotatied.json 모델 input 설정을 어떻게 할 것인가
- test.json 파일은 레이블이 없으므로 성능 지표 파악엔 부적합

# 할 일
## 메인 Task
- 문서 내 객체, 문서 - 문서 사이 객체 간 관계 파악

박정현
- 문서: Train data의 title을 하나의 문서로 취급
- 관계: rel_info.json에 적혀있는 관계들을 객체 간 관계의 한도로 정의

## 서브 Task
1. 어디까지를 객체로 볼 것인가?
2. 관계를 무엇으로 정의할 것인지?

## 모델 아키텍쳐 설정

### 박재윤
- DREEAM

### 박정현
- Gain

### 이수민
- REBEL 


데이터 전처리는 모델 아키텍처 설정 후 진행

## 평가지표 설정
- F1 SCORE (Precision이랑 Recall 각각 측정도 진행)
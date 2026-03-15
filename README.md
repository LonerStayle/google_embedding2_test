# Gemini Embedding 2 - 멀티모달 RAG 테스트 프로젝트
아래 pdf 다운 받아서 data/ 폴더에 넣을 것     
[다운로드 링크](https://www.nrich.go.kr/kor/subscriptionDataUsrView.do?menuIdx=1651&idx=289&gubun=J)

## 프로젝트 개요

Google `gemini-embedding-2-preview` 모델의 멀티모달 임베딩 성능을 다양한 입력 방식으로 테스트하고, 임베딩 방식 간 검색 품질을 비교하는 프로젝트.

## 환경

- Python 3.12 / uv 가상환경
- 주피터 노트북 기반 테스트
- `.env`에 `GOOGLE_API_KEY` 설정 필요

## 테스트 데이터

- `data/헤리티지 역사와 과학 제58권 제4호(통권 제110권).pdf`
- 국립문화유산연구원 학술지 (한국어, 유물사진/표/차트/도면 포함)
- 논문 13편 포함된 통권 PDF (260페이지)

## 노트북 구성

### 1. `test_google_embedding.ipynb` - 페이지 단위 RAG
- PDF → 1페이지 PDF로 분할 → `application/pdf`로 직접 임베딩
- FAISS IndexFlatIP (코사인 유사도), 차원 3072
- task_type 분리: RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY
- 검색 히트 시 앞뒤 ±1 페이지 컨텍스트 포함
- 검색 결과에 텍스트 + 페이지 내 개별 이미지(캡션 매칭) 표시

### 2. `test_bge_m3.ipynb` - Visualized BGE-M3 비교
- PDF → 페이지별 PNG 이미지 → BGE-M3로 임베딩
- 설치: `pip install "visual_bge @ git+https://github.com/FlagOpen/FlagEmbedding.git#subdirectory=research/visual_bge"`
- 가중치: `weights/Visualized_m3.pth` (1.6GB, huggingface_hub로 다운로드)

### 3. `test_image_search.ipynb` - 이미지 개별 검색 (멀티모달)
- PDF에서 개별 이미지 추출 + 캡션/주변 텍스트 함께 추출
- 이미지 + 컨텍스트 텍스트를 하나의 Content로 묶어 멀티모달 임베딩
- 텍스트→이미지 검색, 이미지→이미지 검색 모두 지원

### 4. `test_embedding_comparison.ipynb` - 단일 모달리티 비교
- 동일 쿼리로 3가지 방식 비교: A.PDF바이트 / B.텍스트 / C.페이지이미지
- **결과**: 텍스트 임베딩이 압도적 (Top-1 평균: B=0.64 > C=0.45 > A=0.44)

### 5. `test_multimodal_combination.ipynb` - 멀티모달 조합 비교
- 두 가지 모달리티를 한 Content에 섞어 임베딩할 수 있는지 테스트
- D.PDF+텍스트 / E.이미지+텍스트 / F.PDF+이미지
- 단일 모달리티(A/B/C) 결과와 비교

## 핵심 발견사항

1. **텍스트 쿼리 RAG에는 텍스트 임베딩이 가장 효과적** (유사도 0.64 vs 0.44)
2. **PDF 바이트 직접 임베딩은 장점 없음** — 느리고 정확도도 낮음
3. **이미지 개별 검색 시 컨텍스트 부재 문제** — 이미지만 임베딩하면 캡션/주변 텍스트 정보가 없어 검색 품질 저하
4. **멀티모달 조합 임베딩** — 이미지+텍스트를 한 Content에 묶어 임베딩하면 컨텍스트 보강 가능 (테스트 진행 중)

## gemini-embedding-2-preview 스펙

- **입력**: 텍스트, 이미지(PNG/JPEG, 최대 6장), PDF(최대 6페이지)
- **출력 차원**: 128~3072 (권장: 768, 1536, 3072)
- **토큰 한도**: 8,192
- **task_type**: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, QUESTION_ANSWERING, FACT_VERIFICATION, CODE_RETRIEVAL_QUERY
- **API**: Google AI (Gemini API), `GOOGLE_API_KEY`로 사용 (Vertex AI 불필요)
- **멀티모달 임베딩**: 서로 다른 모달리티(텍스트, 이미지, PDF)를 같은 벡터 공간에 매핑. 크로스모달 검색 가능 (텍스트 쿼리→이미지 검색 등)

## 벡터 검색 구조

```
임베딩 시: 원본 데이터 → embed_content → 벡터(3072) → FAISS 인덱스 저장
                                                    + 원본/메타데이터 별도 저장

검색 시:   쿼리 → 벡터 → FAISS에서 유사 idx 검색 → 저장된 원본에서 꺼내서 표시
```

벡터는 검색용이고, 사용자에게 보여주는 것은 별도 저장된 원본(텍스트, 이미지, 메타데이터)입니다.

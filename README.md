# Gemini Embedding 2 - 멀티모달 RAG 테스트 프로젝트
아래 pdf 다운 받아서 data/ 폴더에 넣을 것     
[다운로드 링크](https://www.nrich.go.kr/kor/subscriptionDataUsrView.do?menuIdx=1651&idx=289&gubun=J)

## 프로젝트 개요

Google `gemini-embedding-2-preview` 모델의 멀티모달 RAG 성능을 테스트하는 프로젝트.
비교 대상으로 `Visualized BGE-M3`도 함께 테스트하여 임베딩 모델 간 성능을 비교한다.

## 환경

- Python 3.12 / uv 가상환경
- 주피터 노트북 기반 테스트
- `.env`에 `GOOGLE_API_KEY` 설정 완료

## 테스트 데이터

- `data/헤리티지 역사와 과학 제58권 제4호(통권 제110권).pdf`
- 국립문화유산연구원 학술지 (한국어, 유물사진/표/차트/도면 포함)
- 논문 13편 포함된 통권 PDF (260페이지)

## 노트북 현황

### 1. `test_google_embedding.ipynb` - 페이지 단위 RAG (메인)
- PDF → 1페이지 PDF로 분할 → `application/pdf`로 직접 임베딩
- FAISS IndexFlatIP (코사인 유사도), 차원 3072
- task_type 분리: RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY
- 검색 히트 시 앞뒤 ±1 페이지 컨텍스트 포함 로직 구현 완료
- 검색 결과에 텍스트 + 페이지 내 개별 이미지(캡션 매칭) 표시
- LLM 호출은 제거된 상태 (검색 결과만 확인용)

### 2. `test_bge_m3.ipynb` - Visualized BGE-M3 테스트
- PDF → 페이지별 PNG 이미지 → BGE-M3로 임베딩
- `visual_bge` 설치 이슈 있음 (ModuleNotFoundError)
- 설치 명령: `pip install "visual_bge @ git+https://github.com/FlagOpen/FlagEmbedding.git#subdirectory=research/visual_bge"`
- 가중치: `weights/Visualized_m3.pth` (1.6GB, huggingface_hub로 다운로드)
- LLM 호출은 제거된 상태

### 3. `test_image_search.ipynb` - 이미지 개별 검색
- PDF에서 개별 이미지 추출 → 각각 임베딩
- 텍스트→이미지 검색, 이미지→이미지 검색 모두 지원
- 추출 시 PNG 변환 처리 완료 (API 호환성)

### 4. `test_embedding_comparison.ipynb` - 임베딩 방식 비교 (완료)
- 동일 쿼리로 3가지 방식 비교: A.PDF바이트 / B.텍스트 / C.페이지이미지
- **결과**: 텍스트 임베딩이 압도적 (Top-1 평균: B=0.64 > C=0.45 > A=0.44)
- 단, 텍스트 쿼리 기준이라 텍스트가 유리한 건 당연
- PDF 바이트 임베딩은 속도도 가장 느리고 정확도도 가장 낮음

## 핵심 발견사항

1. **텍스트 쿼리 RAG에는 텍스트 임베딩이 가장 효과적** (유사도 0.64 vs 0.44)
2. **PDF 바이트 직접 임베딩은 장점 없음** — 느리고 정확도도 낮음
3. **이미지 개별 검색 시 컨텍스트 부재 문제** — 이미지만 임베딩하면 왜 그 이미지가 등장했는지(캡션, 주변 텍스트)를 모름

## 다음 세션에서 할 일

### `test_image_search.ipynb` 개선: 이미지+텍스트 멀티모달 임베딩
- 현재 문제: 이미지만 임베딩하면 컨텍스트(캡션, 주변 설명)가 없어서 검색 품질 낮음
- 해결: 이미지 + 캡션/주변 텍스트를 하나의 Content로 묶어서 임베딩
```python
# 이미지 + 컨텍스트 텍스트를 함께 임베딩
contents = types.Content(
    parts=[
        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        types.Part.from_text(text="그림 7. 출토 유물의 XRF 분석 결과. 본 연구에서는...")
    ]
)
```
- 캡션 추출 로직은 `test_google_embedding.ipynb`의 `extract_page_images_with_captions()`에 이미 구현됨
  - 이미지 bbox 아래 50px 이내 텍스트에서 "그림/Figure/표/사진" 등 캡션 패턴 매칭
  - 폴백: 페이지 텍스트에서 순서대로 매칭
- 추가로 캡션뿐 아니라 주변 문단도 가져오면 더 좋을 수 있음

### 기타 TODO
- [ ] `test_google_embedding.ipynb` 임베딩 방식을 텍스트로 전환 검토 (비교 결과 기반)
- [ ] 임베딩 결과 FAISS 디스크 캐싱 (`faiss.write_index()`)
- [ ] BGE-M3 설치 문제 해결 후 실행

## gemini-embedding-2-preview 핵심 스펙

- **입력**: 텍스트, 이미지(PNG/JPEG, 최대 6장), PDF(최대 6페이지), 비디오, 오디오
- **출력 차원**: 128~3072 (권장: 768, 1536, 3072)
- **토큰 한도**: 8,192
- **task_type**: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, QUESTION_ANSWERING, FACT_VERIFICATION, CODE_RETRIEVAL_QUERY
- **API**: Google AI (Gemini API), `GOOGLE_API_KEY`로 사용 (Vertex AI 불필요)
- **멀티모달 임베딩**: 하나의 Content에 이미지+텍스트를 함께 넣어 임베딩 가능

## Python SDK 사용 예시

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_KEY")

# 텍스트 임베딩
result = client.models.embed_content(
    model='gemini-embedding-2-preview',
    contents='검색할 텍스트',
    config=types.EmbedContentConfig(output_dimensionality=3072, task_type="RETRIEVAL_QUERY"),
)

# 이미지 + 텍스트 멀티모달 임베딩
result = client.models.embed_content(
    model='gemini-embedding-2-preview',
    contents=types.Content(
        parts=[
            types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
            types.Part.from_text(text='캡션 및 주변 텍스트'),
        ]
    ),
    config=types.EmbedContentConfig(output_dimensionality=3072, task_type="RETRIEVAL_DOCUMENT"),
)
```

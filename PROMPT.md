# Gemini Embedding 2 - 멀티모달 RAG 테스트 프로젝트

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
- 논문 13편 포함된 통권 PDF

## 현재 작성된 코드

### 1. `test_google_embedding.ipynb` - Gemini Embedding 2 테스트
- 방법 A: PDF 직접 임베딩 (`application/pdf` MIME 타입)
- 방법 B: PDF → 페이지별 PNG 이미지 → 개별 임베딩 (메인 파이프라인)
- FAISS IndexFlatIP (코사인 유사도)로 벡터 검색
- `gemini-2.5-flash`로 RAG 답변 생성

### 2. `test_bge_m3.ipynb` - Visualized BGE-M3 테스트
- 동일한 파이프라인이지만 임베딩 모델만 BGE-M3로 교체
- 로컬 GPU 추론 (Visualized_m3.pth 가중치 필요)
- 답변 LLM은 동일하게 `gemini-2.5-flash` 사용

## gemini-embedding-2-preview 핵심 스펙

- **입력**: 텍스트, 이미지(PNG/JPEG, 최대 6장), PDF(최대 6페이지), 비디오, 오디오
- **출력 차원**: 128~3072 (권장: 768, 1536, 3072)
- **토큰 한도**: 8,192
- **task_type**: SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, CODE_RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION
- **API**: Google AI (Gemini API), `GOOGLE_API_KEY`로 사용 (Vertex AI 불필요)

## 합의된 청킹 전략

- **임베딩 단위**: 1페이지씩 (정밀한 검색을 위해)
- **페이지 경계 끊김 보완**: 검색 히트된 페이지의 앞뒤 1페이지를 함께 LLM에 전달
  - 예: 5페이지가 히트 → LLM에 4, 5, 6페이지 전달
- PDF 최대 6페이지 제한이 있지만, 6페이지씩 청킹하면 LLM에 너무 많은 정보가 들어가므로 1페이지 단위가 적절

## 다음 단계 (TODO)

- [ ] 두 노트북을 실제 실행하여 결과 비교
- [ ] 검색 히트 시 앞뒤 페이지 포함 로직 구현 (현재 코드에는 아직 미반영)
- [ ] task_type 파라미터 활용 (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY 분리)
- [ ] 임베딩 결과 캐싱 (매번 재생성 방지)
- [ ] 두 모델 간 검색 정확도/속도 비교 리포트 작성

## Python SDK 사용 예시

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_KEY")

# 텍스트 임베딩
result = client.models.embed_content(
    model='gemini-embedding-2-preview',
    contents='검색할 텍스트',
    config=types.EmbedContentConfig(output_dimensionality=768),
)

# 이미지 임베딩
result = client.models.embed_content(
    model='gemini-embedding-2-preview',
    contents=types.Content(
        parts=[types.Part.from_bytes(data=img_bytes, mime_type='image/png')]
    ),
    config=types.EmbedContentConfig(output_dimensionality=768),
)

# PDF 직접 임베딩 (최대 6페이지)
result = client.models.embed_content(
    model='gemini-embedding-2-preview',
    contents=types.Content(
        parts=[types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf')]
    ),
    config=types.EmbedContentConfig(output_dimensionality=768),
)
```

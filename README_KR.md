<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype - 단일세포 RNA 시퀀싱 데이터의 다중 대형 언어 모델 세포 유형 주석 프레임워크" width="300"/>
</div>

<div align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a>
</div>

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=mLLMCelltype%3A%20단일세포%20RNA%20시퀀싱%20데이터의%20세포%20유형%20주석을%20위한%20다중%20LLM%20합의%20프레임워크%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="Tweet"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="Stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="Forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-채팅방참여-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="Issues">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="Release">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
</div>

# mLLMCelltype: 단일세포 RNA 시퀀싱을 위한 다중 대형 언어 모델 합의 프레임워크

mLLMCelltype는 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터에서 정확하고 신뢰할 수 있는 세포 유형 주석을 위한 고급 반복적 다중 LLM 합의 프레임워크입니다. 여러 대형 언어 모델(OpenAI GPT-4o/4.1, Anthropic Claude-3.7/3.5, Google Gemini-2.0, X.AI Grok-3, DeepSeek-V3, Alibaba Qwen2.5, Zhipu GLM-4, MiniMax, Stepfun, OpenRouter 등)의 집단 지능을 활용함으로써, 이 프레임워크는 주석 정확도를 크게 향상시키면서 생물정보학 및 전산 생물학 연구를 위한 투명한 불확실성 정량화를 제공합니다.

## 초록

mLLMCelltype는 단일세포 전사체학 분석을 위한 오픈 소스 도구로, 여러 대형 언어 모델을 사용하여 유전자 발현 데이터에서 세포 유형을 식별합니다. 이 소프트웨어는 여러 모델이 동일한 데이터를 분석하고 그들의 예측을 결합하는 합의 방식을 구현하여, 오류를 줄이고 불확실성 지표를 제공합니다. mLLMCelltype는 Scanpy와 Seurat과 같은 인기 있는 단일세포 분석 플랫폼과 통합되어 연구자가 기존 생물정보학 작업흐름에 통합할 수 있습니다. 일부 전통적인 방법과 달리, 주석을 위한 참조 데이터셋이 필요하지 않습니다.

## 목차
- [뉴스](#뉴스)
- [주요 기능](#주요-기능)
- [최신 업데이트](#최신-업데이트)
- [디렉토리 구조](#디렉토리-구조)
- [설치](#설치)
- [사용 예제](#사용-예제)
- [시각화 예제](#시각화-예제)
- [인용](#인용)
- [기여하기](#기여하기)

🌟 **GitHub 스타 히스토리**

<div align="center">
  <a href="https://star-history.com/#cafferychen777/mLLMCelltype">
    <img src="https://api.star-history.com/svg?repos=cafferychen777/mLLMCelltype&type=Date" alt="Star History Chart" width="600"/>
  </a>
  <p><em>그림 1: 시간에 따른 커뮤니티 채택과 성장을 보여주는 mLLMCelltype GitHub 스타 히스토리.</em></p>
</div>

## 뉴스

🚀 **웹 애플리케이션 출시 (2025-06-18)**

mLLMCelltype 웹 애플리케이션 출시를 발표하게 되어 기쁩니다! 이제 설치 없이 웹 브라우저를 통해 직접 mLLMCelltype의 강력한 세포 유형 주석 기능에 액세스할 수 있습니다.

**✨ 주요 기능:**
- **사용하기 쉬운 인터페이스**: scRNA-seq 데이터를 업로드하고 몇 분 내에 주석을 얻으세요
- **다중 LLM 합의**: GPT-4, Claude, Gemini 등 다양한 AI 모델 중에서 선택
- **실시간 처리**: 실시간 업데이트로 주석 진행 상황 모니터링
- **다양한 내보내기 형식**: CSV, TSV, Excel 또는 JSON 형식으로 결과 다운로드
- **설정 불필요**: 패키지 설치 없이 즉시 주석 시작

**🌐 웹 앱 액세스**: [https://mllmcelltype.com](https://mllmcelltype.com)

**⚠️ 베타 테스트 단계**: 웹 애플리케이션은 현재 베타 테스트 중입니다. 플랫폼 개선을 위해 피드백과 제안을 환영합니다. [GitHub Issues](https://github.com/cafferychen777/mLLMCelltype/issues) 또는 [Discord 커뮤니티](https://discord.gg/pb2aZdG4)를 통해 문제를 신고하거나 경험을 공유해 주세요.

**📢 중요: Gemini 모델 마이그레이션 (2025-06-02)**

Google은 여러 Gemini 1.5 모델을 중단했으며 2025년 9월 24일에 더 많은 모델을 중단할 예정입니다:
- **이미 중단됨**: Gemini 1.5 Pro 001, Gemini 1.5 Flash 001
- **2025년 9월 24일 중단 예정**: Gemini 1.5 Pro 002, Gemini 1.5 Flash 002, Gemini 1.5 Flash-8B -001

**권장 마이그레이션**: 더 나은 성능과 지속적인 지원을 위해 `gemini-2.0-flash` 또는 `gemini-2.0-flash-lite`를 사용하세요. 별칭 `gemini-1.5-pro`와 `gemini-1.5-flash`는 -002 버전을 가리키므로 2025년 9월 24일까지 계속 작동합니다.

🎉 **2025년 4월**: 저희는 프리프린트 발표 후 단 2주만에 mLLMCelltype가 GitHub에서 200개의 별표를 넘었다는 사실을 기쁘게 발표합니다! 또한 다양한 미디어와 콘텐츠 크리에이터들의 많은 보도를 보았습니다. 별표, 공유, 기여를 통해 이 프로젝트를 지원해주신 모든 분들께 진심어린 감사를 드립니다. 여러분의 열정이 mLLMCelltype의 지속적인 개발과 개선을 이끄는 원동력입니다.

## 주요 기능

- **다중 LLM 합의 아키텍처**: 다양한 LLM의 집단 지능을 활용하여 단일 모델의 한계와 편향을 극복
- **구조화된 심의 과정**: LLM이 여러 차례의 협력적 토론을 통해 추론을 공유하고, 증거를 평가하며, 주석을 개선할 수 있게 함
- **투명한 불확실성 정량화**: 전문가 검토가 필요한 모호한 세포 집단을 식별하기 위한 정량적 지표(합의 비율 및 섀넌 엔트로피) 제공
- **환각 감소**: 모델 간 심의를 통해 비판적 평가를 통해 부정확하거나 근거 없는 예측을 적극적으로 억제
- **입력 노이즈에 강인함**: 집단적 오류 수정을 통해 불완전한 마커 유전자 목록에서도 높은 정확도 유지
- **계층적 주석 지원**: 부모-자식 일관성을 갖춘 다중 해상도 분석을 위한 선택적 확장
- **참조 데이터셋 불필요**: 사전 훈련이나 참조 데이터 없이 정확한 주석 수행
- **완전한 추론 체인**: 투명한 의사 결정을 위해 전체 심의 과정 문서화
- **원활한 통합**: 표준 Scanpy/Seurat 워크플로우 및 마커 유전자 출력과 직접 작동
- **모듈식 설계**: 새로운 LLM이 사용 가능해지면 쉽게 통합 가능

## 최신 업데이트

### v1.2.3 (2025-05-10)

#### 버그 수정
- API 응답이 NULL이거나 유효하지 않을 때 합의 확인에서 오류 처리 수정
- OpenRouter API 오류 응답에 대한 오류 로깅 개선
- check_consensus 함수에 견고한 NULL 및 타입 확인 추가

#### 개선사항
- OpenRouter API 오류에 대한 향상된 오류 진단
- API 오류 메시지 및 응답 구조의 자세한 로깅 추가
- 예상치 못한 API 응답 형식 처리 시 견고성 향상

### v1.2.2 (2025-05-09)

#### 버그 수정
- API 응답 처리 중 발생하는 '비문자 인수' 오류 수정
- check_consensus 함수에서 캐릭터 벡터와 NULL 값의 적절한 처리 보장
- API 응답 파싱 중 예상치 못한 타입에 대한 오류 처리 강화

#### 개선사항
- Python에서 R 함수 호출 시 인수 변환 개선
- API 응답 검증에 더 견고한 타입 확인 추가

### v1.2.1 (2025-05-08)

#### 버그 수정
- Python 환경에서 캐시 디렉토리 생성 문제 수정
- 로그 파일 권한 문제 및 디렉토리 접근 오류 해결
- Windows 시스템에서 파일 경로 처리 개선

#### 개선사항
- 크로스 플랫폼 호환성 향상
- 더 상세한 오류 메시지 및 디버깅 정보 제공

### v1.2.0 (2025-05-07)

#### 새로운 기능
- **단일 무료 OpenRouter 모델 지원**: 이제 Llama-3.1-405B-free, Mistral-large-free, Qwen-QwQ-32B-free와 같은 OpenRouter의 무료 모델을 사용할 수 있습니다
- **향상된 OpenRouter 통합**: 통합 API를 통해 여러 제공업체의 모델에 간편하게 액세스
- **개선된 모델 목록**: 각 모델의 가용성과 요금 정보가 포함된 포괄적인 모델 목록

#### 개선사항
- OpenRouter API 호출에 대한 향상된 오류 처리
- 무료 모델 식별을 위한 더 나은 모델 이름 구문 분석
- 향상된 API 키 관리 및 검증

### v1.1.5 (2025-05-01)

#### 새로운 기능
- **고급 합의 구성**: `consensus_check_model` 매개변수로 합의 확인을 위한 특정 모델 지정 가능
- **자동 폴백 선택**: 지정된 합의 모델이 실패할 경우 자동 폴백 시스템
- **향상된 불확실성 정량화**: 개선된 Shannon 엔트로피 및 합의 비율 계산

#### 개선사항
- 모델 간 의견 불일치에 대한 더 나은 처리
- 복잡한 조직 유형에 대한 향상된 성능
- 향상된 문서화 및 예제

### 지원되는 모델

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([API 키](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([API 키](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([API 키](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([API 키](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([API 키](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([API 키](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([API 키](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([API 키](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([API 키](https://accounts.x.ai/))
- **OpenRouter**: 단일 API로 여러 모델에 액세스 ([API 키](https://openrouter.ai/keys))
  - OpenAI, Anthropic, Meta, Google, Mistral 등의 모델 지원
  - 형식: 'provider/model-name' (예: 'openai/gpt-4o', 'anthropic/claude-3-opus')

## 디렉토리 구조

- `R/`: R 언어 인터페이스 및 구현
- `python/`: Python 인터페이스 및 구현

## 설치

### R 버전

```r
# GitHub에서 설치
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Python 버전

```bash
# PyPI에서 설치
pip install mllmcelltype

# 또는 GitHub에서 설치 (하위 디렉토리 매개변수 참고)
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

#### 의존성에 대한 중요 사항

mLLMCelltype는 서로 다른 LLM 제공업체 라이브러리가 선택적 의존성인 모듈식 설계를 사용합니다. 사용할 모델에 따라 해당 패키지를 설치해야 합니다:

```bash
# OpenAI 모델 사용 시 (GPT-4o 등)
pip install "mllmcelltype[openai]"

# Anthropic 모델 사용 시 (Claude)
pip install "mllmcelltype[anthropic]"

# Google 모델 사용 시 (Gemini)
pip install "mllmcelltype[gemini]"

# 모든 선택적 의존성을 한 번에 설치
pip install "mllmcelltype[all]"
```

`ImportError: cannot import name 'genai' from 'google'`와 같은 오류가 발생하면 해당 제공업체 패키지를 설치해야 합니다. 예를 들어:

```bash
# Google Gemini 모델용
pip install google-genai
```

## 사용 예제

### 단일 무료 OpenRouter 모델 사용

하나의 모델로 더 간단한 접근 방식을 선호하는 사용자를 위해, OpenRouter를 통한 Microsoft MAI-DS-R1 무료 모델이 우수한 결과를 제공합니다:

```python
import os
from mllmcelltype import annotate_clusters

# 참고: 로깅이 자동으로 구성됩니다

# OpenRouter API 키 설정
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# 각 클러스터의 마커 유전자 정의
marker_genes = {
    "0": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T 세포
    "1": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B 세포
    "2": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # 단핵구
}

# Microsoft MAI-DS-R1 무료 모델을 사용한 주석
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider='openrouter',
    model='microsoft/mai-ds-r1:free'  # 무료 모델
)

# 주석 출력
for cluster, annotation in annotations.items():
    print(f"클러스터 {cluster}: {annotation}")
```

이 접근 방식은 빠르고 정확하며 API 크레딧이 필요하지 않아, 빠른 분석이나 API 액세스가 제한적일 때 이상적입니다.

### AnnData 객체에서 마커 유전자 추출

Scanpy와 AnnData 객체를 사용하는 경우, `rank_genes_groups` 결과에서 직접 마커 유전자를 쉽게 추출할 수 있습니다:

```python
import scanpy as sc

# 차등 발현 분석 수행 (아직 수행하지 않은 경우)
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# 마커 유전자를 사전 형식으로 추출
marker_genes = {}
for cluster in adata.obs['leiden'].cat.categories:
    genes = [adata.uns['rank_genes_groups']['names'][cluster][i] for i in range(10)]
    marker_genes[cluster] = genes

print("추출된 마커 유전자:", marker_genes)
```

## 빠른 시작

### R 사용 예시

> **참고**: 더 자세한 R 튜토리얼 및 문서는 [mLLMCelltype 문서 웹사이트](https://cafferyang.com/mLLMCelltype/)를 방문하세요.

```r
library(mLLMCelltype)
library(Seurat)

# 마커 유전자 목록 준비
markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# 세포 유형 주석 수행
consensus_results <- interactive_consensus_annotation(
  input = markers,
  tissue_name = "human PBMC",
  models = c("gpt-4o", "claude-3-7-sonnet-20250219", "gemini-2.0-pro"),
  api_keys = list(
    openai = "your_openai_api_key",
    anthropic = "your_anthropic_api_key",
    gemini = "your_gemini_api_key"
  ),
  top_gene_count = 10
)

# 결과 확인
print(consensus_results$final_annotations)

# Seurat 객체에 주석 추가
current_clusters <- as.character(Idents(seurat_obj))
cell_types <- as.character(current_clusters)
for (cluster_id in names(consensus_results$final_annotations)) {
  cell_types[cell_types == cluster_id] <- consensus_results$final_annotations[[cluster_id]]
}
seurat_obj$cell_type <- cell_types
```

### Python 사용 예시

```python
# 단일세포 RNA-seq 데이터에서 mLLMCelltype를 사용한 세포 유형 주석 예제
import scanpy as sc
import mllmcelltype as mct

# 단일세포 RNA-seq 데이터를 AnnData 형식으로 로드
adata = sc.read_h5ad("your_data.h5ad")  # 자신의 scRNA-seq 데이터로 교체

# 세포 집단 식별을 위한 네트워크 구축 및 Leiden 클러스터링 수행
sc.pp.neighbors(adata)  # 유사한 세포를 연결하는 KNN 그래프 구축
sc.tl.leiden(adata)     # 커뮤니티 검출 알고리즘을 사용하여 세포 집단 식별

# 각 클러스터의 마커 유전자 식별을 위한 차별 발현 분석 수행
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")  # Wilcoxon 랭크합 검정을 사용한 마커 검출

# Scanpy의 마커 유전자 결과를 mLLMCelltype에서 사용할 수 있는 형식으로 변환
markers_dict = mct.utils.convert_scanpy_markers(adata)  # 마커 유전자 사전 형태로 변환

# 다중 대형 언어 모델을 사용한 세포 유형 주석 수행
consensus_results = mct.annotate.interactive_consensus_annotation(
    input=markers_dict,                                             # 마커 유전자 사전
    tissue_name="human PBMC",                                      # 조직 정보 지정(주석 정확도 향상)
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-1.5-pro"],  # 다양한 대형 언어 모델 사용
    api_keys={                                                     # 각 LLM 제공업체의 API 키 설정
        "openai": "your_openai_api_key",                             # OpenAI GPT-4o API 키
        "anthropic": "your_anthropic_api_key",                       # Anthropic Claude API 키
        "gemini": "your_gemini_api_key"                              # Google Gemini API 키
    },
    top_gene_count=10                                              # 각 클러스터마다 사용할 상위 마커 유전자 수
)

# 주석 결과를 AnnData 객체에 추가하여 후속 분석 및 시각화 준비
adata.obs["cell_type"] = adata.obs["leiden"].map(
    lambda x: consensus_results["final_annotations"].get(x, "Unknown")
)
```

### 고급 합의 구성: 합의 확인 모델 지정

`consensus_check_model` 매개변수(R) / `consensus_model` 매개변수(Python)를 사용하면 합의 확인 및 토론 조정에 사용할 LLM 모델을 지정할 수 있습니다. 이 매개변수는 합의 확인 모델이 다음을 수행하기 때문에 합의 주석의 정확성에 **중요**합니다:

1. 서로 다른 세포 유형 주석 간의 의미적 유사성 평가
2. 합의 지표 계산 (비율 및 엔트로피)
3. 논란이 있는 클러스터에 대한 모델 간 토론 조정 및 통합
4. 모델이 의견을 달리할 때 최종 결정

**⚠️ 중요: 주석 품질에 직접적인 영향을 미치므로 합의 확인에는 사용 가능한 가장 우수한 모델을 사용할 것을 강력히 권장합니다.**

#### 합의 확인을 위한 권장 모델 (성능 순으로 순위)

1. **Anthropic Claude 모델** (최고 권장)
   - `claude-opus-4-20250514` - 최고 전체 성능
   - `claude-3-7-sonnet-20250219` - 성능과 속도의 우수한 균형
   - `claude-3-5-sonnet-20241022` - 빠른 응답으로 좋은 성능

2. **OpenAI 모델**
   - `o1` / `o1-pro` - 고급 추론 능력
   - `gpt-4o` - 다양한 세포 유형에서 강력한 성능
   - `gpt-4.1` - 최신 GPT-4 변형

3. **Google Gemini 모델**
   - `gemini-2.5-pro` - 최고 계층 성능
   - `gemini-2.0-flash` - 빠른 처리로 좋은 성능

4. **기타 고성능 모델**
   - `deepseek-r1` / `deepseek-reasoner` - 강력한 추론 능력
   - `qwen-max-2025-01-25` - 과학적 맥락에서 우수
   - `grok-3-latest` - 고급 언어 이해

#### Python 패키지 사용법

```python
# 예제 1: 합의 확인을 위한 최고 사용 가능 모델 사용 (권장)
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-2.0-flash", "qwen-max-2025-01-25"],
    consensus_model="claude-opus-4-20250514",  # 가장 우수한 모델 사용
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# 예제 2: 고성능 모델과 함께 사전 형식 사용
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="mouse",
    tissue="liver",
    models=["gpt-4o", "gemini-2.0-flash", "qwen-max-2025-01-25"],
    consensus_model={"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

#### 합의 모델 선택 모범 사례

1. **비용보다 정확성 우선**: 합의 확인 모델은 최종 주석 결정에 중요한 역할을 합니다. 여기서 덜 우수한 모델을 사용하면 전체 주석 프로세스가 손상될 수 있습니다.

2. **모델 가용성**: 선택한 합의 모델에 대한 API 액세스가 있는지 확인하세요. 주 선택이 사용할 수 없는 경우 시스템이 폴백 모델을 사용합니다.

3. **일관성**: 일관된 평가 기준을 보장하기 위해 프로젝트 내의 모든 합의 확인에 동일한 고성능 모델을 사용하세요.

4. **복잡한 조직**: 도전적인 조직(예: 뇌, 면역 시스템)의 경우 Claude Opus, O1 또는 Gemini 2.5 Pro와 같은 가장 고급 모델 사용을 고려하세요.

#### 합의 확인에서 모델 품질이 중요한 이유

합의 확인 모델은 다음을 수행해야 합니다:
- 서로 다른 세포 유형 이름 간의 의미적 유사성을 정확히 평가 (예: "T 림프구"와 "T 세포"가 동일한 세포 유형을 지칭한다는 것을 인식)
- 생물학적 맥락과 계층적 관계 이해
- 여러 모델의 토론을 종합하여 정확한 결론 도출
- 하위 분석을 위한 신뢰할 수 있는 신뢰도 지표 제공

이러한 중요한 작업에 능력이 떨어지는 모델을 사용하면 다음과 같은 결과를 초래할 수 있습니다:
- 논란이 있는 클러스터의 오식별
- 부정확한 합의 계산
- 모델 간 의견 불일치의 부적절한 해결
- 궁극적으로 덜 정확한 세포 유형 주석

## 시각화 예제

### 세포 유형 주석 시각화

다음은 세포 유형 주석과 함께 불확실성 지표(합의 비율 및 섀넌 엔트로피)를 보여주는 mLLMCelltype와 SCpubr로 생성된 출판 품질 시각화의 예입니다:

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*그림: 왼쪽 패널은 UMAP 투영에서 세포 유형 주석을 보여줍니다. 중간 패널은 노란색-녹색-파란색 그라디언트를 사용하여 합의 비율을 표시합니다(진한 파란색은 LLM 간의 강한 합의를 나타냅니다). 오른쪽 패널은 주황색-빨간색 그라디언트를 사용하여 섀넌 엔트로피를 보여줍니다(진한 빨간색은 낮은 불확실성, 밝은 주황색은 높은 불확실성을 나타냅니다).*

### 마커 유전자 시각화

mLLMCelltype에는 이제 합의 주석 워크플로우와 원활하게 통합되는 향상된 마커 유전자 시각화 기능이 포함되어 있습니다:

```r
# 필요한 라이브러리 로드
library(mLLMCelltype)
library(Seurat)
library(ggplot2)

# 합의 주석 실행 후
consensus_results <- interactive_consensus_annotation(
  input = markers_df,
  tissue_name = "human PBMC",
  models = c("anthropic/claude-3.5-sonnet", "openai/gpt-4o"),
  api_keys = list(openrouter = "your_api_key")
)

# Seurat를 사용한 마커 유전자 시각화 생성
# Seurat 객체에 합의 주석 추가
pbmc_data$cell_type_consensus <- consensus_results$final_annotations[Idents(pbmc_data)]

# 마커 유전자의 dotplot 생성
DotPlot(pbmc_data, 
        features = top_markers,
        group.by = "cell_type_consensus") + 
  RotatedAxis()

# 마커 유전자의 히트맵 생성
DoHeatmap(pbmc_data, 
          features = top_markers,
          group.by = "cell_type_consensus")
```

**마커 유전자 시각화의 주요 특징:**

- **DotPlot**: 각 유전자를 발현하는 세포의 백분율(점 크기)과 평균 발현 수준(색상 강도) 표시
- **DoHeatmap**: 유전자의 계층적 클러스터링과 함께 스케일된 발현 값 표시
- **표준 Seurat 함수**: 익숙한 Seurat 시각화 함수를 사용하여 일관된 워크플로우 유지
- **원활한 통합**: 합의 주석 결과를 Seurat 객체와 직접 작동

자세한 지침과 고급 사용자 정의 옵션은 [시각화 가이드](R/vignettes/06-visualization-guide.html)를 참조하세요.

## 불확실성 시각화

mLLMCelltype는 주석 불확실성을 정량화하기 위한 두 가지 지표를 제공합니다:

1. **합의 비율**: 특정 예측에 동의하는 모델의 비율
2. **섀넌 엔트로피**: 예측 분포의 불확실성 측정

이러한 지표를 시각화하려면:

```r
library(Seurat)
library(ggplot2)
library(cowplot)
library(SCpubr)

# 불확실성 지표 계산
uncertainty_metrics <- calculate_uncertainty_metrics(consensus_results)

# Seurat 객체에 불확실성 지표 추가
current_clusters <- as.character(Idents(pbmc))
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# 세포 유형 주석 시각화
p1 <- SCpubr::do_DimPlot(sample = pbmc,
                       group.by = "cell_type",
                       label = TRUE,
                       repel = TRUE,
                       pt.size = 0.1) +
      ggtitle("Cell Type Annotations") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# 합의 비율 시각화
p2 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "consensus_proportion",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("yellow", "green", "blue"),
                         limits = c(min(pbmc$consensus_proportion),  # 최소값 설정
                                   max(pbmc$consensus_proportion)),  # 최대값 설정
                         na.value = "lightgrey") +  # 결측값 색상
      ggtitle("Consensus Proportion") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# 엔트로피 시각화
p3 <- SCpubr::do_FeaturePlot(sample = pbmc,
                          features = "entropy",
                          pt.size = 0.1) +
      scale_color_gradientn(colors = c("darkred", "red", "orange"),
                         limits = c(min(pbmc$entropy),  # 최소값 설정
                                   max(pbmc$entropy)),  # 최대값 설정
                         na.value = "lightgrey") +  # 결측값 색상
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# 동일한 너비로 플롯 결합
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

#### CSV 입력 예제

Seurat을 사용하지 않고 CSV 파일에서 직접 mLLMCelltype를 사용할 수도 있으며, 이는 마커 유전자가 이미 CSV 형식으로 사용 가능한 경우에 유용합니다:

```r
# 최신 버전의 mLLMCelltype 설치
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# 필요한 패키지 로드
library(mLLMCelltype)

# 캐시 및 로그 디렉토리 생성
cache_dir <- "path/to/your/cache"
log_dir <- "path/to/your/logs"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

# CSV 파일 내용 읽기
markers_file <- "path/to/your/markers.csv"
file_content <- readLines(markers_file)

# 헤더 행 건너뛰기
data_lines <- file_content[-1]

# 데이터를 리스트 형식으로 변환, 숫자 인덱스를 키로 사용
marker_genes_list <- list()
cluster_names <- c()

# 먼저 모든 클러스터 이름 수집
for(line in data_lines) {
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  cluster_names <- c(cluster_names, parts[1])
}

# 그런 다음 숫자 인덱스가 있는 marker_genes_list 생성
for(i in 1:length(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # 첫 부분은 클러스터 이름
  cluster_name <- parts[1]

  # 인덱스를 키로 사용 (0-based 인덱스, Seurat과 호환)
  cluster_id <- as.character(i - 1)

  # 나머지 부분은 유전자
  genes <- parts[-1]

  # NA와 빈 문자열 필터링
  genes <- genes[!is.na(genes) & genes != ""]

  # marker_genes_list에 추가
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# API 키 설정
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# 콘센서스 주석 실행
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # 예: "human heart"
    models = c("gemini-2.0-flash",
              "gemini-1.5-pro",
              "qwen-max-2025-01-25",
              "grok-3-latest",
              "anthropic/claude-3-7-sonnet-20250219",
              "openai/gpt-4o"),
    api_keys = api_keys,
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 3,
    cache_dir = cache_dir,
    log_dir = log_dir
  )

# 결과 저장
saveRDS(consensus_results, "your_results.rds")

# 결과 요약 출력
cat("\n결과 요약:\n")
cat("사용 가능한 필드:", paste(names(consensus_results), collapse=", "), "\n\n")

# 최종 주석 출력
cat("최종 세포 유형 주석:\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**CSV 형식 관련 참고사항**:
- CSV 파일의 첫 번째 열에는 어떤 값도 사용할 수 있습니다(클러스터 이름, 0,1,2,3 또는 1,2,3,4와 같은 숫자 시퀀스 등), 이 값들은 인덱스로 사용됩니다
- 첫 번째 열의 값들은 참조용으로만 사용되며, LLM 모델에 전달되지 않습니다
- 다음 열에는 각 클러스터의 마커 유전자를 포함해야 합니다
- 패키지에는 고양이 심장 조직용 예제 CSV 파일이 포함되어 있습니다: `inst/extdata/Cat_Heart_markers.csv`

CSV 파일 구조 예제:
```
cluster,gene
Fibroblasts,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
Cardiomyocytes,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
Endothelial cells,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
T cells,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

다음 코드를 사용하여 R 스크립트에서 예제 데이터에 액세스할 수 있습니다:
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### 단일 LLM 모델 사용

API 키가 하나만 있거나 특정 LLM 모델을 사용하고 싶은 경우, `annotate_cell_types()` 함수를 사용할 수 있습니다:

```r
# 전처리된 Seurat 객체 로드
pbmc <- readRDS("your_seurat_object.rds")

# 각 클러스터의 마커 유전자 찾기
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# 지원되는 어떤 공급업체에서나 모델 선택
# 지원되는 모델 목록:
# - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: 단일 API로 여러 모델에 액세스. 형식: 'provider/model-name'
#   - OpenAI 모델: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic 모델: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Meta 모델: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google 모델: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral 모델: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - 기타 모델: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# 단일 LLM 모델로 세포 유형 주석 실행
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # 조직 문맥 제공
  model = "claude-3-7-sonnet-20250219",  # 단일 모델 지정
  api_key = "your-anthropic-key",  # API 키 직접 제공
  top_gene_count = 10
)

# 결과 출력
print(single_model_results)

# Seurat 객체에 주석 추가
# single_model_results는 각 클러스터당 하나의 주석을 가진 문자 벡터
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# 결과 시각화
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("단일 LLM 모델로 주석된 세포 유형")
```

#### 다양한 모델 비교

다른 모델로 `annotate_cell_types()`를 여러 번 실행하여 다양한 모델의 주석을 비교할 수도 있습니다:

```r
# 다양한 모델을 사용하여 주석
models <- c("claude-3-7-sonnet-20250219", "gpt-4o", "gemini-2.0-pro", "qwen-max-2025-01-25", "grok-3")
api_keys <- c("your-anthropic-key", "your-openai-key", "your-google-key", "your-qwen-key", "your-xai-key")

# 각 모델에 대한 열 생성
for (i in 1:length(models)) {
  results <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = models[i],
    api_key = api_keys[i],
    top_gene_count = 10
  )

  # 모델에 기반한 열 이름 생성
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", models[i]))

  # Seurat 객체에 주석 추가
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results)-1)),
    to = results
  )
}

# 다양한 모델의 결과 시각화
p1 <- DimPlot(pbmc, group.by = "cell_type_claude_3_7_sonnet_20250219", label = TRUE) + ggtitle("Claude 3.7")
p2 <- DimPlot(pbmc, group.by = "cell_type_gpt_4o", label = TRUE) + ggtitle("GPT-4o")
p3 <- DimPlot(pbmc, group.by = "cell_type_gemini_2_0_pro", label = TRUE) + ggtitle("Gemini 2.0 Pro")
p4 <- DimPlot(pbmc, group.by = "cell_type_qwen_max_2025_01_25", label = TRUE) + ggtitle("Qwen Max")
p5 <- DimPlot(pbmc, group.by = "cell_type_grok_3", label = TRUE) + ggtitle("Grok-3")

# 플롯 결합
cowplot::plot_grid(p1, p2, p3, p4, p5, ncol = 3)
```

## 시각화 예시

다음은 mLLMCelltype과 SCpubr을 사용하여 작성된 출판 품질의 시각화 예시로, 세포 유형 주석과 불확실성 지표(합의 비율 및 섀넌 엔트로피)를 보여줍니다:

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*그림: 왼쪽 패널은 UMAP 투영에 세포 유형 주석을 보여줍니다. 중간 패널은 노란색-녹색-파란색 그라데이션을 사용하여 합의 비율을 표시합니다(더 깊은 파란색은 LLM 간의 더 강한 동의를 나타냄). 오른쪽 패널은 주황색-빨간색 그라데이션을 사용하여 섀넌 엔트로피를 보여줍니다(더 깊은 빨간색은 낮은 불확실성을, 더 밝은 주황색은 높은 불확실성을 나타냄).*

## 인용

연구에서 mLLMCelltype을 사용하는 경우 다음을 인용해 주세요:

```bibtex
@article{Yang2025.04.10.647852,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data},
  elocation-id = {2025.04.10.647852},
  year = {2025},
  doi = {10.1101/2025.04.10.647852},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2025/04/17/2025.04.10.647852},
  journal = {bioRxiv}
}
```

일반 텍스트 형식으로도 인용할 수 있습니다:

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. https://doi.org/10.1101/2025.04.10.647852

## 기여하기

커뮤니티의 기여을 환영하고 감사드립니다! mLLMCelltype에 기여할 수 있는 다양한 방법이 있습니다:

### 문제 보고

버그를 발견하거나, 기능 요청이 있거나, mLLMCelltype 사용에 관한 질문이 있는 경우 GitHub 저장소에서 [이슈를 열어주세요](https://github.com/cafferychen777/mLLMCelltype/issues). 버그를 보고할 때는 다음 내용을 포함해주세요:

- 문제에 대한 명확한 설명
- 문제를 재현하는 단계
- 예상 동작과 실제 동작
- 운영 체제 및 패키지 버전 정보
- 관련 코드 스니펙 또는 오류 메시지

### 풀 리퀘스트

코드 개선 또는 새로운 기능을 풀 리퀘스트를 통해 기여하는 것을 권장합니다:

1. 저장소를 포크하세요
2. 기능을 위한 새 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m '멋진 기능 추가'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. 풀 리퀘스트를 열어주세요

### 기여 영역

다음은 기여가 특히 가치 있는 영역입니다:

- 새로운 LLM 모델 지원 추가
- 문서 및 예제 개선
- 성능 최적화
- 새로운 시각화 옵션 추가
- 특정 세포 유형이나 조직에 대한 기능 확장
- 문서를 다양한 언어로 번역

### 코드 스타일

저장소의 기존 코드 스타일을 따라주세요. R 코드의 경우 일반적으로 [tidyverse 스타일 가이드](https://style.tidyverse.org/)를 따릅니다. Python 코드의 경우 [PEP 8](https://www.python.org/dev/peps/pep-0008/)을 따릅니다.

mLLMCelltype를 개선하는 데 도움을 주셔서 감사합니다!
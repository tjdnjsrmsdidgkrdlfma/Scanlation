# Scanlation

자기 소유의 계약(contract) 기반 만화 **OCR + 제자리 번역**(일본어 → 한국어):
만화 텍스트를 검출·deskew·인식·번역하는 FastAPI 서버 + 결과를 이미지 위에 그대로
오버레이하는 클린룸 MV3 브라우저 확장.

Crivella의 `ocr_translate` + `ocr_extension` 스택을 대체합니다. **정확도 최우선** —
만화의 진짜 병목은 *검출*이라, 엔진 계약이 회전 기하(4점 폴리곤 + 각도 + 마스크)를
품고, 파이프라인이 기울어진/세로 텍스트를 인식 전에 deskew합니다. 모든 엔진은 교체
가능한 플러그인입니다.

> 설계 근거 & 단계별 로드맵 전체: **[YOMU_DESIGN.md](YOMU_DESIGN.md)**

---

## 진행 상태

전체 파이프라인이 동작하며, 실제 만화 페이지로 **브라우저에서 검증 완료**(검출 + OCR +
오버레이, 로컬 http). 실제 한국어 번역은 코드·단위테스트 완료, 라이브 구동은 GPU 호스트의
LLM 백엔드만 붙이면 됩니다.

| 단계 | 내용 | 상태 |
|---|---|---|
| P0 | FastAPI 스켈레톤, handshake, CORS 허용 | ✅ |
| P1 | contracts · registry · state · cache · pipeline · **전체 wire 라우트** · dummy 엔진 | ✅ 테스트 |
| P2 | deskew 기하(OpenCV homography, 축정렬 fast-path) | ✅ 테스트 |
| P3 | `ctd` 검출기(comic-text-detector ONNX) + 마스크→회전quad 디코드 + `visualize.py` | ✅ 실페이지 검증 |
| P4 | `mangaocr` 인식기 | ✅ 실제 일본어 브라우저 검증 |
| P5 | `ollama` 번역기 (ROCm) | ✅ 단위테스트; 라이브는 호스트 |
| P5b | `llamacpp` 번역기 (OpenAI 호환; **Vulkan**/vllm/LM Studio) | ✅ 단위테스트 |
| P6 | 클린룸 **MV3 확장** ([extension/](extension/)) | ✅ 브라우저 검증 |
| P7 | Docker / ROCm + Vulkan 배포 | ⬜ 예정 |
| P8 | 튜닝(라틴 라벨 폴백 OCR, merge_px 스케일링, 세로쓰기) | ⬜ 예정 |

**end-to-end 검증됨:** 브라우저 → 서버 → `ctd` 검출 → deskew → `mangaocr` OCR →
`%` 위치 오버레이, 실제 862×1200 만화 페이지에서. 단위테스트 30개 통과
(`pytest -m "not slow"`); `ctd`/`mangaocr` 스모크는 `@slow`.

---

## 아키텍처

```
[ MV3 확장 ]  ──HTTP (md5 / box / lazy)──►  [ FastAPI 서버 ]
 이미지 발견                                 detect ─► deskew ─► recognize ─► translate
 + %-오버레이                                (ctd)     (geometry) (mangaocr)  (ollama|llamacpp)
```

- **서버** ([server/](server/)): FastAPI + uvicorn(단일 워커), async GPU 락 1개, 블로킹
  모델 작업은 threadpool. SQLite 결과 캐시 + 수동 번역 메모리(TM). 엔진은 entry_points +
  내장 fallback으로 발견, **첫 사용 시 lazy 인스턴스화**(그때 가중치 로드).
- **확장** ([extension/](extension/)): MV3, 번들러/npm 없음, 순수 ES. 자족형 content
  script(이미지 → base64 → md5 → lazy/work → 오버레이). 클린룸 MD5는 파이썬
  `hashlib.md5`와 **바이트 단위 일치** 검증됨.

### 와이어 계약 (구 클라이언트 drop-in; 바꾸지 말 것)
- `md5`는 **base64 문자열** 기준으로 계산(raw 바이트 아님) → 불일치 시 400.
- box는 `[x_min, y_min, x_max, y_max]`(클라는 `[l, b, r, t]`로 읽음).
- `POST /run_ocrtsl/`는 **lazy**(`{md5, options}` → 캐시 히트, 미스 시 non-2xx) 후
  **work**(`{md5, contents, options}`). 그 외: `/run_tsl/`, `/get_trans/`,
  `/set_manual_translation/`, `/set_models/`, `/set_lang/`, `/get_active_options/`,
  `/get_plugin_data/`, `/manage_plugins/`, 그리고 `GET /` handshake.

---

## 레포 구조

```
server/
  app/        contracts, geometry, pipeline, registry, cache, state, config, schemas, routes/
  plugins/    dummy/  detector_ctd/  recognizer_mangaocr/  translator_ollama/
              translator_llamacpp/  llm_prompt.py (공유 LLM 프롬프트)
  tools/      run_image.py, visualize.py
  tests/      단위 ~30개 + @slow 모델 스모크
  models/     CTD 가중치 (gitignore)        data/  sqlite 캐시+state (gitignore)
extension/
  manifest.json  popup.{html,css}  content.css  icons/
  src/content.js (md5+파이프라인+오버레이)  service-worker.js  popup.js
YOMU_DESIGN.md   전체 설계 / 핸드오프
```

---

## 빠른 시작

Python 3.11+, Node는 확장 md5를 건드릴 때만 필요. 의존성은 repo 루트 `.venv`(gitignore)에 —
**절대 전역 pip install 금지.**

```bash
python -m venv .venv
# Windows: ./.venv/Scripts/python -m pip install -e "./server[ctd,mangaocr,dev]"
# Linux:   source .venv/bin/activate && pip install -e "./server[ctd,mangaocr,dev]"
```

**모델 가중치는 첫 사용 시 자동 다운로드**됩니다(HF): CTD onnx(~95MB) + manga-ocr 모델 둘 다.
미리 두거나 오프라인이면: `server/models/ctd/`에 `.onnx` 배치 / `SCANLATION_CTD_MODEL=/path.onnx` /
`SCANLATION_CTD_URL`로 미러 지정.

### 서버 실행

```bash
cd server
# 빠른 스모크(모델 0): dummy 엔진
python -m uvicorn app.main:app --host 127.0.0.1 --port 4000
# 실엔진(CTD + manga-ocr는 CPU, ollama는 GPU):
SCANLATION_DEVICE=cpu SCANLATION_DETECTOR=ctd SCANLATION_RECOGNIZER=mangaocr \
SCANLATION_TRANSLATOR=ollama OLLAMA_MODEL=<your-model> \
python -m uvicorn app.main:app --host 0.0.0.0 --port 4000
```
첫 실엔진 요청은 느림(CTD ONNX + manga-ocr 모델 로드); 같은 이미지 재요청은 md5 캐시로 즉시.

### 확장 로드

`chrome://extensions` → 개발자 모드 → **압축해제 로드** → [extension/](extension/).
Firefox: `about:debugging` → 임시 부가 기능 로드 → [extension/manifest.json](extension/manifest.json).
그 다음: 페이지 열고 **F5**(content script는 로드 시 주입), 아이콘 → **Connect** →
**Enable on tab**.

> **크로스오리진 이미지**(예: pixiv `i.pximg.net`)는 임의 HTTPS 페이지에서 못 읽음
> (CORS + Referer 핫링크). 확실한 경로는 **로컬 페이지를 http로 서빙**(same-origin) —
> 설계의 `make_viewer.py` 흐름. 확장은 핫링크 없는 크로스오리진 사이트 커버용으로 이미지
> 페치를 서비스워커(host_permissions)로 우회하기도 함. 혼합 콘텐츠: `http://127.0.0.1:4000`은
> 허용됨(localhost는 보안 컨텍스트). 다른 호스트의 http 서버는 HTTPS 페이지에서 차단 →
> SSH 터널 사용.

---

## 엔진 & 플러그인

3개 역할, 각각 독립 선택(팝업 드롭다운, `state.json`에 영속, 또는 env):

| 역할 | 플러그인 | 기본값 |
|---|---|---|
| detector | `ctd`, `dummy` | dummy |
| recognizer | `mangaocr`, `dummy` | dummy |
| translator | `ollama`, `llamacpp`, `dummy` | dummy |

나머지를 `dummy`로 두면 실엔진 하나만 격리 검증 가능
(도구에선 `--engines ctd,dummy,dummy`, 또는 `set_models`).

**플러그인 추가:** `EngineBase` 상속, 역할 메서드(`detect`/`recognize`/`translate`) +
클래스 메타데이터 + `OPTION_SCHEMA` 구현, 그 다음 `app/registry.py`(`_BUILTIN`)와
`pyproject.toml` entry_points에 등록. `scanlation.<role>` entry-point 그룹을 선언하는
서드파티 패키지는 자동 발견됨.

### 번역 백엔드
- **`ollama`** → `POST /api/generate` (ollama 내부 llama.cpp로 ROCm). env:
  `OLLAMA_ENDPOINT`(`http://127.0.0.1:11434/api`), `OLLAMA_MODEL`.
- **`llamacpp`** → OpenAI `POST /v1/chat/completions` — 최신 AMD에서 ROCm이 불안할 때
  **Vulkan**(`llama-server`)용, 또는 임의 OpenAI 호환 서버. env:
  `LLAMACPP_ENDPOINT`(`http://127.0.0.1:8080`), `LLAMACPP_MODEL`. `<think>` 구간 제거.

둘 다 사용자 튜닝 시스템 프롬프트 + 템플릿([server/plugins/llm_prompt.py](server/plugins/llm_prompt.py))
공유: 번역만, OCR 오류 감안, 추론 한 문장.

---

## 설정 (env)

| 변수 | 기본 | 의미 |
|---|---|---|
| `SCANLATION_DEVICE` | `cpu` | `cpu` / `rocm` / `dml` provider 힌트(항상 CPU fallback) |
| `SCANLATION_DETECTOR` / `_RECOGNIZER` / `_TRANSLATOR` | `dummy` | 시작 시 역할별 엔진 |
| `SCANLATION_LANG_SRC` / `_DST` | `ja` / `ko` | 시작 언어 |
| `SCANLATION_BASE_DIR` | server/ | `data/`(캐시, state.json) 루트 |
| `SCANLATION_MODELS_DIR` | `<base>/models` | 가중치 루트 |
| `SCANLATION_CTD_MODEL` / `_CTD_URL` | — / HF | CTD `.onnx` 명시 경로 / 자동 다운로드 URL |
| `OLLAMA_ENDPOINT` / `OLLAMA_MODEL` | `…:11434/api` / — | ollama 백엔드 |
| `LLAMACPP_ENDPOINT` / `LLAMACPP_MODEL` | `…:8080` / `local` | llama.cpp/OpenAI 백엔드 |

---

## 테스트

```bash
cd server
../.venv/Scripts/python -m pytest -m "not slow"        # ~30개, 모델/GPU 불필요
../.venv/Scripts/python -m pytest -m slow              # CTD/manga-ocr 스모크(가중치 필요)
```

**검출 육안 확인**(정확도 핵심 루프 — 검출이 병목):
```bash
python tools/visualize.py page.jpg --detector ctd --out annotated.png   # 폴리곤 + 인덱스
python tools/run_image.py  page.jpg --engines ctd,mangaocr,dummy         # wire JSON
```
`visualize.py`는 `annotated.png` + deskew된 `crops/`를 저장 → 박스 위치와 crop이 똑바른지
눈으로 판단.

---

## 알려진 이슈 / 주의

- **gfx1200(RDNA4) ROCm**이 최대 리스크 — 사전빌드 휠 부재 가능. 완화: CTD + manga-ocr는
  **CPU**(충분히 빠름), GPU는 LLM 전용; ollama/ROCm 말썽이면 **`llamacpp` + Vulkan**.
- **CPU 속도**: manga-ocr이 영역마다 트랜스포머 1패스 → CPU에선 느리고 GPU에선 빠름.
  같은 페이지 재방문은 즉시(md5 캐시).
- **라틴/영숫자 라벨**(예: `正1L=1000ml`)은 manga-ocr(일본어 모델)이 오인식. 향후:
  라틴 비중 높은 영역을 폴백 OCR로 라우팅(P8).
- **pixiv 라이브**는 안 됨(크로스오리진 + Referer 핫링크) — 로컬 서빙 페이지 사용.
- `merge_px`(CTD 글자→말풍선 묶음)는 고정 기본값(13); 텍스트 크기에 따라 스케일해야 함(P8).
  오버레이 세로쓰기는 현재 가로 렌더.

---

## 배포 (리눅스 호스트)

**bare-metal 먼저**(빠른 반복); Docker(P7)는 동작한 뒤에.

```bash
git clone https://github.com/tjdnjsrmsdidgkrdlfma/Scanlation.git
cd Scanlation && python -m venv .venv && source .venv/bin/activate
pip install -e "./server[ctd,mangaocr]"
# CTD onnx + manga-ocr 모델은 첫 실행 시 자동 다운로드됨
cd server
SCANLATION_DEVICE=cpu SCANLATION_DETECTOR=ctd SCANLATION_RECOGNIZER=mangaocr \
SCANLATION_TRANSLATOR=ollama OLLAMA_MODEL=<your-model> \
python -m uvicorn app.main:app --host 0.0.0.0 --port 4000
```
ROCm 불안? `llama-server`(Vulkan) 띄우고 `SCANLATION_TRANSLATOR=llamacpp
LLAMACPP_ENDPOINT=http://127.0.0.1:8080`으로 전환. 확장은 호스트를 가리키게(HTTPS 페이지
혼합콘텐츠 피하려면 `127.0.0.1:4000`으로 SSH 터널).

---

## 다른 머신 / 새 세션에서 이어받기

1. 이 README(위 상태) + [YOMU_DESIGN.md](YOMU_DESIGN.md)(왜) 읽기.
2. `pip install -e "./server[ctd,mangaocr,dev]"` 후 `pytest -m "not slow"` green 확인.
3. 서버 파이프라인(P0–P6) 완료·검증됨; **다음은 P7(Docker/ROCm+Vulkan)** 이어서 P8 튜닝.
   솔로 프로젝트: `main`에 직접 커밋, 의존성은 `.venv`.

라이선스: 프로젝트 라이선스 미정(TBD). 트리는 **GPLv3-free**(클린룸; 엔진은 런타임 의존만) —
그대로 유지할 것.

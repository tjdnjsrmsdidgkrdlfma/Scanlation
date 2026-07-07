# Scanlation

자기 소유의 계약(contract) 기반 만화 **OCR + 제자리 번역**(일본어 → 한국어):
만화 텍스트를 검출·deskew·인식·번역하는 FastAPI 서버 + 결과를 이미지 위에 그대로
오버레이하는 클린룸 MV2 브라우저 확장(Firefox).

Crivella의 `ocr_translate` + `ocr_extension` 스택을 대체합니다. **정확도 최우선** —
만화의 진짜 병목은 *검출*이라, 엔진 계약이 회전 기하(4점 폴리곤 + 각도 + 마스크)를
품고, 파이프라인이 기울어진/세로 텍스트를 인식 전에 deskew합니다. 모든 엔진은 교체
가능한 플러그인입니다. (현 기본 검출기 `comic-text-and-bubble-detector`는 **축정렬** 박스만 내므로 지금은 deskew가
회전을 실제로 펴진 않습니다 — 계약은 회전 quad를 그대로 지원하니 회전 quad 검출기로 교체하면 되살아납니다.)

> 설계 근거 & 단계별 로드맵 전체: **[SCANLATION_DESIGN.md](SCANLATION_DESIGN.md)**

---

## 진행 상태

전체 파이프라인이 **리눅스/gfx1200(RDNA4) 호스트에서 end-to-end 라이브 검증 완료** — 실제 Pixiv
만화 페이지에서 `comic-text-and-bubble-detector` 검출 → `manga-ocr` 일본어 OCR → `Ollama`(gemma4-26b, **GPU**)
**실제 한국어 번역**까지. (검출·인식 연산 장치는 `/admin`에서 CPU/GPU로 전환, 기본 CPU.) 브라우저 오버레이도 별도 검증(로컬 http). Docker 배포(P7)도
도메인(`scanlation.takecontr.lol`)에 **라이브**. 남은 건 튜닝(P8).

| 단계 | 내용 | 상태 |
|---|---|---|
| P0 | FastAPI 스켈레톤, handshake, CORS 허용 | ✅ |
| P1 | contracts · registry · state · cache · pipeline · **전체 wire 라우트** (엔진은 미번들) | ✅ 테스트 |
| P2 | deskew 기하(OpenCV homography, 축정렬 fast-path) | ✅ 테스트 |
| P3 | `comic-text-and-bubble-detector` 검출기(RT-DETR text/bubble 클래스 검출 + text-class 필터 + IoU/IoS 중복제거, 축정렬 박스) + `visualize.py` | ✅ 실페이지 검증 |
| P4 | `manga-ocr` 인식기 | ✅ 실제 일본어 브라우저 검증 |
| P5 | `Ollama` 번역기 (ROCm) | ✅ **gfx1200 라이브 검증** (gemma4-26b) |
| P5b | `llama.cpp` 번역기 (OpenAI 호환; **Vulkan**/vllm/LM Studio) | ✅ 단위테스트 |
| P6 | 클린룸 **MV2 확장(Firefox)** ([extension/](extension/)) | ✅ 브라우저 검증 |
| P7 | Docker 배포 (core-only 이미지 + 어드민 런타임 플러그인 설치) | ✅ 라이브 (도메인 배포·E2E 검증) |
| P8 | 튜닝 — 노이즈/SFX 컷·세로 병합 ✅; 라틴 폴백 OCR·세로쓰기 렌더 ⬜ | 🟨 진행 중 |

**검증됨:**
- **리눅스/gfx1200 호스트** — 실제 Pixiv 페이지 → `comic-text-and-bubble-detector,manga-ocr,ollama` → 진짜 ja→ko. 예:
  `背中のチャックを閉めてもらってもいいですか…？` → `등에 있는 지퍼 좀 올려주실 수 있을까요...?`
- **브라우저(윈도우)** — `comic-text-and-bubble-detector` 검출 + `manga-ocr` OCR + `%` 오버레이, 실제 862×1200 페이지.
- 서버 설정(엔진·모델·언어·프롬프트)은 전부 **`/admin`**에서(→ `state.json` 영속). 실행 명령엔 env·플래그 없음.
- 코어 단위테스트 **42개** 통과(`python -m tests`, pytest 미사용·모델/GPU 불필요); 엔진 패키지는
  각자 suite(`comic-text-and-bubble-detector`/`manga-ocr` 스모크는 가중치/패키지 없으면 자동 skip, `Ollama`/`llama.cpp`는 HTTP mock).
- **Docker(P7) — 라이브 배포 완료.** core-only 이미지가 리눅스 호스트에서 `docker compose up`으로 기동,
  `/admin` 원클릭으로 `comic-text-and-bubble-detector`+`manga-ocr`+`Ollama`를 **공개 레포에서 자격증명 없이** 설치(볼륨 영속 → 재시작
  유지). 호스트 nginx + Cloudflare로 **`https://scanlation.takecontr.lol`** 외부 노출(컨테이너는
  `127.0.0.1:4010`만 publish). 라이브 handshake + `/run_pipeline/` ja→ko 검증됨.

---

## 아키텍처

```
[ MV2 확장 ]  ──HTTP (lookup / run / box)──►  [ FastAPI 서버 ]
 이미지 발견                                 detect ─► deskew ─► recognize ─► translate
 + %-오버레이                                (comic-text-and-bubble-detector)  (geometry) (manga-ocr)  (ollama|llama.cpp)
```

- **서버** ([packages/scanlation-server/](packages/scanlation-server/)): FastAPI + uvicorn(단일 워커),
  async GPU 락 1개, 블로킹 모델 작업은 threadpool. SQLite 페이지 결과 캐시. 엔진은
  **`entry_points`로만 발견**(설치된 패키지 = 탑재 엔진), **첫 사용 시 lazy 인스턴스화**(그때 가중치 로드).
- **확장** ([extension/](extension/)): MV2(Firefox), 번들러/npm 없음, 순수 ES. `browser_action`
  =설정 팝업, `page_action`(주소창 아이콘)=원클릭 번역 토글. 자족형 content
  script(이미지 → base64 → md5 → lookup/run → 오버레이). 클린룸 MD5는 파이썬
  `hashlib.md5`와 **바이트 단위 일치** 검증됨.

### 와이어 계약 (서버 ↔ 번들 확장 공유)
- 역할 이름은 끝단까지 **detector / recognizer / translator**로 통일. 구 `ocr_extension`의
  BOX/OCR/TSL 어휘는 폐기했고 **옛 확장 호환은 지원하지 않음**(서버·확장 둘 다 이 레포 소유).
- `md5`는 **base64 문자열** 기준으로 계산(raw 바이트 아님) → 불일치 시 400.
- `bounds`는 `[x_min, y_min, x_max, y_max]`(클라는 `[l, b, r, t]`로 읽음). 결과 아이템 키는
  `{bounds, source, destination}` — 폐기된 역할 라벨(BOX/OCR/TSL)을 데이터 필드에서도 걷어냈다.
- **캐시 조회 / 작업 2단계**: 클라이언트가 먼저 `POST /run_lookup/`(`{md5, options}`)로 캐시를 **조회**한다 —
  적중이면 `200 {result:[...]}`, 미스도 **`200 {result: null}`**(이미지 업로드 없이 대역폭 절약; 404 같은 제어신호
  안 씀). 미스일 때만 `POST /run_pipeline/`(`{md5, contents, options}`)로 **실제 작업**을 돌린다(항상 200; contents
  없으면 400). 그 외: `/set_engines/`, `/set_languages/`, `/set_device/`, `/set_options/`, `/install_plugins/`,
  `/clear_cache/`, 그리고 `GET /` handshake.

---

## 레포 구조

```
packages/
  scanlation-sdk/       공유 계약: contracts · context(models_dir/device/langs) · prompt · testing
  scanlation-server/    코어(FastAPI): app/ · tools/ · tests/ — 엔진은 미번들(전부 별도 패키지)
  scanlation-comic-text-and-bubble-detector/    detector 플러그인 (transformers + torch)
  scanlation-manga-ocr/  recognizer 플러그인 (manga-ocr)
  scanlation-ollama/    translator 플러그인 (httpx)
  scanlation-llama-cpp/  translator 플러그인 (httpx)
extension/
  manifest.json  popup.{html,css}  content.css  icons/ (icon*·icon-off* = page_action off 상태)
  src/content.js (md5+파이프라인+오버레이)  background.js  popup.js
Dockerfile  docker-compose.yml  deploy/nginx.conf.example   Docker 배포(core-only 이미지 + 런타임 플러그인)
SCANLATION_DESIGN.md   전체 설계 / 핸드오프
```

> 코어는 엔진을 전혀 모르고 **`entry_points`로만 발견**합니다. 각 엔진 패키지는 `scanlation-sdk`만
> 의존하고 자기 백엔드 라이브러리를 담습니다("설치한 패키지 = 탑재 엔진"). 런타임 상태
> (`data/` sqlite·state, `models/` 가중치)는 실행 위치의 gitignore 디렉터리에 생깁니다.

> **모노레포 (서버 패키지들 + 확장 한 리포)** — 서버·확장이 **와이어 계약(JSON API)** 을, 코어·엔진
> 패키지가 **`EngineBase` 계약(`scanlation-sdk`)** 을 공동 진화하므로 한 리포에서 원자적으로 바꿉니다.
> 배포엔 분리 불필요 — 확장은 `extension/` zip을 AMO에, 서버는 `packages/`를 Docker로. 실제 독립
> 배포/공개가 필요해지면 히스토리 보존해 분리 가능:
> `git subtree split --prefix=packages/scanlation-comic-text-and-bubble-detector -b comic-text-and-bubble-detector-only`. 근거: [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md).

---

## 빠른 시작

Python 3.11+, Node는 확장 md5를 건드릴 때만 필요. 의존성은 repo 루트 `venv`(gitignore)에 —
**절대 전역 pip install 금지.** 로컬 패키지라 **`scanlation-sdk`를 항상 먼저** 설치합니다.

```bash
python -m venv venv
# 코어만(엔진 미포함): sdk + server
./venv/Scripts/python -m pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server
# 실엔진까지 전부(원하는 엔진만 골라 설치 가능):
./venv/Scripts/python -m pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server \
  -e ./packages/scanlation-comic-text-and-bubble-detector -e ./packages/scanlation-manga-ocr \
  -e ./packages/scanlation-ollama -e ./packages/scanlation-llama-cpp
# (Linux: source venv/bin/activate 후 pip install -e ... 동일)
```
설치한 엔진 패키지만 `/admin`·팝업 드롭다운에 나타납니다(= 탑재 엔진).

**모델 가중치는 명시적으로 설치**합니다(`load()`는 자동 다운로드 안 함 — 숨은 기본 동작 금지).
코어 디렉터리에서 한 번만 실행:
```bash
cd packages/scanlation-server
../../venv/Scripts/python tools/install.py   # 설치된 엔진의 가중치 (= 팝업 원클릭 / POST /install_plugins/)
```
또는 `models/comic-text-and-bubble-detector/`에 transformers 스냅샷(config.json + preprocessor_config.json + model.safetensors)을
직접 배치 / `SCANLATION_COMIC_TEXT_AND_BUBBLE_DETECTOR_MODEL=/path/to/dir` 로 디렉터리 지정. (ollama/llama.cpp는 별도 서비스라
설치 대상 아님 — `ollama pull <모델>`은 따로.)

### 서버 실행

```bash
cd packages/scanlation-server
../../venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 4000
```
**엔진·모델·언어·프롬프트·연산 장치는 전부 `/admin`에서 선택**합니다(→ `state.json`에 영속, 다음 기동부터 기본값).
실행 명령엔 플래그·env가 없습니다. **연산 장치(CPU/GPU)는 `/admin`(엔진 옵션 탭)에서 엔진마다 따로 고릅니다**
(비우면 각 엔진의 코드 기본값 — PaddleOCR-VL-For-Manga은 GPU, comic-text-and-bubble-detector·manga-ocr은 CPU; 저장 시 그 엔진만 새 장치로 재로드).
첫 실엔진 요청은 느림(RT-DETR transformers + manga-ocr 모델 로드); 같은 이미지 재요청은 md5 캐시로 즉시.

### 관리자 페이지 (`/admin`)

서버를 띄운 뒤 브라우저로 **`http://<host>:<port>/admin`** 접속(예: `http://127.0.0.1:4000/admin`).
클라이언트(확장)가 매번 모델을 정할 필요 없이 **서버에 설정을 저장**합니다 — 마지막 선택이 곧 기본값
(`data/state.json`에 영속). 할 수 있는 것:

- **모델/언어 선택** — detector·recognizer·translator + src/dst. 저장 시 기본값이 됨.
- **번역 프롬프트** — 기본 LLM 시스템 프롬프트(`default`)를 쓰거나 직접 편집·저장(커스텀 프리셋).
  활성 프롬프트는 캐시 키에 포함되어 바꾸면 재번역됨.
- **엔진 옵션** — 선택된 엔진의 옵션. 번역기 **모델**(설치된 것 중 드롭다운 선택·영속; env 없음),
  `num_ctx`/`temperature` 등. 빈칸 = 환경변수/스키마 기본값으로 복귀.
- **플러그인 설치** — 설치된 엔진뿐 아니라 **미설치 엔진 패키지까지** 원클릭 설치(패키지 pip 설치 → 가중치;
  = `POST /install_plugins/`). Docker에선 볼륨에 영속돼 재시작해도 유지.
- **유지보수** — **캐시 비우기**(`POST /clear_cache/`): 저장된 페이지 결과 캐시(`page_runs`)를 지워
  다음 접속 때 전 과정(검출·인식·번역)을 재실행.

> **인증**: `SCANLATION_AUTH_TOKEN`을 설정하면 API·admin이 `X-Auth-Token` 헤더를 요구한다(미설정=무인증, 로컬/LAN 기본값). 외부 노출(공개 도메인) 시엔 이 토큰을 설정하고 확장 팝업·`/admin`에 같은 값을 입력할 것 — 안 그러면 번역 API로 GPU가 무단 사용될 수 있다. `OPTIONS`(CORS preflight)와 `/admin` 정적 쉘은 면제(쉘은 토큰 없으면 API가 401이라 무해).

### 확장 로드

**Firefox 전용(MV2)**: `about:debugging#/runtime/this-firefox` → 임시 부가 기능 로드 →
[extension/manifest.json](extension/manifest.json). (Chrome은 MV2 미지원)
- **설정**: 툴바 아이콘(`browser_action`) 클릭 → 팝업(다크)에서 **Server 연결 · 언어/엔진** 선택.
- **번역**: 페이지에서 **주소창의 아이콘(`page_action`) 클릭 = 번역 on/off 토글** — 아이콘 색으로
  상태 표시(회색=off, 컬러=on). 팝업의 Enable/Disable로도 가능. 박스 클릭 = 원문 복사.

> **크로스오리진 이미지**(예: pixiv `i.pximg.net`)는 임의 HTTPS 페이지에서 못 읽음
> (CORS + Referer 핫링크). 확실한 경로는 **로컬 페이지를 http로 서빙**(same-origin) —
> 설계의 `make_viewer.py` 흐름. 확장은 핫링크 없는 크로스오리진 사이트 커버용으로 이미지
> 페치를 백그라운드(`<all_urls>` 호스트 권한)로 우회하기도 함. 혼합 콘텐츠: `http://127.0.0.1:4000`은
> 허용됨(localhost는 보안 컨텍스트). 다른 호스트의 http 서버는 HTTPS 페이지에서 차단 →
> SSH 터널 사용.

---

## 엔진 & 플러그인

3개 역할, 각각 독립 선택(팝업 드롭다운, `state.json`에 영속, 또는 env):

| 역할 | 플러그인 | 기본값 |
|---|---|---|
| detector | `comic-text-and-bubble-detector` | comic-text-and-bubble-detector |
| recognizer | `manga-ocr` | (없음) |
| translator | `Ollama`, `llama.cpp` | (없음) |

코어는 **엔진을 하나도 번들하지 않습니다**(placeholder `dummy` 제거됨). 설치·선택된 엔진이 없는
역할로 요청하면 **`400`**(“no \<role\> engine installed — /admin에서 설치·선택”)입니다. 검출만 단독
확인하려면 [tools/visualize.py](packages/scanlation-server/tools/visualize.py)를 쓰세요.

**엔진 설치(명시적, 숨은 기본 동작 아님) — 두 층:**
1. **패키지(코드)** — 엔진 플러그인은 별도 pip 패키지다. bare-metal은 `pip install`로, **Docker/런타임은
   `/admin` 원클릭**으로 설치한다: `POST /install_plugins/ {"plugins": {"comic-text-and-bubble-detector": true}}`가 미설치 시 엔진을
   **GitHub에서 `pip install`**(`git+<repo>@<ref>#subdirectory=packages/scanlation-comic-text-and-bubble-detector`)해
   `SCANLATION_PLUGINS_DIR`(Docker 볼륨)에 넣고, 무거운 백엔드 의존(transformers/torch…)을 그때 끌어오며,
   `entry_points`를 라이브 재발견한다. 이미지엔 엔진 코드가 **아예 없고**(core만), 설치 전엔 `/admin`에 "미설치"로만
   뜬다. (dev/오프라인은 `SCANLATION_ENGINES_SRC`를 로컬 `packages/`로 두면 GitHub 대신 로컬 소스에서 설치.)
2. **가중치** — 패키지가 깔린 뒤 `comic-text-and-bubble-detector`(transformers 스냅샷)·`manga-ocr`(HF) 가중치를 이어서 다운로드(같은 원클릭이 연달아 수행,
   또는 CLI `python tools/install.py`). `GET /get_settings/`가 엔진별 `installed_package`(패키지)
   /`installed`(가중치) 상태를 보고한다.

계약에 `is_installed()`/`install()`만 구현하면 어떤 엔진이든 같은 방식으로 설치된다("설치한 패키지 = 탑재 엔진").

**플러그인 추가(= 새 패키지):** `scanlation-sdk`만 의존하는 패키지를 만들어 `EngineBase` 상속,
역할 메서드(`detect`/`recognize`/`translate`) + 클래스 메타데이터 + `OPTION_SCHEMA` 구현, 그리고
`pyproject.toml`에 `[project.entry-points."scanlation.<role>"]`로 등록. `pip install`하면 코어가
`entry_points`로 **자동 발견**합니다 — 코어 수정 불필요(기존 엔진 패키지가 그 예시).

### 번역 백엔드
- **`Ollama`** → `POST /api/generate` (ollama 내부 llama.cpp로 ROCm). env:
  `OLLAMA_ENDPOINT`(`http://127.0.0.1:11434/api`). **모델은 `/admin`에서 선택**(env 없음).
- **`llama.cpp`** → OpenAI `POST /v1/chat/completions` — 최신 AMD에서 ROCm이 불안할 때
  **Vulkan**(`llama-server`)용, 또는 임의 OpenAI 호환 서버. env:
  `LLAMACPP_ENDPOINT`(`http://127.0.0.1:8080`). 모델은 `/admin`(서버 `/v1/models`에서 조회). `<think>` 구간 제거.

둘 다 사용자 튜닝 시스템 프롬프트 + 템플릿([scanlation_sdk/prompt.py](packages/scanlation-sdk/scanlation_sdk/prompt.py))
공유: 번역만, OCR 오류 감안, 추론 한 문장.

---

## 설정 (env)

| 변수 | 기본 | 의미 |
|---|---|---|
| `SCANLATION_AUTH_TOKEN` | (빈 값) | 설정 시 모든 API·admin이 `X-Auth-Token` 헤더를 요구(빈 값=무인증). 확장 팝업·`/admin`에 같은 값 입력. `OPTIONS`·`/admin` 정적은 면제 |
| `SCANLATION_LOG_LEVEL` | `INFO` | `scanlation.*` 로거 레벨(서드파티는 root WARNING으로 고정). access 로그는 자체 미들웨어가 타임스탬프+소요시간(`METHOD PATH -> STATUS Nms`)으로 대체. `DEBUG`로 상세화 |
| `SCANLATION_MIN_IMAGE_DIM` | `80` | 확장 이미지 필터 최초 기본값: **짧은 변**이 이 px 미만이면 아이콘·배너로 보고 스킵. `/admin` **동작** 탭에서 조절(→ handshake로 확장에 전달). `0`=전부 번역 |
| `SCANLATION_DETECTOR` / `_RECOGNIZER` / `_TRANSLATOR` | `comic-text-and-bubble-detector` / (빈 값) / (빈 값) | 최초 기동 기본 엔진(detector 기본 `comic-text-and-bubble-detector`; 나머지 빈 값=미선택; 이후 `/admin`이 덮어씀) |
| `SCANLATION_LANG_SRC` / `_DST` | `ja` / `ko` | 최초 기동 기본 언어(이후 `/admin`) |
| `SCANLATION_BASE_DIR` | 실행 위치(CWD) | `data/`(캐시, state.json) 루트; Docker/테스트는 명시 지정 |
| `SCANLATION_MODELS_DIR` | `<base>/models` | 가중치 루트 |
| `SCANLATION_PLUGINS_DIR` | `<base>/plugins` | `/admin`이 엔진 패키지를 pip 설치하는 위치(Docker: `/plugins` 볼륨) |
| `SCANLATION_ENGINE_REPO` | 이 레포 GitHub URL | `/admin` 설치가 엔진을 받아오는 git 소스 |
| `SCANLATION_ENGINE_REF` | `main` | 받아올 브랜치/태그(배포 고정용) |
| `SCANLATION_ENGINES_SRC` | (미설정) | 설정 시 GitHub 대신 로컬 `packages/` 소스에서 설치(dev/오프라인) |
| `HF_HOME` | HF 기본 | manga-ocr 가중치 캐시(Docker: `/data/hf`, 볼륨 영속) |
| `SCANLATION_COMIC_TEXT_AND_BUBBLE_DETECTOR_MODEL` | — | RT-DETR transformers 스냅샷 **디렉터리** 명시 경로(미설정 시 `models/comic-text-and-bubble-detector/`, `install()`이 HF 레포에서 다운로드) |
| `OLLAMA_ENDPOINT` | `…:11434/api` | ollama 백엔드 주소 (모델은 `/admin`) |
| `LLAMACPP_ENDPOINT` | `…:8080` | llama.cpp/OpenAI 백엔드 주소 (모델은 `/admin`) |

> 모델 태그는 이제 env가 아니라 **`/admin` 엔진 옵션의 드롭다운**에서만 정합니다(백엔드에 설치된 모델을 조회). `state.json`에 영속.

> **ollama 동시성**: 번역은 이미지 단위 배치로 GPU 락 밖에서 여러 이미지를 **동시에** 처리합니다(동시 이미지 수는 `/admin` **동작** 탭의 "동시 번역 이미지 수", **기본 1** — `state.json`에 영속, 재시작 없이 조절). 기본 1은 안전값(어떤 ollama 설정에서도 큐 대기·타임아웃 없음)이며, 1이어도 검출·인식↔번역이 겹칩니다. GPU에서 **실제 병렬 생성**을 켜려면 이 값과 **호스트 ollama 데몬**의 `OLLAMA_NUM_PARALLEL`을 **함께** 올려야 합니다(둘 중 낮은 쪽이 실제 병렬도) — 설정 방법·VRAM 계산은 아래 **배포 → 호스트 ollama 튜닝** 참고.

---

## 테스트

```bash
# 코어 빠른 단위 42개, 모델/GPU 불필요 (pytest 미사용, 자체 러너)
cd packages/scanlation-server && ../../venv/Scripts/python -m tests
# 엔진 패키지별 suite (각자 self-contained; 스모크는 가중치/패키지 없으면 자동 skip)
cd packages/scanlation-comic-text-and-bubble-detector   && ../../venv/Scripts/python -m tests
cd packages/scanlation-manga-ocr && ../../venv/Scripts/python -m tests
cd packages/scanlation-ollama   && ../../venv/Scripts/python -m tests   # HTTP mock
cd packages/scanlation-llama-cpp && ../../venv/Scripts/python -m tests   # HTTP mock
```

**검출 육안 확인**(정확도 핵심 루프 — 검출이 병목):
```bash
cd packages/scanlation-server
../../venv/Scripts/python tools/visualize.py page.jpg --detector comic-text-and-bubble-detector --out annotated.png   # 박스 + 인덱스
```
`visualize.py`는 `annotated.png` + deskew된 `crops/`를 저장 → 박스 위치와 crop이 똑바른지
눈으로 판단. (실제 ja→ko 확인은 서버+`/admin`+브라우저로.)

---

## 알려진 이슈 / 주의

- **gfx1200(RDNA4) ROCm**이 최대 리스크 — 사전빌드 휠 부재 가능. 완화: `comic-text-and-bubble-detector`는 CUDA/ROCm torch가
  있으면 **GPU**(없으면 CPU fallback), manga-ocr는 **CPU**(충분히 빠름), ollama/ROCm 말썽이면
  **`llama.cpp` + Vulkan**.
- **CPU 속도**: manga-ocr이 영역마다 트랜스포머 1패스 → CPU에선 느리고 GPU에선 빠름.
  같은 페이지 재방문은 즉시(md5 캐시).
- **라틴/영숫자 라벨**(예: `正1L=1000ml`)은 manga-ocr(일본어 모델)이 오인식. 향후:
  라틴 비중 높은 영역을 폴백 OCR로 라우팅(P8).
- **pixiv 라이브**는 안 됨(크로스오리진 + Referer 핫링크) — 로컬 서빙 페이지 사용.
- **검출 노이즈/중복 (RT-DETR)**: RT-DETR은 NMS-free라 겹치거나 중첩된 박스를 그대로 낼 수 있어,
  플러그인이 `text_bubble`+`text_free` 클래스만 남기고(말풍선 컨테이너 `bubble`은 버림) IoU(`nms_iou`)+
  IoS(`contain_thresh`)로 중복·포함 박스를 제거한다. 신뢰도 하한은 `conf`(기본 0.6). 문장부호-only OCR
  스킵은 미구현.
- **검출 옵션(RT-DETR)** — 튜닝 가능한 float 3개: `conf`(신뢰도 하한, 기본 0.6),
  `nms_iou`(중복 IoU 임계, 0.6), `contain_thresh`(포함 IoS 임계, 0.85). `/admin` 엔진 옵션에서 조절.
- 오버레이 **세로쓰기**는 현재 가로 렌더(P8 예정).
- **SFX 무한반복 → 배치 JSON 깨짐 (미해결)**: ollama(gemma4-26b)가 늘어진 SFX/의성어(「びゅうううう」 등)에서
  같은 토큰을 무한반복 → 배치 `format`(JSON schema) 문자열이 안 닫혀 `JSONDecodeError` → per-text 폴백도
  같은 모델·입력이라 같은 루프. **샘플러로는 못 잡는다**: near-greedy(정확도용 저온)에선 `frequency_penalty`가
  무력, 온도를 올리면 루프는 깨지나 번역 정확도가 손상, `repeat_penalty`는 강한 루프의 argmax를 못 뒤집음.
  남은 선택지 — (1) SFX를 덜 무는 모델 교체, (2) `pipeline.py`에서 온도 무관 반복붕괴 후처리(입력·출력 run
  collapse). 진단: `/admin` 동작탭 verbose 로그 + `docker compose logs` + `tools/run_image.py`(force, all-at-once).

---

## 배포 (리눅스 호스트) — 검증된 절차

**bare-metal 먼저**(빠른 반복); Docker(P7)는 동작한 뒤에.

```bash
# 0) 받기 + 설치 (sdk 먼저; 원하는 엔진 패키지만 골라 설치 가능)
git clone https://github.com/tjdnjsrmsdidgkrdlfma/Scanlation.git
cd Scanlation && python -m venv venv && source venv/bin/activate
pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server \
  -e ./packages/scanlation-comic-text-and-bubble-detector -e ./packages/scanlation-manga-ocr \
  -e ./packages/scanlation-ollama -e ./packages/scanlation-llama-cpp
cd packages/scanlation-server
python tools/install.py            # 모델 가중치 설치 (한 번; = 원클릭 / POST /install_plugins/)
ollama pull <your-model>           # ollama 모델 (별도 서비스)
```

**1) 서버 띄우기** (포트 자유; 끄지 말 것 — tmux/nohup). 플래그·env 없음:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 4001
#  확인:  curl -s http://127.0.0.1:4001/ | head -c 150
```
> 검출·인식을 GPU에서 돌리려면 `/admin` **엔진 옵션 탭**에서 해당 엔진의 연산 장치를 GPU로 저장하면 됨
> (비우면 엔진 기본값 — PaddleOCR-VL-For-Manga은 GPU, comic-text-and-bubble-detector·manga-ocr은 CPU).

**2) `/admin`에서 설정** (SSH 터널 뒤 브라우저 → 3번의 터널 사용):
`http://127.0.0.1:4001/admin` → **모델·언어 탭**에서 detector `comic-text-and-bubble-detector` · recognizer `manga-ocr` ·
translator `Ollama` · **연산 장치**(CPU/GPU) 선택 → **엔진 옵션 탭**에서 translator `model` **드롭다운**으로
pull해둔 모델 선택 → 저장. 전부 `state.json`에 영속(다음 기동부터 기본값). **모델은 여기서만 지정**(env 없음).

**3) 확장 연결** (브라우저가 다른 PC면 SSH 터널로 혼합콘텐츠·CORS 회피):
```bash
# 브라우저 PC에서:
ssh -L 4001:localhost:4001 -L 8001:localhost:8001 root@<host>
```
팝업 Server = `http://127.0.0.1:4001` → Connect (드롭다운에 `comic-text-and-bubble-detector/manga-ocr/ollama`).

**4) 로컬 만화를 http로 서빙** (same-origin이라야 content script가 읽음):
```bash
cd /path/to/manga && python -m http.server 8001    # 호스트에서
```
→ 브라우저 `http://127.0.0.1:8001/` 만화 페이지 → **F5 → Enable** → 한국어 오버레이.

**ROCm 불안하면**: `llama-server`(Vulkan) 띄우고 `/admin`에서 translator를 `llama.cpp`로 바꾸면 됨
(백엔드 주소만 다르면 `LLAMACPP_ENDPOINT`를 서버 실행 앞에 env로; 기본 `http://127.0.0.1:8080`).

### Docker (P7)

> **현재 운영 배포가 이 방식이다** — `https://scanlation.takecontr.lol`이 이 구성(호스트 Docker + nginx + Cloudflare)으로 떠 있다.

이미지는 **core만**(엔진 미포함) — 엔진 코드는 이미지에 **아예 없다.** 실엔진은 `/admin`에서 설치할 때
**GitHub에서 `pip install`**(`git+…#subdirectory=packages/scanlation-<name>`)돼 `plugins` 볼륨에 들어간다.
"설치한 패키지 = 탑재 엔진"이 컨테이너에서도 그대로. LLM 백엔드(ollama)는 **컨테이너 밖**, HTTPS는
**호스트 nginx**가 담당(호스트에 `127.0.0.1:4010`만 publish; 컨테이너 내부 uvicorn은 4000).

> **레포 접근 필요:** 컨테이너엔 git 자격증명이 없으므로 엔진을 받으려면 `SCANLATION_ENGINE_REPO`가
> **공개 레포**여야 한다(비공개면 토큰 박은 URL 필요). 배포를 고정하려면 `SCANLATION_ENGINE_REF`를 태그로.

```bash
docker compose up -d --build                 # core-only 이미지 빌드 + 기동
curl -s http://127.0.0.1:4010/ | head -c 120 # handshake = 엔진 목록 비어있음(아직 미설치)
```
1. **엔진 설치** — `http://127.0.0.1:4010/admin` 플러그인 탭에서 `comic-text-and-bubble-detector`·`manga-ocr`(+원하면 `Ollama`/`llama.cpp`)
   **설치**. 패키지가 GitHub에서 `plugins` 볼륨에, 가중치가 `data` 볼륨에 받아진다(둘 다 재시작해도 유지).
   첫 설치는 느림(git clone + transformers/torch + RT-DETR 스냅샷 ~172MB 다운로드) — 인터넷 필요.
2. **모델·언어·프롬프트** — 같은 `/admin`에서 선택(→ `data/state.json` 영속). translator=`Ollama`면 옵션 탭에서
   호스트 ollama의 pull된 모델을 드롭다운 선택. 컨테이너는 `host.docker.internal`로 호스트 ollama(`:11434`)에 접속.
3. **HTTPS** — 호스트 nginx에 [deploy/nginx.conf.example](deploy/nginx.conf.example)을 반영(도메인·인증서 경로만 수정)
   → 도메인으로 `/admin`·확장 접속. 컨테이너 nginx·인증서 중복 없음.

> **⚠️ 호스트 ollama 접근:** ollama가 호스트 `127.0.0.1:11434`(loopback)에 떠 있으면 기본 compose의
> `host.docker.internal`로는 **컨테이너가 못 붙는다**(컨테이너의 127.0.0.1은 자기 자신). 단일 호스트라면
> compose에 **`network_mode: host`**를 주는 게 제일 깔끔하다(컨테이너가 호스트 네트워크를 그대로 써서
> `127.0.0.1:11434`에 바로 접속; 이땐 `ports`/`extra_hosts`가 무시되고 uvicorn 바인드 포트가 곧 노출 포트 —
> 기본 4000, 4010으로 노출하려면 Dockerfile `CMD`의 `--port`를 4010으로).
> 아니면 ollama를 `OLLAMA_HOST=0.0.0.0:11434`로 열어 두면 된다(방화벽 주의).

#### GPU (ROCm/CUDA) 배포

이미지엔 torch가 없고 설치 때 **CPU torch가 기본**이라, 그대로면 검출·인식이 CPU로 돈다. GPU로 돌리려면 **둘 다** 필요하다:

1. **패스스루** — 벤더별 compose 오버라이드를 base에 얹는다:
   ```bash
   # AMD (ROCm)
   docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build
   # NVIDIA (CUDA — 호스트에 nvidia-container-toolkit 필요)
   docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
   ```
2. **백엔드 선택** — `/admin` **동작** 탭에서 **연산 백엔드 = GPU**로 저장. 패스스루된 장치(`/dev/kfd` vs `/dev/nvidia*`)로 벤더(AMD ROCm / NVIDIA CUDA)를 **자동 판별**해, **엔진 설치 시** 맞는 torch 휠을 받는다 → **엔진 설치 *전에* 정할 것**(바꾸면 재설치해야 적용).

- 어느 GPU(다중)로 돌릴지는 **엔진 옵션 탭의 per-engine 장치**(`cuda:N`)로 따로 고른다. `torch`는 한 빌드 = 한 벤더라 AMD+NVIDIA를 **동시엔 못 쓴다**(동작 탭에서 벤더 하나).
- **RDNA4(gfx1200) 등 최신 카드**는 사전빌드 rocm 휠이 없을 수 있다 — 동작 탭의 **torch index URL**(예 `…/whl/rocm6.2`)이나 `HSA_OVERRIDE_GFX_VERSION`(rocm 오버라이드 env)로 맞춘다.
- 확인: `docker compose … exec server python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → `+rocm`/`+cu##` `True`면 성공.

#### 호스트 ollama 튜닝 (번역 동시성 — 권장)

번역은 **이미지 단위 전체 배치**(한 이미지의 말풍선을 한 LLM 호출로, 상호 문맥으로 일관성↑)이고,
검출·인식을 지키는 GPU 락 **밖**에서 돈다. 그래서 서버는 여러 이미지의 번역을 **동시에** ollama로
던진다(동시 이미지 수는 `/admin` **동작** 탭의 "동시 번역 이미지 수", **기본 1** — 안전값). 이를 늘려 GPU에서 **실제 병렬로 생성**하려면
아래를 **호스트 ollama 데몬**에 설정해야 한다(우리 compose가 아니라 ollama 자체 env — 컨테이너 밖에 있음):

| ollama env | 권장 | 의미 |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `4` | 한 모델이 동시에 처리할 요청 수. `/admin` 동작 탭의 "동시 번역 이미지 수"와 맞춘다 |
| `OLLAMA_KEEP_ALIVE` | `-1` | 모델을 계속 상주(페이지 사이 콜드 리로드 방지; `30m` 등도 가능) |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | 번역 모델 1벌만 — 불필요한 VRAM 점유 방지 |

**Linux (systemd — 운영 배포):**
```bash
sudo systemctl edit ollama       # [Service] 아래에 추가:
#   Environment="OLLAMA_NUM_PARALLEL=4"
#   Environment="OLLAMA_KEEP_ALIVE=-1"
#   Environment="OLLAMA_MAX_LOADED_MODELS=1"
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**Windows (개발 PC):** 시스템 환경 변수에 같은 이름으로 추가(또는 `setx OLLAMA_NUM_PARALLEL 4`) 후
ollama 재시작(트레이 아이콘 종료 → 재실행). 최신 ollama는 메모리를 보고 자동으로 잡기도 하지만 명시 권장.

**VRAM:** KV캐시 ≈ `num_ctx × OLLAMA_NUM_PARALLEL`. `num_ctx` 기본값은 단일·배치 공통 `2048`(경로 전환 시 모델
리로드 방지)이라 4슬롯이면 KV가 크게 늘 수 있다 — `ollama ps` / `nvidia-smi`로 여유를 확인하고 빠듯하면 `OLLAMA_NUM_PARALLEL=2`로.
`NUM_PARALLEL=1`이어도 검출·인식↔번역이 겹쳐 이득이 있고, `>1`이 **생성까지** 병렬화한다.
배치가 컨텍스트를 넘치거나 JSON 파싱이 깨지면 **말풍선 단위 순차로 자동 폴백**하므로 결과 정확성은 항상 보장된다.

> LLM을 컨테이너로 옮기고 싶으면 `ollama/ollama:rocm` 이미지를 별도 서비스로 추가하고
> `OLLAMA_ENDPOINT`만 그쪽으로 돌리면 된다(gfx1200은 `/dev/kfd`·`/dev/dri` 패스스루 필요). 기본 compose는
> 이미 검증된 호스트 ollama를 재사용하도록 밖을 가리킨다. (컨테이너 ollama에도 위 `OLLAMA_*` env를 그대로 준다.)

---

## 다른 머신 / 새 세션에서 이어받기

1. 이 README(위 상태) + [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md)(왜) 읽기.
2. `pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server`(+ 원하는 엔진 패키지) 후
   `cd packages/scanlation-server && ../../venv/Scripts/python -m tests` green 확인.
3. P0–P7 완료 — **리눅스/gfx1200에서 실제 번역까지 라이브 검증**, Docker도 도메인
   (`scanlation.takecontr.lol`)에 **라이브 배포**됨(위 Docker 절 참고, `network_mode: host` 주의).
   다음은 P8(노이즈 필터·세로쓰기·라틴 폴백 등). 솔로 프로젝트: `main`에 직접 커밋, 의존성은 `venv`.
4. 번역은 **이미지 단위 전체 배치 + GPU 락 밖 실행**(말풍선 상호 문맥 → 일관성, 이미지 간 병렬).
   라이브에서 여러 이미지 동시 번역을 쓰려면 **호스트 ollama에 `OLLAMA_NUM_PARALLEL`을 설정**해야 한다
   (위 배포 → **호스트 ollama 튜닝** 참고). 서버 쪽 상한은 `/admin` 동작 탭의 "동시 번역 이미지 수"(기본 1 — 병렬 생성은 이 값과 함께 올릴 때).

라이선스: 프로젝트 라이선스 미정(TBD). 트리는 **GPLv3-free**(클린룸; 엔진은 런타임 의존만) —
그대로 유지할 것.

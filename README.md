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

전체 파이프라인이 **리눅스/gfx1200(RDNA4) 호스트에서 end-to-end 라이브 검증 완료** — 실제 Pixiv
만화 페이지에서 `ctd`(CPU) 검출 → `mangaocr`(CPU) 일본어 OCR → `ollama`(gemma4-26b, **GPU**)
**실제 한국어 번역**까지. 브라우저 오버레이도 별도 검증(로컬 http). 남은 건 배포 마감(P7) + 튜닝(P8).

| 단계 | 내용 | 상태 |
|---|---|---|
| P0 | FastAPI 스켈레톤, handshake, CORS 허용 | ✅ |
| P1 | contracts · registry · state · cache · pipeline · **전체 wire 라우트** · dummy 엔진 | ✅ 테스트 |
| P2 | deskew 기하(OpenCV homography, 축정렬 fast-path) | ✅ 테스트 |
| P3 | `ctd` 검출기(comic-text-detector ONNX) + 마스크→회전quad 디코드 + `visualize.py` | ✅ 실페이지 검증 |
| P4 | `mangaocr` 인식기 | ✅ 실제 일본어 브라우저 검증 |
| P5 | `ollama` 번역기 (ROCm) | ✅ **gfx1200 라이브 검증** (gemma4-26b) |
| P5b | `llamacpp` 번역기 (OpenAI 호환; **Vulkan**/vllm/LM Studio) | ✅ 단위테스트 |
| P6 | 클린룸 **MV3 확장** ([extension/](extension/)) | ✅ 브라우저 검증 |
| P7 | Docker / ROCm + Vulkan 배포 | ⬜ 예정 |
| P8 | 튜닝(라틴 라벨 폴백 OCR, merge_px 스케일링, 세로쓰기) | ⬜ 예정 |

**검증됨:**
- **리눅스/gfx1200 호스트** — 실제 Pixiv 페이지 → `ctd,mangaocr,ollama` → 진짜 ja→ko. 예:
  `背中のチャックを閉めてもらってもいいですか…？` → `등에 있는 지퍼 좀 올려주실 수 있을까요...?`
- **브라우저(윈도우)** — `ctd` 검출 + `mangaocr` OCR + `%` 오버레이, 실제 862×1200 페이지.
- 서버 설정(엔진·모델·언어·프롬프트)은 전부 **`/admin`**에서(→ `state.json` 영속). 실행 명령엔 env·플래그 없음.
- 코어 단위테스트 **26개** 통과(`python -m tests`, pytest 미사용·모델/GPU 불필요); 엔진 패키지는
  각자 suite(`ctd`/`mangaocr` 스모크는 가중치/패키지 없으면 자동 skip, `ollama`/`llamacpp`는 HTTP mock).

---

## 아키텍처

```
[ MV3 확장 ]  ──HTTP (md5 / box / lazy)──►  [ FastAPI 서버 ]
 이미지 발견                                 detect ─► deskew ─► recognize ─► translate
 + %-오버레이                                (ctd)     (geometry) (mangaocr)  (ollama|llamacpp)
```

- **서버** ([packages/scanlation-server/](packages/scanlation-server/)): FastAPI + uvicorn(단일 워커),
  async GPU 락 1개, 블로킹 모델 작업은 threadpool. SQLite 결과 캐시 + 수동 번역 메모리(TM). 엔진은
  **`entry_points`로만 발견**(설치된 패키지 = 탑재 엔진), **첫 사용 시 lazy 인스턴스화**(그때 가중치 로드).
- **확장** ([extension/](extension/)): MV3, 번들러/npm 없음, 순수 ES. 자족형 content
  script(이미지 → base64 → md5 → lazy/work → 오버레이). 클린룸 MD5는 파이썬
  `hashlib.md5`와 **바이트 단위 일치** 검증됨.

### 와이어 계약 (서버 ↔ 번들 확장 공유)
- 역할 이름은 끝단까지 **detector / recognizer / translator**로 통일. 구 `ocr_extension`의
  BOX/OCR/TSL 어휘는 폐기했고 **옛 확장 호환은 지원하지 않음**(서버·확장 둘 다 이 레포 소유).
- `md5`는 **base64 문자열** 기준으로 계산(raw 바이트 아님) → 불일치 시 400.
- box는 `[x_min, y_min, x_max, y_max]`(클라는 `[l, b, r, t]`로 읽음). 결과 아이템 키
  `{ocr, tsl, box}`는 역할이 아니라 **데이터 필드**라 그대로.
- `POST /run_ocrtsl/`는 **lazy**(`{md5, options}` → 캐시 히트, 미스 시 non-2xx) 후
  **work**(`{md5, contents, options}`). 그 외: `/run_tsl/`, `/get_trans/`,
  `/set_manual_translation/`, `/set_models/`, `/set_lang/`, `/get_active_options/`,
  `/get_plugin_data/`, `/manage_plugins/`, 그리고 `GET /` handshake.

---

## 레포 구조

```
packages/
  scanlation-sdk/       공유 계약: contracts · context(models_dir/device/langs) · prompt · testing
  scanlation-server/    코어(FastAPI): app/ · plugins/dummy/ · tools/ · tests/ — dummy 엔진만 번들
  scanlation-ctd/       detector 플러그인 (onnxruntime)
  scanlation-mangaocr/  recognizer 플러그인 (manga-ocr)
  scanlation-ollama/    translator 플러그인 (httpx)
  scanlation-llamacpp/  translator 플러그인 (httpx)
extension/
  manifest.json  popup.{html,css}  content.css  icons/
  src/content.js (md5+파이프라인+오버레이)  service-worker.js  popup.js
YOMU_DESIGN.md   전체 설계 / 핸드오프
```

> 코어는 엔진을 전혀 모르고 **`entry_points`로만 발견**합니다. 각 엔진 패키지는 `scanlation-sdk`만
> 의존하고 자기 백엔드 라이브러리를 담습니다("설치한 패키지 = 탑재 엔진"). 런타임 상태
> (`data/` sqlite·state, `models/` 가중치)는 실행 위치의 gitignore 디렉터리에 생깁니다.

> **모노레포 (서버 패키지들 + 확장 한 리포)** — 서버·확장이 **와이어 계약(JSON API)** 을, 코어·엔진
> 패키지가 **`EngineBase` 계약(`scanlation-sdk`)** 을 공동 진화하므로 한 리포에서 원자적으로 바꿉니다.
> 배포엔 분리 불필요 — 확장은 `extension/` zip을 AMO에, 서버는 `packages/`를 Docker로. 실제 독립
> 배포/공개가 필요해지면 히스토리 보존해 분리 가능:
> `git subtree split --prefix=packages/scanlation-ctd -b ctd-only`. 근거: [YOMU_DESIGN.md](YOMU_DESIGN.md).

---

## 빠른 시작

Python 3.11+, Node는 확장 md5를 건드릴 때만 필요. 의존성은 repo 루트 `venv`(gitignore)에 —
**절대 전역 pip install 금지.** 로컬 패키지라 **`scanlation-sdk`를 항상 먼저** 설치합니다.

```bash
python -m venv venv
# 코어만(dummy 엔진): sdk + server
./venv/Scripts/python -m pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server
# 실엔진까지 전부(원하는 엔진만 골라 설치 가능):
./venv/Scripts/python -m pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server \
  -e ./packages/scanlation-ctd -e ./packages/scanlation-mangaocr \
  -e ./packages/scanlation-ollama -e ./packages/scanlation-llamacpp
# (Linux: source venv/bin/activate 후 pip install -e ... 동일)
```
설치한 엔진 패키지만 `/admin`·팝업 드롭다운에 나타납니다(= 탑재 엔진).

**모델 가중치는 명시적으로 설치**합니다(`load()`는 자동 다운로드 안 함 — 숨은 기본 동작 금지).
코어 디렉터리에서 한 번만 실행:
```bash
cd packages/scanlation-server
../../venv/Scripts/python tools/install.py   # 설치된 엔진의 가중치 (= 팝업 원클릭 / POST /manage_plugins/)
```
또는 `models/ctd/`에 `.onnx` 직접 배치 / `SCANLATION_CTD_MODEL=/path.onnx` 지정. 미러는
`SCANLATION_CTD_URL`. (ollama/llamacpp는 별도 서비스라 설치 대상 아님 — `ollama pull <모델>`은 따로.)

### 서버 실행

```bash
cd packages/scanlation-server
../../venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 4000
```
**엔진·모델·언어·프롬프트는 전부 `/admin`에서 선택**합니다(→ `state.json`에 영속, 다음 기동부터 기본값).
실행 명령엔 플래그·env가 없습니다. GPU면 provider 힌트로 `SCANLATION_DEVICE=rocm`만 앞에 붙이면 됩니다.
첫 실엔진 요청은 느림(CTD ONNX + manga-ocr 모델 로드); 같은 이미지 재요청은 md5 캐시로 즉시.

### 관리자 페이지 (`/admin`)

서버를 띄운 뒤 브라우저로 **`http://<host>:<port>/admin`** 접속(예: `http://127.0.0.1:4000/admin`).
클라이언트(확장)가 매번 모델을 정할 필요 없이 **서버에 설정을 저장**합니다 — 마지막 선택이 곧 기본값
(`data/state.json`에 영속). 할 수 있는 것:

- **모델/언어 선택** — detector·recognizer·translator + src/dst. 저장 시 기본값이 됨.
- **번역 프롬프트** — LLM 시스템 프롬프트 프리셋(`default`/`literal`/`natural`)을 고르거나 직접
  편집·저장(커스텀 프리셋). 활성 프롬프트는 캐시 키에 포함되어 바꾸면 재번역됨.
- **엔진 옵션** — 선택된 엔진의 옵션. 번역기 **모델**(설치된 것 중 드롭다운 선택·영속; env 없음),
  `num_ctx`/`temperature` 등. 빈칸 = 환경변수/스키마 기본값으로 복귀.
- **플러그인 설치** — CTD/manga-ocr 가중치 원클릭 설치(= `tools/install.py` / `POST /manage_plugins/`).

> 인증 없음(로컬/LAN 전용). 외부 노출 시 리버스 프록시 뒤에 둘 것.

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

**모델 설치(명시적, 숨은 기본 동작 아님):** 무거운 가중치가 있는 엔진(`ctd`, `mangaocr`)은
설치해야 동작합니다. `GET /get_plugin_data/`가 엔진별 `installed` 상태를,
`POST /manage_plugins/ {"plugins": {"ctd": true}}`가 설치(다운로드)를 수행 — 팝업 플러그인 탭의
**원클릭**이 이걸 호출하고, CLI는 `python tools/install.py`. 계약에 `is_installed()`/`install()`만
구현하면 어떤 엔진이든 같은 방식으로 설치됨.

**플러그인 추가(= 새 패키지):** `scanlation-sdk`만 의존하는 패키지를 만들어 `EngineBase` 상속,
역할 메서드(`detect`/`recognize`/`translate`) + 클래스 메타데이터 + `OPTION_SCHEMA` 구현, 그리고
`pyproject.toml`에 `[project.entry-points."scanlation.<role>"]`로 등록. `pip install`하면 코어가
`entry_points`로 **자동 발견**합니다 — 코어 수정 불필요(기존 엔진 패키지가 그 예시).

### 번역 백엔드
- **`ollama`** → `POST /api/generate` (ollama 내부 llama.cpp로 ROCm). env:
  `OLLAMA_ENDPOINT`(`http://127.0.0.1:11434/api`). **모델은 `/admin`에서 선택**(env 없음).
- **`llamacpp`** → OpenAI `POST /v1/chat/completions` — 최신 AMD에서 ROCm이 불안할 때
  **Vulkan**(`llama-server`)용, 또는 임의 OpenAI 호환 서버. env:
  `LLAMACPP_ENDPOINT`(`http://127.0.0.1:8080`). 모델은 `/admin`(서버 `/v1/models`에서 조회). `<think>` 구간 제거.

둘 다 사용자 튜닝 시스템 프롬프트 + 템플릿([scanlation_sdk/prompt.py](packages/scanlation-sdk/scanlation_sdk/prompt.py))
공유: 번역만, OCR 오류 감안, 추론 한 문장.

---

## 설정 (env)

| 변수 | 기본 | 의미 |
|---|---|---|
| `SCANLATION_DEVICE` | `cpu` | `cpu` / `rocm` / `dml` provider 힌트(항상 CPU fallback) |
| `SCANLATION_DETECTOR` / `_RECOGNIZER` / `_TRANSLATOR` | `dummy` | 최초 기동 기본 엔진(이후 `/admin` 선택이 덮어씀) |
| `SCANLATION_LANG_SRC` / `_DST` | `ja` / `ko` | 최초 기동 기본 언어(이후 `/admin`) |
| `SCANLATION_BASE_DIR` | 실행 위치(CWD) | `data/`(캐시, state.json) 루트; Docker/테스트는 명시 지정 |
| `SCANLATION_MODELS_DIR` | `<base>/models` | 가중치 루트 |
| `SCANLATION_CTD_MODEL` / `_CTD_URL` | — / HF | CTD `.onnx` 명시 경로 / 설치 다운로드 URL |
| `OLLAMA_ENDPOINT` | `…:11434/api` | ollama 백엔드 주소 (모델은 `/admin`) |
| `LLAMACPP_ENDPOINT` | `…:8080` | llama.cpp/OpenAI 백엔드 주소 (모델은 `/admin`) |

> 모델 태그는 이제 env가 아니라 **`/admin` 엔진 옵션의 드롭다운**에서만 정합니다(백엔드에 설치된 모델을 조회). `state.json`에 영속.

---

## 테스트

```bash
# 코어 빠른 단위 26개, 모델/GPU 불필요 (pytest 미사용, 자체 러너)
cd packages/scanlation-server && ../../venv/Scripts/python -m tests
# 엔진 패키지별 suite (각자 self-contained; 스모크는 가중치/패키지 없으면 자동 skip)
cd packages/scanlation-ctd      && ../../venv/Scripts/python -m tests
cd packages/scanlation-mangaocr && ../../venv/Scripts/python -m tests
cd packages/scanlation-ollama   && ../../venv/Scripts/python -m tests   # HTTP mock
cd packages/scanlation-llamacpp && ../../venv/Scripts/python -m tests   # HTTP mock
```

**검출 육안 확인**(정확도 핵심 루프 — 검출이 병목):
```bash
cd packages/scanlation-server
../../venv/Scripts/python tools/visualize.py page.jpg --detector ctd --out annotated.png   # 폴리곤 + 인덱스
```
`visualize.py`는 `annotated.png` + deskew된 `crops/`를 저장 → 박스 위치와 crop이 똑바른지
눈으로 판단. (실제 ja→ko 확인은 서버+`/admin`+브라우저로.)

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
- **노이즈 영역**: 검출기가 SFX 조각·말줄임표(`．．．`, `マッ`, `みちっ`)까지 잡아 같이 번역됨.
  향후: 문장부호-only OCR 스킵 / 작은 SFX area 컷(P8).

---

## 배포 (리눅스 호스트) — 검증된 절차

**bare-metal 먼저**(빠른 반복); Docker(P7)는 동작한 뒤에.

```bash
# 0) 받기 + 설치 (sdk 먼저; 원하는 엔진 패키지만 골라 설치 가능)
git clone https://github.com/tjdnjsrmsdidgkrdlfma/Scanlation.git
cd Scanlation && python -m venv venv && source venv/bin/activate
pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server \
  -e ./packages/scanlation-ctd -e ./packages/scanlation-mangaocr \
  -e ./packages/scanlation-ollama -e ./packages/scanlation-llamacpp
cd packages/scanlation-server
python tools/install.py            # 모델 가중치 설치 (한 번; = 원클릭 / POST /manage_plugins/)
ollama pull <your-model>           # ollama 모델 (별도 서비스)
```

**1) 서버 띄우기** (포트 자유; 끄지 말 것 — tmux/nohup). 플래그·env 없음:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 4001
#  확인:  curl -s http://127.0.0.1:4001/ | head -c 150
```
> GPU면 provider 힌트로 `SCANLATION_DEVICE=rocm`만 앞에 붙이면 됨.

**2) `/admin`에서 설정** (SSH 터널 뒤 브라우저 → 3번의 터널 사용):
`http://127.0.0.1:4001/admin` → **모델·언어 탭**에서 detector `ctd` · recognizer `mangaocr` ·
translator `ollama` 선택 → **엔진 옵션 탭**에서 translator `model` **드롭다운**으로 pull해둔 모델 선택
→ 저장. 전부 `state.json`에 영속(다음 기동부터 기본값). **모델은 여기서만 지정**(env 없음).

**3) 확장 연결** (브라우저가 다른 PC면 SSH 터널로 혼합콘텐츠·CORS 회피):
```bash
# 브라우저 PC에서:
ssh -L 4001:localhost:4001 -L 8001:localhost:8001 root@<host>
```
팝업 Server = `http://127.0.0.1:4001` → Connect (드롭다운에 `ctd/mangaocr/ollama`).

**4) 로컬 만화를 http로 서빙** (same-origin이라야 content script가 읽음):
```bash
cd /path/to/manga && python -m http.server 8001    # 호스트에서
```
→ 브라우저 `http://127.0.0.1:8001/` 만화 페이지 → **F5 → Enable** → 한국어 오버레이.

**ROCm 불안하면**: `llama-server`(Vulkan) 띄우고 `/admin`에서 translator를 `llamacpp`로 바꾸면 됨
(백엔드 주소만 다르면 `LLAMACPP_ENDPOINT`를 서버 실행 앞에 env로; 기본 `http://127.0.0.1:8080`).

---

## 다른 머신 / 새 세션에서 이어받기

1. 이 README(위 상태) + [YOMU_DESIGN.md](YOMU_DESIGN.md)(왜) 읽기.
2. `pip install -e ./packages/scanlation-sdk -e ./packages/scanlation-server`(+ 원하는 엔진 패키지) 후
   `cd packages/scanlation-server && ../../venv/Scripts/python -m tests` green 확인.
3. P0–P6 완료, **리눅스/gfx1200에서 실제 번역까지 라이브 검증됨**; 다음은 P7(Docker) +
   P8(노이즈 필터·세로쓰기·라틴 폴백 등). 솔로 프로젝트: `main`에 직접 커밋, 의존성은 `venv`.

라이선스: 프로젝트 라이선스 미정(TBD). 트리는 **GPLv3-free**(클린룸; 엔진은 런타임 의존만) —
그대로 유지할 것.

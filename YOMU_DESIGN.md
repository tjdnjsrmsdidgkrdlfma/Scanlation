# Greenfield 만화 OCR+번역 시스템 — 설계 & 핸드오프 문서

> **작업 코드네임:** `yomu` (임시 — 자유롭게 변경. 폴더/패키지/도커 서비스명으로 쓰임)
> **라이선스:** TBD (단 전제: GPLv3 코드는 **복사 금지**, 알고리즘만 독립 재구현)
> **이 문서의 목적:** 컨텍스트가 길어져 **새 세션에서 이어서 빌드**하기 위한 자족적(self-contained) 설계서. 사전 지식 없이 이 문서만으로 구현을 시작할 수 있도록 작성됨.

> ⚠ **이후 구현에서 바뀐 점 (이 문서는 원 설계 기록):** 현재 상태·사용법은 [README.md](README.md), 에이전트 지침은 [CLAUDE.md](CLAUDE.md)를 우선한다. 원 설계 대비 주요 divergence:
> - **역할 어휘 통일** — 구 `ocr_extension` drop-in 호환(BOX/OCR/TSL)을 **폐기**하고 서버·와이어·확장·admin 전 계층을 `detector`/`recognizer`/`translator`로 통일. (결과 키 `{ocr,tsl,box}`만 데이터 필드로 유지)
> - **설정 = `/admin` 단일 소스** — 엔진·모델·언어·프롬프트를 서버 관리 페이지(`/admin`, `state.json` 영속)에서 지정. `OLLAMA_MODEL`/`LLAMACPP_MODEL` 등 **모델 env 폴백 제거**(미설정 시 에러). 모델은 백엔드 설치 목록 드롭다운으로 선택.
> - **테스트 = 자체 핸드롤 러너** — pytest 미사용. `python -m tests`(빠른 스위트) / `python -m tests.test_ctd`(개별 스모크).
> - **CLI 최소화** — `tools/run_image.py` 삭제(실 검증은 `/admin`+브라우저). `tools/visualize.py`(검출 육안 확인)만 유지.
> - **다중 패키지 분리** — `server/` 단일 패키지를 `packages/`의 여러 pip 패키지로 분리: 공유 계약 `scanlation-sdk`(contracts·context·prompt·testing) + 코어 `scanlation-server`(dummy 엔진만 번들) + 엔진별 패키지(`scanlation-{ctd,mangaocr,ollama,llamacpp}`). 코어는 엔진을 전혀 모르고 **`entry_points`로만 발견**(`registry._BUILTIN` 하드코딩 맵 제거) — "설치한 패키지 = 탑재 엔진". 아래 §의 `server/plugins/*`·`app.contracts`·`plugins.llm_prompt` 경로 참조는 이 구조로 이동됨(계약 = `scanlation_sdk`).

---

## 1. Context — 왜 이걸 만드는가

사용자는 브라우저에서 **일본 만화를 읽으며 한국어로 in-place 번역 오버레이**를 띄우는 용도로 Crivella의 오픈소스 스택(`django-ocr_translate` 코어 + 3개 플러그인 + `ocr_extension` 브라우저 확장)을 써왔다. (이 스택은 현재 `c:/Users/MicroJackson/Projects/OCR/` 에 있음.)

**불만 / 동기:**
- OCR 품질이 아쉬움 — **작은 글씨**는 영역 검출/인식 실패, **기울어진 글씨(SFX)**는 인식 실패.
- PaddleOCR로 교체 시도 → 문제 잦고 성능도 아쉬움.
- 남이 만든 코드를 `patch_plugins.py`로 계속 패치하는 구조가 한계.

**진단(검증됨):** 일본 만화에서 진짜 병목은 **인식(recognition)이 아니라 검출/세그멘테이션(detection)**이다. 일반 장면-텍스트 검출기(easyocr)는 만화에서 정확도 <30%. 인식기 manga-ocr은 이미 SOTA. → "깔끔하게 잘린 영역"을 일반 검출기가 못 만들어서 manga-ocr이 쓰레기 입력을 받는 게 문제.

**핵심 통찰:** Crivella의 box 계약은 **축정렬 사각형 전용**(`{single:[lbrt...], merged:lbrt}`)이라 기울어진 글씨를 **구조적으로** 못 고친다. 새 계약이 **회전 기하(폴리곤/각도/마스크)**를 품으면, 검출→deskew→인식 파이프라인으로 기울기를 근본 해결할 수 있다.

**검증 결과(소스 직접 확인):** 우리가 쓸 엔진들이 실제로 회전 기하를 주고받는다 →
- **comic-text-detector**(검출): 4점 폴리곤(`polys.reshape(-1,8)`) + 세그멘테이션 마스크 반환. ONNX 버전 존재(`mayocream/comic-text-detector-onnx`) → torch 불필요, onnxruntime로 CPU/ROCm/DirectML.
- **manga-ocr**(인식): 순수 인식기. PIL crop 입력, 내부 검출/회전보정 **없음**, **똑바로 세운 crop을 기대**. Apache-2.0.
- **manga-image-translator**(deskew 레퍼런스): `get_transformed_region()`이 `cv2.findHomography`+`warpPerspective`로 회전 quad를 똑바로 펴고 세로글자는 `cv2.rotate`. ⚠️ GPLv3 → **코드 복사 금지, 알고리즘만 재구현**.
- 일반성: PaddleOCR/RapidOCR도 4점 quad 반환, easyocr는 축박스(=직각 quad, 퇴화 케이스) → 폴리곤 계약이 모든 엔진에 우아하게 적용됨.

**결정 사항(사용자 확정):**
1. 그린필드로 **자기 소유의 계약 기반 서버 + MV3 브라우저 확장**을 새로 만든다 (Crivella 포크/패치 아님).
2. 원문 = 일본어 만화, 번역 = 한국어, **정확도 최우선**.
3. 첫 플러그인 3개(전부 교체 가능): 검출=comic-text-detector(ONNX), 인식=manga-ocr, 번역=ollama(기존 셋업 재활용).
4. 확장은 MV3로 재작성 (Chrome+Firefox). 기존 확장은 MV2/Firefox라 어차피 Chrome 지원 불가.

**환경:** Docker/Linux 호스트 + AMD **gfx1200(RDNA4)** GPU(ROCm). VRAM 빡빡 — Ollama ≈14GB / 16GB 카드라 OCR과 경합.

---

## 2. 검증된 Ground Truth (구현 시 반드시 지킬 것)

### 2.1 확장이 말하는 wire 프로토콜 (원 설계: drop-in 호환 대상)
출처: `ocr_translate/ocr_translate/views.py`, `ocr_extension/src/utils/API.js`, `.../content.js`.

> ⚠ **이후 변경됨:** 구 `ocr_extension` drop-in 호환 목표는 폐기했다. 서버·번들 확장 둘 다
> 이 레포 소유라, 역할 필드를 끝단까지 **detector/recognizer/translator**로 개명했다
> (`BOXModels→detectors`, `box_selected→detector_selected`, `box_model_id→detector` 등).
> 결과 아이템 키 `{ocr, tsl, box}`만 데이터 필드로 유지. 아래 표는 원 설계 기록.

| 엔드포인트 | 요청 | 응답 |
|---|---|---|
| `GET /` (handshake) | 없음 | `{version:[M,m,p], Languages, Languages_src, Languages_dst, Languages_hr, BOXModels, OCRModels, TSLModels, box_selected, ocr_selected, tsl_selected, lang_src, lang_dst}` |
| `POST /run_ocrtsl/` | `{md5, contents?(base64), force?, options?}` | `{result:[{ocr, tsl, box}]}` |
| `POST /run_tsl/` | `{text}` | `{text}` |
| `GET /get_trans/` | `?text=&lang_src=&lang_dst=` | `{translations:[{model,text}]}` |
| `POST /set_manual_translation/` | `{text, translation}` | `{}` |
| `POST /set_models/` | `{box_model_id, ocr_model_id, tsl_model_id}` | `{}` |
| `POST /set_lang/` | `{lang_src, lang_dst}` | `{}` |
| `GET /get_active_options/` | 없음 | `{options:{box_model:{opt:{type,default,description}}, ocr_model:{}, tsl_model:{}}}` |
| `GET /get_plugin_data/` | 없음 | `{name:{homepage,warning,description,version,installed}}` |
| `POST /manage_plugins/` | `{plugins:{name:bool}}` | `{status:'success'}` |

**Lazy 흐름:** 클라이언트는 먼저 `md5`만 POST(`/run_ocrtsl/`) → 캐시 미스로 non-2xx 받으면 `contents`(base64) 포함해 재요청. (`API.js:getOcr` = `try{lazy}catch{work}`.)

### 2.2 ⚠️ 절대 바꾸면 안 되는 디테일 2개
1. **md5는 base64 *문자열* 기준으로 계산.** 클라 `content.js:249` `md5(base64data)`; 서버 `hashlib.md5(b64.encode('utf-8')).hexdigest()`. raw 바이트로 해시하면 매 요청 400.
2. **box 포맷은 `[left, bottom, right, top]` (lbrt) 이고 의미가 특수.** 서버는 `(x_min, y_min, x_max, y_max)`(원점 좌상단, 픽셀)를 내보내되 클라가 `[l,b,r,t]`로 읽음 → 여기서 `b=y_min`(위쪽 모서리), `t=y_max`(아래쪽 모서리), 즉 수치상 `b<t`. 클라 `utils/textbox.js`가 `top=b/H, height=(t-b)/H, left=l/W, width=(r-l)/W`로 그린다. **이 순서 그대로 유지.**

### 2.3 Ollama 번역 (그대로 재사용)
출처: `ocr_translate-ollama/ocr_translate_ollama/plugin.py` + `commons.py`, 사용자 튜닝 `model_test.py`.
- 엔드포인트 기본 `http://127.0.0.1:11434/api`; 문장별 `POST /generate` `{model, prompt, system, stream:False, think:False, options:{num_ctx, num_gpu, temperature, seed, top_p}}`.
- **모델명 프리픽스 `oct_ollama_` 필수** (없으면 번역 깨짐 — 메모리 확인됨). load 시 프리픽스 제거 후 pull/create.
- 패치: **`think:False`** = 추론모델에서 ~11x 속도(숨은 `<think>` 제거), **`num_ctx:512`** = KV캐시 VRAM ~1GiB 절약(번역 입력 <200토큰이라 충분).
- 사용자 working config: `temperature:0, seed:42, top_p:1.0, num_gpu:31, num_ctx:512`.
- 시스템 프롬프트: "번역만 하라 + OCR 오류 감안 + 컨텍스트 활용 + 추론은 한 문장으로". `_translate`는 한 문장의 토큰을 `'. '`로 join, ≤2자 입력은 모델 호출 스킵. src/dst는 평문("japanese"/"korean").

### 2.4 기존 Docker/런타임 (패턴 재사용)
`ocr_translate/Dockerfile`: `python:3.13-slim`, `libgl1`/`nginx`/`tesseract` 설치, EXPOSE 4000, gunicorn, `DEVICE` 환경변수(cpu 기본; cuda/rocm는 override + 해당 torch 빌드), `OCT_BASE_DIR` 데이터 루트, 플러그인 scope 디렉터리(`generic`/`cpu`/`cuda`/`rocm`). **Ollama는 별도 서비스**(`OCT_OLLAMA_ENDPOINT`).

### 2.5 버릴 것 (단순화)
- 기존 `queues.py`의 스테이지별 worker 큐(`NUM_BOX/OCR/TSL_WORKERS`) + `messaging.py` 커스텀 dedupe/batch 레이어 → Django sync/WSGI 때문에 존재. FastAPI async + GPU asyncio 락으로 대체.
- 2단 entry_points(class 그룹 + data 그룹) + DB행 materialize + Django proxy 모델 → 1단 entry_points + 클래스 속성 메타데이터로 단순화.
- 런타임 pip 설치 플러그인 매니저(`plugin_manager.py`) → v1 범위 밖(스텁 처리).

---

## 3. 아키텍처 — 서버

**스택:** FastAPI + uvicorn(async), Pydantic v2; **onnxruntime**(CTD 검출, provider 선택 CPU/ROCm/DirectML, torch 불필요); **manga-ocr** 패키지(인식); **httpx**(ollama async); **opencv-python-headless + numpy + Pillow**(기하); **sqlite**(stdlib, 캐시+TM); **pytest + TestClient**(테스트). Python 3.11+.

**근거:** Django ORM/마이그레이션/커스텀 워커큐 제거. 실제 작업은 단일 GPU 락 뒤의 CPU/GPU 연산이지 고동시성 웹이 아님 → async + semaphore가 스테이지별 워커풀보다 단순/적합.

### 3.1 파일 구조
```
yomu-server/
  pyproject.toml            # FastAPI 앱 + 내장 플러그인 entry_points
  docker/{Dockerfile, docker-compose.yml, docker-compose.cpu.yml, nginx.default}
  app/
    main.py                 # app factory, 라우트, lifespan
    config.py               # env 설정(pydantic-settings) + iso1 언어 테이블
    contracts.py            # Region + Detector/Recognizer/Translator Protocol
    geometry.py             # deskew: homography/warpPerspective + 세로처리
    pipeline.py             # detect -> deskew -> recognize -> translate
    registry.py             # entry_points 발견 + 선택 + 옵션 스키마
    cache.py                # sqlite 결과 캐시 + 수동 번역(TM)
    state.py                # 선택된 엔진/언어/옵션 + GPU 락
    schemas.py              # pydantic 요청/응답 모델
    routes/{handshake,run,manual,settings_routes,plugins}.py
  plugins/
    detector_ctd/{plugin.py, decode.py}
    recognizer_mangaocr/plugin.py
    translator_ollama/plugin.py
    dummy/plugin.py
  tests/{conftest.py, fixtures/, test_geometry.py, test_contracts.py,
         test_pipeline.py, test_routes.py, test_ctd.py}
  tools/{run_image.py, visualize.py, bench_translate.py}
  models/ data/             # 볼륨, gitignore
```

> **repo 구조 결정: 모노레포 (서버 + 확장 한 리포)**
> Crivella는 `ocr_translate`(PyPI 패키지)와 `ocr_extension`(스토어 확장)을 **별도 리포**로 뒀다 —
> 서로 다른 생태계에 독립 배포·버전·라이선스되는 두 제품이라서. 우리는 다르다:
> - **솔로 + 와이어 계약 공동 진화** — 서버·확장이 같은 JSON 계약을 쓰고 함께 바뀐다. 한 리포면
>   계약 변경(예: 역할 어휘 rename)을 **한 커밋**으로 반영; 나누면 두 리포·버전맞추기·비호환 창이 생긴다.
> - **배포에 분리 불필요** — AMO는 `extension/` zip, Docker(P7)는 `server/`에서 빌드.
> - **락인 없음** — 나중에 독립 배포/라이브러리 공개 같은 트리거가 생기면 히스토리 보존해 분리:
>   `git subtree split --prefix=extension`(또는 `git filter-repo --subdirectory-filter extension`).
>
> → 지금은 모노레포 유지. **개발을 다 하고 나눠도 늦지 않다.**

### 3.2 플러그인 발견 — 권장: entry_points 1단 + 내장 fallback
프로젝트가 자기 내장 엔진을 pyproject에 3개 그룹으로 선언:
```toml
[project.entry-points."yomu.detectors"]
ctd   = "plugins.detector_ctd.plugin:CTDDetector"
dummy = "plugins.dummy.plugin:DummyDetector"
[project.entry-points."yomu.recognizers"]
mangaocr = "plugins.recognizer_mangaocr.plugin:MangaOcrRecognizer"
dummy    = "plugins.dummy.plugin:DummyRecognizer"
[project.entry-points."yomu.translators"]
ollama = "plugins.translator_ollama.plugin:OllamaTranslator"
dummy  = "plugins.dummy.plugin:DummyTranslator"
```
`registry.py`가 시작 시 각 그룹 `entry_points(group=...)` 읽어 `{name→class}` 맵 구성. 선택 엔진은 **첫 사용 시 lazy 인스턴스화**(그때 VRAM 로드). 서드파티 플러그인은 같은 그룹 선언하는 패키지를 `pip install`하면 자동 발견(코어 수정 0). 소스 체크아웃(pip 미설치) 대비 **내장+dummy용 작은 하드코딩 fallback**을 두고 entry_points로 병합.

`state.py`: 역할별 선택 엔진 이름 + src/dst 언어 + 엔진별 옵션 오버라이드를 작은 json/sqlite에 영속화(재시작 시 선택 유지).

### 3.3 계약 모듈 (`contracts.py`)
```python
@dataclass
class Region:
    polygon: np.ndarray          # (4,2) float32, 이미지 px, 비(非)축정렬
    bbox: tuple[int,int,int,int] # 축정렬 (x_min,y_min,x_max,y_max), polygon에서 유도
    angle: float = 0.0           # deskew용 부호있는 각도(deg)
    vertical: bool = False       # 일본어 세로쓰기
    mask: np.ndarray | None = None  # 선택: 영역별 seg 마스크(향후 inpaint/타이트crop)
    score: float = 1.0
    order: int = 0               # 읽기 순서(파이프라인이 부여; 만화 R->L, T->B)

class Detector(Protocol):
    def detect(self, image: PIL.Image.Image, options: dict) -> list[Region]: ...
class Recognizer(Protocol):
    def recognize(self, crop: PIL.Image.Image, region: Region, options: dict) -> str: ...  # crop은 이미 deskew된 upright
class Translator(Protocol):
    def translate(self, text: str, src: str, dst: str, options: dict) -> str: ...
    def translate_batch(self, texts, src, dst, options) -> list[str]: ...  # 선택
```
- 모든 엔진은 라이프사이클 mixin 구현: 클래스 속성 `name, display_name, homepage, warning, description, OPTION_SCHEMA, SUPPORTED_SRC, SUPPORTED_DST` + `load()` / `unload()`.
- `Region.from_quad(quad)` 헬퍼 → 축박스만 주는 검출기도 직각 quad로 valid Region 생성(angle=0).
- **wire 전송:** `bbox`만 `[x_min,y_min,x_max,y_max]`(= 클라 `[l,b,r,t]`)로 직렬화. polygon/angle/mask는 서버 내부용(deskew/향후 inpaint), v1 wire 포맷 불변.

**옵션 스키마(`OPTION_SCHEMA`)** — 기존 popup이 그대로 렌더되도록 `{type, default, description}` 삼중 유지. `/get_active_options/`는 `type.__name__`으로 직렬화. 기존 "cascade" 디폴트 메커니즘은 **제거**(과설계), 디폴트는 평문 리터럴.

### 3.4 엔드포인트 매핑 (FastAPI, 확장에 drop-in)
모든 POST는 CSRF 면제(FastAPI엔 CSRF 없음). **CORS 미들웨어 활성화**(기존엔 주석처리됨).
- `GET /` → 2.1 키 그대로. 언어 리스트는 `config.py`의 정적 iso1 테이블(ja/ko/en…)에서. handshake는 모델 로드 안 함(가벼움).
- `POST /run_ocrtsl/` → lazy/work 흐름(2.1) + md5 검증(2.2) + GPU 락 후 `pipeline.run` + 결과/TM 저장. 동시 동일요청은 in-memory `dict[id→asyncio.Future]`로 dedupe.
- `POST /run_tsl/`, `GET /get_trans/`, `POST /set_manual_translation/` → TM(`favor_manual` 기본 True: 수동번역 우선).
- `set_models`/`set_lang` → `state.py` 갱신(이름 검증, eager 로드 안 함). `get_active_options` → 선택 엔진 스키마. `get_plugin_data`/`manage_plugins` → 발견된 엔진 목록 / manage는 v1 no-op 스텁 `{status:'success'}`.

### 3.5 파이프라인 (`pipeline.py` + `geometry.py`)
```python
regions = detector.detect(img, opt_box)
regions = assign_reading_order(regions, vertical_hint=(src=="ja"))  # 만화 R->L, T->B
for r in regions:
    crop = geometry.deskew_crop(img, r)
    text = recognizer.recognize(crop, r, opt_ocr)
    if not text.strip(): continue
    tsl = translator.translate(text, src, dst, opt_tsl)
    out.append({"ocr": text, "tsl": tsl, "box": list(r.bbox)})
```
**deskew(`geometry.deskew_crop`) — GPLv3 코드 복사 금지, 표준 OpenCV 레시피로 독립 재구현:**
1. `region.polygon`에서 `cv2.minAreaRect`로 최소회전사각 → 목표 W,H.
2. `M = cv2.getPerspectiveTransform(src_quad, dst_rect)`; `crop = cv2.warpPerspective(img, M, (W,H))`.
3. 방향 복원(2.3/0.7 규칙): 세로/가로 판단(aspect + 검출기 힌트). **일본어 세로글자는 세로 유지**(manga-ocr이 세로 native). warp가 텍스트 자연방향 대비 누웠으면 `cv2.rotate`/transpose로 "사람이 읽는 방향"으로.
4. `PIL.Image.fromarray`. 엣지: 영(0)면적 quad 스킵, 초소형 crop은 최소크기 패딩, grayscale→RGB.

**동시성:** 전역 `asyncio.Lock`("gpu_lock")으로 detect+recognize 보호(CTD ONNX + manga-ocr torch가 GPU 경합; ollama는 별 프로세스). 핸들러는 `async def`, CPU/GPU 작업은 락 안에서 `run_in_threadpool`로 실행(이벤트 루프 비차단). `uvicorn --workers 1`(VRAM 모델 1벌). 배치는 v1에서 순차 루프(페이지당 5~30영역).

### 3.6 캐시 + 수동 TM (`cache.py`)
단일 sqlite(`data/yomu.sqlite`, WAL + 스레드락):
```
images(md5 PK)
ocr_runs(md5, src, dst, box, ocr, tsl, opt_hash, result_json, created_at)  PK(md5,src,dst,box,ocr,tsl,opt_hash)
translations(src_text, src_lang, dst_lang, model, dst_text, created_at)   -- model='manual' 우선
```
lazy = ocr_runs PK SELECT; `force=True` 덮어쓰기; get_trans = translations SELECT; set_manual = UPSERT(model='manual'). `opt_hash` = 정규화 json 옵션의 sha256.

---

## 4. 첫 3개 플러그인 (개요)

**4.1 CTDDetector** (`plugins/detector_ctd/`) — `mayocream/comic-text-detector-onnx`(ONNX).
- `load()`: `ort.InferenceSession(path, providers=<선택>)`; provider는 env `DEVICE`로 `ROCMExecutionProvider`(Linux ROCm) / `DmlExecutionProvider`(Windows 로컬개발) / 아니면 CPU — **항상 CPU fallback 추가**.
- `detect()`: letterbox 리사이즈→정규화→추론; `decode.py`가 출력`(blks, mask, mask_refined)`→라인 quad(`polys.reshape(-1,8)`)+세그; thresh/unclip(pyclipper)/NMS; letterbox 역변환으로 원본 px 매핑; 라인별 Region.
- ⚠️ **출력 텐서 이름/순서/후처리는 실제 onnx + 레퍼런스 `comic_text_detector/inference.py`(TextDetector.__call__)로 반드시 검증** — 최대 미지수(6번).
- 가중치는 첫 로드시 `models/ctd/`로 다운로드 또는 이미지에 굽기.

**4.2 MangaOcrRecognizer** (`plugins/recognizer_mangaocr/`) — `from manga_ocr import MangaOcr; self.m=MangaOcr()`; `self.m(pil_crop)`→str(Apache-2.0). 또는 `kha-white/manga-ocr-base` VED를 transformers로 직접(디바이스 제어 필요시; 기존 `ocr_translate-hugging_face/.../ved.py` 패턴: image_processor→generate→decode). **패키지 우선**, 디바이스 제어 필요하면 수동 VED. 회전은 파이프라인이 이미 처리(2.3) → region.vertical로 회전 안 함.

**4.3 OllamaTranslator** (`plugins/translator_ollama/`) — 기존 plugin+commons+model_test.py 튜닝 이식. env `OLLAMA_ENDPOINT`(기본 `.../api`)/`OLLAMA_MODEL`(프리픽스 `oct_ollama_`), 옵션 `num_ctx:512,num_gpu:31,temperature:0,seed:42,top_p:1.0,think:False`. 프롬프트 템플릿(src/dst/context/text) + 시스템프롬프트(2.3 verbatim); `/generate stream:False think:False`; ≤2자 스킵; httpx async 넉넉한 타임아웃; iso1→언어명 매핑(ja→"japanese", ko→"korean").

**4.4 Dummy** (`plugins/dummy/`) — DummyDetector(하드코딩 1~2 Region, 회전 quad 1개 포함해 deskew 검증용), DummyRecognizer("REGION-N"), DummyTranslator(`f"[{src}->{dst}] {text}"`). P1 스켈레톤/CPU CI/가중치 없는 Claude 반복용.

---

## 5. 브라우저 확장 (MV3)

**그대로 가져옴(엔진 무관, API만 교체):** content 스크립트 파이프라인(이미지 발견 + MutationObserver + base64 + md5 + lazy getOcr + 텍스트박스 오버레이 + 클릭=원문복사 + 컨텍스트메뉴=대체번역 + 원문/번역 토글 + writing-mode). `content.js`, `utils/{textbox,blob,image,wrapper,contextmenu,API}.js`. **md5(base64) + box `[l,b,r,t]` 처리(2.2) 정확히 유지.** popup 기능셋 동일.

**MV3로 바꿔야 할 것:** `manifest_version:3`; `browser_action`/`page_action`→단일 `action`; `background.scripts`→`background.service_worker`(ES 모듈, 비영속); `host_permissions:["<all_urls>"]`; `permissions:["activeTab","scripting","storage","contextMenus","tabs"]`; `tabs.executeScript/insertCSS`→`chrome.scripting.executeScript/insertCSS`(또는 message 토글 content_scripts); 서비스워커는 장기 상태 못 가짐 → hub 변수(endpoint/scale/color…)를 `storage.local`을 SoT로; `pageAction`→`action.setIcon`+탭플래그; `browser.menus`→`chrome.contextMenus`(onInstalled에서 생성). **webextension-polyfill**로 Chrome+Firefox 동일 `browser.*` 코드.

**React popup 유지 권장**(이미 동작: handshake 라이브쿼리 + 동적 모델/옵션/플러그인 목록 + 테마). 번들러만 MV3(서비스워커 모듈 엔트리)로, API base만 재지정, 동일 JSON으로 재테스트.

**구조:** `extension/{manifest.json(v3), popup.html/css, icons, content.css, dist/{service-worker.js, content.js, popup.js}}`; `src/{service-worker.js(구 background.js), content.js, popup.js, utils/*, components/*, utils/browser.js(polyfill)}`; 엔트리 3개 번들러.

---

## 6. Docker/빌드 — 사용자 질문 답변: "빌드 도커로 해야 하나?"

**판단의 핵심(ROCm 현실):** gfx1200(RDNA4)은 최신이라 ROCm 지원이 최근/버전민감. ort-rocm·torch-rocm 휠이 호스트 ROCm과 맞아야 하고 gfx1200 사전빌드 휠이 없을 수 있음(`HSA_OVERRIDE_GFX_VERSION` 또는 소스빌드 필요할 수도) — Docker가 핀(pin)하기 딱 좋은 취약한 호스트 의존성. 사용자는 이미 구 서버+ollama를 이 Linux/ROCm 호스트에서 Docker로 돌리는 중.

**권장: 하이브리드 — 개발은 로컬 CPU, 배포는 Docker+ROCm.**
- **로컬 개발**(Windows / Claude 샌드박스): `DEVICE=cpu`(또는 CTD에 DirectML). Dummy + CTD-CPU + manga-ocr-CPU로 Claude가 GPU 없이 완전 반복. 빠른 루프, 리빌드 없음.
- **배포**(Linux/ROCm): compose 2서비스 — `ocr-server`(우리 FastAPI, `DEVICE=rocm`, ort-rocm+torch-rocm 호스트 ROCm에 핀, `/dev/kfd`+`/dev/dri` + `video` 그룹, 필요시 `HSA_OVERRIDE_GFX_VERSION`)와 `ollama`(공식 이미지 11434, 자기 VRAM). 서버는 `OLLAMA_ENDPOINT=http://ollama:11434/api`로 접근. 볼륨: `./models:/models`(CTD+manga-ocr 가중치+`HF_HOME`), `./data:/data`(sqlite), ollama 자체 볼륨. GPU 없는 스모크용 `docker-compose.cpu.yml` override + 무도커 `pip install -e .` 경로도 제공.
- **Dockerfile**(구 패턴 재사용): 멀티스테이지(builder venv→slim+`libgl1`), EXPOSE 4000, `uvicorn app.main:app --host 0.0.0.0 --port 4000 --workers 1`. 타깃 `cpu`(평범한 휠)/`rocm`(ort-rocm+torch-rocm index URL). nginx 선택(uvicorn 직접 서빙 가능).

**결론: 예, Docker — 단 배포/ROCm 용으로만.** 일상 개발과 Claude 테스트는 속도를 위해 컨테이너 없이 로컬 CPU로. 컨테이너는 재현 가능한 ROCm 배포 타깃.

---

## 7. Claude 직접 테스트 — 사용자 질문 답변: "개발 중 Claude가 직접 테스트하면 좋겠다"

**필요 조건:** CPU 전용 빠른 경로(dummy + CTD/manga-ocr CPU provider, `DEVICE=cpu`); 서버 타격 수단 — pytest의 FastAPI `TestClient`/`httpx.AsyncClient` 우선(네트워크 없이 빠르고 단언 가능) + `uvicorn & + curl` 수동; 샘플 만화 픽스처 `tests/fixtures/`(세로글자 1, 기울어진 SFX 1, 작은글씨 1 — 사용자 실샘플은 `pixic/` 및 `make_viewer.py` 흐름에 있음; private면 합성 placeholder).

**하네스:**
- `tools/run_image.py <image> [--src ja --dst ko] [--engines ctd,mangaocr,ollama|dummy…]` → 파이프라인 JSON `[{ocr,tsl,box}]` 출력.
- `tools/visualize.py <image> [--out annotated.png]` → 각 영역의 **폴리곤**(bbox 아닌) + 인덱스 + 인식텍스트를 이미지 복사본에 그려 저장 → **사용자가 눈으로 검출/deskew 검증**. 추가로 deskew된 crop을 `crops/region_NN.png`로 덤프 → upright 잘 됐는지 확인. **검출이 병목이므로 이게 핵심 정확도 디버깅 도구.**
- pytest:
  - `test_geometry.py` — 합성 회전사각으로 deskew **단위테스트**(upright 크기/방향; 축 quad는 no-op; 세로처리). 모델 없이 빠름 — 최고위험 수학 커버.
  - `test_routes.py` — dummy로 프로토콜 호환: handshake 키; WORK(정확한 md5(base64)→200 + box 길이4; 틀린 md5→400); LAZY(미지 md5→non-2xx; WORK 후 동일 md5 lazy→200 캐시); run_tsl/get_trans/set_manual 왕복; get_active_options 형태; set_models/set_lang 반영.
  - `test_pipeline.py` — dummy end-to-end 픽스처 → 안정 JSON.
  - `test_ctd.py`(`@pytest.mark.slow`) — CTD 픽스처, 영역수 > N 스모크.
- 골든 vs 스모크: 모델 구동 테스트는 **스모크**(개수/비어있지않음/박스 이미지내), dummy 결정적 출력만 **골든** json.
- `tools/bench_translate.py` — `model_test.py` 이식, ja→ko 셋으로 ollama 재튜닝(model/num_gpu/num_ctx).
- Make 타깃: `make test`(`-m "not slow"`), `test-all`, `run-cpu`, `viz IMG=…`, `serve`. README에 문서화 → 새 Claude 세션이 self-serve.

---

## 8. 단계별 빌드 순서 (각 단계 독립 테스트 가능)
- **P0 스켈레톤** — scaffold, pyproject, `main.py`(FastAPI + handshake + CORS). TEST: `curl /` valid; popup 연결되어 (빈) 드롭다운 표시.
- **P1 계약+파이프라인+dummy** — contracts/registry/state/cache/dummy 파이프라인/전 라우트. TEST: `test_routes.py`+`test_pipeline.py` green; 확장이 dummy로 실제 만화에 박스+가짜텍스트(=md5/box순서/lazy를 모델위험 0으로 검증).
- **P2 기하/deskew** — `deskew_crop` 구현+단위테스트, 파이프라인 연결(dummy 검출기가 회전 quad 방출해 검증).
- **P3 CTD 검출** — CTDDetector + decode.py(CPU 먼저). TEST: `visualize.py`로 폴리곤 육안 검사 + test_ctd 스모크. **여기서 ONNX 디코딩 검증.** 정확도 검증의 핵심.
- **P4 manga-ocr** — MangaOcrRecognizer(CPU). TEST: P3 crop 먹여 일본어 비어있지않음, 알려진 패널 스팟체크.
- **P5 ollama** — 이식 + 프롬프트 + 튜닝. TEST: bench_translate.py ja→ko; run_ocrtsl 실제 ocr+tsl end-to-end.
- **P6 확장 MV3** — manifest v3, 서비스워커, scripting 주입, polyfill, popup 재빌드. TEST: Chrome+Firefox unpacked 로드; 저장된 pixiv `viewer.html` 및 실사이트 오버레이.
- **P7 Docker/ROCm** — Dockerfile(cpu+rocm), compose, 볼륨, 디바이스 마운트, gfx1200 env. TEST: 호스트 `compose up`; active provider 로그; 컨테이너 서버 end-to-end.
- **P8 튜닝** — 작은+SFX 텍스트 thresh/unclip; deskew 품질; 읽기순서; VRAM 예산(CTD+manga-ocr vs ollama 14GB/16GB — detect/recognize를 CPU로 돌리고 GPU는 ollama 전용 등).

---

## 9. 열린 리스크 / 미결정
1. **gfx1200(RDNA4) ort-ROCm 성숙도:** 사전빌드 휠 부재 가능 → DirectML(Win 개발)/CPU fallback/소스빌드. 완화: CTD는 작아 CPU로도 충분; provider 선택은 CPU fallback + active provider 로그.
2. **manga-ocr(torch-rocm) VRAM 경합:** ollama ≈14GB/16GB라 CTD+manga-ocr OOM 위험. 선택지(P8 결정): (a) **CTD+manga-ocr를 CPU로, GPU는 ollama 전용**[정확도우선+빡빡VRAM+페이지당 영역 적음 → 가장 안전한 기본]; (b) ollama 전후 load/unload; (c) ollama `num_gpu` 캡.
3. **CTD ONNX 디코딩 미검증:** 출력 이름/순서, thresh/unclip/NMS, letterbox 좌표역변환을 실제 모델+레퍼런스 `inference.py`로 확인 필수. 최대 미지수 — P3에 실시간 확보, `visualize.py`를 먼저 만들어 육안 검증.
4. **라이선스(중요):** manga-image-translator=GPLv3 → `get_transformed_region` **복사 시 우리 서버도 GPLv3 전염**. deskew는 표준 homography → **알고리즘 설명만 보고 OpenCV 프리미티브로 독립 재구현**(§3.5). manga-ocr=Apache-2.0(런타임 의존 OK). comic-text-detector 가중치 라이선스는 번들 전 확인. 구 Crivella 코드(GPLv3)는 **학습만, 복사 금지**. → **프로젝트 라이선스 미정(TBD)**, 단 트리에 GPLv3 코드 미포함, 런타임 의존만.
5. **세로/방향 정확성:** "기울기는 펴되 세로는 세로 유지"는 실제 세로 일본어+기울어진 SFX로 경험적 검증 필요 → visualize.py + crop 덤프로 P3~P4 튜닝.
6. **엔진별 디바이스 배치는 런타임 정책(계약 아님):** CTD≠manga-ocr≠ollama 디바이스를 독립 설정 가능하게 → 코드변경 없이 CPU/GPU 이동.
7. **uvicorn 단일 워커** 필수(VRAM 모델 1벌) → 프로세스 병렬 없음; asyncio 락+threadpool로 단일사용자 충분(멀티유저 원하면 재검토).
8. **프로젝트 이름 미정:** 작업명 `yomu` — 폴더/패키지/도커명 일괄 변경 가능.

---

## 10. 먼저 읽을 기존 파일 (구현 시작 시)
- `c:/Users/MicroJackson/Projects/OCR/ocr_translate/ocr_translate/views.py` — 엔드포인트 정확한 형태(handshake/run_ocrtsl/get_active_options 등)
- `c:/Users/MicroJackson/Projects/OCR/ocr_translate/ocr_translate/ocr_tsl/full.py` — 구 파이프라인 흐름(detect→ocr→tsl)
- `c:/Users/MicroJackson/Projects/OCR/ocr_extension/src/content.js` — 확장 이미지 처리 파이프라인(md5/lazy/오버레이)
- `c:/Users/MicroJackson/Projects/OCR/ocr_extension/src/utils/textbox.js` — box `[l,b,r,t]` 렌더 의미
- `c:/Users/MicroJackson/Projects/OCR/ocr_extension/src/utils/API.js` — 클라 API 호출/lazy 흐름
- `c:/Users/MicroJackson/Projects/OCR/ocr_translate-ollama/ocr_translate_ollama/{plugin.py,commons.py}` — ollama 번역 이식 원본
- `c:/Users/MicroJackson/Projects/OCR/model_test.py` — 사용자 ollama 튜닝(num_gpu/num_ctx/시스템프롬프트)
- `c:/Users/MicroJackson/Projects/OCR/ocr_translate-easyocr/ocr_translate_easyocr/plugin.py` — 교체 대상(축정렬 검출기) 참고
- (외부) comic-text-detector `inference.py` + `mayocream/comic-text-detector-onnx` — CTD 디코딩 검증

## 11. 검증 (end-to-end 테스트 방법)
- **단위:** `make test` → geometry(deskew) + routes(프로토콜, dummy) + pipeline(dummy). GPU/모델 불필요, 빠름.
- **검출 육안:** `python tools/visualize.py <manga.png> --out annotated.png` → 폴리곤/인덱스/인식텍스트 오버레이 + `crops/region_NN.png` deskew 결과. 사용자가 파일로 확인.
- **실엔진 스모크:** `make test-all`(`@slow` 포함) → CTD 영역수 > N, manga-ocr 비어있지않음.
- **번역:** `python tools/bench_translate.py` → ja→ko 셋으로 ollama 품질/속도.
- **확장 통합:** Chrome/Firefox에 unpacked 로드 → 저장된 pixiv `viewer.html` 및 실 만화 사이트에서 오버레이 확인(원문/번역 토글, 세로쓰기, 클릭복사).
- **배포:** Linux/ROCm 호스트 `docker compose up --build` → active provider(ROCm/CPU) 로그 확인 → 컨테이너 서버로 확장 end-to-end.

---
*작성: 설계 단계(plan mode). 다음 세션은 이 문서 + §10 파일들로 P0부터 시작. 첫 작업 추천: 저장소 scaffold(§3.1) → P0/P1(dummy로 프로토콜 검증) → P3에서 `visualize.py` 먼저.*

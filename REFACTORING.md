# 리팩토링 백로그

트리 전체를 네 축(**벤치 하네스 통합 / 어휘·이름 정리 / 대형 파일 분할 / 하드코딩→`/admin` 노출**)으로 훑은
결과. 원칙은 **동작 보존** — 외부 동작(와이어 계약·CLI 출력·엔진 결과)은 그대로 두고 내부 구조만 바꾼다.
[CLAUDE.md](CLAUDE.md)의 코드 규칙과 커밋 규칙을 따른다: 항목 하나 = 커밋 하나.

기준선: 코어 단위테스트(`python -m tests`)가 회귀 게이트다.

옵션 어휘(`opt_detect`/`opt_recognize`/`opt_translate`)와 결과 키(`{bounds, source, destination}`)는 규칙을
온전히 지키고 있고, 플러그인 튜닝 값은 대부분 `OPTION_SCHEMA`로 `/admin`에 노출된다. 부채는 세 지점에
몰려 있다 — **`tools/` 벤치의 중복**, **확장의 하드코딩**, **`plugins_install.py`의 이중 구현**.

---

## Tier 0 — 리팩토링이 아니라 버그 (수정 여부 별도 결정)

동작 보존 원칙상 리팩토링 커밋에 끼워 고치지 않는다. 전부 코드에서 확인됨.

| # | 위치 | 내용 |
|---|---|---|
| **B1** | [pipeline.py:22](packages/scanlation-server/app/pipeline.py#L22), [:57](packages/scanlation-server/app/pipeline.py#L57) | `assign_reading_order(regions, vertical_hint)`의 본문(:28-40)이 `vertical_hint`를 **한 번도 읽지 않는다.** 호출부는 `vertical_hint=(src == "ja")`를 계산해 넘긴다. 세로쓰기 일본어 읽기 순서가 적용되지 않고 있을 수 있다. 파라미터를 지울지, 실제로 구현할지 결정 필요 |
| **B2** | [logconfig.py:56](packages/scanlation-server/app/logconfig.py#L56) + [main.py:65-69](packages/scanlation-server/app/main.py#L65-L69) | `apply_verbose(False)`가 `INFO`를 리터럴로 세팅한다. lifespan이 `configure_logging(settings.log_level)` 직후 이걸 부르므로 `SCANLATION_LOG_LEVEL=WARNING` + `verbose_log=False`면 레벨이 조용히 INFO로 되돌아간다. off 분기는 `settings.log_level`로 복귀해야 한다 |
| **B3** | [registry.py:76-78](packages/scanlation-server/app/registry.py#L76-L78) | `registry.get(name, device=...)`의 lock-free 캐시 히트 경로가 `device` 인자를 검사하지 않는다. `set_engine_device` → `unload_one` 순서 덕에 현재는 안전하지만 [orchestrator.py:86-87](packages/scanlation-server/app/orchestrator.py#L86-L87)이 매 요청 `device=`를 넘기므로 계약이 거짓이다 |
| **B4** | [orchestrator.py:129-132](packages/scanlation-server/app/orchestrator.py#L129-L132) | `_run_deduped`에서 대기자가 없을 때 `fut.set_exception()` 후 pop → 아무도 await하지 않아 GC 시점에 asyncio "Future exception was never retrieved" 경고 |
| **B5** | [bench_recognize_batch.py:101](packages/scanlation-server/tools/bench_recognize_batch.py#L101) vs [bench_recognize_threads.py:188](packages/scanlation-server/tools/bench_recognize_threads.py#L188) | 같은 이름 `_detected_crops`, 같은 4px 필터, 같은 detector — 그런데 하나는 `deskew_crop(img, r)`, 다른 하나는 `img.crop(bbox)`. **크롭 픽셀이 다르다.** [recognize-crop-batching.md](packages/scanlation-server/tools/recognize-crop-batching.md)가 "배칭 1.27x는 멀티워커 1.8x에 진다"고 단언하는데 1.8x는 deskew 안 한 세트에서 나온 수치라 apples-to-apples 비교가 아니다 |
| **B6** | [bench_recognize_threads.py:366](packages/scanlation-server/tools/bench_recognize_threads.py#L366) | 생성되는 리포트 산문에 `"best ~1.8x over base"`가 측정값과 무관하게 하드코딩돼 있다. 다른 머신에서 돌리면 표는 1.2x인데 산문은 1.8x |

> B5·B6은 이미 커밋된 `tools/*.md`의 결론에 영향을 준다. R1이 B5를 계약으로 승격시키므로 묶어서 처리한다.

---

## Tier 1 — 높은 가치, 낮은 위험

### R1. `tools/_bench_common.py` 추출 — 축 ①

코드가 이미 필요성을 자백하고 있다 — [bench_recognize_gpu_concurrency.py:49](packages/scanlation-server/tools/bench_recognize_gpu_concurrency.py#L49)가 다른 벤치의 언더스코어 프라이빗 3개를 가로질러 import한다:

```python
from bench_recognize_batch import _load_crops, _paddle_device, _silenced
```

나머지 하나(threads)는 같은 함수를 복붙했고, 그 복붙이 B5를 낳았다.

추출 대상 (`_bootstrap` 다음에 import):

- `IMAGE_EXTS` — 현재 4곳에 4가지 버전 (`batch:114`, `threads:232`, `compare_models:914`, `run_image:42`)
- `silenced()` — `batch:63-79` ≡ `threads:44-61` (바이트 단위 동일)
- `load_crops(path, *, detect, deskew: bool, as_paths: bool)` — **`deskew`를 명시 인자로 승격해 B5를 계약으로 전환**
- `pick_device(force_cpu)`, `load_paddle(device, attn)` / `load_manga_ocr(force_cpu)`
- `sec_per_call(call, reps)`, `warm(fn, *args)` — 워밍업이 gpuconc 안에서만 4번 반복된다
- `score_texts(ref, got)` — `gpuconc:293`과 `:418`이 같은 식(difflib ratio + exact + mean chars)을 따로 구현
- `Report` — `rows: list[str]`에 `"| a | b |"`를 append하고 구분선을 손으로 세는 패턴이 전 벤치에 산재. `.table()`이 구분선 자동 생성, `.emit(prefix)`가 cwd→tempdir 폴백 포함
- `add_data_args(ap)` — threads만 `--data` 옵션, 나머지는 positional
- `sweep_baseline(points)` — 세 벤치 모두 "스윕 첫 점 = baseline"을 가정하면서 열 이름(`c_B/c_1`, `vs base`)은 하드코딩. `--batch 2,4,8`을 주면 열은 조용히 `c_B/c_2`가 된다

부수 효과: `bench_recognize_threads.py`는 `_bootstrap`을 import하지 않아 **우연히** 굴러간다(`app`을 안 쓰기
때문). 공통 모듈을 쓰는 순간 정상화된다.

예상: batch −120줄, threads −70줄, gpuconc −60줄.

### R2. 어휘·이름 정리 — 축 ③

`BOX`/`TSL`은 역할 라벨로 이미 깨끗하다. 남은 건 역할 라벨로서의 `OCR`과 `plugin`/`engine` 주석 혼용.

- **코드 식별자 1건**: [content.js:151](extension/src/content.js#L151), [:311](extension/src/content.js#L311)의 `runOcr()` — 실제로는 `/run_lookup/` + `/run_pipeline/`를 부르는 파이프라인 클라이언트 → `runPipeline()`
- **사용자 노출 문자열**: [app/web/app.js:84](packages/scanlation-server/app/web/app.js#L84), [:175](packages/scanlation-server/app/web/app.js#L175) — `/admin` UI에 "OCR/번역 결과", "OCR text and translation"이 그대로 보인다 (ko/en 양쪽)
- **주석**: [pipeline.py:65-66](packages/scanlation-server/app/pipeline.py#L65-L66) "what OCR read", [http_translator.py:84](packages/scanlation-sdk/scanlation_sdk/http_translator.py#L84) "blank OCR", ctbd `plugin.py:5` / `postprocess.py:5`, [routes/run.py:1](packages/scanlation-server/app/routes/run.py#L1) 모듈 docstring 첫 줄
- **명시적 예외**: [prompt.py:26](packages/scanlation-sdk/scanlation_sdk/prompt.py#L26)의 `"Treat any odd or garbled input as an OCR error."` 는 LLM 시스템 프롬프트 본문이라 문구를 바꾸면 모델 동작이 바뀐다. 건드리지 않는다
- **`plugin` vs `engine`**: [catalog.py:1](packages/scanlation-server/app/catalog.py#L1)("installable **engines**")과 [:72](packages/scanlation-server/app/catalog.py#L72)("installable **plugins**, keyed by engine name")가 같은 파일 안에서 뒤집힌다. [config.py:27](packages/scanlation-server/app/config.py#L27)의 "role -> plugin name"은 런타임 엔진 선택이므로 `engine name`. `install_plugin(name)`의 `name`이 엔진명인 것도 docstring이 자백 중
- **죽은 import 삭제**: [engine_meta.py:10](packages/scanlation-server/app/engine_meta.py#L10)의 `from .registry import registry  # noqa: F401 (kept for callers importing from here)` — 그런 호출자는 0개다(전부 `..registry`에서 직접 가져온다). 이 한 줄 때문에 순수 직렬화 모듈이 entry_points 스캔 전체를 끌고 온다
- **중복 역사 기록 통합**: `schemas.py:4`와 `routes/handshake.py:4`에 "BOX/OCR/TSL 어휘는 폐기됨" 문장이 복붙돼 있다

### R3. SDK 계약 표면 정리

- **`translate_batch`를 `Translator` Protocol에 추가** ([contracts.py:130-131](packages/scanlation-sdk/scanlation_sdk/contracts.py#L130-L131)). 실질 주 경로인데 [pipeline.py:88](packages/scanlation-server/app/pipeline.py#L88)이 `hasattr`로 덕타이핑한다
- **결과 아이템 키의 소유자 지정.** `{bounds, source, destination}`은 SDK 어디에도 정의돼 있지 않고 [pipeline.py:116](packages/scanlation-server/app/pipeline.py#L116) / [content.js:238-246](extension/src/content.js#L238-L246) / `test_routes_run.py:16` 세 곳이 손으로 동기화 중이다. `Region.to_wire_item(source, destination)` 또는 응답 모델로 승격
- **죽은 표면 제거 또는 소비**: `SUPPORTED_DST`(설정자 0·소비자 0), `SUPPORTED_SRC`(플러그인 3개가 채우지만 `app/`에 소비자 없음 — handshake는 `LANGUAGES` 전체를 그대로 낸다), `EngineBase.warning`(설정자 0), `Region.mask`(설정 0·읽기 0)
- `Recognizer.recognize(crop, region, options)`의 `region`을 어떤 recognizer도 읽지 않는다

---

## Tier 2 — 중간

### R4. `plugins_install.py` 중복 제거 + 순환 의존 해소

- **2단계 설치 알고리즘이 통째로 두 벌**: `install_plugin()`(:170-193)과 `install_plugin_events()` 내부 `worker()`(:300-324). find_class → catalog 조회 → pip → refresh → 재확인 → `install()` 흐름이 에러 문자열까지 복붙. 차이는 `_stream_pip` vs `install_package` + 이벤트 push뿐 → `put=lambda _: None` 싱크로 통합
- pip 실행 + 에러 tail도 두 벌(`install_package:145-148`의 `stderr[-800:]` vs `_stream_pip:241-255`의 `deque(maxlen=40)` → `[-6:]`) — 같은 실패를 다른 포맷으로 보고한다
- `install_plugin_events`는 트리 유일의 50줄 초과 함수(66줄)이고 **스트리밍 설치 경로 전체가 미테스트**다 (`/install_plugin_stream/` 엔드포인트, `_LineTee`, `_begin_install` 중복 방지 포함)
- **순환 의존**: `registry.py:19` → `plugins_install.ensure_on_path`(모듈 레벨) ↔ `plugins_install.py:153,162` → `registry`(함수 내부 지연 import). `ensure_on_path()`는 sys.path 조작이니 lifespan에서 한 번 부르면 사이클이 사라진다
- **역방향 의존**: `_torch_pip_args`가 `state`·`gpus`를 직접 읽는다(:103-104). 인자로 받으면 순수 함수가 되고, `test_routes_plugins.py:67-82`가 전역을 통째로 저장/복원하는 이유도 사라진다

### R5. 하드코딩 → `/admin` 노출 — 축 ②

규칙(env 기본값 + `state.json` + `/admin` UI + 필요시 handshake)을 온전히 지키는 값은 `min_image_dim` 하나뿐이다.

**확장** — 동작을 좌우하는데 서버 설정이 아니다:

- 박스 채움 비율 `0.8` ([content.js:266](extension/src/content.js#L266)), 최소 폰트 `7`px ([:267](extension/src/content.js#L267)), resize 디바운스 `150`ms ([:395](extension/src/content.js#L395))
- **요청 타임아웃 부재** — `content.js:137,143,156`이 전부 `AbortController` 없는 맨 `fetch`. 서버가 매달리면 `processing` WeakSet이 안 풀린다. 신설 대상
- 엔드포인트 `"http://127.0.0.1:4010"` **3중 중복** — `background.js:12` / `content.js:91` / `popup.js:110`
- [background.js:12](extension/src/background.js#L12)의 `DEFAULTS`에 `minImageDim`이 빠져 있어, popup에서 Connect를 한 번도 안 누르면 `content.js:91`의 리터럴 `80`이 영구 적용된다
- 하드코딩 한국어 `"번역 실패"` ([content.js:295](extension/src/content.js#L295)) — i18n 없음

**SDK/플러그인**:

- [http_translator.py:53](packages/scanlation-sdk/scanlation_sdk/http_translator.py#L53)의 `httpx.Client(timeout=10.0)` — 주석이 튜닝 근거까지 적어 놓고도 `OPTION_SCHEMA`에 안 뺐다. 번역기 2종 공통이므로 SDK 레벨 옵션 후보
- `ollama/plugin.py:36`의 `num_gpu` 기본값 `31` — 특정 머신의 레이어 수를 보편 기본값으로 제시한다
- `paddleocr/plugin.py:112`의 `do_sample=False` — `max_new_tokens`는 스키마에 있는데 샘플링 결정만 리터럴

**서버 A등급** (env·state·admin 어디에도 없음): `geometry.py:41`의 `eps=1.0`, `:49`의 `min_size=8`, 패딩색
`(255,255,255)` 3회; `cache.py:33`의 `hexdigest()[:16]`(충돌 확률 결정); `plugins_install.py`의 torch wheel
인덱스 URL들; 에러 tail 길이 4종(`[-800:]`, `maxlen=40`, `[-6:]`, `[:200]`).

**서버 B등급** (state+admin은 있으나 env 기본값 없음): `translate_concurrency`, `torch_backend`,
`torch_vendor`, `torch_index`, `prompt_active`.

**중복 검증**: `min_image_dim`과 `translate_concurrency`가 JS clamp / 라우트 400 / state clamp **3계층에
서로 다른 규칙**으로 존재한다. JS가 먼저 clamp하므로 라우트의 400은 UI에서 도달 불가이고 state의 clamp도
죽은 코드다. clamp와 reject 중 하나만 남긴다.

### R6. 플러그인 보일러플레이트를 SDK로

로컬 모델 3종(ctbd / manga-ocr / paddleocr)이 같은 것을 각자 구현한다.

**올릴 것**: `EngineBase._log`(이미 `HttpTranslatorBase._log`가 같은 패턴 — 비대칭), `to_rgb()`,
`hf_cached(repo, file)`, `hf_download()`, `INSTALL_HINT` 템플릿(`self.name`에서 생성), 모델 경로 env
오버라이드 규칙(manga-ocr에만 없어 비일관), `COMMON_LLM_OPTIONS`(번역기 2종의 `model`/`temperature`/
`seed`/`top_p`가 설명 문구까지 동일), `testing.recognizer_smoke()` / `http_translator_contract()`
(테스트 5종이 seam 이름만 바꾼 동일 본문).

**그대로 둘 것**: `is_installed()`(SDK가 `local_engine.py:8`에서 "checks genuinely differ"라 명시),
`_load()`, `OPTION_SCHEMA` 값, pyproject의 transformers 핀(주석이 실제 제약을 담은 load-bearing 문서).

**제거할 것**: 추론 진입부의 `if self._model is None` lazy 가드 — `registry.get()`이 이미 `load()`를
부르므로 죽은 가드이자 `LocalModelEngineBase._loaded`와의 이중 상태다. `_generate`/`_chat`(ollama/llama-cpp)
도 `self._post` 한 줄 래퍼로, 존재 이유가 "unit-test seam"인데 `_post`가 이미 seam이다.

**별개 문제**: `description` 문자열이 각 `plugin.py`와 [catalog.py](packages/scanlation-server/app/catalog.py)에
두 번 있다(5쌍 전부). 카탈로그 쪽은 미설치 플러그인을 `/admin`에 보여주는 용도라 지울 수 없지만 드리프트한다.

---

## Tier 3 — 대형 파일 분할 — 축 ④

순수 이동 위주라 마지막에 둔다(diff는 크고 리뷰 가치는 낮다).

### R7. `compare_models.py` (1404줄) → `tools/compare/` 패키지

어댑터/렌더/리포트가 자연 경계로 갈라져 있어 기계적 분할이 가능하다: `boxes.py` /
`adapters/{base,detect,ocr}.py` / `registry.py` / `render.py` / `report_md.py` / `report_html.py` /
`commands.py` / `__main__.py`.

내부 중복 2개를 같이 처리:

- `_write_ocr_html`(:1128)과 `_write_box_html`(:1216)이 사실상 같은 투표 페이지다. CSS 11줄이 문자 단위 동일, JS는 `_HTML_JS`(:1077) 공유, `VK` 네임스페이스만 다르다 → `render_vote_page(images, cell_fn, vote_ns, ...)` 하나로
- 엔진×디바이스 실행 루프가 `cmd_ocr:742`와 `cmd_ocrbatch:843`에 두 벌 (주석까지 동일). 검출 루프도 `cmd_detect:598` / `cmd_batch:947`에 두 벌

`_HTML_JS`(파이썬 문자열 속 JS 48줄)와 CSS blob 2개는 `assets/`로. `all_adapters()`(:425-451)는 데이터이므로
TOML/JSON으로 빼면 어댑터 추가에 파이썬 수정이 불필요해진다.

주의: `"compare_out"` 리터럴이 9곳, `"ogkalu_rtdetr"`이 3곳에 반복된다 — 분할과 함께 상수화.

### R8. `app/web/app.js` (994줄)

i18n 번역 테이블이 :14-218 약 200줄을 차지한다. `i18n.js` 분리 → 나머지를 render/actions로. 번들러가
없으므로 `<script type="module">` 또는 script 태그 나열로 처리.

### R9. `extension/src/content.js` (414줄)

책임 8가지(md5 구현 · 설정/상태 · 이미지 획득 3단 폴백 · 서버 프로토콜 클라이언트 · DOM 레이아웃 해킹 ·
오버레이 렌더 + 타이포그래피 휴리스틱 · 라이프사이클 · 메시지 라우터)가 한 파일에 있다.
[extension/README.md](extension/README.md)가 "manifest content scripts는 ES import를 못 쓴다"고 단일 파일
이유를 설명하지만, **md5 구현(:14-88, 75줄)**과 **서버 프로토콜 클라이언트(:150-171)**는 별도 파일로 떼어
[manifest.json](extension/manifest.json)의 `js` 배열에 나열하면 된다.

참고: `popup.js`(117줄)보다 장식용 `starfield.js`(186줄)가 1.6배 크다.

---

## Tier 4 — 환경·문서 위생 (코드 변경 아님)

| # | 내용 |
|---|---|
| **H1** | 개발 venv에 구 이름 editable 설치가 남으면 **유령 엔진이 entry_points로 등록**된다: `scanlation_ctd` → `[scanlation.detectors] ctd`, `scanlation_mangaocr` → `[scanlation.recognizers] mangaocr`. `packages/scanlation-ctd/`·`scanlation-mangaocr/`도 git 미추적 잔해(`.pyc` + `egg-info`)로 남는다. `pip uninstall` + 디렉터리 삭제 |
| **H2** | `scanlation_ollama.egg-info/entry_points.txt`는 `ollama`(소문자), [pyproject.toml](packages/scanlation-ollama/pyproject.toml)은 `Ollama`(대문자). editable 설치에서는 egg-info가 `importlib.metadata`가 읽는 실제 메타데이터라 엔진명이 어긋난다. 재설치로 해결 |
| **H3** | **라이선스 미결.** [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md) §9-4가 "트리에 GPLv3 코드 미포함"을 약속하는데 `tools/vendored/`에 manga-image-translator(GPL) 코드 1,003줄이 실재한다(`_mit_ocr_48px.py` 635 + `_mit_ocr_ctc.py` 368). tools 전용·프로덕션 의존 0인 건 확인됐지만 배포 단위가 같은 저장소라면 문서의 불변식은 깨져 있다. 또 `_mit_xpos.py:2`가 `[see LICENSE for details]`를 가리키는데 **저장소에 LICENSE 파일이 없다**(MIT 고지 요건 미충족) |
| **H4** | `compare_models.py:397`의 `"weights auto-included in tools/vendored/_mit_weights/"`는 사실이 아니다 — [.gitignore](.gitignore)가 그 디렉터리를 제외한다. 새 clone에서는 수동 다운로드가 필요 |
| **H5** | [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md)의 stale 서술: `CTD` 12회(§3.1은 "원 설계 기록"으로 표시돼 일부는 의도적), `SCANLATION_TRANSLATE_CONCURRENCY` env(§3.5) — 현재는 `/admin` 전용 |
| **H6** | entry-point 이름 케이싱 규칙 부재: `comic-text-and-bubble-detector`, `manga-ocr`(kebab) vs `PaddleOCR-VL-For-Manga`, `Ollama`, `llama.cpp`. 이 이름은 `state.json`에 영속되고 **캐시 키의 일부**([orchestrator.py:65](packages/scanlation-server/app/orchestrator.py#L65))라 변경하면 캐시가 무효화된다 — 동작 변경이므로 별도 결정 |
| **H7** | 5개 플러그인 pyproject 모두 `scanlation-sdk`를 버전 제약 없이 의존한다. SDK가 git ref로 배포되므로 불일치가 조용히 통과한다 |
| **H8** | [recognize-gpu-speed.md](packages/scanlation-server/tools/recognize-gpu-speed.md)가 "해상도 캡 150k + pow2 → 1.66x, 채택 방향"이라 적었으나 `scanlation-paddleocr-vl-for-manga`에 `max_pixels`/downscale이 없다. `_downscale_one`과 `GRID = 28`(`gpuconc:213-245`)이 **프로덕션에 가야 할 코드인데 벤치에 갇혀 있다** |

---

## 권장 순서

1. **B5+B6 + R1** — 벤치 하네스 통합. B5를 `load_crops(deskew=...)` 계약으로 승격, B6의 하드코딩된 결론 산문 제거. `tools/*.md`의 1.27x vs 1.8x 결론 재검토가 따라온다
2. **R2** — 어휘 정리. 리스크 최저
3. **R3** — SDK 계약 표면
4. **B1~B4** — 버그 4건 (각각 개별 판단; B1은 세로 읽기 순서 구현 여부 결정 필요)
5. **R4 → R6** — 설치기 통합, 플러그인 보일러플레이트
6. **R5** — 하드코딩 → `/admin`. 새 설정 필드가 늘어나므로 handshake·확장·i18n 동시 수정
7. **R7 → R9** — 대형 파일 분할
8. **H1~H8** — 환경/문서 위생 (H3, H6은 판단 필요)

---

## 검증

리팩토링 항목마다 변경 전후로 동일하게 통과해야 한다.

```bash
# 코어 단위 (pytest 미사용·모델/GPU 불필요)
cd packages/scanlation-server && ../../venv/Scripts/python -m tests

# 엔진 패키지별 suite (가중치/패키지 없으면 자동 skip)
cd packages/scanlation-<engine>  && ../../venv/Scripts/python -m tests
```

축별 추가 확인:

- **R1** — 통합 전후로 세 벤치를 같은 입력에 돌려 표의 수치가 일치하는지. B5 때문에 threads는 크롭이 바뀌므로 **수치가 달라지는 것이 정상**이다 — 이 경우 새 값을 기록하고 [recognize-cpu-threads.md](packages/scanlation-server/tools/recognize-cpu-threads.md)·[recognize-crop-batching.md](packages/scanlation-server/tools/recognize-crop-batching.md)의 결론을 갱신한다
- **R2** — `grep -rnE '\b(BOX|TSL)\b'` 및 역할 라벨 `OCR` 재검색이 0건(제품명·기하학 `bounding box`·`prompt.py:26` 예외 제외). `/admin`을 열어 UI 문자열 확인(ko/en 둘 다)
- **R3** — `python -m tests`가 `tests/fake_engines.py`로 Protocol 준수를 검증한다. `translate_batch` 추가 후 `test_pipeline.py`에 batch 분기 테스트 신설(현재 `hasattr` True 경로 미검증)
- **R5** — `/admin`에서 새 필드를 바꾼 뒤 `GET /`(handshake) 응답에 실려 나오는지, 확장 재주입 후 실제로 적용되는지. `state.json` 라운드트립은 `test_state.py`
- **R7~R9** — 순수 이동이므로 `python -m tests` green + `/admin` 수동 클릭 + 확장으로 실제 페이지 1장 번역

엔드투엔드(구조 변경이 큰 R4·R5·R9 후 권장):

```bash
make serve                        # 또는 uvicorn app.main:app --port 4000
python tools/run_image.py <img>   # 서버 왕복
python tools/visualize.py <img>   # 검출 폴리곤 육안 확인
```

> H1의 유령 엔진이 남아 있으면 `/admin` 엔진 목록에 `ctd`·`mangaocr`가 뜬다. 검증 전에 H1을 먼저 처리하면
> 혼선이 적다.

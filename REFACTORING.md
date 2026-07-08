# 리팩토링 백로그

트리 전체를 네 축(**벤치 하네스 통합 / 어휘·이름 정리 / 대형 파일 분할 / 하드코딩→`/admin` 노출**)으로 훑은
결과. 원칙은 **동작 보존** — 외부 동작(와이어 계약·CLI 출력·엔진 결과)은 그대로 두고 내부 구조만 바꾼다.
[CLAUDE.md](CLAUDE.md)의 코드 규칙과 커밋 규칙을 따른다: 항목 하나 = 커밋 하나.

**범위 제약: 구조·가독성만.** 실행 방식을 바꾸는 최적화는 이 백로그에 넣지 않는다 — 멀티프로세스·병렬화·
알고리즘 교체는 결과가 같아도 제외. 이미 계산한 값을 저장해 재사용하는 정도(중복 호출 제거)는 허용.
문자열·주석·문서 문구 변경은 허용.

기준선: 코어 단위테스트(`python -m tests`)가 회귀 게이트다.

옵션 어휘(`opt_detect`/`opt_recognize`/`opt_translate`)와 결과 키(`{bounds, source, destination}`)는 규칙을
온전히 지키고 있고, 플러그인 튜닝 값은 대부분 `OPTION_SCHEMA`로 `/admin`에 노출된다. 부채는 세 지점에
몰려 있었다 — **`tools/` 벤치의 중복**, **확장의 하드코딩**, **`plugins_install.py`의 이중 구현**.

---

## 현재 상태

| 완료 | 커밋 | 내용 |
|---|---|---|
| R1 | `565144d` | 벤치 공통 모듈 `tools/_bench_common.py` 추출 (순수 이동) |
| R2 | `bb7f769` | 역할 라벨 `OCR` 제거, `plugin`↔`engine` 어휘, 죽은 import |
| R3 | `425237d` | `BatchTranslator` Protocol, `ResultItem` TypedDict |
| R4 | `79d4b24` → `ed24280` | 스트리밍 설치 테스트 7개 → 설치 알고리즘 통합 + 순환 의존 해소 |
| B1 | `608fde7` | 읽기 순서가 소스 언어를 따른다 (죽은 `vertical_hint` 교체) |
| B2 | `df36bae` | `apply_verbose(False)`가 `SCANLATION_LOG_LEVEL`을 파괴하던 것 |
| B4 | `f053aa6` | dedup future의 예외를 읽어 asyncio 경고 제거 |
| H1·H2 | `b6985ad` | 개발 venv의 낡은 설치 메타데이터 (**PC별 작업** — 아래 참조) |

코어 테스트 58개 → **75개**.

**결정을 기다리는 것:**

- **B3** — `registry.get()`의 `device` 인자가 캐시 히트 시 무시된다. 시그니처를 진실에 맞출지, 코드를 시그니처에 맞출지 (Tier 0 표)
- **H3** — `tools/vendored/`의 GPLv3 코드 1,003줄이 설계 문서의 약속과 충돌하고 LICENSE 파일이 없다 (아래 전용 절)
- **H6** — entry-point 이름 케이싱 규칙. 이 이름은 `state.json`에 영속되고 캐시 키의 일부라 바꾸면 캐시가 무효화된다

**측정 장비가 필요한 것:** B5·B6 — 벤치 크롭 세트 불일치. GPU 호스트에서 재측정해야 `tools/*.md`의 결론을 갱신할 수 있다.

**남은 리팩토링:** R5(하드코딩→`/admin`), R6(플러그인 보일러플레이트), R7~R9(대형 파일 분할).

---

## 다른 PC에서 이어받기

`git pull` 후, **저장소에 담기지 않는 두 가지**를 먼저 확인한다.

1. **개발 venv의 설치 메타데이터** (H1·H2). editable 설치의 dist-info는 저장소가 아니라 그 PC의 venv에 있다.
   패키지 이름이 바뀌었거나 `pyproject.toml`의 entry-point가 바뀐 뒤 재설치하지 않았다면, `importlib.metadata`는
   **낡은 이름을 그대로 본다**. 증상: `registry`와 `catalog`가 같은 엔진을 다른 이름으로 부른다.

   ```bash
   # 유령 entry_point 점검 — 로드에 실패하는 게 있으면 낡은 설치다
   venv/Scripts/python -c "
   from importlib.metadata import entry_points
   for g in ('scanlation.detectors','scanlation.recognizers','scanlation.translators'):
       for ep in entry_points(group=g):
           try: ep.load(); s='ok'
           except Exception as e: s=type(e).__name__
           print(g, ep.name, s)"

   # 고치기: 사라진 패키지를 지우고, 남은 것을 editable 재설치
   venv/Scripts/python -m pip uninstall -y <구-패키지명>
   venv/Scripts/python -m pip install -e packages/scanlation-sdk -e packages/scanlation-server --no-deps
   ```

   `registry._discover`가 실패한 `ep.load()`를 조용히 삼키므로(H9) 유령은 `/admin`에 뜨지 않는다 — 즉
   **증상 없이 이름만 갈라진다.** 위 스크립트가 유일한 진단 수단이다.

2. **엔진 가중치.** `manga-ocr`/`comic-text-and-bubble-detector` 스모크 테스트는 가중치가 없으면 자동 skip한다.
   코어 75개는 모델 없이 돌지만, 엔진을 건드리는 변경은 가중치 있는 머신에서 검증해야 한다.

---

## Tier 0 — 리팩토링이 아니라 버그 (수정 여부 별도 결정)

동작 보존 원칙상 리팩토링 커밋에 끼워 고치지 않는다. 전부 코드에서 확인됨.

| # | 위치 | 내용 |
|---|---|---|
| ~~**B1**~~ ✅ | [pipeline.py](packages/scanlation-server/app/pipeline.py) | `assign_reading_order(regions, vertical_hint)`의 본문이 `vertical_hint`를 **한 번도 읽지 않았다** — 첫 커밋(`741bf4b`)부터 죽어 있었고, [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md) §3.5 의사코드를 시그니처만 옮겨 적은 결과다. 읽기 순서는 `src`와 무관하게 **항상 만화 R→L**이었다. 이름도 틀렸다: 호출부는 *언어*(`src == "ja"`)로 계산하는데 정작 분기해야 할 건 세로/가로가 아니라 **수평 방향**이다(진짜 세로쓰기는 `Region.vertical`이 따로 들고 있다). `rtl` 인자 + `LANG_RTL` 언어 표로 실제 배선했다 |
| ~~**B2**~~ ✅ | [logconfig.py](packages/scanlation-server/app/logconfig.py) | `apply_verbose(False)`가 `INFO`를 리터럴로 세팅했다. lifespan이 `configure_logging(settings.log_level)` 직후 이걸 부르므로 `SCANLATION_LOG_LEVEL=WARNING`은 조용히 INFO가 됐고, `/admin`에서 상세로그를 켰다 끄면 그때도 INFO로 떨어졌다. `configure_logging`이 연 레벨을 기억했다가 그리로 복귀한다 |
| **B3** | [registry.py:76-78](packages/scanlation-server/app/registry.py#L76-L78) | `registry.get(name, device=...)`의 lock-free 캐시 히트 경로가 `device` 인자를 검사하지 않는다. `set_engine_device` → `unload_one` 순서 덕에 현재는 안전하지만 [orchestrator.py:86-87](packages/scanlation-server/app/orchestrator.py#L86-L87)이 매 요청 `device=`를 넘기므로 **시그니처가 거짓말을 한다.** 아래 두 갈래 중 결정 필요 |
| ~~**B4**~~ ✅ | [orchestrator.py](packages/scanlation-server/app/orchestrator.py) | `_run_deduped`에서 대기자가 없을 때 `fut.set_exception()` 후 pop → 아무도 읽지 않아 future finalize 시점에 asyncio가 "Future exception was never retrieved"와 트레이스백을 로그에 뿌렸다(실패 요청마다). `set_exception` 직후 `fut.exception()`으로 읽음 표시 — 대기자는 여전히 같은 예외 객체를 받는다 |
| **B5** | `_bench_common.deskewed_crops` vs `bench_recognize_threads._raw_bbox_crop_files` | batch/gpuconc는 `deskew_crop(img, r)`, threads는 `img.crop(bbox)` — **크롭 픽셀이 다르다.** [recognize-crop-batching.md](packages/scanlation-server/tools/recognize-crop-batching.md)가 "배칭 1.27x는 멀티워커 1.8x에 진다"고 단언하는데 1.8x는 deskew 안 한 세트에서 나온 수치라 apples-to-apples 비교가 아니다. R1이 두 함수의 이름을 갈라 차이를 드러냈으니, 고치는 것은 threads의 크롭 방식을 바꾸고 재측정하는 일이다 |
| **B6** | [bench_recognize_threads.py:366](packages/scanlation-server/tools/bench_recognize_threads.py#L366) | 생성되는 리포트 산문에 `"best ~1.8x over base"`가 측정값과 무관하게 하드코딩돼 있다. 다른 머신에서 돌리면 표는 1.2x인데 산문은 1.8x |

> B5·B6은 이미 커밋된 `tools/*.md`의 결론에 영향을 준다. R1이 B5를 계약으로 승격시키므로 묶어서 처리한다.

### B3 — 두 갈래

**(a) 계약을 코드에 맞춘다.** `device`를 `get()` 시그니처에서 빼고, 인스턴스 생성 시점에만 주입한다. 그러면
"device는 로드 시점에 정해지고, 바꾸려면 `unload_one`한다"는 **현재의 진실**이 시그니처에 드러난다.
호출부(`orchestrator._read_sync`)는 `state.resolve_device_for(...)`를 registry가 대신 읽게 하거나, device 변경을
`settings_routes`가 지금처럼 unload로 처리한다. 동작 변경 0.

**(b) 코드를 시그니처에 맞춘다.** 캐시 키를 `(role, name, device)`로 넓히거나, 히트 시 device 불일치를 감지해
재로드한다. 동작 변경이고, 같은 모델이 device당 한 벌씩 VRAM에 뜰 수 있다.

**(a) 권장.** 이 프로젝트는 uvicorn 단일 워커에 VRAM 모델 한 벌이 전제다(`SCANLATION_DESIGN.md` §9-7).
device당 인스턴스를 캐싱하는 건 그 전제와 어긋난다. 지금 코드가 하는 일이 맞고, 거짓말하는 건 시그니처다.

---

## Tier 1 — 높은 가치, 낮은 위험

### ~~R1. `tools/_bench_common.py` 추출 — 축 ①~~ ✅ 완료

세 벤치가 공유하는 것을 [_bench_common.py](packages/scanlation-server/tools/_bench_common.py)로 옮겼다:
`IMAGE_EXTS`, `silenced()`, `deskewed_crops()` / `load_crops()`, `paddle_device()`, `load_paddle()`,
`write_report()`. gpuconc가 batch의 언더스코어 프라이빗 3개를 가로질러 import하던 줄이 사라졌다.

옮긴 함수 본문은 원본과 **AST 동일**(docstring 제외). 유일한 예외는 `load_crops`로, 로컬 `exts` 리터럴이
같은 값의 모듈 상수 `IMAGE_EXTS`가 됐다. 벤치 3종 −179줄, 공통 모듈 +179줄 — **순 감소 0인 순수 이동**이다.

`load_paddle()`은 플러그인 내부(`_repo()`, `_model`)에 손을 뻗는 유일한 지점이 됐다. 그 결합이 한 곳에
모인 것이 이 항목의 실질 수확이다.

**범위 제약 때문에 하지 않은 것** (서두 참조):

- `warm()` 통합 — 워밍업 횟수가 툴마다 다르다(threads는 `workers * 2`)
- `add_data_args()` — threads만 `--data`, 나머지는 positional. 통일하면 CLI가 바뀐다
- `sweep_baseline()` — 없던 검증 추가
- `Report` 클래스 — 마크다운 표 조립은 `write_report()`만 공용화하고 `rows: list[str]` 패턴은 남겼다

**`load_crops(deskew=...)`로 합치지 못한 이유** (당초 계획을 뒤집는다): threads의 크롭 함수는 in-process가
아니라 **격리된 fork 서브프로세스**에서 돈다 — torch를 초기화한 프로세스를 fork하면 자식이 상속된 스레드풀
락에서 데드락하기 때문이다([`_detect_isolated`](packages/scanlation-server/tools/bench_recognize_threads.py)).
합치면 실행 모델이 바뀐다. 대신 이름을 갈랐다: `deskewed_crops()`(batch/gpuconc, deskew) vs
`_raw_bbox_crop_files()`(threads, raw bbox). **B5는 이제 이름에 드러나 있고, 고치는 것은 한 줄이다.**

부수 효과: `bench_recognize_threads.py`는 여전히 `_bootstrap`을 import하지 않는다(`app`을 안 쓴다).
`_bench_common`이 모듈 스코프에서 무거운 것을 하나도 import하지 않게 만든 이유다 — threads는 부모
프로세스에 torch가 들어오면 안 되고, 그 풀 워커는 모듈을 재import한다.

### ~~R2. 어휘·이름 정리 — 축 ③~~ ✅ 완료

`app/` · `scanlation_sdk` · `extension/` · 플러그인에서 역할 라벨 `OCR`을 걷어냈다. `runOcr()` →
`runPipeline()`, `/admin` UI 문자열(ko/en), 주석·docstring 5곳, `plugin`↔`engine` 혼용(`catalog.py`의
자기모순 docstring, `config.py`의 "plugin name", `unknown plugin:` 에러 문자열), `engine_meta.py`의 죽은
`registry` import.

폐기 어휘를 "버렸다"고 서술하던 `schemas.py` / `handshake.py`의 중복 문장은 지웠다 — CLAUDE.md가 회고적
이력 주석을 금하고, 그 문장이 `BOX`/`TSL`을 코드에 살려두는 유일한 이유였다. 이력은 [README.md](README.md)
와이어 계약 절에 있다.

**의도적으로 남긴 것** (지우려면 별도 판단):

- [prompt.py:26](packages/scanlation-sdk/scanlation_sdk/prompt.py#L26)의 `"Treat any odd or garbled input as an OCR error."` — LLM 시스템 프롬프트 본문이라 문구를 바꾸면 모델 동작이 바뀐다. `:16`의 요약 주석도 그에 맞춰 둔다
- `tools/compare_models.py`의 역할 라벨 `BOX`/`OCR` — 투표 페이지의 `'ocrsel:'` / `'boxsel:'`는 **localStorage 네임스페이스**라 개명하면 저장된 투표가 고아가 된다. 개명이 아니라 데이터 변경이므로 **R7과 함께** 다룬다
- `Image.BOX`(`bench_recognize_gpu_concurrency.py`) — PIL 리샘플 필터 상수, 역할 라벨 아님
- `catalog.py`의 `"Manga OCR"` / `"Japanese OCR"`, `extension/README.md`의 `ocr_extension` — 제품명·고유명사

### ~~R3. SDK 계약 표면 정리~~ ✅ 완료 (일부는 남김 — 아래 참조)

- **`BatchTranslator` Protocol 신설** ([contracts.py](packages/scanlation-sdk/scanlation_sdk/contracts.py)). `translate_batch`가 LLM 번역기의 실질 주 경로인데 계약에 없어 `pipeline._translate_all`이 `hasattr`로 덕타이핑하고 있었다. `runtime_checkable` protocol의 `isinstance`는 메서드 멤버에 대해 `hasattr`와 **동치**라(Python 3.13에서 실측), 옵셔널 능력을 계약에 적어 넣으면서 동작은 그대로다
- **`ResultItem` TypedDict** ([pipeline.py](packages/scanlation-server/app/pipeline.py)). `{bounds, source, destination}`을 만드는 곳에서 선언한다. `TypedDict` 호출은 런타임에 그냥 dict라 와이어 바이트가 같다. `test_routes_run.py`가 리터럴 대신 `ResultItem.__annotations__`를 단언하므로 키를 바꾸면 테스트가 먼저 깨진다. `schemas.py` docstring은 모양을 재서술하는 대신 소유자를 가리킨다

**조사 결과 "죽은 표면"이 아니었던 것** (당초 진단의 오류 정정):

- `EngineBase.warning` — [engine_meta.py:43](packages/scanlation-server/app/engine_meta.py#L43)이 읽어 `/admin`에 싣고 [app.js:579](packages/scanlation-server/app/web/app.js#L579)가 `⚠`로 렌더한다. 설정하는 플러그인이 아직 없을 뿐, 소비자가 있는 확장점
- `Region.mask` — [contracts.py](packages/scanlation-sdk/scanlation_sdk/contracts.py) docstring이 "polygon/angle/mask stay server-internal (deskew, future inpaint)"라 명시한다. [README.md](README.md)도 계약을 "4점 폴리곤 + 각도 + 마스크"로 서술. 의도된 계약 표면
- `Region.angle` — `tools/visualize.py`와 테스트가 읽는다
- `_translate_all`의 batch 분기 — `test_pipeline.py`가 이미 `_BatchRecorder`로 덮고 있었다(진단이 "미검증"이라 한 것은 오류)

**정말로 소비자가 없는 것** — 지울지 소비할지는 설계 결정이라 남긴다:

- `SUPPORTED_SRC`(플러그인 3개가 채우지만 `app/`에 소비자 0 — handshake는 `LANGUAGES` 전체를 그대로 낸다), `SUPPORTED_DST`(설정자·소비자 모두 0). **소비**하면(엔진별 언어 필터링) 동작이 바뀌고, **삭제**하면 플러그인이 제공하던 정보가 사라진다
- `Recognizer.recognize(crop, region, options)`의 `region` — 어떤 recognizer도 읽지 않는다. 지우면 엔진 계약이 깨진다

---

## Tier 2 — 중간

### ~~R4. `plugins_install.py` 중복 제거 + 순환 의존 해소~~ ✅ 완료

**먼저 안전망부터.** 스트리밍 설치 경로(`install_plugin_events`, `_LineTee`, `_stream_pip`,
`_begin_install`, `/install_plugin_stream/`)는 테스트가 하나도 없었다. 합칠 대상이 바로 그 절반이라
테스트 7개를 먼저 붙였고, 각각을 **뮤테이션으로 검증**했다(`_LineTee`의 `\r`/`\n` 판별을 뒤집고
`_stream_pip`의 에러 tail을 줄이면 실제로 빨개진다).

- **설치 알고리즘 통합**: `install_plugin()`과 `install_plugin_events()`의 `worker()`가 같은 흐름을
  에러 문자열까지 복붙하고 있었다 → `_run_install(name, put)` 하나. `put=None`이면 조용한 블로킹,
  `put`이 있으면 phase 이벤트 + 라인 스트리밍. 두 pip 실행기(`install_package` / `_stream_pip`)는 남긴다 —
  에러 tail 포맷이 다르고(`stderr[-800:]` vs 마지막 6줄) 그 차이가 각 경로의 계약이다
- `install_plugin_events` 65줄 → 37줄, `install_plugin` 24줄 → 4줄. **트리에 50줄 초과 함수가 사라졌다**
- **순환 의존 해소**: `plugins_dir()` / `ensure_on_path()`를 잎 모듈 [plugins_path.py](packages/scanlation-server/app/plugins_path.py)로 내렸다. `registry`가 설치기 전체를 끌어오던 유일한 이유가 그 한 줄이었다. 이제 `import app.registry`는 `app.plugins_path`만 함께 로드한다(전엔 `plugins_install` + `catalog`까지)
- **`_torch_pip_args`를 순수 함수로**: `(backend, vendor, index)`를 받는다. `state`/`gpus` 조회는
  `_resolve_torch_pip_args`가 맡는다. 역방향 의존은 그 함수에 남았다 — 완전히 걷어내려면 시그니처가
  라우트까지 전파된다. **CPU 백엔드에서 `detect_gpu_vendor()`를 부르지 않는 짧은 회로**를 보존했고, 그
  지연을 깨는 뮤테이션이 테스트를 빨갛게 만든다

`plugins_install.py` 346 → 327줄, 새 잎 모듈 34줄. 줄 수는 늘었고, 사라진 것은 중복 알고리즘과 사이클이다.

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
| ~~**H1**~~ ✅ | 개발 venv에 **낡은 설치 메타데이터**가 남으면 죽은 entry_point가 등록된다: `scanlation_ctd` → `[scanlation.detectors] ctd`, `scanlation_mangaocr` → `[scanlation.recognizers] mangaocr`, 그리고 `scanlation_server`의 dist-info가 현 pyproject에 없는 `dummy` 3개. **다만 `/admin`에는 뜨지 않는다** — `registry._discover`의 `except Exception: pass`가 실패한 `ep.load()`를 삼킨다. 비용은 discovery마다 헛도는 import와, 아래 H2가 그러듯 이름이 갈라지는 것. `pip uninstall` + 잔해 디렉터리 삭제 + editable 재설치 |
| ~~**H2**~~ ✅ | 설치된 `scanlation_ollama` 메타데이터는 `ollama`(소문자), [pyproject.toml](packages/scanlation-ollama/pyproject.toml)은 `Ollama`(대문자). editable 설치에서는 그 메타데이터가 `importlib.metadata`가 읽는 실물이라 **registry는 `ollama`, catalog는 `Ollama`로 같은 엔진을 두 이름으로 본다**. `pip install -e` 재설치로 dist-info 재생성 |
| **H3** | **라이선스 미결.** [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md) §9-4가 "트리에 GPLv3 코드 미포함"을 약속하는데 `tools/vendored/`에 manga-image-translator(GPL) 코드 1,003줄이 실재한다(`_mit_ocr_48px.py` 635 + `_mit_ocr_ctc.py` 368). tools 전용·프로덕션 의존 0인 건 확인됐지만 배포 단위가 같은 저장소라면 문서의 불변식은 깨져 있다. 또 `_mit_xpos.py:2`가 `[see LICENSE for details]`를 가리키는데 **저장소에 LICENSE 파일이 없다**(MIT 고지 요건 미충족) |
| **H4** | `compare_models.py:397`의 `"weights auto-included in tools/vendored/_mit_weights/"`는 사실이 아니다 — [.gitignore](.gitignore)가 그 디렉터리를 제외한다. 새 clone에서는 수동 다운로드가 필요 |
| **H5** | [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md)의 stale 서술: `CTD` 12회(§3.1은 "원 설계 기록"으로 표시돼 일부는 의도적), `SCANLATION_TRANSLATE_CONCURRENCY` env(§3.5) — 현재는 `/admin` 전용. §3.5 의사코드는 `opt_box`/`opt_ocr`/`opt_tsl`과 `{"ocr","tsl","box"}` 결과 키, `vertical_hint`를 그대로 보여준다 — 넷 다 폐기된 어휘/시그니처다 |
| **H6** | entry-point 이름 케이싱 규칙 부재: `comic-text-and-bubble-detector`, `manga-ocr`(kebab) vs `PaddleOCR-VL-For-Manga`, `Ollama`, `llama.cpp`. 이 이름은 `state.json`에 영속되고 **캐시 키의 일부**([orchestrator.py:65](packages/scanlation-server/app/orchestrator.py#L65))라 변경하면 캐시가 무효화된다 — 동작 변경이므로 별도 결정 |
| **H7** | 5개 플러그인 pyproject 모두 `scanlation-sdk`를 버전 제약 없이 의존한다. SDK가 git ref로 배포되므로 불일치가 조용히 통과한다 |
| **H9** | [registry.py](packages/scanlation-server/app/registry.py)의 `_discover`가 `ep.load()` 실패를 `except Exception: pass`로 삼킨다. "깨진 엔진 하나가 discovery 전체를 죽이면 안 된다"는 의도는 맞지만, **왜 안 뜨는지 알 길이 없다** — H1의 유령 entry_point 3종이 조용히 사라지고 있었다. 최소한 `logger.warning`은 남겨야 한다 |
| **H8** | [recognize-gpu-speed.md](packages/scanlation-server/tools/recognize-gpu-speed.md)가 "해상도 캡 150k + pow2 → 1.66x, 채택 방향"이라 적었으나 `scanlation-paddleocr-vl-for-manga`에 `max_pixels`/downscale이 없다. `_downscale_one`과 `GRID = 28`(`gpuconc:213-245`)이 프로덕션에 가야 할 코드인데 벤치에 갇혀 있다. **성능 변경이자 신규 기능이라 이 백로그 밖** — 별도 결정 |

### H3 상세 — `tools/vendored/`의 GPL 코드와 라이선스 부재

**사실** (전부 실측):

| 파일 | 줄 | 출처 |
|---|---|---|
| `_mit_ocr_48px.py` | 635 | `zyddnys/manga-image-translator` **복사본, GPL** |
| `_mit_ocr_ctc.py` | 368 | 같은 곳, **GPL** |
| `_mit_xpos.py` | 103 | `Copyright (c) 2022 Microsoft`, **MIT** |
| `_mit_ocr.py` | 138 | 우리 코드(라인 분할 러너). 위 GPL 모듈을 import한다 |

- 소비자는 [compare_models.py:411](packages/scanlation-server/tools/compare_models.py#L411) **한 곳뿐**. `app/`·`scanlation_sdk`·플러그인의 참조는 0건이고, `tools/`는 wheel(`include = ["app*"]`)에도 Docker 이미지에도 들어가지 않는다
- [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md) §9-4가 약속한다: *"프로젝트 라이선스 미정(TBD), 단 **트리에 GPLv3 코드 미포함**, 런타임 의존만."* [geometry.py](packages/scanlation-server/app/geometry.py)는 그 원칙을 지키려 deskew를 일부러 독립 재구현했다고 명시한다
- **저장소에 `LICENSE`도 `COPYING`도 없다.** `_mit_xpos.py:2`가 `[see LICENSE for details]`를 가리키는데 그 파일이 없다 — MIT 고지 요건 미충족. 프로젝트 자체 라이선스도 없어(=기본값은 모든 권리 유보) GPL 코드를 담은 채 공개 배포되는 모양이다

즉 **"research-only"라는 주석은 사실이지만 파일의 위치를 바꾸지 않는다.** 배포 단위는 저장소다.

**세 갈래:**

- **(a) 트리에서 뺀다** — `git rm -r tools/vendored/`. 필요하면 실행 시 upstream에서 받아 오는 스크립트로 대체한다. 설계 문서의 불변식이 복원되고 라이선스 결정을 미룰 수 있다. 잃는 것: bake-off 재실행에 한 단계가 는다. `tools/vendored/`는 이미 프로덕션 의존이 0이므로 **제품은 아무것도 잃지 않는다**
- **(b) 안고 간다** — 프로젝트 라이선스를 GPLv3로 확정하고, `vendored/`에 upstream COPYING + `_mit_xpos.py`용 MIT LICENSE를 넣고 NOTICE를 쓴다. 정직하지만 저장소 전체가 GPL로 묶인다
- **(c) 분리한다** — bake-off 하네스를 별도 저장소로 (`git subtree split --prefix=packages/scanlation-server/tools`). 이 저장소는 깨끗해지고 그쪽이 GPL을 진다

**(a) 권장.** `vendored/`가 tools 전용이고 프로덕션 의존이 0이라 제품 손실이 없다. (b)는 "라이선스 미정" 상태를
끝내야 하는데 그건 리팩토링이 아니라 프로젝트 차원의 결정이다. **법적 함의가 있으므로 소유자 판단 사항.**

곁가지: **H4** — `compare_models.py:397`의 `"weights auto-included in tools/vendored/_mit_weights/"`는 사실이
아니다. `.gitignore`가 그 디렉터리를 제외하므로 새 clone에선 수동 다운로드가 필요하다. (a)를 고르면 함께 사라진다.

---

## 범위 밖 (최적화로 분류)

서두의 범위 제약에 걸려 이 백로그가 다루지 않는 것들. 발견은 했으니 기록만 남긴다.

- `gpuconc:122` — spawn 워커마다 PIL 이미지 42장 전체를 pickle 전송. 워커 수에 선형
- R1의 `warm()` 통합 — 워밍업 횟수가 툴마다 다르므로(threads는 `workers * 2`) 통합하면 측정 조건이 바뀐다
- R1의 `add_data_args` — threads의 `--data`를 positional로 통일하면 CLI가 바뀐다
- R1의 `sweep_baseline` 검증 신설, R5의 확장 fetch 타임아웃 신설 — 없던 동작 추가
- **H8** — 프로덕션 recognizer에 해상도 캡 도입

`batch:315`(배치 크기마다 per-crop 레퍼런스 재계산)는 **이미 계산한 값의 재사용**이므로 R1에 남긴다.

---

## 다음 순서 (완료분은 「현재 상태」 참조)

**바로 착수 가능** — 결정도 장비도 필요 없다:

1. **H9** — `registry._discover`의 조용한 `except`에 `logger.warning` 추가. 한 줄 + 테스트
2. **R6** — 플러그인 보일러플레이트를 SDK로. 5개 패키지에 걸치고, **가중치 없는 머신에선 엔진 스모크가 skip되어 검증이 얕아진다**
3. **R5** — 하드코딩 → `/admin`. 새 설정 필드가 늘어나므로 handshake·확장·i18n 동시 수정
4. **R7 → R9** — 대형 파일 분할 (R7은 R2가 남긴 `compare_models.py`의 어휘 잔여를 함께 처리)

**결정이 먼저** — 위 「현재 상태」의 "결정을 기다리는 것":

5. **B3** (권장 (a)), **H3** (권장 (a)), **H6** (캐시 무효화 감수 여부)

**GPU 호스트에서만:**

6. **B5+B6** — threads의 `_raw_bbox_crop_files`를 deskew 크롭으로 바꾸고 리포트의 하드코딩된 결론 산문을 제거.
   측정값이 바뀌므로 `tools/*.md`의 1.27x vs 1.8x 결론 재검토가 따라온다
7. **H8** — 해상도 캡을 프로덕션 recognizer로. 성능 변경이자 신규 기능이라 이 백로그 밖

**문서:** H4·H5·H7

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

> 개발 venv의 설치 메타데이터가 낡으면(H1/H2) 엔진 이름이 registry와 catalog에서 갈라진다. 엔진을 건드리는
> 검증 전에 `pip install -e`로 재설치해 두면 혼선이 적다.

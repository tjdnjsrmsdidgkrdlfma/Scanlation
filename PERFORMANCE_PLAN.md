# Scanlation 종합 성능 향상 로드맵 (위험·규모별 분류)

## Context — 왜

이 저장소는 이미 **측정 기반 성능 엔지니어링**이 성숙합니다: 벤치 툴 [run_report.py](packages/scanlation-server/tools/run_report.py), 설계 문서 [recognize-crop-batching.md](packages/scanlation-server/tools/recognize-crop-batching.md)·[recognize-gpu-speed.md](packages/scanlation-server/tools/recognize-gpu-speed.md), `InferenceGate`·워커풀·SQLite 결과 캐시·idle-unload·in-flight dedup. 큰 구조적 병목은 대부분 다뤄졌고 **crop 배치처럼 이미 측정으로 기각된 최적화는 다시 제안하지 않습니다.** 남은 것은 (a) 아직 손대지 않은 **저위험 위생 이슈**, (b) **클라이언트(확장) 체감 병목**, (c) opt-in 동시성이 **실제로 이득을 내도록** 하는 조절·문서 개선입니다.

**결정된 제약:** 실행 환경 CPU·GPU **양쪽** / **전 계층 로드맵** / 서버 동시성 기본값은 **opt-in 유지**(올리지 않고 조절 경로만 개선).

> **읽는 법 — 전부 다 할 필요 없습니다.** 아래는 **위험이 낮고 변경이 작은 것(Tier 0)** 에서 **크고 불확실한 것(Tier 4)** 순입니다. Tier 0~1만 해도 안전한 순이득이 나고, Tier 2~3은 체감이 큰 대신 손이 더 갑니다. Tier 4는 "측정해보고 결정"하는 후보입니다.

### Tier 분류 기준
| Tier | 성격 | 변경 표면 | 되돌리기 | 검증 |
|---|---|---|---|---|
| **0** | 측정·env·문서·도움말 텍스트 | 코드 동작 변경 없음 | 즉시 | 눈으로 확인 |
| **1** | 국소 저위험 코드 | 파일 1~2개, **출력 불변** | 커밋 revert | 기존 pytest + 벤치 A/B |
| **2** | 새 조절값 배관 | 여러 파일이지만 **확립된 패턴 미러** | 커밋 revert | pytest 케이스 미러 + 수동 |
| **3** | 클라이언트 구조 변경 | 신규 로직, 동작 표면 변화 | 커밋 revert(수동 재확인) | 브라우저 수동 A/B |
| **4** | 대규모·불확실 | 스파이크 + A/B 선행, 채택 미정 | 스파이크 브랜치 폐기 | 전용 A/B 후 판단 |

### 한눈에 보기
| Tier | 항목 | 대상 파일 | 위험 | 기대 효과 |
|---|---|---|---|---|
| 0 | GPU 레버 실적용 점검(flash-attn env·다운스케일 캡·멀티워커) | 배포 env, README | 없음 | GPU recognize 최대 ~3.7x(문서 실측) 실제 반영 |
| 0 | `/admin` 조절 힌트(동반 backend `--parallel`) | admin i18n | 없음 | 켠 동시성이 실제로 이득 나게 |
| 0 | `detect_lock` 직렬 천장 문서화 | README/주석 | 없음 | 오해 방지 |
| 0 | 베이스라인 측정 수립 | 없음(툴 실행) | 없음 | 모든 A/B의 기준선 |
| 1 | 이벤트 루프 동기 SQLite → threadpool/deferred | server 2파일 | 낮음 | 동시성 시 루프·캐시락 병목 제거 |
| 1 | deskew `np.asarray` 페이지당 1회 캐시 | `geometry.py` | 낮음 | 회전 크롭 많은 페이지 CPU 절감 |
| 1 | `torch.inference_mode`(detector) | detector plugin | 낮음 | detect 소폭 |
| 1 | MutationObserver 디바운스 | `content.js` | 낮음 | 동적 페이지 오버헤드 감소 |
| 2 | `client_concurrency` 조절값 + 확장 동시성 큐 | server 6 + ext 4 | 중간 | 무제한 병렬 폭주 → 유한 상한 |
| 3 | 메인스레드 md5 → Web Worker | ext 2~3 | 중간 | 이미지당 UI 프리즈 제거 |
| 3 | lazy-load/뷰포트(IntersectionObserver) | `content.js` | 중간 | 누락 이미지 해소 + 가시 우선 |
| 3 | 렌더 read/write 분리 + DocumentFragment | `content.js` | 낮음~중간 | resize/오버레이 reflow 폭풍 완화 |
| 4 | RT-DETR ONNX Runtime(CPU detect) | detector plugin | 미지수 | CPU detect 큰 이득 가능 |
| 4 | 검출기 풀화 / batch-lookup / async httpx / 부팅 워밍업 | 다수 | 미지수 | 상황별 천장 제거 |

---

## Tier 0 — 측정·설정·문서 (코드 동작 변경 없음, 가장 안전)

가장 먼저·가장 안전. 새 코드 로직이 없고 되돌리기가 즉시입니다.

### 0-A. 베이스라인 측정 수립 (모든 A/B의 기준)
- **현황**: 툴은 있으나 "현재 성능 스냅샷"이 고정돼 있지 않으면 이후 변경의 이득을 증명할 수 없음.
- **할 일**: 대표 샘플로 3종 리포트 확보 — `run_report.py samples/`(serial, 단계별 분해), `--parallel`(throughput_pps), `--no-translate`(recognize 격리). CPU·GPU 경로 각각. 결과 md/json을 기준선으로 보관.
- **효과/위험**: 이득 측정의 토대. 위험 0.

### 0-B. GPU 경로 레버가 실제로 켜져 있는지 점검·명문화
- **현황**: [recognize-gpu-speed.md](packages/scanlation-server/tools/recognize-gpu-speed.md)에 실측된 레버들이 **배포 env에 달림**. flash/AOTriton(`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) → vision attention **~3.7x**, 다운스케일 캡(`downscale_to_cap`, PaddleOCR-VL `recognize` 입력 전) → **~1.66x**(이미 구현). 코드가 아니라 **환경에서 빠지면 그냥 손해**.
- **할 일**: 실제 배포에서 env가 적용되는지 확인하고, README/`/admin` 도움말에 "GPU 배포 시 이 env" 체크리스트로 명문화. 멀티워커(`recognize_concurrency`)·게이트(`gpu_concurrency`)는 opt-in 유지하되 GPU에서 켤 때의 값 가이드만.
- **효과/위험**: 이미 검증된 큰 배수를 **실제로 받게** 함. 코드 변경 없음.

### 0-C. `/admin` 조절 힌트 — 동시성이 실제로 이득 나게
- **현황**: 실효 병렬성은 `min(translate_sem, backend --parallel)`([state.py:100](packages/scanlation-server/app/state.py#L100) 주석). 운영자가 `/admin`에서 `translate_concurrency`만 올리고 host의 `OLLAMA_NUM_PARALLEL`/llama-server `--parallel`을 안 올리면 **아무 이득이 없음**.
- **할 일**: `/admin` 동작 탭 해당 필드에 **동반 조정 안내 도움말**(ko/en i18n)만 추가. 동작·기본값 불변.
- **효과/위험**: opt-in 정책 유지하면서 "켜면 실제로 빨라지는" 경로. 순수 텍스트.

### 0-D. `detect_lock` 직렬 천장 문서화
- **현황**: `gpu_concurrency>1`로 이미지 겹침을 켜도 detect forward는 `detect_lock`([state.py:146](packages/scanlation-server/app/state.py#L146))로 전 이미지 직렬 — 공유 in-process 검출기가 동시 forward에 안전하지 않기 때문(**설계상 의도**).
- **할 일**: GPU 겹침 사용 시 "detect는 여전히 직렬"이라는 **알려진 천장**을 README/주석에 기록. (해소는 Tier 4 검출기 풀화 후보.)
- **효과/위험**: 오해 방지. 코드 변경 없음.

---

## Tier 1 — 국소 저위험 코드 (파일 1~2개, 출력 불변)

출력 결과는 그대로이고 스케줄링/내부 계산만 바뀝니다. 기존 pytest가 회귀를 잡습니다.

### 1-A. 이벤트 루프의 동기 SQLite 제거
- **현황**: async 경로에서 **동기 SQLite가 이벤트 루프에서 직접** 실행 — `cache.get_run`([run.py:68](packages/scanlation-server/app/routes/run.py#L68), async 라우트), `cache.put_run`([orchestrator.py:222](packages/scanlation-server/app/orchestrator.py#L222)), `cache.record_stats`([orchestrator.py:241](packages/scanlation-server/app/orchestrator.py#L241), compute 코루틴). 모두 단일 `cache._lock`([cache.py:60](packages/scanlation-server/app/cache.py#L60)) + JSON 직렬화. 동시성이 켜지면 **모든 요청이 캐시 락에서 직렬화되고 루프가 짧게 블록**. (`/run_lookup/`은 동기 `def`라 이미 threadpool — 문제 아님.)
- **변경**: 세 호출을 `run_in_threadpool`로 래핑(orchestrator엔 이미 import됨, [orchestrator.py:19](packages/scanlation-server/app/orchestrator.py#L19)). `record_stats`는 통계라 크리티컬 패스가 아니므로 **응답 반환 후 fire-and-forget**로 분리(지연 무관). `put_run`은 결과 캐싱 정합상 완료돼야 하나 threadpool로 오프로드.
- **효과**: 단독 이득은 작지만 **Tier 2 동시성의 필수 전제** — 미리 넣어야 처리량 A/B가 캐시 락에 오염되지 않음.
- **위험/규모**: 낮음. 서버 2파일(`routes/run.py`, `orchestrator.py`), 결과 불변. 되돌리기 revert.
- **검증**: 기존 `pytest`(캐시/라우트 테스트) green + `--parallel` throughput 전/후.

### 1-B. deskew `np.asarray` 페이지당 1회 캐시
- **현황**: [geometry.py](packages/scanlation-server/app/geometry.py)의 `deskew_crop`이 **회전 크롭마다** `np.asarray(img)`로 전체 페이지 ndarray를 재실체화. 축정렬 크롭은 PIL 빠른 경로라 무관하지만, 회전 영역이 많은 페이지에서 같은 배열을 N번 다시 만드는 CPU 낭비.
- **변경**: 페이지 ndarray를 **1회 계산해 회전 크롭들이 공유**. `deskew_crop`(및 호출부 [pipeline.py:179](packages/scanlation-server/app/pipeline.py#L179))에 선계산 배열을 넘기거나, 페이지 단위로 lazy 캐시. 축정렬 빠른 경로는 그대로. (참고: `geometry.py`는 **서버 app 내부** 모듈이라 SDK version bump 규칙과 무관.)
- **효과**: 회전 크롭 많은 페이지의 recognize 직전 CPU 절감. 출력 픽셀 동일.
- **위험/규모**: 낮음. 1파일 중심. 결과 불변(같은 크롭 산출).
- **검증**: `run_report.py`의 detect/recognize 분해 + `region_stats`로 회전 많은 샘플 A/B.

### 1-C. `torch.inference_mode` (detector)
- **현황**: detector가 `torch.no_grad`(comic-text-and-bubble-detector `plugin.py`의 forward). recognizer의 `generate()`는 내부적으로 이미 no_grad이므로 대상 아님.
- **변경**: detector forward를 `torch.inference_mode()`로 교체(autograd-view 부작용 없는지 확인).
- **효과/위험/규모**: detect 소폭. 낮음. detector plugin 1곳.
- **검증**: 검출 결과 동일 확인 + detect_ms 전/후.

### 1-D. MutationObserver 디바운스
- **현황**: [content.js:297](extension/src/content.js#L297) 콜백이 추가 노드마다 `scan(n)`→`querySelectorAll`를 스로틀 없이 실행 — SPA/광고 페이지에서 잦은 DOM 변동 시 오버헤드.
- **변경**: 변경을 rAF/짧은 타이머로 **코얼레스**해 한 프레임에 1회 스캔. (Tier 3의 IntersectionObserver 도입 시 스캔 자체가 더 가벼워짐 — 함께 가면 좋지만 독립적으로도 안전.)
- **효과/위험/규모**: 동적 페이지 CPU 절감. 낮음. `content.js` 국소.
- **검증**: 동적 페이지에서 DevTools Performance long task 감소.

---

## Tier 2 — 새 조절값 배관 (여러 파일, 확립된 패턴 미러)

파일 수는 많지만 **`min_image_dim`이 지나가는 경로를 그대로 미러**하므로 배관 자체는 저위험입니다. 신규 위험은 확장의 "동시성 큐 로직" 한 군데뿐입니다.

### 2-A. 확장 무제한 병렬 → 유한 동시성 상한 (`client_concurrency`)
- **현황**: [content.js:248](extension/src/content.js#L248) `els.forEach(processImage)`가 `await` 없이 페이지의 **모든** 이미지에 대해 base64 인코딩+md5+네트워크를 **동시에** 발사. 상한이 전혀 없어 대형 챕터에서 브라우저 메모리/소켓 폭주. 서버는 어차피 게이트/세마포어로 직렬화하므로 클라이언트 폭주는 **이득 없이 낭비**.
- **변경 (두 부분)**:
  1. **확장 동시성 큐(신규 로직, 유일한 새 위험)**: `scan()`은 대상 엘리먼트를 큐에 넣기만 하고 동시 N개 슬롯이 `processImage`를 소비. N개가 끝나면 다음.
  2. **조절값 `client_concurrency` 배관(패턴 미러)** — `min_image_dim`과 **동일 경로**:
     - env 기본 + floor 1: [config.py:59](packages/scanlation-server/app/config.py#L59) 옆에 `_env_int("SCANLATION_CLIENT_CONCURRENCY", …, floor=1)`.
     - Selection 필드 + 클램프: [state.py:95](packages/scanlation-server/app/state.py#L95), [state.py:253-254](packages/scanlation-server/app/state.py#L253) `set_client_config`.
     - 요청 스키마: [schemas.py:84](packages/scanlation-server/app/schemas.py#L84) 옆 `Optional[int]`.
     - `/admin` 동작 탭 영속 + 노출 + i18n: [admin.py:100](packages/scanlation-server/app/routes/admin.py#L100)(get_settings), [admin.py:187-195](packages/scanlation-server/app/routes/admin.py#L187)(set_client_config).
     - handshake payload: [handshake.py:48](packages/scanlation-server/app/routes/handshake.py#L48) 옆.
     - 확장 내장 폴백 상수: [constants.js:9](extension/src/constants.js#L9)에 `CLIENT_CONCURRENCY`(= `MIN_IMAGE_DIM: 80` 미러) — **리터럴을 content.js에 박지 않음**. 소비: [content.js:19](extension/src/content.js#L19) cfg + [content.js:37](extension/src/content.js#L37) `loadConfig`, popup의 storage 기록([popup.js:85-89](extension/src/popup.js#L85)).
     - 테스트 미러: [test_routes_admin.py:103-118](packages/scanlation-server/tests/test_routes_admin.py#L103), [test_state.py:35](packages/scanlation-server/tests/test_state.py#L35)에 `client_concurrency` 케이스 추가.
- **효과**: 클라이언트/서버 발사량 분리, 메모리·소켓 안정, 메인스레드 부하 분산. **무한→유한 완화**라 "opt-in 기본값 유지" 정책과 상충하지 않음(동시성 상향이 아님).
- **위험/규모**: 배관은 저위험(패턴 존재), 큐 로직만 신규 중간. server 6 + ext 4 파일.
- **검증**: pytest 미러 케이스 green + 브라우저 Network waterfall로 동시 요청이 상한 준수.

---

## Tier 3 — 클라이언트 구조 변경 (신규 로직, 브라우저 수동 A/B)

체감이 가장 크지만 자동 테스트가 없어 수동 검증이 필요합니다. Tier 2의 큐 위에 얹으면 자연스럽습니다.

### 3-A. 메인스레드 md5 → Web Worker
- **현황**: [content.js:229](extension/src/content.js#L229) `md5(base64)`가 동기([md5.js:5](extension/src/md5.js#L5)). 만화 페이지 base64는 수 MB → 이미지마다 메인스레드 프리즈. `crypto.subtle`은 MD5 미지원이고 와이어 계약이 **base64 문자열에 대한 md5**([run_report.py:89](packages/scanlation-server/tools/run_report.py#L89))라 알고리즘 교체 불가.
- **변경**: 기존 `md5.js`를 **Web Worker**에서 실행(계약 그대로). MV2에서 워커 스크립트는 `runtime.getURL`로 로드. Tier 2 큐와 결합하면 인코딩+해싱이 슬롯 단위로 흩어져 long task 소멸. 워커 미가용 프레임은 동기 폴백.
- **효과/위험/규모**: 이미지당 UI 프리즈 제거. 중간(워커 배선). ext 2~3파일.
- **검증**: DevTools Performance에서 md5 long task 소멸, 결과 동일(md5 값 불변).

### 3-B. lazy-load / 뷰포트 대응 (IntersectionObserver)
- **현황**: MutationObserver가 `childList`만 감시([content.js:300](extension/src/content.js#L300)) — 다수 웹툰 뷰어는 같은 `<img>`의 `src`만 스왑해 **누락**. 미로드 이미지는 `naturalSize` 0이라 스킵([content.js:223](extension/src/content.js#L223))되고 `load` 재시도 없음. 뷰포트 우선순위도 없어 off-screen까지 즉시 처리.
- **변경**: **IntersectionObserver**로 뷰포트 진입 시 처리(가시 이미지 우선 = 체감 개선 + off-screen 작업 지연). 미로드 이미지엔 1회성 `load` 리스너(또는 `attributes: ['src','srcset']` 감시)로 재시도.
- **효과**: 정확성(누락 해소) + 성능(불필요 작업 억제) 동시. Tier 2 큐와 결합 시 "보이는 것부터 N개씩".
- **위험/규모**: 중간(스캔 트리거 로직 교체). `content.js` 중심.
- **검증**: lazy 뷰어에서 스크롤 시 이미지가 빠짐없이 잡히는지 수동 확인.

### 3-C. 렌더 read/write 분리 + DocumentFragment
- **현황**: [sizeFonts](extension/src/content.js#L175)가 박스마다 읽기(`clientWidth/Height`)와 쓰기(`style.fontSize`)를 교차 → forced reflow. [onResize](extension/src/content.js#L317)의 `tracked.forEach(sizeFonts)`가 전 이미지로 확대. [applyResult](extension/src/content.js#L198)는 박스를 하나씩 `appendChild`.
- **변경**: `sizeFonts`/`onResize`를 **측정 일괄 → 쓰기 일괄**로 분리하고 `requestAnimationFrame` 안에서 수행. `applyResult`는 **DocumentFragment**에 박스를 모아 1회 append. 150ms 디바운스는 유지.
- **효과/위험/규모**: resize·오버레이 reflow 폭풍 완화. 낮음~중간. `content.js` 국소.
- **검증**: 이미지 많은 페이지 resize 시 Performance layout 시간 감소.

---

## Tier 4 — 대규모·불확실 (스파이크 + A/B 선행, 채택 미정)

이 팀은 채택 전 실측하는 문화이므로([recognize-crop-batching.md](packages/scanlation-server/tools/recognize-crop-batching.md)에서 배치 기각) 모두 **스파이크 브랜치에서 A/B 후 판단**. 계획엔 후보로만 둡니다.

- **RT-DETR ONNX Runtime (CPU detect)**: CPU 경로 detect를 ONNX Runtime으로 — CPU 배포에서 큰 이득 가능성. 정확도/속도 A/B 필수. 현재 transformers safetensors만 fetch.
- **검출기 풀화 (Tier 0-D 연장)**: detect를 워커 프로세스로 옮겨 GPU 겹침의 `detect_lock` 직렬 천장 제거. 복잡도 높음.
- **batch-lookup 엔드포인트**: 콜드 캐시에서 이미지마다 `/run_lookup/`+`/run_pipeline/` 2 RTT([content.js:83](extension/src/content.js#L83)) — N개 md5를 1회 프로브로 묶어 RTT 절감. **localhost에선 무의미**, 원격 배포에서만 가치.
- **async httpx translator**: [http_translator.py:72](packages/scanlation-sdk/scanlation_sdk/http_translator.py#L72) `httpx.Client`(동기) — 지금은 threadpool에서 돌아 루프를 막지 않으므로 우선순위 낮음. 동시성 상향으로 threadpool 포화가 **관측될 때만**.
- **부팅 후 1회 워밍업 (의도적 결정 재검토)**: `recognize_pool`은 현재 **의도적으로** 합성 워밍업이 없어 첫 요청이 커널 JIT(PaddleOCR-VL ~12s)를 흡수. 조절값(기본 off) 뒤의 선택적 워밍업으로 "부팅 시간 ↔ 첫 요청 지연"을 맞바꿈. 기존 결정을 뒤집는 것이라 GPU 배포 A/B 후 판단.
- **pool 경로 deskew 워커 이동**: 1-B 연장 — deskew를 워커로 옮겨 recognize와 병렬화(현재 fan-out 전 메인 프로세스 CPU 직렬).

---

## 권장 시퀀스 & 규칙 준수

**시퀀스**: Tier 0(측정·문서) → Tier 1(저위험 코드) → Tier 2(조절값+큐) → Tier 3(클라이언트 구조) → Tier 4(스파이크). 각 코드 변경마다 해당 벤치로 전/후 A/B.

**CLAUDE.md 준수 체크**:
- **하드코딩 금지 → `/admin`**: `client_concurrency`(및 선택적 워밍업 토글)는 env 기본 + `state.json` + `/admin` 동작 탭(ko/en i18n) + handshake→storage→content 전 경로. 확장 폴백은 `constants.js` `SCAN.*` 상수(리터럴 금지).
- **역할 어휘**: detector/recognizer/translator, `{bounds, source, destination}`. BOX/OCR/TSL 미사용.
- **plugin vs engine** 구분 준수 / **내부 SDK version bump 금지**(`geometry.py`는 서버 app이라 무관).
- **출력**: 회고적 이력 주석 금지, 현재형만. 커밋 메시지 영어 + `Co-Authored-By` 트레일러, 커밋 후 바로 push.

## 검증 (계층별)
- **서버**: `pytest`(packages/scanlation-server) green. `run_report.py` serial(단계별) + `--parallel`/`--concurrency K`(throughput) + `--no-translate`(recognize 격리)로 각 서버 변경 전/후.
- **확장(수동)**: 대표 긴 챕터에서 (1) Network로 동시 요청이 `client_concurrency` 준수, (2) Performance로 md5 워커 후 long task 소멸, (3) 스크롤 시 lazy 이미지 포착, (4) resize reflow 완화.
- **전달 경로**: `/admin` 동작 탭 새 필드 + ko/en i18n, handshake payload → 확장 `storage.local` 반영을 popup 연결로 확인.

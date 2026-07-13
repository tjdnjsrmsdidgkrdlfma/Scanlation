# recognize GPU 속도 — PaddleOCR-VL은 왜 느렸나 (그리고 스위치는 어디 있었나)

작성 2026-07-08. 측정 도구: [bench_recognize_gpu_concurrency.py](bench_recognize_gpu_concurrency.py). crop-batching이 기각된([recognize-crop-batching.md](recognize-crop-batching.md)) 뒤 GPU recognizer(PaddleOCR-VL)의 남은 레버 **"동시성(멀티워커)"**을 재려다, **"애초에 왜 느린가"**를 끝까지 판 기록. 환경: 서버 9060 XT(gfx1200/RDNA4) + ROCm 7.1, torch rocm7.0, 실제 Pixiv 챕터 crop 42개(detect + deskew, 파이프라인과 동일).

## 결론 먼저

| 레버 | 이득 | 대가 | 판정 |
|---|---|---|---|
| **AOTriton flash attention** (env 한 줄) | **3.7x** | 없음 | **채택 — `docker-compose.rocm.yml`에 반영됨** |
| 해상도 캡 150k + `pow2` | 1.66x | 24개 중 3개가 뭉개진 말줄임/작은 가나(줄 소실 없음), 나머지는 표기 차 | **채택 — 구현 완료(`3503181`)** |
| 멀티워커 (W=4) | **1.38x** (정점 W=3 1.41x) | VRAM 4배, per-crop 지연 | **채택 확정 · 린 코어 구현 완료(W=1 off 기본)** — recognize-bound + MI50 GPU분리, 2의 배수 정렬로 W=4 (§동시성 판정) |

병목은 **vision attention**이 맞다 — 큰 crop이 수백 개 vision 토큰을 만들고 attention이 그걸 매 스텝 문다. **그런데 해법이 막힌 게 아니었다:** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`이 sdpa를 flash/mem-efficient 커널로 태운다.

| 같은 스크립트(`--diag`) · 같은 crop | crops/sec |
|---|---|
| AOTriton **OFF** | 0.094 |
| AOTriton **ON** | **0.345** |

## ⚠️ 측정 사고 — 이 문서의 flash-OFF 수치들

조사 중반의 측정(decode 프로파일, 초기 캡 sweep, 초기 동시성 프로브)이 **AOTriton을 끈 채** 이뤄졌다. 경위:

- "AOTriton이 효과 있나"를 본다며 **플래그를 켠 채 두 번 돌려** 같은 값(0.35 / 0.34)이 나오자 **"무효"로 판정**했다. 비교 기준으로 삼은 이전 baseline(0.34)도 **역시 켜진 값**이었다(§셋업 부채가 "수동 `-e`로 우회해 돌렸다"고 명시).
- 즉 **가진 숫자가 전부 flash-ON**이라 "켜도 안 변한다"는 필연이었고 정보가 0이었다. **OFF 기준선을 한 번도 안 쟀다.**
- 그 오판을 근거로 이후 커맨드에서 플래그를 빼면서, 뒤의 측정이 flash-OFF(3.7배 느림)로 진행됐다.

**교훈: 플래그 A/B는 반드시 OFF 기준선을 따로 잰다. 벤치 env는 커맨드마다 명시적으로 고정한다.**

## 배제된 가설들 (전부 실측)

- **캐시/컴파일** — 같은 커맨드 2회차가 1회차와 바이트 단위로 동일(root로 `PYTORCH_KERNEL_CACHE_PATH`를 쓰기 가능하게 해도 동일). 캐시가 채워졌는데 안 빨라지면 캐시 문제가 아니다.
- **decode 캡 / 런어웨이 생성** — 느린 crop의 출력은 `♥` 반복이 아니라 **진짜 다중 대사(58자)**. `--probe-cap 64`로 잘라도 max가 안 줄었다(출력이 이미 ≤64토큰).
- **crop 집합** — `--diag`의 crop 인벤토리(내용 md5)를 두 런 diff → **바이트까지 동일.** 검출기는 결정적이다.
- **스크립트 변경** — git 확인: 세션 중 배치 벤치의 유일한 변경은 env-gated `BENCH_ATTN` 블록(unset이면 no-op). baseline 루프 무변경.
- **GPU 전력/클럭 상태** — perflevel을 건드리지 않고 AOTriton만 켜도 0.345가 재현된다.
- **`attn_implementation="sdpa"`** — 명시해도 변화 없음. 모델이 **이미 sdpa를 기본으로** 쓰기 때문(`ValueError` 없이 로드). AOTriton은 그 sdpa가 느린 math 폴백 대신 **flash 커널**을 타게 하는 스위치다.

→ 남은 변수는 **AOTriton 하나**였다.

## flash 효과 — 큰 crop일수록 크다

같은 24 crop, `--workers 1`:

| per-crop | flash-OFF | flash-ON | |
|---|---|---|---|
| min | 538ms | 239ms | 2.2x |
| med | 5613ms | 1973ms | 2.8x |
| **max** | **40568ms** | **8830ms** | **4.6x** |

**crop이 클수록 이득이 크다**(2.2x → 4.6x). vision attention의 O(n²)를 flash가 없앤다는 진단의 서명이다.

## 왜 attention이 병목인가 — decode 프로파일 (flash-OFF)

`--profile-decode`로 생성 스텝별 wall time(`StoppingCriteria` + `cuda.synchronize`):

| crop | tokens | prefill+t1 | steady/token | first steps ms |
|---|---|---|---|---|
| 1709x343 (586k px) | 34 | 38065ms | **1293ms** | 38065 1279 1286 1287 … (flat) |
| 438x928 (406k px) | 50 | 25987ms | 719ms | 25987 707 708 711 … (flat) |
| 102x316 (32k px) | 9 | 141ms | 135ms | 141 136 136 135 … (flat) |

1. **per-token이 flat** → **KV 캐시는 정상**(O(n) 증가도, 스텝마다 vision 재계산도 아님).
2. **per-token이 crop 크기에 비례**(32k=135ms → 586k=1293ms) → 각 decode 스텝이 그 crop의 **vision 토큰 수만큼** 일한다.
3. **prefill이 지배** — 큰 crop 25~38s, decode 스텝 하나의 ~29배. patch28 그리드로 586k px ≈ **800 vision 토큰**.

flash 없는 sdpa(math 폴백)는 이 둘을 O(n²)로 문다. **AOTriton이 그걸 flash로 바꾸는 것 = 3.7x.**

**참고 — `prefill+t1`의 12s 스파이크:** 같은 149k crop 셋 중 첫 번째만 12245ms, 나머지는 137ms. 이건 **프로세스 첫 forward의 1회성 커널 JIT**(shape 무관 generic 커널 대량 컴파일)다. 디스크 영속이 안 돼 새 프로세스마다 다시 물지만, **long-running 서버에선 부팅 후 1회성**이라 per-crop 비용이 아니다(벤치 crops/sec가 비관적인 또 다른 이유).

## 동시성(멀티워커) — 원래 질문의 답 (flash-ON 실측)

각 워커가 별도 프로세스로 **B=1 recognize**를 돌린다(배치가 아니라 동시성이라 padding·correctness 문제 없음). `--workers 1,2,4,6,8`:

| W | crops/sec | speedup | per-crop med | est VRAM |
|---|---|---|---|---|
| 1 | 0.34 | 1.00x | 1973ms | 2.0GB |
| 2 | 0.32 | **0.92x** | 4706ms | 4.0GB |
| 4 | 0.45 | **1.31x** | 6652ms | 7.9GB |
| 6 | 0.44 | 1.29x | 9551ms | 11.9GB |
| 8 | — | `BrokenProcessPool`(OOM) | | >16GB |

- **ROCm엔 MPS가 없다** → GPU 커널이 동시 실행되지 않고 **타임슬라이스**된다. 동시성으로 벌 수 있는 건 워커의 **CPU 구간**(전처리·토크나이저·Python decode 루프)에 생기는 **GPU 유휴를 다른 워커가 채우는 것**뿐이다.
- 그래서 천장 = `1/duty`. **측정 천장 1.31x → W=1에서 GPU가 이미 ~76% 바쁘다.** 남은 24%를 회수한 게 전부이고, 구조적으로 더 짜낼 여지가 없다. (W=4에서 per-crop이 3.4배 느려지는데 4-way라 net 1.31x — 딱 그 그림. W=2는 낮은 큐 깊이라 유휴를 못 채우고 경합 오버헤드만 먹어 오히려 손해.)
- **VRAM 천장**: 모델 사본 하나당 ~2.0GB(torch alloc, HIP 컨텍스트 별도) → 16GB에 W=6까지, W=8은 OOM.

**판정: 지금은 안 넣는다.** 1.31x가 **VRAM 4배 + per-crop 지연 3.4배**(1973→6652ms) 값을 못 한다. 참고로 manga-ocr CPU 멀티워커는 1.88x에 VRAM 비용 0([recognize-cpu-threads.md](recognize-cpu-threads.md)). **재고 조건**: ROCm이 MPS급 co-execution을 제공하거나, VRAM이 남고 지연이 무관한 순수 처리량 워크로드일 때.

> 즉 *"처리량은 배치가 아니라 동시성"*은 **CPU 엔진엔 맞지만 GPU 엔진엔 안 맞는다.** GPU 쪽 진짜 레버는 flash attention이었다. (하드웨어 확장 시의 "GPU별 역할 분리"는 한 GPU를 나눠 쓰는 게 아니라 물리적 분리라 이 천장과 무관하다.)

### 후속 — 캡 켠 현재 스택, 워커 1~8 full 스윕 (2026-07-12)

해상도 캡(150k + `pow2`)을 넣은 프로덕션 스택에서 워커 1~8을 다 재측정(2026-07-10의 W=1,2,4,6 부분 측정을 W=3,5,7까지 채운 것).

| W | crops/sec | speedup | per-crop med | est VRAM |
|---|---|---|---|---|
| 1 | 0.58 | 1.00x | 1612ms | ~1.9GB |
| 2 | 0.52 | **0.91x** ↓ | 3500ms | ~3.9GB |
| **3** | 0.81 | **1.41x** (정점) | 2968ms | ~5.8GB |
| 4 | 0.80 | **1.38x** | 4369ms | ~7.7GB |
| 5 | 0.78 | 1.35x | 5886ms | ~9.7GB |
| 6 | 0.76 | 1.31x | 7148ms | ~11.6GB |
| 7 | — | OOM (BrokenProcessPool) | | >16GB |

- **W=1 baseline 0.34→0.58(1.71x)** — 동시성이 아니라 **캡의 처리량 이득**이다(run_report의 1.75x와 교차 검증).
- **정점 W=3(1.41x), 이후 완만한 하락**(W=4 1.38 · W=6 1.31). W=2는 **딥(0.91x)** — 큐 깊이가 낮아 유휴를 못 채우고 경합 오버헤드만 먹는다. 구조적 천장(~1.4x)은 ROCm MPS 부재로 GPU 유휴(~28%) 회수가 상한이라, 캡 전(1.31x)보다 살짝 오른 것뿐.
- **VRAM 천장 W=6** — 캡을 써도 per-worker ~1.9GB × 7 + activation/HIP 오버헤드가 16GB를 넘어 W=7이 OOM.
- **→ W=4 채택.** 정점은 W=3(1.41x)이나 W=4(1.38x)와 사실상 동일하고, **2의 배수 정렬**(설정·확장 일관성)로 W=4를 택한다. `chars max 62`가 전 W 동일 → 동시성은 배치와 달리 correctness가 온전.

> **판정 전환 (2026-07-12) — 재고조건 충족.** 위 "안 넣음"은 **manga-ocr가 기본**이라 파이프라인이 translate-bound(recognize가 전체의 18.7%)이던 전제였다. **PaddleOCR-VL로 전환하면 recognize가 병목**이 된다 — recognize-only 68.6s = translate 31.7s의 **2.2배**, 전체의 **64.7%**(run_report 2건 실측, manga-ocr는 18.7% ↔ PaddleOCR-VL는 64.7%로 역전). 그리고 **MI50 32GB 도입으로 translate(Gemma)를 별 GPU로 분리**하면 이 절이 든 반대가 다 사라진다: VRAM 4배 → 9060 XT 16GB를 recognize가 전용, per-crop 지연 → 처리량 워크로드라 무관, `gpu_lock` 경합 → GPU 물리 분리로 0. correctness는 배치와 달리 원래 안전. **→ W=4 채택 확정**(정점 W=3 1.41x이나 2의 배수 정렬로 W=4 1.38x, 위 full 스윕 표). 구현: **린 코어(프로세스풀 + crop fan-out + per-engine W 배관)** — env `SCANLATION_RECOGNIZE_CONCURRENCY`(기본 1 = off, 기존 per-crop 경로와 동일) + `/admin` 플러그인별 설정, [`recognize_pool.py`](../app/recognize_pool.py). W>1이면 한 페이지의 crop을 W개 워커 프로세스(각 B=1)로 fan-out. `gpu_lock`→gate 전환·크로스이미지 오버랩(K)은 `--no-translate` 벤치로 **MI50 전 구현·측정 완료**(§크로스이미지 오버랩, K=2 sweet spot); 프로덕션 상시(translate 포함)는 MI50 GPU 분리 후. [배치](recognize-crop-batching.md)는 2026-07-12 공정 실측(같은 크롭·캡 대칭·no_grad)에서 per-crop보다 **1.3x 느려 폐기**됐다(옛 2.04x는 stale) — 멀티워커가 recognize 동시성의 유일한 레버다.

### 크로스이미지 오버랩 — 크롭 천장 회수 (K, 2026-07-13)

위 멀티워커(W)는 **한 페이지의 crop만** 워커 풀에 fan-out한다 — `gpu_lock`이 detect+recognize를 이미지 단위로 직렬화하기 때문. 그래서 페이지 crop 수 < W면 워커가 논다(**크롭 천장**). 실제 챕터는 페이지당 말풍선이 대부분 2개라, W=4를 켜도 워커 절반이 항상 논다. bench의 W=4 1.38x는 crop 42개를 **이미지 경계 없이 한 스트림**으로 밀어넣은 값이라, 사실상 크롭 천장이 없는(= cross-image가 있는) 상한이다.

**구현**: `gpu_lock`(단일 mutex) → `InferenceGate(K)` reader/writer 게이트 — K개 이미지가 detect+recognize를 동시에 통과해, 여러 페이지의 crop이 **공유 워커 풀을 함께 채운다**. K는 per-recognizer(`recognize_concurrency`(W)와 대칭), K=1 기본=직렬(기존과 byte-identical). detect는 공유 torch 모델이라 `detect_lock`으로 직렬, recognize(워커 프로세스)만 겹친다. RecognizePool은 in-flight refcount로 self-protected(gpu_lock 의존 제거). 커밋 `74ba94e`, [`state.py InferenceGate`](../app/state.py)·[`recognize_pool.py`](../app/recognize_pool.py).

**실측** — `run_report.py --parallel --no-translate`, 21장(실제 챕터), W=4 고정, K 스윕. 지표는 **batch 전체 벽시계 = `total_ms`의 최대값**(21장이 다 끝나는 실제 시간):

| 설정 | batch 벽시계 | lockwait 평균 | vs baseline |
|---|---|---|---|
| W1·K1 (baseline) | 79.7s | 36.9s | 1.00x |
| W4·K1 | 71.5s | 33.4s | 1.11x |
| **W4·K2** | **52.6s** | 21.2s | **1.52x** |
| W4·K4 | 49.4s | 18.0s | 1.61x |

- **W만 올림(W1K1→W4K1)은 1.11x뿐** — crop 2개 페이지가 W=4의 절반만 쓰니 크롭 천장 그대로.
- **K를 켬(W4K1→W4K2)이 1.36x 점프** — 놀던 워커 2개가 옆 페이지 crop으로 채워진다. lockwait 33.4→21.2s로 붕괴(오버랩 동작 신호).
- **K=2→K4는 1.06x뿐.** per-image recognize는 3배 폭증(3.2s→8.4s, ROCm MPS 부재로 GPU를 4장이 타임슬라이스)하는데 batch 이득은 미미 — 자원만 더 쓴다.

**판정: K=2 sweet spot.** 이론 `K ≈ W ÷ 페이지당 평균 crop = 4 ÷ 2 = 2`와 일치(2 crop × 2 이미지 = W=4 풀 참). → **프로덕션 W=4·K2**, baseline 대비 1.5x. 최적 K는 W가 아니라 **crop 분포**에 달렸으므로(K=W로 자동 묶으면 crop-적은 챕터에서 오버) K는 별도 축으로 노출/기본 2가 낫다.

**단, recognize-only 실측이다.** translate까지 켠 end-to-end는 9060 XT에 gemma가 VRAM을 공유 못 해 여기선 못 잰다 — **MI50(translate 전용 GPU) 도입 후** 재측정해야 최종 그림(recognize-bound가 얼마나 풀리는지)이 나온다.

## 해상도 캡 (flash-ON 실측 — 판정 보류)

crop을 픽셀 상한으로 다운스케일해 vision 토큰을 줄인다(`--max-pixels` / `--sweep-pixels`).

**정확도는 세 관점으로 보되, 캡끼리 비교되는 건 "무언가를 고정한" 둘뿐이다.**

| 지표 | 고정하는 것 | 답하는 질문 | 캡 간 비교 |
|---|---|---|---|
| `바뀐 crop /24` | 표본 24개 전부 | 캡이 **몇 개**를 바꾸나 | ✅ |
| `coh-*` | 코호트 = 가장 큰 캡 초과 crop 6개(= affected 집합들의 교집합, **모든 캡이 줄인다**) | 제일 세게 맞는 애들을 **얼마나 심하게** | ✅ |
| `aff-sim` | 아무것도 | — | ❌ |

`aff-sim`이 무효인 이유: 캡을 낮추면 **"간신히 넘어 살짝만 줄어든" crop이 새로 껴서** sim≈1.0으로 평균을 올린다. 그래서 100k(0.906)가 150k(0.870)보다 좋아 *보인다*.

| cap | crops/sec | 속도 | 바뀐 /24 | coh-sim | **coh-chars** |
|---|---|---|---|---|---|
| uncapped | 0.35 | 1.00x | 0 | 1.000 | 54.2 |
| 250k | 0.43 | 1.23x | 4 | 0.891 | 54.0 |
| 200k | 0.51 | 1.46x | 9 | **0.769** | **46.2** |
| **150k** | 0.58 | **1.66x** | 10 | 0.868 | 54.8 |
| 100k | 0.57 | 1.63x | 12 | 0.860 | 54.3 |
| 50k | 0.60 | 1.71x | 15 | 0.789 | **46.3** |

### `coh-chars`가 잡아낸 것 — 줄 소실(truncation)

`coh-chars`(같은 6개의 평균 출력 길이)가 200k·50k에서 54 → 46으로 떨어진다. 그 지점에서 crop #4가 **통째로 잘린다**:

```
ref 'お...ほぉ...っっ\nや、やっぱこの体勢...\n奥まで入るう...♥♥\n子宮ごりごりくるう...♥♥'
got 'お...ほぉ...'                                        ← 4줄 중 3줄 소실
```

**출력이 짧아지면 decode가 줄어 "빨라진다".** 50k가 이 런에서 가장 빠른(0.60) 것은 **정확도를 잃어서 산 속도**다. 같은 이유로 저 구간의 crops/sec는 런마다 흔들린다(직전 런에선 50k가 0.47).

### 손상이 단조롭지 않다

`coh-sim`: 250k 0.891 → **200k 0.769** → 150k 0.868 → 100k 0.860 → 50k 0.789.
**덜 줄인 200k가 더 줄인 150k보다 나쁘다** — #4가 200k에선 잘리고 150k에선 안 잘렸다. 잘림은 매끄러운 열화가 아니라 **임계·확률적 사건**이라, 특정 캡이 "안전하다"고 보장할 수 없다.

### 150k에서 실제로 바뀌는 것 (10개)

- **코스메틱만** (#9 #10 #17 #21 #22): `・・・`↔`...`, ♥ 개수, ♥↔♡
- **실제 오독·쓰레기 문자**: #4 `ほぉ→ぼぉ` · #7 `ばっかり→ばつかり`, `:・・` · #13 `・・・→::` · #19 `::`
- **오히려 고쳐짐**: #3 `思いつきり→思いっきり` · #13 `つ→っ`

**원본(uncapped)도 틀린다.** 작은 가나(っ/つ)와 말줄임 표기는 입력이 조금만 흔들려도 뒤집히는 **모델 노이즈**다(`ばっかり`는 50k에선 오히려 맞게 나온다). 그러니 **"바뀜 = 나빠짐"이 아니다.** 다만 `::` 같은 쓰레기 문자와 줄 소실은 노이즈가 아니라 손상이다.

## 다운스케일 *방식* — 대가의 일부는 리샘플링 탓이었다 (`--sweep-modes`)

**캡을 150k로 고정하고 줄이는 방식만 바꾼다.** 캡이 고정이라 **같은 16개 crop이 모든 방식에서 다운스케일**되므로 정확도 열이 코호트 트릭 없이 직접 비교된다.

| mode | 필터 | mean-px | crops/sec | 바뀐 /24 | **sim** | chars |
|---|---|---|---|---|---|---|
| uncapped | — | 259k | 0.35 | 0 | 1.000 | 38.7 |
| **area** (현행) | LANCZOS | 150k | 0.58 | 10 | **0.870** | 38.6 |
| grid28 | LANCZOS | 139k | 0.58 | 10 | 0.886 | 38.5 |
| box | BOX | 150k | 0.59 | 10 | 0.904 | 38.6 |
| boxgrid | BOX | 139k | 0.58 | 12 | 0.908 | 38.4 |
| **pow2** | BOX(정수배) | **65k** | 0.58 | 11 | **0.916** | 37.8 |

1. **속도는 모든 방식이 같다(0.58).** `pow2`가 65k px(2.3배 작음)인데도 같다 — 150k 아래에선 vision이 이미 바닥이고 decode가 지배한다. **따라서 방식 선택은 공짜다: 가장 정확한 걸 고르면 된다.**
2. **필터가 지배 요인이다.** LANCZOS 최대(0.886) **<** BOX 최소(0.904)로 **구간이 겹치지 않는다.** LANCZOS는 windowed-sinc라 얇은 획·`・・・` 점에 링잉을 남기고, BOX(면적 평균)는 안 남긴다. 정수배 다운스케일이 좋은 이유도 *"2의 배수"라서가 아니라* **각 출력 픽셀이 k×k 블록의 정확한 평균**이기 때문이다.
3. **`pow2`는 2.3배 더 줄였는데도 원본에 더 가깝다**(0.916 vs `area` 0.870). 이 구간에선 **"얼마나 줄이냐"보다 "어떻게 줄이냐"가 크다.**
4. **28 그리드 정렬은 부차적이다** — LANCZOS 위에선 +0.016, BOX 위에선 +0.004. 범인은 프로세서의 두 번째 리사이즈가 아니라 **첫 번째 리샘플의 품질**이었다.

### 그러나 "해결"은 아니다

- **바뀐 crop 수는 안 준다**(10 → 11~12). 변화가 *덜 심해질* 뿐 *덜 일어나진* 않는다.
- **`::` / `:・・`는 대부분 모드에 남는다**(#7 #13 #19). 저해상도에서 `・・・` 점이 뭉쳐 콜론처럼 읽히는 **해상도 효과**라 필터로 못 없앤다.
- **개별 crop 개선은 카오스적이다.** `box`(281x533)는 #7의 `ばっかり`와 #13의 `・・・`를 살렸는데, **1픽셀 다른** `boxgrid`(280x532)는 못 살렸다. **24개 표본에서 개별 crop으로 튜닝하지 말 것** — 믿을 신호는 **필터 계열의 sim 서열**뿐이다.
- **150k에선 어떤 모드도 줄 소실이 없다**(chars 38.4~38.6 vs 원본 38.7).

### 판정 — `pow2` + 150k (구현 완료)

- **현행 `area`(임의 배율 + LANCZOS)는 다섯 중 최악이다.** 같은 속도에 가장 부정확하다. **캡을 쓴다면 `pow2`로 쓴다** — 이건 캡 채택/기각과 무관하게 참이다. (벤치 기본값도 아직 `area`이니 `--downscale-mode pow2`를 줘야 한다.)
- **채택: cap 150k + `pow2`, 1.66x.** 잔여 대가는 24개 중 **3개**가 뭉개진 말줄임/작은 가나를 얻는 것(#7 `:・・`+`ばつかり`, #13 `::`, #19 `お::·`)이고 나머지는 `・・・`↔`...`·♥ 개수·♥↔♡ 같은 **표기 차**다. **줄 소실은 없다.** #3은 오히려 원본 오류를 고친다(`思いつきり→思いっきり`).
- **구현 완료(`3503181`)** — `max_pixels`(env `SCANLATION_RECOGNIZE_MAX_PIXELS`·기본 150000)·`downscale_mode`(기본 `pow2`)를 [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)의 `OPTION_SCHEMA`로 `/admin` 노출, `recognize` 입력 전 `downscale_to_cap` 적용.

## 실투입 — 셋업 부채

이 문서의 모든 GPU 측정은 **벤치가 수동 `-e`로 우회해** 돌린 것이다. `/admin`에서 PaddleOCR-VL을 GPU로 설치하는 경로의 부채는 네 갈래였고 서로 독립적이다. **#1(accelerate)·#4(캐시 핀)·#5(AOTriton)는 해결**됐고, 남은 건 torch 인덱스 **#2·#3**이다.

> 참고: PaddleOCR-VL은 **native transformers 경로**로 로드된다(`AutoModelForImageTextToText`, `trust_remote_code` 아님 — transformers 5.x가 `transformers/models/paddleocr_vl/`로 지원). "remote-code 모델"이 아니다.

### 1. ~~`accelerate`가 의존성에 없다~~ — deps에 추가로 해결
- **증상(이었던 것)**: GPU 로드가 `ValueError: Using a device_map … requires accelerate`로 죽어 전 이미지 실패(2026-07-10 실측 확인 — 실재하는 하드 블로커였다).
- **원인**: [plugin.py `_load`](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)의 `device_map=device`가 transformers에서 `accelerate`를 요구하는데 [pyproject](../../scanlation-paddleocr-vl-for-manga/pyproject.toml) `dependencies`에 없었다.
- **해결**: `dependencies`에 `accelerate` 추가(`device_map` 로드 경로 유지). 배포 때 `/plugins`에 안 딸려오면 `pip install --target /plugins accelerate`(또는 플러그인 `--force-reinstall`)로 반영.

### 2. AMD torch 설치가 PyPI의 CUDA 빌드로 샌다
- **증상**: 백엔드=GPU + AMD인데 `torch 2.12.1+cu130`(CUDA 빌드)이 깔린다.
- **원인**: [plugins_install.py `_torch_pip_args`](../app/plugins_install.py#L114-L116)의 amd 경로가 `--index-url <rocm>`과 `--extra-index-url https://pypi.org/simple`을 **함께** 준다. **pip엔 인덱스 우선순위가 없다** — `extra-index-url`은 같은 네임스페이스로 합쳐지고 pip은 **모든 인덱스를 통틀어 최고 버전**을 고른다. rocm6.2 인덱스는 torch 2.5.1까지인데 PyPI엔 2.12.x가 있으니 **PyPI(CUDA)가 이긴다.**
- **고칠 것**: **2단계 설치.** ① torch를 **rocm 인덱스만** 줘서 먼저 설치(PyPI 없이). ② 그다음 플러그인을 평소대로 설치(torch는 이미 충족되어 안 건드리고, 나머지 의존성만 PyPI에서).
- **주의**: 3번만 고쳐도 버전이 맞아떨어져 PEP 440 local-version 규칙(`2.12.1+rocm7.0` > `2.12.1`)으로 rocm이 우연히 이길 수 있다. 하지만 인덱스가 다시 뒤처지면 재발하므로 **2단계 분리가 근본 해법**이다.

### 3. AMD 기본 torch 인덱스가 `rocm6.2`
- **원인**: [plugins_install.py:115](../app/plugins_install.py#L115)에 `https://download.pytorch.org/whl/rocm6.2`가 기본값으로 박혀 있다. 이 인덱스는 **torch 2.5.1까지**라 현 스택과 어긋난다.
- **고칠 것**: 기본값을 `rocm7.0`으로(호스트 ROCm 7.1에 `rocm7.0` wheel이 맞았다). 사용자 오버라이드는 이미 `state.selection.torch_index`(/admin)로 열려 있으니 **기본값만** 바꾸면 된다.

### 4. ~~`HOME`이 `/root`라 캐시가 `Permission denied`~~ — 캐시별 경로 핀으로 해결
`setpriv`가 uid/gid만 바꾸고 `HOME`은 `/root` 그대로 둬서([docker-entrypoint.sh](../../../docker-entrypoint.sh)) app 유저가 `/root`에 못 쓰는 건 여전하다. 하지만 **`HOME`을 바꿀 필요 없이 캐시별로 경로를 직접 핀**하면 된다 — 셋 다 app-writable·영속인 `/data`로, 모두 [Dockerfile](../../../Dockerfile)에서:
- **MIOpen DB** — `MIOPEN_USER_DB_PATH`/`MIOPEN_CUSTOM_CACHE_DIR=/data/miopen`(`a36e92d`). 안 풀렸을 땐 `miopenStatusUnknownError`로 매 GPU conv가 죽었다(하드 블로커).
- **torch JIT 커널 캐시** — `PYTORCH_KERNEL_CACHE_PATH=/data/torch-kernels`. `libtorch_hip`가 이 env를 읽는 걸 확인. 첫 forward의 1회성 hiprtc 컴파일이 재생성을 넘어 영속 — **콜드 스타트만 절약, 정상 처리량엔 무영향**.
- **HF 캐시** — `HF_HOME=/data/hf`로 원래 빠져 있어 안 깨진다.

### 5. ~~compose에 AOTriton env~~ — 반영 완료
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`([docker-compose.rocm.yml](../../../docker-compose.rocm.yml)). 3.7x.

**남은 순서**: 3(기본값) → 2(설치 분리). 1·4·5는 완료. 2·3은 함께 검증하는 게 낫다.

**부수 발견**: `_torch_pip_args`의 기본이 `torch_backend="cpu"`라 **GPU 호스트에서도 CPU wheel을 받는다**. device-node 자동 감지(`detect_gpu_vendor`) 기반 "auto" 기본값으로 개선 여지가 있다(별도 건).

## 실투입 — 나머지 결정

- **해상도 캡** — **채택: cap 150k + `pow2`**(§다운스케일 방식), 1.66x. **구현 완료(`3503181`)** — env 기본값(`SCANLATION_RECOGNIZE_MAX_PIXELS`) + `/admin` 노출, [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)의 `recognize` 입력 전 `downscale_to_cap`. **`area`로 넣지 않았다** — `pow2` 기본(같은 속도에 가장 정확).
- **멀티워커** — 안 넣음(위 §동시성의 재고 조건; 캡 켠 뒤 재측정 1.38x로도 재확인).

## 참고 — 4070 Ti Super 대비

flash-ON 2.9s/crop 대 4070 Ti Super ~1s/crop ≈ **2.9배**. CUDA는 SDPA가 기본으로 flash를 타므로 이제 **양쪽 다 flash-ON의 공정 비교**다. 대역폭 672 vs ~320 GB/s(2.1x, decode가 대역폭 바운드) + 상위 급의 compute 우위(vision prefill)로 설명되는 범위이고, 남는 잔차가 없다. 단 "~1s/crop"은 다른 이미지 기준의 대략치라, 엄밀히는 4070 박스에서 같은 챕터로 `--diag`를 돌려 비교해야 한다.

## 관련

- 배치 축(단일 forward에 N크롭)은 [recognize-crop-batching.md](recognize-crop-batching.md) — 양쪽 엔진 다 기각.
- CPU 멀티워커(manga-ocr)는 [recognize-cpu-threads.md](recognize-cpu-threads.md) — 8w×1t 1.88x.
- 동시성·`gpu_lock`·translate 배치 그림은 아티팩트 [동시성과 번역 배치](https://claude.ai/code/artifact/543ff4c0-d2be-4d4f-9d70-fc35fac17c1f).

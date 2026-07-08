# recognize GPU 속도 — PaddleOCR-VL은 왜 느린가 (그리고 유일한 레버)

작성 2026-07-08. 측정 도구: [bench_recognize_gpu_concurrency.py](bench_recognize_gpu_concurrency.py). crop-batching이 기각된([recognize-crop-batching.md](recognize-crop-batching.md)) 뒤, GPU recognizer(PaddleOCR-VL)의 처리량 레버로 남은 "동시성(멀티워커)"을 재려다 **"PaddleOCR-VL이 애초에 왜 느린가"**를 끝까지 판 기록. 환경: 서버 9060 XT(gfx1200/RDNA4) + ROCm 7.1, torch rocm7.0, 실제 Pixiv 챕터 크롭 42개.

## 결론 먼저

느림의 정체는 **캐시도, 하드웨어 체급도, 배치도, 동시성도 아니다** — **거대 crop이 만든 수백 개 vision 토큰 × flash 없는 eager O(n²) attention**이다. 유일하게 당길 수 있는 레버는 **recognize 전 해상도 다운스케일(max-pixels)**: 오버사이즈 crop을 바닥으로 끌어내려 이 챕터에서 ~2.1x. 단 그 바닥(**~135ms/token**)은 flash attention 없이는 못 내리고, 캡 후에도 manga-ocr(CPU ~6 crops/sec)보다 느리다. **캡은 "느린 crop을 평범하게" 만드는 것이지 "전체를 빠르게"가 아니다.**

## 배제된 가설들 (전부 실측)

- **동시성(멀티워커)** — GPU 동시성 프로브를 만들었으나, 재보니 병목이 여기가 아니었다(아래). 한 GPU를 나눠 쓰는 거라 상한이 CPU 멀티워커와 다르고, 애초에 per-crop이 초 단위라 동시성 이전에 per-crop을 봐야 했다.
- **캐시/컴파일** — 같은 커맨드 2회차가 1회차와 **바이트 단위로 동일**(root로 `PYTORCH_KERNEL_CACHE_PATH` 써도 동일). 캐시가 채워졌는데 안 빨라지면 캐시 문제가 아니다. (단 프로세스 첫 forward의 ~12s JIT는 별개 — 아래.)
- **decode 캡/런어웨이** — 느린 crop의 출력이 `♥♥♥` 반복이 아니라 **진짜 다중 대사(58자)**. `--probe-cap 64`로 잘라도 max가 안 줄었다(출력이 이미 ≤64토큰이라) → 런어웨이 아님.
- **attention 백엔드(sdpa / AOTriton)** — `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`도 `attn_implementation="sdpa"`도 수치 불변 → 이 모델의 attention이 그 크롭에서 flash/SDPA 경로를 안 타고 eager로 돈다(이게 O(n²)의 원인).

## decode 프로파일 — 정체를 잡은 곳

`--profile-decode`로 생성 스텝별 wall time을 잰다(`StoppingCriteria` + `cuda.synchronize`). full-res:

| crop | tokens | prefill+t1 | steady/token | first steps ms |
|---|---|---|---|---|
| 1709x343 (586k px) | 34 | 38065ms | **1293ms** | 38065 1279 1286 1287 … (flat) |
| 438x928 (406k px) | 50 | 25987ms | 719ms | 25987 707 708 711 … (flat) |
| 102x316 (32k px) | 9 | 141ms | 135ms | 141 136 136 135 … (flat) |

세 가지가 드러난다:

1. **per-token이 flat**(1279, 1286, 1287…) → **KV 캐시는 정상**. O(n) 증가도, 스텝마다 vision 재계산도 아니다.
2. **per-token이 crop 크기에 비례**(32k=135ms → 586k=1293ms). 각 decode 스텝이 그 crop의 vision 토큰 수만큼 일한다 = **eager attention이 매 스텝 vision KV 전체를 문다**(flash면 near-free).
3. **prefill이 지배** — 큰 crop은 25~38초, decode 스텝 하나의 ~29배. patch28 그리드로 586k px ≈ **800 vision 토큰**, eager O(n²)가 prefill을 수십 초로 만든다.

**12245ms의 정체:** cap 후 같은 149k crop 셋 중 첫 번째만 12s, 나머지는 137ms. 이건 **프로세스 첫 forward의 1회성 커널 JIT**(shape 무관 generic 커널 대량 컴파일)다. 디스크 영속이 안 돼 새 프로세스마다 다시 물지만, **long-running 서버에선 부팅 후 첫 recognize 1회성**이라 실사용 per-crop 비용이 아니다. (벤치 crops/sec가 비관적인 이유 — 매 `docker run`이 이걸 먹는다.)

## 레버 — 해상도 캡 (`--max-pixels`)

crop을 픽셀 상한으로 다운스케일해 vision 토큰을 줄인다. cap 150k:

| | full-res | cap 150k |
|---|---|---|
| per-token (steady) | 719~1293ms | **~135ms** (crop 크기 무관 바닥) |
| warm prefill | 25~38s | **~137ms** |
| crops/sec (24-crop) | 0.09 | **0.19** (2.1x) |
| max / med ms | 40568 / 5613 | 16134 / 3843 |

**단 10x는 다운스케일된 crop에만 걸린다.** 42개 중 26개만 >150k라 다운스케일됐고, 원래 작은 crop은 그대로다:

| crop | full-res | cap 150k |
|---|---|---|
| `いいよ…一番奥まできて…`(큰) | 40568ms | 6790ms (6x↓) |
| `おまんこ…イってる…`(작음) | 3512ms | 3512ms (동일) |
| `おッッ`(작음) | 538ms | 539ms (동일) |

crops/sec는 전부의 합이라 큰 몇 개만 6x 빨라지고 작은 다수는 그대로 → 블렌드해서 **2.1x**. 그래서 캡의 이득은 챕터의 crop 크기 분포에 종속이다.

**정확도 knee — `--sweep-pixels` 실측**(캡별 속도 vs uncapped, 열화는 다운스케일된 crop만 비교):

| cap | 속도(uncapped 대비) | exact | |
|---|---|---|---|
| 250k | 1.3x | 20/24 | 거의 안 빨라짐 |
| 200k | 2.5x | 15/24 | |
| **150k** | **3.7x** | **14/24** | **knee** |
| 100k | 3.8x | 12/24 | 속도 정체, 정확도만↓ |

**knee = 150k px.** 200k→150k에서 속도가 확 오르고(2.5→3.7x) 100k는 정체(exact만 14→12 손해). 열화 대부분은 코스메틱(♥ 개수·っ/つ·`・・・`)이라 `いいよ…` 같은 실제 대사는 원본과 동일하게 읽힌다. **실투입 다운스케일 캡 후보 = 150k.**

**⚠️ 절대 crops/sec는 신뢰하지 말 것.** 같은 코드·같은 이미지인데 per-crop baseline이 런마다 **0.34 / 0.09 / 0.06**으로 5배 넘게 흔들린다(원인 미확정 — crop 집합은 동일할 것이므로 GPU 상태/캐시/공유 같은 런타임 변동으로 추정, 조사 중). 위 수치는 절대값이 아니라 **한 런 안의 상대 비교(캡 knee, 큰 crop 배수)로만** 읽어야 한다.

## 남은 바닥 — 캡으로 못 내린다

캡 후 모든 crop의 남은 시간 = **출력 토큰 수 × ~135ms/token**. 캡은 해상도를 줄이지 출력 길이를 못 줄인다(진짜 텍스트). 135ms/token은 대역폭 바닥(~5.6ms)의 ~25배 — eager attention + 커널 런치 오버헤드의 per-step 바닥이다. 더 내리려면:

- **flash/mem-efficient attention** = 진짜 해법(O(n²)→선형, per-step도 급감). 이 ROCm/RDNA4에선 막힘(AOTriton 무효). 미래 ROCm/드라이버가 열어주면 이 문서의 대부분이 사라진다.
- **토큰 줄이기** = 불가(실제 텍스트).

## 판단 / 실투입

- **manga-ocr(CPU)가 처리량 기본** — 캡 후 PaddleOCR-VL 0.19 crops/sec도 manga-ocr ~6보다 느리다. PaddleOCR-VL은 정확도(88%) 프리미엄이 필요한 곳에만.
- **PaddleOCR-VL을 쓸 거면 recognize 전 다운스케일을 config로** — 하드코딩 금지 규칙대로 env 기본값 + `state.json` + `/admin` 노출(값은 `--sweep-pixels` knee로 확정). 통합 지점은 [pipeline.py `detect_and_recognize`](../app/pipeline.py) 또는 [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)의 `recognize` 입력 전.
- **동시성(멀티워커)은 이 병목 위에선 무의미** — per-crop이 초 단위인 한 GPU를 나눠 써도 vision O(n²)는 그대로. 캡으로 per-crop을 낮춘 뒤에야 동시성 재평가 값어치가 생긴다.

## 관련

- 배치 축(단일 forward에 N크롭)은 [recognize-crop-batching.md](recognize-crop-batching.md) — 양쪽 엔진 다 기각.
- CPU 멀티워커(manga-ocr)는 [recognize-cpu-threads.md](recognize-cpu-threads.md) — 8w×1t 1.88x.
- 동시성·`gpu_lock`·translate 배치 그림은 아티팩트 [동시성과 번역 배치](https://claude.ai/code/artifact/543ff4c0-d2be-4d4f-9d70-fc35fac17c1f).

# recognize GPU 속도 — PaddleOCR-VL은 왜 느렸나 (그리고 스위치는 어디 있었나)

작성 2026-07-08. 측정 도구: [bench_recognize_gpu_concurrency.py](bench_recognize_gpu_concurrency.py). crop-batching이 기각된([recognize-crop-batching.md](recognize-crop-batching.md)) 뒤 GPU recognizer(PaddleOCR-VL)의 남은 레버 "동시성"을 재려다, **"애초에 왜 느린가"**를 끝까지 판 기록. 환경: 서버 9060 XT(gfx1200/RDNA4) + ROCm 7.1, torch rocm7.0, 실제 Pixiv 챕터 crop 42개(detect + deskew, 파이프라인과 동일).

## 결론 먼저

병목은 **vision attention**이 맞다 — 큰 crop이 수백 개 vision 토큰을 만들고 attention이 그걸 매 스텝 문다. **그런데 해법이 막힌 게 아니었다:** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`이 sdpa를 flash/mem-efficient 커널로 태우고, **그것만으로 3.7배**다.

| 같은 스크립트(`--diag`) · 같은 crop | crops/sec |
|---|---|
| AOTriton **OFF** | 0.094 |
| AOTriton **ON** | **0.345** |

→ **실투입 필수: `docker-compose.rocm.yml`에 이 env를 넣어야 한다.** [crop-batching](recognize-crop-batching.md)의 §셋업 부채 5번은 cosmetic이 아니라 **load-bearing**이다.

## ⚠️ 측정 사고 — 아래 flash-OFF 수치들의 출처

이 조사 중반의 측정(decode 프로파일, 해상도 캡 sweep, 동시성 프로브)이 **AOTriton을 끈 채** 이뤄졌다. 경위:

- "AOTriton이 효과 있나"를 본다며 **플래그를 켠 채 두 번 돌려** 같은 값(0.35 / 0.34)이 나오자 **"무효"로 판정**했다. 비교 기준으로 삼은 이전 baseline(0.34)도 **역시 켜진 값**이었다(§셋업 부채가 "수동 `-e`로 우회해 돌렸다"고 명시).
- 즉 **가진 숫자가 전부 flash-ON**이라 "켜도 안 변한다"는 필연이었고 정보가 0이었다. **OFF 기준선을 한 번도 안 쟀다.**
- 그 오판을 근거로 이후 커맨드에서 플래그를 빼면서, 뒤의 모든 측정이 flash-OFF(3.7배 느림)로 진행됐다.

**교훈: 플래그 A/B는 반드시 OFF 기준선을 따로 잰다. 벤치 env는 커맨드마다 명시적으로 고정한다.** 아래 flash-OFF 표들은 *"왜 attention이 병목인가"*의 근거로는 유효하지만 **절대 수치는 flash-ON에서 재측정해야 한다.**

## 배제된 가설들 (전부 실측)

- **캐시/컴파일** — 같은 커맨드 2회차가 1회차와 바이트 단위로 동일(root로 `PYTORCH_KERNEL_CACHE_PATH`를 쓰기 가능하게 해도 동일). 캐시가 채워졌는데 안 빨라지면 캐시 문제가 아니다.
- **decode 캡 / 런어웨이 생성** — 느린 crop의 출력은 `♥` 반복이 아니라 **진짜 다중 대사(58자)**. `--probe-cap 64`로 잘라도 max가 안 줄었다(출력이 이미 ≤64토큰).
- **crop 집합** — `--diag`의 crop 인벤토리(내용 md5)를 두 런 diff → **바이트까지 동일.** 검출기는 결정적이다.
- **스크립트 변경** — git 확인: 세션 중 배치 벤치의 유일한 변경은 env-gated `BENCH_ATTN` 블록(unset이면 no-op). baseline 루프 무변경.
- **GPU 전력/클럭 상태** — perflevel을 건드리지 않고 AOTriton만 켜도 0.345가 재현된다.
- **`attn_implementation="sdpa"`** — 명시해도 변화 없음. 모델이 **이미 sdpa를 기본으로** 쓰기 때문(`ValueError` 없이 로드). AOTriton은 그 sdpa가 **느린 math 폴백 대신 flash 커널**을 타게 하는 스위치다.

→ 남은 변수는 **AOTriton 하나**였다.

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

flash 없는 sdpa(math 폴백)는 이 둘을 O(n²)로 문다. **AOTriton이 그걸 flash로 바꾸는 것 = 3.7x.** 즉 위 표는 "attention이 범인"의 증거이고, 그 수치들은 flash-ON에서 크게 낮아진다.

**참고 — `prefill+t1`의 12s 스파이크:** cap 후 같은 149k crop 셋 중 첫 번째만 12245ms, 나머지는 137ms. 이건 **프로세스 첫 forward의 1회성 커널 JIT**(shape 무관 generic 커널 대량 컴파일)다. 디스크 영속이 안 돼 새 프로세스마다 다시 물지만, **long-running 서버에선 부팅 후 1회성**이라 per-crop 비용이 아니다(벤치 crops/sec가 비관적인 또 다른 이유).

## 해상도 캡 (flash-OFF 측정 — flash-ON 재측정 필요)

crop을 픽셀 상한으로 다운스케일해 vision 토큰을 줄인다(`--max-pixels` / `--sweep-pixels`). **flash-OFF**에서:

| cap | 속도(uncapped 대비) | exact | |
|---|---|---|---|
| 250k | 1.3x | 20/24 | 거의 안 빨라짐 |
| 200k | 2.5x | 15/24 | |
| **150k** | **3.7x** | **14/24** | knee (flash-OFF 기준) |
| 100k | 3.8x | 12/24 | 속도 정체, 정확도만↓ |

이득은 **다운스케일된 crop에만** 걸린다(42개 중 26개만 >150k). 작은 crop은 그대로라 전체 blend는 2.1x였다. 정확도 열화는 대부분 코스메틱(♥ 개수·っ/つ·`・・・`)이고 `いいよ…` 같은 실제 대사는 원본과 동일하게 읽혔다.

**단 이 knee는 flash-OFF 기준이다.** flash-ON에선 O(n²)가 완화돼 큰 crop의 패널티가 줄므로 **캡의 이득이 크게 작아질 수 있다** — 캡 자체가 불필요해질 가능성도 있다. flash-ON에서 재측정해 knee를 다시 잡는다.

## 실투입

1. **`docker-compose.rocm.yml`에 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 추가 — 3.7x, 최우선.** MIOPEN 캐시 env도 함께.
2. **해상도 캡**은 flash-ON 재측정 후 값어치를 판정한다. 넣게 되면 하드코딩 금지 규칙대로 env 기본값 + `state.json` + `/admin` 노출, 위치는 detect 다음·recognize 전([pipeline.py `detect_and_recognize`](../app/pipeline.py) 또는 [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)의 `recognize` 입력 전).
3. **동시성(멀티워커)**은 per-crop을 먼저 낮춘 뒤에야 재평가 값어치가 생긴다.

## 관련

- 배치 축(단일 forward에 N크롭)은 [recognize-crop-batching.md](recognize-crop-batching.md) — 양쪽 엔진 다 기각.
- CPU 멀티워커(manga-ocr)는 [recognize-cpu-threads.md](recognize-cpu-threads.md) — 8w×1t 1.88x.
- 동시성·`gpu_lock`·translate 배치 그림은 아티팩트 [동시성과 번역 배치](https://claude.ai/code/artifact/543ff4c0-d2be-4d4f-9d70-fc35fac17c1f).

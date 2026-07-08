# recognize GPU 속도 — PaddleOCR-VL은 왜 느렸나 (그리고 스위치는 어디 있었나)

작성 2026-07-08. 측정 도구: [bench_recognize_gpu_concurrency.py](bench_recognize_gpu_concurrency.py). crop-batching이 기각된([recognize-crop-batching.md](recognize-crop-batching.md)) 뒤 GPU recognizer(PaddleOCR-VL)의 남은 레버 **"동시성(멀티워커)"**을 재려다, **"애초에 왜 느린가"**를 끝까지 판 기록. 환경: 서버 9060 XT(gfx1200/RDNA4) + ROCm 7.1, torch rocm7.0, 실제 Pixiv 챕터 crop 42개(detect + deskew, 파이프라인과 동일).

## 결론 먼저

| 레버 | 이득 | 대가 | 판정 |
|---|---|---|---|
| **AOTriton flash attention** (env 한 줄) | **3.7x** | 없음 | **채택 — `docker-compose.rocm.yml`에 반영됨** |
| 해상도 캡 (150k px) | 1.66x | 다운스케일된 crop 대부분의 출력이 바뀜 | 보류 — 변경이 코스메틱인지 확인 중 |
| 멀티워커 (W=4) | 1.31x | VRAM 4배, per-crop 지연 3.4배 | 지금은 안 넣음 (천장에 구조적 이유) |

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

## 해상도 캡 (flash-ON 실측 — 판정 보류)

crop을 픽셀 상한으로 다운스케일해 vision 토큰을 줄인다(`--max-pixels` / `--sweep-pixels`). 정확도는 **다운스케일된 crop만** 채점한다(캡이 안 건드린 crop은 자동 일치라 metric을 희석시킨다).

| cap | crops/sec | 속도 | downscaled | exact | char-sim |
|---|---|---|---|---|---|
| uncapped | 0.35 | 1.00x | 0 | — | — |
| 250k | 0.43 | 1.23x | 6 | **2/6** | 0.891 |
| 200k | 0.51 | 1.46x | 12 | **3/12** | 0.839 |
| **150k** | 0.58 | **1.66x** | 16 | **6/16** | 0.870 |
| 100k | 0.57 | 1.63x | 19 | 7/19 | 0.906 |

- **flash가 캡의 값어치를 깎았다** — flash-OFF에서 3.7x였던 이득이 **1.66x**로 줄었다. 캡은 원래 O(n²)를 우회하려던 것인데 flash가 그걸 이미 없앴기 때문.
- **150k에서 속도가 정체**(100k는 오히려 0.57). 그 아래로는 vision 비용이 decode 바닥 밑으로 내려가 **속도 0, 정확도만 손해**다. 따라서 **판단할 캡은 150k 하나.**
- **가장 약한 250k조차 건드린 6개 중 4개의 출력이 바뀐다.** 희석 metric일 땐 "20/24 exact"로 좋아 보였는데, 그 18개는 캡이 손도 안 댄 crop이었다.
- char-sim의 캡 간 비교는 무의미하다(affected 집합이 다르다 — 낮은 캡일수록 "간신히 넘어 살짝만 줄어든" crop이 섞여 평균을 올린다). **exact 비율이 더 정직.**

**판정 보류.** 결정은 *무엇이* 바뀌는지에 달렸다 — `♥`/`♡` 개수, `・・・`↔`...` 표기 같은 **코스메틱**이면 1.66x는 사실상 공짜고, `ばっかり→ばつかり`처럼 **작은 가나가 바뀌면** 정확도 프리미엄(PaddleOCR-VL을 쓰는 이유)을 깎는 것이라 안 하는 게 맞다. `--sweep-pixels`가 이제 바뀐 crop마다 `ref`/`got`을 찍으니 눈으로 확정한다.

## 실투입

1. ~~`docker-compose.rocm.yml`에 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 추가~~ — **반영 완료**(3.7x). 순수 env라 파일시스템 의존이 없다. MIOPEN 커널 캐시 env는 app 유저가 쓸 수 있는 영속 디렉터리가 전제라 2번(entrypoint `HOME`)과 묶는다.
2. **§셋업 부채 나머지**([crop-batching](recognize-crop-batching.md))가 실제 배포 블로커다 — 지금까지 전부 벤치가 수동 env로 우회한 상태의 측정이고, `/admin`에서 GPU로 설치하면 아직 **로드조차 안 된다**(`accelerate` 누락, torch가 CUDA 빌드로 샘, rocm6.2 기본, entrypoint `HOME=/root`). MIOPEN 캐시 env를 compose에 넣는 것도 이 entrypoint 수정이 전제다.
3. **해상도 캡** — `ref`/`got` 확인 후 채택/보류 확정. 넣게 되면 하드코딩 금지 규칙대로 env 기본값 + `state.json` + `/admin` 노출, 위치는 detect 다음·recognize 전([pipeline.py `detect_and_recognize`](../app/pipeline.py) 또는 [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)의 `recognize` 입력 전).
4. **멀티워커** — 안 넣음(위 재고 조건).

## 참고 — 4070 Ti Super 대비

flash-ON 2.9s/crop 대 4070 Ti Super ~1s/crop ≈ **2.9배**. CUDA는 SDPA가 기본으로 flash를 타므로 이제 **양쪽 다 flash-ON의 공정 비교**다. 대역폭 672 vs ~320 GB/s(2.1x, decode가 대역폭 바운드) + 상위 급의 compute 우위(vision prefill)로 설명되는 범위이고, 남는 잔차가 없다. 단 "~1s/crop"은 다른 이미지 기준의 대략치라, 엄밀히는 4070 박스에서 같은 챕터로 `--diag`를 돌려 비교해야 한다.

## 관련

- 배치 축(단일 forward에 N크롭)은 [recognize-crop-batching.md](recognize-crop-batching.md) — 양쪽 엔진 다 기각.
- CPU 멀티워커(manga-ocr)는 [recognize-cpu-threads.md](recognize-cpu-threads.md) — 8w×1t 1.88x.
- 동시성·`gpu_lock`·translate 배치 그림은 아티팩트 [동시성과 번역 배치](https://claude.ai/code/artifact/543ff4c0-d2be-4d4f-9d70-fc35fac17c1f).

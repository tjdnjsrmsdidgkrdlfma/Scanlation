# recognize GPU 속도 — PaddleOCR-VL은 왜 느렸나 (그리고 스위치는 어디 있었나)

작성 2026-07-08. 측정 도구: [bench_recognize_gpu_concurrency.py](bench_recognize_gpu_concurrency.py). crop-batching이 기각된([recognize-crop-batching.md](recognize-crop-batching.md)) 뒤 GPU recognizer(PaddleOCR-VL)의 남은 레버 **"동시성(멀티워커)"**을 재려다, **"애초에 왜 느린가"**를 끝까지 판 기록. 환경: 서버 9060 XT(gfx1200/RDNA4) + ROCm 7.1, torch rocm7.0, 실제 Pixiv 챕터 crop 42개(detect + deskew, 파이프라인과 동일).

## 결론 먼저

| 레버 | 이득 | 대가 | 판정 |
|---|---|---|---|
| **AOTriton flash attention** (env 한 줄) | **3.7x** | 없음 | **채택 — `docker-compose.rocm.yml`에 반영됨** |
| 해상도 캡 (150k px) | 1.63x | 24개 중 10개 crop의 출력이 바뀜 | 보류 — 변경이 코스메틱인지 확인 중 |
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

| cap | crops/sec | 속도 | downscaled | **바뀐 crop /24** | (exact) | (char-sim) |
|---|---|---|---|---|---|---|
| uncapped | 0.35 | 1.00x | 0 | **0** | — | — |
| 250k | 0.43 | 1.23x | 6 | **4** | 2/6 | 0.891 |
| 200k | 0.51 | 1.46x | 12 | **9** | 3/12 | 0.839 |
| **150k** | 0.57 | **1.63x** | 16 | **10** | 6/16 | 0.870 |
| 100k | 0.57 | 1.63x | 19 | **12** | 7/19 | 0.906 |
| 50k | 0.47 | 1.34x | 24 | **15** | 9/24 | 0.887 |

**⚠️ 괄호 친 `exact`/`char-sim`은 캡끼리 비교하면 안 된다.** 그 캡이 다운스케일한 crop만 채점하는데 **그 집합이 캡마다 다르다** — 캡을 낮추면 "간신히 넘어 살짝만 줄어든" crop이 새로 끼어 거의 안 망가지므로 평균을 위로 끌어올린다(그래서 100k의 char-sim 0.906이 150k의 0.870보다 높아 보인다). 캡 간 비교가 성립하는 유일한 지표는 **분모가 고정된 "바뀐 crop /24"**다.

- **flash가 캡의 값어치를 깎았다** — flash-OFF에서 3.7x였던 이득이 **1.63x**로 줄었다. 캡은 원래 O(n²)를 우회하려던 것인데 flash가 그걸 이미 없앴기 때문.
- **속도는 U자다: 150k/100k가 정점(1.63x), 50k는 1.34x로 역주행.** per-crop ≈ `prefill(vision) + 출력토큰 × per-token`인데, 150k에서 이미 앞항이 바닥에 닿아 더 줄일 게 없고(150k→100k 정체), 50k에선 글자가 뭉개져 모델이 **더 길게 뱉으면서 뒷항(decode)이 늘어난다.** vision을 아끼려다 decode를 사는 셈이라 교환이 손해로 뒤집힌다.
- **150k가 100k를 약우위로 지배한다** — 속도 동일(0.57)인데 **바뀐 crop이 10 vs 12**. 따라서 **판단할 캡은 150k 하나.**
- **가장 약한 250k조차 건드린 6개 중 4개의 출력이 바뀐다.** 캡을 낮출수록 바뀌는 crop이 단조 증가(4 → 9 → 10 → 12 → 15).

**판정 보류.** 결정은 *무엇이* 바뀌는지에 달렸다 — `♥`/`♡` 개수, `・・・`↔`...` 표기 같은 **코스메틱**이면 1.66x는 사실상 공짜고, `ばっかり→ばつかり`처럼 **작은 가나가 바뀌면** 정확도 프리미엄(PaddleOCR-VL을 쓰는 이유)을 깎는 것이라 안 하는 게 맞다. `--sweep-pixels`가 이제 바뀐 crop마다 `ref`/`got`을 찍으니 눈으로 확정한다.

## 실투입 — 셋업 부채

이 문서의 모든 GPU 측정은 **벤치가 수동 `-e`로 우회해** 돌린 것이다. `/admin`에서 PaddleOCR-VL을 GPU로 설치하면 아직 **로드조차 안 된다.** 네 갈래이고 서로 독립적이다.

> 참고: PaddleOCR-VL은 **native transformers 경로**로 로드된다(`AutoModelForImageTextToText`, `trust_remote_code` 아님 — transformers 5.x가 `transformers/models/paddleocr_vl/`로 지원). "remote-code 모델"이 아니다.

### 1. `accelerate`가 의존성에 없다 — GPU 로드 하드 블로커
- **증상**: GPU 로드가 `ValueError`로 죽는다.
- **원인**: [plugin.py `_load`](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)가 `device_map=device`를 쓰는데 transformers의 `device_map` 경로는 `accelerate`를 요구한다. [pyproject](../../scanlation-paddleocr-vl-for-manga/pyproject.toml)의 `dependencies`엔 없다(`scanlation-sdk`, `transformers`, `torch`, `huggingface_hub`, `pillow`뿐).
- **고칠 것**: `dependencies`에 `accelerate` 추가.

### 2. AMD torch 설치가 PyPI의 CUDA 빌드로 샌다
- **증상**: 백엔드=GPU + AMD인데 `torch 2.12.1+cu130`(CUDA 빌드)이 깔린다.
- **원인**: [plugins_install.py `_torch_pip_args`](../app/plugins_install.py#L114-L116)의 amd 경로가 `--index-url <rocm>`과 `--extra-index-url https://pypi.org/simple`을 **함께** 준다. **pip엔 인덱스 우선순위가 없다** — `extra-index-url`은 같은 네임스페이스로 합쳐지고 pip은 **모든 인덱스를 통틀어 최고 버전**을 고른다. rocm6.2 인덱스는 torch 2.5.1까지인데 PyPI엔 2.12.x가 있으니 **PyPI(CUDA)가 이긴다.**
- **고칠 것**: **2단계 설치.** ① torch를 **rocm 인덱스만** 줘서 먼저 설치(PyPI 없이). ② 그다음 플러그인을 평소대로 설치(torch는 이미 충족되어 안 건드리고, 나머지 의존성만 PyPI에서).
- **주의**: 3번만 고쳐도 버전이 맞아떨어져 PEP 440 local-version 규칙(`2.12.1+rocm7.0` > `2.12.1`)으로 rocm이 우연히 이길 수 있다. 하지만 인덱스가 다시 뒤처지면 재발하므로 **2단계 분리가 근본 해법**이다.

### 3. AMD 기본 torch 인덱스가 `rocm6.2`
- **원인**: [plugins_install.py:115](../app/plugins_install.py#L115)에 `https://download.pytorch.org/whl/rocm6.2`가 기본값으로 박혀 있다. 이 인덱스는 **torch 2.5.1까지**라 현 스택과 어긋난다.
- **고칠 것**: 기본값을 `rocm7.0`으로(호스트 ROCm 7.1에 `rocm7.0` wheel이 맞았다). 사용자 오버라이드는 이미 `state.selection.torch_index`(/admin)로 열려 있으니 **기본값만** 바꾸면 된다.

### 4. `docker-entrypoint.sh`가 `HOME`을 안 바꾼다 — 캐시가 전부 `Permission denied`
- **증상**: torch JIT 커널 캐시(`$HOME/.cache/torch/kernels`), MIOpen DB, HF 캐시가 `Permission denied`로 꺼진다.
- **원인**: [docker-entrypoint.sh:18](../../../docker-entrypoint.sh)의 `setpriv --reuid --regid --init-groups`는 **uid/gid만 바꾸고 `HOME`은 `/root` 그대로** 둔다. app 유저는 `/root`에 못 쓴다.
- **고칠 것**: exec 전에 `HOME`을 app 유저의 홈으로 설정. 이게 풀려야 **MIOPEN 커널 캐시 env(`MIOPEN_USER_DB_PATH`/`MIOPEN_CUSTOM_CACHE_DIR`)를 compose에 넣을 수 있다**(영속 볼륨, warm 시 4.4x).

### 5. ~~compose에 AOTriton env~~ — 반영 완료
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`([docker-compose.rocm.yml](../../../docker-compose.rocm.yml)). 3.7x.

**순서 제안**: 1(한 줄, 하드 블로커) → 4(캐시 뿌리, MIOPEN env를 풀어줌) → 3(기본값) → 2(설치 분리). 1·4는 독립적이고, 2·3은 함께 검증하는 게 낫다.

**부수 발견**: `_torch_pip_args`의 기본이 `torch_backend="cpu"`라 **GPU 호스트에서도 CPU wheel을 받는다**. device-node 자동 감지(`detect_gpu_vendor`) 기반 "auto" 기본값으로 개선 여지가 있다(별도 건).

## 실투입 — 나머지 결정

- **해상도 캡** — `ref`/`got` 확인 후 채택/보류 확정. 넣게 되면 하드코딩 금지 규칙대로 env 기본값 + `state.json` + `/admin` 노출, 위치는 detect 다음·recognize 전([pipeline.py `detect_and_recognize`](../app/pipeline.py) 또는 [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)의 `recognize` 입력 전).
- **멀티워커** — 안 넣음(위 §동시성의 재고 조건).

## 참고 — 4070 Ti Super 대비

flash-ON 2.9s/crop 대 4070 Ti Super ~1s/crop ≈ **2.9배**. CUDA는 SDPA가 기본으로 flash를 타므로 이제 **양쪽 다 flash-ON의 공정 비교**다. 대역폭 672 vs ~320 GB/s(2.1x, decode가 대역폭 바운드) + 상위 급의 compute 우위(vision prefill)로 설명되는 범위이고, 남는 잔차가 없다. 단 "~1s/crop"은 다른 이미지 기준의 대략치라, 엄밀히는 4070 박스에서 같은 챕터로 `--diag`를 돌려 비교해야 한다.

## 관련

- 배치 축(단일 forward에 N크롭)은 [recognize-crop-batching.md](recognize-crop-batching.md) — 양쪽 엔진 다 기각.
- CPU 멀티워커(manga-ocr)는 [recognize-cpu-threads.md](recognize-cpu-threads.md) — 8w×1t 1.88x.
- 동시성·`gpu_lock`·translate 배치 그림은 아티팩트 [동시성과 번역 배치](https://claude.ai/code/artifact/543ff4c0-d2be-4d4f-9d70-fc35fac17c1f).

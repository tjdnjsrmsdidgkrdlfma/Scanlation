# recognize가 왜 느린가 — decode는 호스트 오버헤드 바운드, 답은 런타임 교체(llama.cpp)

작성 2026-07-25. 측정 도구: [bench_recognize_gpu_concurrency.py](bench_recognize_gpu_concurrency.py) `--profile-decode`, [bench_recognize_llamacpp.py](bench_recognize_llamacpp.py). flash·cap·멀티워커를 다 짜낸([recognize-gpu-speed.md](recognize-gpu-speed.md)) 뒤 "그래도 recognize가 느리다 / per-crop을 더 내릴 레버가 뭐냐"를 끝까지 판 기록.

**결론 세 줄:**
1. cap을 켠 스택에서 per-crop은 **~95%가 decode**이고, 그 decode는 대역폭도 연산도 아닌 **호스트 측 오버헤드**다 — steady/token이 crop 크기와 무관하게 flat ~64ms인데 그중 weight read 4% + compute <1%. **카드 대역폭의 5%만 쓰고 있었다**(§3).
2. 그래서 **torch 안의 레버는 전부 죽는다** — 양자화도, MI50도, `torch.compile`도(§1·§4).
3. 답은 튜닝이 아니라 **런타임 교체**였다. 같은 모델을 llama.cpp가 서빙하면 **per-crop 10x·VRAM 1/4**이다(0.557 → 5.585 crops/sec, §5). 단 그 10배는 **GPU 핀을 env로 걸 때만** 나온다 — `--device`로 걸면 vision 인코더가 GPU를 못 타고 4.4배를 잃는다(§6).

## 전체 변천사 — PaddleOCR-VL-For-Manga는 어디서 어디까지 왔나

두 문서에 흩어진 단계들을 한 줄에 세운 것. 같은 42 crop 기준.

| # | 단계 | crops/sec | 직전 | 누적 | 출처 |
|---|---|---|---|---|---|
| 0 | **CPU** | ~0.017 | — | **1x** | 플러그인 docstring `~60s/crop` |
| 1 | GPU, flash OFF | 0.094 | ~5.6x | ~5.6x | [recognize-gpu-speed.md](recognize-gpu-speed.md) |
| 2 | **+ AOTriton flash attention** | 0.345 | **3.7x** | ~21x | 〃 — env 한 줄. sdpa가 math 폴백 대신 flash 커널을 탄다 |
| 3 | + 해상도 캡 150k / `pow2` | 0.58 | 1.7x | ~35x | 〃 — vision 토큰 감소 |
| 4 | + 멀티워커 W4·K2 | ~0.77 | 1.3x | ~46x | 〃 — 호스트 오버헤드가 만든 GPU 유휴를 다른 워커가 채운다 |
| 5 | **llama.cpp로 런타임 교체** | 1.274 | 1.7x | ~76x | §5 — decode의 ~95%인 호스트 오버헤드를 계층째 제거 |
| 6 | **+ GPU 핀을 env로** | **5.585** | **4.4x** | **~340x** | §6 — vision 인코더가 옆 GPU로 새던 것 |
| 7 | 현재 — `box` / 300k | 4.277 | 0.77x | **~260x** | §7 — **일부러 되돌린 단계.** 절단되던 크롭을 산다 |

> **누적에 `~`가 붙은 이유**: 0번의 `~60s/crop`은 잰 값이 아니라 어림이라 유효숫자가 없다. 배수를 정밀하게 인용해야 하면 실측된 첫 점인 **1번을 기준으로 쓴다 — 6번 59x, 현재 46x.** 크기만 말하면 **크롭 하나에 1분 → 1초에 4개**다.

**가장 큰 두 도약(2·6번)은 튜닝이 아니라 잘못된 설정을 고친 것이다.** 캡·멀티워커·양자화·`torch.compile`을 다 짜낸 3·4번의 합이 2.2배인데, 런타임 교체+핀(5·6번)이 7.3배다. 그리고 둘 다 **의도한 실험이 아니라 사고를 수습하다** 나왔다 — 6번은 `InaccessiblePaths`로 서비스를 망가뜨려 로그를 뒤지다 `prompt eval 739ms`가 눈에 들어온 게 계기다(§9).

## 결론 먼저

| 레버 | 판정 | 근거 |
|---|---|---|
| **llama.cpp로 서빙** | ✅ **채택 — 10x, VRAM 1/4** (§5) | Python/torch 디스패치 계층이 없어 오버헤드를 애초에 안 문다. 정확도는 실질 동등 |
| **GPU 핀을 `GGML_VK_VISIBLE_DEVICES`로** | ✅ **필수 — 이걸 놓치면 4.4x를 잃는다** (§6) | `--device`는 LM만 GPU에 올리고 vision 인코더를 남긴다. 같은 229 토큰이 739ms → 93ms |
| **`downscale_mode=box` + cap 300k** | ✅ **채택 — 절단 하나를 페이지당 105ms(+4.7%)에 산다** (§7) | `pow2`는 배율이 2의 거듭제곱뿐이라 예산의 최대 4배를 버린다. 302k 크롭이 150k 캡에서 **75k로** 접혀 4줄이 1줄로 잘렸다 |
| **동시성 W4·K2** | 🔸 transformers 한정 채택(1.5x) | llama.cpp 쪽 동시성은 잘못된 GPU 핀 상태에서만 재봤다 — 현 구성 기준 미측정(§6) |
| **양자화 (weight-only INT8/INT4)** | ❌ 폐기(transformers 기준) | decode가 읽는 건 LM(893MB)뿐이라 weight read가 토큰당 ~2.8ms = **전체의 ~4%**. 다 없애도 3%다 |
| **MI50로 recognize** | ❌ 폐기 | torch rocm7.0 rocBLAS에 **gfx906 Tensile 라이브러리가 없음** — 로드는 되지만 첫 matmul에서 죽는다. (llama.cpp는 gfx906에서 도니 **이 판정은 torch 한정**이다) |
| **decode 오버헤드 제거** (`torch.compile` / HIP graph) | ❌ **실측 폐기(이 스택)** — §4 | 벽 다 넘어도 inductor+dynamic는 **1.11x뿐** + **출력 2/8 오독**(っ→コ·♥→✓) + ~11s/shape라 dynamic-res서 상각 불가. 큰 이득(그래프 캡처)은 동적 shape과 원천 충돌 |
| vLLM / SGLang / FastDeploy / ONNX / OpenVINO | ❌ 조사 단계에서 배제 | RDNA4에서 vLLM은 Docker 기동 버그 미해결 + 최적화 커널 부재(커뮤니티 실측: **llama.cpp Vulkan이 vLLM ROCm보다 29% 빠름**), FastDeploy는 AMD 미지원, ONNX는 export 경로 자체가 없음(optimum "not planned"), OpenVINO는 Intel 전용 |
| **manga-ocr로 전환/하이브리드** | 🔸 보류 | 정확도 트레이드 대안. llama.cpp가 정확도 손실 없이 10x를 줬으므로 **당장은 불필요** |

## 측정 셋업

- **하드웨어**: 서버 2-GPU — 9060 XT(gfx1200/RDNA4, 16GB, HIP index **1**) + MI50(gfx906/Vega20, 32GB, HIP index **0**). `HIP_VISIBLE_DEVICES=0,1`.
- **스택**: torch `2.10.0+rocm7.0`, AOTriton flash on(`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`). translate는 llama.cpp(host:8080) 별도 프로세스 → recognize GPU와 분리.
- **역할 배치**: recognize = 9060 XT(GPU 1), translate = MI50를 쓰는 llama.cpp.
- **입력**: 실제 챕터 21페이지 → detect+deskew 42 crops(26개가 150k로 다운스케일), cap 150k / `pow2`.
- **명령**(컨테이너 안. 호스트엔 torch가 없고, 미디어는 컨테이너에 안 보여 `docker cp`로 넣어 실행):
  ```bash
  docker cp .../tools scanlation-server:/tmp/tools
  docker cp <chapter_dir> scanlation-server:/data/benchpages
  docker exec -e HIP_VISIBLE_DEVICES=1 -e BENCH_DATA=/data/benchpages scanlation-server \
    python /tmp/tools/bench_recognize_gpu_concurrency.py \
      --profile-decode --detect --max-pixels 150000 --downscale-mode pow2
  ```

## 1. MI50(gfx906)는 torch recognize에 못 쓴다

`HIP_VISIBLE_DEVICES=0`(MI50)로 recognize 한 번 돌리면 모델은 로드되지만 **첫 matmul에서 rocBLAS가 죽는다**:

```
model loaded OK; now recognizing 2400 1800
rocBLAS error: Cannot read /plugins/torch/lib/rocblas/library/TensileLibrary.dat:
  Illegal seek for GPU arch : gfx906
 List of available TensileLibrary Files :
  ... gfx1030 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201 gfx908 gfx90a gfx942 gfx950
```

torch rocm7.0의 rocBLAS가 **gfx906 Tensile 라이브러리를 안 담고** 있다(목록에 gfx1200(9060XT)은 있고 gfx906만 빠짐). 로드가 되는 건 device 열거·가중치 전송까지고, 실제 GEMM에서 아키텍처별 커널을 못 찾아 abort한다. `pick_device`가 조용히 CPU 폴백하는 것과 달리 이건 **하드 크래시**이고, 프로파일 코드가 `generate`를 `silenced()`(fd→devnull) 안에서 돌려 처음엔 "헤더만 찍고 프롬프트 복귀"로 보였다 — silencing을 뺀 단발 recognize로 표출시킨 것이 위 로그.

**함의**:
- **recognize는 9060 XT 전용**, MI50는 **llama.cpp translate 전용**(llama.cpp는 자체 ROCm/Vulkan 백엔드라 torch rocBLAS와 무관하게 gfx906을 씀).
- `HSA_OVERRIDE_GFX_VERSION`으로 gfx908 등으로 스푸핑하는 건 **비권장** — Vega20(GCN)과 CDNA(gfx908)는 ISA가 달라 오작동/오답 위험.
- 그래서 "recognize를 MI50에 얹어 대역폭을 벌자"는 애초에 실행 불가. (설령 됐어도 §3에서 대역폭 바운드가 아님이 드러나 이득도 없었다.)

## 2. cap을 켜면 prefill은 무시할 수준, decode가 지배

9060 XT(GPU 1), cap 150k/`pow2`, 스텝별 wall time:

| crop px | tokens | prefill+t1 ms | steady/token ms | first steps ms |
|---|---|---|---|---|
| 197x753 (148k) | 18 | **1630** \* | 61 | 1630 66 61 61 61 62 61 62 |
| 855x172 (147k) | 34 | 74 | 68 | 74 68 67 68 68 67 68 68 |
| 258x518 (133k) | 9 | 70 | 64 | 70 64 64 64 64 64 64 65 |
| 102x316 (32k) | 9 | 68 | 64 | 68 64 64 64 64 64 64 65 |

\* 첫 크롭의 1630ms는 **프로세스 첫 forward의 1회성 커널 JIT**이다 — 같은 147k인 두 번째가 74ms인 게 증거. long-running 서버에선 부팅 후 1회성이라 per-crop 비용이 아니다.

1. **prefill(warm)은 ~70ms**로 무시할 수준. flash + cap이 vision attention을 이미 바닥으로 내렸다(문서 [recognize-gpu-speed.md](recognize-gpu-speed.md)의 "150k 아래에선 vision이 바닥, decode가 지배"를 스텝 단위로 재확인).
2. **decode가 압도적으로 지배** — per-crop ≈ prefill + tokens × ~64ms. 18토큰 crop ≈ 1.2s, 34토큰 ≈ 2.3s로 프로덕션 로그의 per-crop과 일치.
3. **steady/token이 크롭 크기(32k~148k)와 무관하게 ~64ms로 완전히 FLAT.**

## 3. decode는 대역폭도 연산도 아니라 호스트 오버헤드 바운드

steady/token이 크롭 크기에 **안 늘어난다**는 건 vision-attention(KV 캐시 읽기)이 병목이 아니라는 결정적 증거다(그랬다면 큰 crop = 더 많은 vision 토큰 = 토큰당 느려야 함). 그럼 ~64ms를 분해해 보자.

B=1 디코드의 물리적 하한은 **토큰마다 LM 가중치를 한 번 읽는 것**이다. 이 모델은 LM과 vision 타워가 갈려 있고(GGUF로 893MB + 841MB), **디코드는 LM만 읽는다** — vision 타워는 prefill에서 한 번 돈다:

| 성분 | 시간 | 비중 |
|---|---|---|
| weight read (LM 893MB ÷ ~320 GB/s) | ~2.8ms | ~4% |
| compute (2×0.45B FLOP ÷ 수십 TFLOPS) | ~0.1ms | <1% |
| **잔차 = 호스트 측 오버헤드** | **~61ms** | **~95%** |

0.45B LM이 **~15.6 tok/s**면 이 하드웨어 능력 대비 비정상적으로 느리다 — GPU가 바빠서가 아니라 **놀아서** 느린 것이고, B=1 eager decode가 스텝마다 작은 커널 수백 개를 던지는 전형적 그림이다.

**대역폭 지붕에 대보면 한 줄로 끝난다** (같은 카드, 같은 모델, 같은 crop):

| | 토큰당 | 실효 대역폭 | ~320 GB/s 대비 |
|---|---|---|---|
| transformers (BF16 LM 893MB) | 64ms | ~14 GB/s | **4.6%** |
| **llama.cpp (Q8_0 LM 476MB)** | **2.38ms** | **~210 GB/s** | **65%** |

transformers는 **카드 대역폭의 5%만** 쓰고 있었고, 나머지 95%는 GPU가 호스트를 기다린 시간이다. llama.cpp의 65%는 B=1 디코드가 실제로 닿는 영역이다(peak에는 원래 못 간다) — 즉 **여기서 더 짜낼 게 없다**는 뜻이기도 하다. llama.cpp 수치는 서버 자체 계측(`slot print_timing`: `eval time = 81.37 ms / 34 tokens (2.39 ms per token)`).

> **잔차의 내부 구성은 증명하지 않았다.** 문서화된 커널 런치 비용은 토큰당 50~100µs 규모라 61ms와 세 자릿수 차이다 — 즉 순수 런치뿐 아니라 per-token CPU↔GPU 동기화, HF `generate` 루프의 파이썬 오버헤드, 연산마다 무는 torch 디스패처 비용 등이 섞여 있다. **확실한 건 "대역폭도 연산도 아닌 호스트 측"이라는 것**이고, 그거면 레버를 고르는 데 충분하다(원인이 런치든 sync든 파이썬이든 처방이 같다 — 그 계층을 없애는 것). 위 65%가 이 진단의 경험적 확증이다: 그 계층을 없앴더니 남은 게 정확히 대역폭이었다.

이게 레버 랭킹을 바꾼다:
- **양자화**는 그 ~4% weight read만 줄인다 → **잘해야 3%**. 작은 모델이라 상한 자체가 낮다. **폐기.**
- **동시성**은 오히려 딱 맞는다 — 런치 사이 GPU 유휴를 다른 워커가 채운다(문서의 "W=1에서 GPU가 ~76% 바쁨, ~24% 회수"가 바로 이 오버헤드의 유휴). **무료 1.5x, 채택.**
- **런치 오버헤드 자체를 접는 것**(static KV cache + `torch.compile`, 또는 HIP graph 캡처)이 유일하게 남았던 큰 per-crop 후보 — 였으나 **프로브 실측 결과 1.11x + 출력 파손으로 폐기(§4)**. accuracy-neutral일 거란 기대와 달리 inductor가 출력을 바꿨다.

정리하면 시도는 전부 이 두 조각 중 하나를 겨눴다: **4%를 겨눈 것**(양자화)은 상한이 3%라 애초에 무의미했고, **95%를 겨눈 것**(동시성 1.5x, `torch.compile` 1.11x)은 torch 안에 있는 한 일부만 회수했다. 계층 자체를 걷어내야 전부 돌아온다는 게 §5다.

## 4. torch.compile 프로브 — 실측으로 폐기

decode가 런치 오버헤드 바운드니 `torch.compile`(런치 제거)이 이론상 정답이라, 플러그인 손대기 전 standalone 프로브([bench_recognize_compile.py](bench_recognize_compile.py))로 9060 XT(GPU 1)에서 재봤다. 셋업 벽 셋을 다 넘어 실측까지 갔고, 결과가 셋 다 폐기 사유다.

**셋업 벽(넘김):**
1. **inductor = Triton 필요, 이미지에 없음** (`TritonMissing`) → `/plugins`에 `pytorch-triton-rocm` 3.5.1 설치.
2. **Triton AMD 백엔드 = C 컴파일러 필요, slim 이미지에 없음** (`Failed to find C compiler`) → 컨테이너에 `build-essential`(gcc/g++) 설치.
3. **그래프 캡처(cudagraphs/reduce-overhead = 런치를 통째로 없애는 큰 이득)는 동적 shape과 원천 충돌** (`size of tensor a (213) must match b (214)`). static KV cache로 캐시는 고정해도 처리 시퀀스가 가변(dynamic-resolution)이라 그래프 replay가 불가 — 그래프 캡처의 본질이라 gcc로도 못 고친다. → 남는 건 동적 shape을 견디는 **inductor + `dynamic=True`(fusion)**뿐.

**실측 (inductor + dynamic, 벽 다 넘은 뒤):**

| | baseline | compiled |
|---|---|---|
| crops/sec | 0.649 | 0.721 (**1.11x**) |
| per-crop med | 1347ms | 1159ms |
| compile warm | — | 86.8s / 8 shapes (~11s/shape) |
| 출력 동일성 | — | **✗ 2/8 변함** (`っ→コ`, `♥→✓` 등 실오독) |

셋 다 폐기 사유:
- **이득이 작다 (1.11x).** fusion은 런치 *횟수*만 줄이고(그래프 캡처는 배제), mrope graph break로 그마저 쪼개져 ~95% 오버헤드의 일부만 회수.
- **정확도가 깨진다.** 8개 중 2개 출력이 바뀌고 `っ→コ`·`♥→✓` 같은 실제 오독이 낀다. inductor의 부동소수 재결합이 노이즈 플로어의 그리디 디코드를 뒤집는다 — accuracy-first 엔진엔 치명적.
- **프로덕션에서 상각 불가.** ~11s/shape 컴파일인데 dynamic-res라 crop마다 새 shape → 매 crop이 컴파일 비용을 물어 오히려 훨씬 느려진다. 프로브의 1.11x는 같은 shape을 재사용한 best case일 뿐.

→ **이 스택에서 `torch.compile`은 실측으로 폐기.** 프로브 스크립트는 남겨둔다(스택/모델이 바뀌면 재실행).

## 5. 답은 런타임 교체 — llama.cpp (채택)

torch 안의 레버가 다 막힌 뒤 "공식/커뮤니티에 우리가 안 본 게 있나"를 웹 조사해 나온 결론: **llama.cpp가 PaddleOCR-VL을 지원**하고([ggml-org/llama.cpp#18825](https://github.com/ggml-org/llama.cpp/pull/18825), build **b8110+**), **우리가 쓰는 fine-tune의 GGUF가 이미 공개**돼 있다([adambarbato/PaddleOCR-VL-For-Manga-GGUF](https://huggingface.co/adambarbato/PaddleOCR-VL-For-Manga-GGUF)). 이게 정확히 §3 진단의 처방이다 — llama.cpp엔 Python/torch 디스패치 계층이 아예 없어 그 오버헤드를 안 문다. 게다가 translate가 이미 llama.cpp라 인프라·플러그인 패턴이 이미 있었다.

**실측** ([bench_recognize_llamacpp.py](bench_recognize_llamacpp.py), 같은 42 crop·같은 cap, 9060 XT / Vulkan 빌드 `--device Vulkan2`):

| 구성 | crops/sec | per-crop med | per-crop max | VRAM |
|---|---|---|---|---|
| transformers 순차 | 0.557 | 1745ms | 5014ms | ~1.9GB |
| transformers 프로덕션(W4·K2) | ~0.77 | 4369ms | — | **7.7GB** |
| **llama.cpp (env 핀, c=1)** | **5.585 (10.0x)** | **168ms** | 305ms | **1.3GB** |

- **per-crop 10x, VRAM 4.3배 절감**(모델 사본이 워커마다가 아니라 서버에 하나), **꼬리 지연 16x**(max 5014→305ms, 편차 19배→2.6배).
- **정확도는 실질 동등.** 42개 중 24 동일 + 11 표기차(`...`↔`・・・`, ♥ 개수/♥↔♡, 줄바꿈, `?`↔`？`) = **35/42**. 나머지는 **개선 3**(`ばつかり`→`ばっかり`, `::`→`・・・` 2건 — transformers의 기존 결함을 고침) 대 **악화 4**(#4 truncation 4줄→1줄이 유일한 실손실, #6·#26 오독, #20 소소).
- 이 수치는 **서버 재시작 + `--pad-uncached`**로 캐시를 배제하고, 42개 crop의 출력 텍스트를 전부 눈으로 확인하고, llama-server 자체 계측(`total 176.9ms`)과 교차 검증한 것이다. 왜 그렇게까지 하는지는 §9.

**대가**: 모델 배포가 서버 관리자 몫이 된다(GGUF 교체·GPU 선택 = `llama-server` 커맨드라인). 그리고 **유휴 언로드가 사라진다** — llama-server는 프로세스 수명 동안 모델을 붙들어 9060 XT가 D3hot(~0W)로 못 내려가고 **상시 ~15W**를 먹는다([idle_unload](../app/idle_unload.py)는 이 engine에 대해 HTTP 클라이언트만 닫는 no-op이 된다). 회수하려면 llama-swap 류의 TTL 프록시가 필요하다(콜드스타트는 로그상 **~1.9초**라 싸다).

**구현**: `scanlation-llama-cpp` 플러그인에 engine 추가([recognizer.py](../../scanlation-llama-cpp/scanlation_llama_cpp/recognizer.py)) — translator는 무수정, transformers 경로도 `/admin`에 그대로 남아 폴백 가능. env `LLAMACPP_RECOGNIZE_ENDPOINT`(기본 `:8090`), 유닛 예시 [deploy/llama.cpp-PaddleOCR-VL-For-Manga.service.example](../../../deploy/llama.cpp-PaddleOCR-VL-For-Manga.service.example).

## 6. GPU 핀은 env로 — `--device`는 vision 인코더를 GPU에 못 올린다 (4.4x)

llama-server를 **어떻게 GPU에 핀하느냐**가 recognize 속도를 4.4배 가른다. 카드가 여럿인 호스트에서 각 서버를 자기 GPU에 묶는 방법이 둘인데, 같은 카드를 가리켜도 결과가 다르다:

| 핀 방법 | prompt eval | 처리량 |
|---|---|---|
| `--device Vulkan2` | **739 ms** / 229 tokens (3.23 ms/token) | 1.274 crops/sec |
| **`GGML_VK_VISIBLE_DEVICES=2`** | **93 ms** / 229 tokens (0.41 ms/token) | **5.585 crops/sec** |

**토큰 수가 229로 동일하다.** 전처리가 달라져 일이 준 게 아니라 **같은 일을 8배 빠르게** 한다. 원인은 기동 로그(`-lv 5`)에 그대로 찍힌다:

```
--device Vulkan2 :
  device_info:  Vulkan1 = VEGA20(MI50, 1968 MiB free)  Vulkan2 = GFX1200(9060 XT, 16246 MiB free)
  [mtmd] adding 927.03 MiB to fit_params_target for device Vulkan1   ← vision 인코더가 MI50로
  llama_prepare_model_devices: using device Vulkan2                  ← LM만 9060 XT로
GGML_VK_VISIBLE_DEVICES=2 :
  device_info:  Vulkan0 = GFX1200 (하나뿐)
  [mtmd] adding 927.03 MiB to fit_params_target for device Vulkan0   ← vision도 9060 XT로
```

**`--device`는 LM 레이어만 제한한다.** vision 인코더(mtmd, 927 MiB)는 그 제한을 받지 않고 **여전히 보이는 목록에서 스스로 고르며, 여기선 MI50를 골랐다** — 번역용 LLM이 올라가 여유가 1968 MiB뿐인 카드다. 그래서 crop마다 이미지 인코딩이 경합 중인 옆 GPU에서 돌고 LM decode만 9060 XT에서 돌았다. env는 다른 카드를 **아예 안 보이게** 만들어 mtmd가 고를 여지를 없앤다.

→ **유닛은 반드시 `Environment="GGML_VK_VISIBLE_DEVICES=<idx>"`로 핀한다**(예시: [deploy/](../../../deploy/)의 두 유닛). 인덱스는 `llama-server --list-devices`를 **env 없이** 돌린 목록 기준이다.

**두 방법을 섞으면 서버가 뜨지 않는다.** env가 보이는 카드를 하나로 줄이면 그 카드가 `Vulkan0`으로 **재번호**되므로, 함께 준 `--device Vulkan2`는 존재하지 않는 이름이 되어 인자 파싱에서 exit 1이다. `InaccessiblePaths=`로 다른 카드의 render 노드를 가리는 것도 **같은 재번호를 일으킨다** — env 인덱스와 함께 쓰면 엉뚱한 디바이스를 골라 CPU로 떨어진다(실측: per-crop 52초).

### 이 구성에서 아직 안 잰 것들

per-crop이 168ms로 내려오면서 §3의 비용 구조가 통째로 바뀌었다. **아래 노브들은 잘못 핀된 서버에서만 재봤으므로 현 구성 기준 미측정이다** — 다시 재기 전에는 결론을 인용하지 말 것. (해상도 캡은 재측정 완료 → §7.)

- **mmproj의 `clip.vision.image_min_pixels`(147384)** — 이 바닥이 작은 crop을 업스케일해 vision 토큰을 188로 고정한다. 낮추면 빨라지지만 작은 SFX crop의 오독이 늘었다(잘못 핀된 서버 기준).
- **동시성** — llama.cpp 쪽 c>1은 재볼 값이 있다.
- **Q8_0** — 출력이 BF16과 **42개 전부 바이트 동일**하고 VRAM을 417MB(893→476MB) 아낀다. 이건 핀과 무관한 사실이라 유효. 속도 이득은 없었다.
- **`-fa`** — on/off 차이 없음(1.240 vs 1.242).
- **PaddleOCR-VL 1.5/1.6** — 공식 GGUF가 있으나 만화 fine-tune을 잃어 정확도가 눈에 띄게 떨어진다(별도 확인). 제외.

## 7. 해상도 캡 재측정 — 범인은 캡이 아니라 `pow2`의 오버슛

핀을 고친 뒤 캡을 다시 쟀다(42 crop, 서버 재시작 후 네 구성 모두 fresh, 참조 패스 없음):

| 구성 | crops/sec | per-crop med | per-crop max |
|---|---|---|---|
| `pow2` 150k (구 기본값) | **5.526** | 171ms | 306ms |
| `box` 150k | 5.158 | 191ms | 319ms |
| **`box` 300k (채택)** | **4.277** | 226ms | **410ms** |
| 캡 off | 4.005 | 218ms | 595ms |

**캡을 푸는 게 답이 아니었다.** `pow2`는 `reduce(2)`를 반복하므로 배율이 1, ½, ¼뿐이고 넓이는 1, ¼, ¹⁄₁₆로 뚝뚝 떨어진다 — 캡을 조금만 넘겨도 한 칸 통째로 내려가 **예산의 최대 4배를 버린다.** 실제로 302k짜리 crop이 150k·200k·300k 캡에서 **전부 같은 75k**로 접혔다(그래서 캡을 올려도 출력이 한 글자도 안 바뀐다). `box`는 `scale = √(cap/넓이)`로 필요한 배율을 한 번에 계산해 캡에 딱 붙인다. 필터는 둘 다 영역 평균이라 같고, **다른 건 허용 배율뿐**이다.

같은 crop(359×840)을 네 설정으로 읽힌 결과:

| 설정 | 보낸 크기 | 출력 |
|---|---|---|
| `pow2` 150k | 180×420 (75k) | `お...ほぉ...` — **4줄이 1줄로 절단** |
| `box` 150k | 253×592 (149k) | 4줄 전부, 단 `ほ`→`ぼ` 탁점 오독 |
| **`box` 300k** | 358×837 (300k) | 4줄 전부, 정확 |
| 캡 off | 359×840 (302k) | 4줄 전부, 작은 `ぅ`를 큰 `う`로 |

**`box` 300k가 캡 off보다 낫다** — 텍스트가 더 정확하고, 빠르고(4.277 vs 4.005), **꼬리가 훨씬 낫다**(max 410 vs 595ms: 캡이 남아 있어 거대 crop이 폭주하지 않는다). 캡 off는 선택지에서 빠진다.

**대가는 파이프라인 기준 +4.7%다.** 두 설정을 각각 **양 llama-server 재시작 직후**(캐시 없음)에 21장 챕터로 돌린 A/B:

| 중앙값 | `pow2` 150k | `box` 300k | 차이 |
|---|---|---|---|
| detect | 301.0ms | 302.1ms | — |
| **recognize** | 397.8ms | **502.8ms** | **+105ms (1.26x)** |
| translate | 1298.0ms | 1293.8ms | — |
| 페이지 total | 2008.4ms | 2101.6ms | **+4.7%** |

**대가는 recognize 한 스테이지에만 붙는다.** 인식이 살아나 텍스트가 늘면 번역도 무거워질 거라 예상했지만(#4가 1줄 → 4줄), translate는 4ms 안에서 같다. 크롭 하나가 통째로 잘리는 걸 페이지당 105ms에 산다.

> 나머지 41개 crop은 캡으로 갈리지 않았다 — 원본이 이미 캡 아래라 어느 설정에서도 손을 안 탄다. 갈린 것들은 작가가 폰트 없이 손으로 쓴 의성어라 인식 문제가 아니라 **detect 단계에서 걸러야 할 대상**이다.

**설정**: `/admin` recognizer 옵션 = `downscale_mode: box`, `max_pixels: 300000`. 역할별 옵션 저장이 되기 전에는 이걸 `/admin`에서 만질 수 없었다(llama.cpp가 translator와 블롭 하나를 공유해 recognizer 필드가 아예 안 떴다).

## 8. 파이프라인에서의 결과 — 이제 translate 바운드

recognize 교체의 값은 결국 파이프라인에서 재야 한다. 같은 21장 챕터, `run_report.py` serial(스테이지별 시간이 깨끗하게 나온다), 채택 설정(`box` 300k), **양 llama-server 재시작 직후**:

| 스테이지 | 중앙값 | 비중 |
|---|---|---|
| decode | 11.9ms | 0.6% |
| detect | 302.1ms | 14.4% |
| recognize | 502.8ms | 23.9% |
| **translate** | **1293.8ms** | **61.6%** |
| lockwait / semwait | 0 / 0 | 서버 측 직렬화 없음 |
| **페이지 total** | **2101.6ms** | serial 0.43 p/s · **동시성 2에서 21장 32.8초(0.64 p/s)** |

| 시점 | recognize 엔진 | recognize 비중 |
|---|---|---|
| 2026-07-12 | PaddleOCR-VL (transformers) | **64.7%** |
| 2026-07-16 | manga-ocr (CPU) | ~26% |
| **2026-07-26** | **PaddleOCR-VL (llama.cpp)** | **23.9%** |

**병목이 recognize에서 translate로 넘어갔다.** recognize는 전체의 1/4이 채 안 되고, 그 자리를 translate가 62%로 대신한다. **다음 레버는 translate이고, 그 다음은 detect**(14.4%, CPU RT-DETR)다.

> **비교할 때 캐시를 맞출 것.** 이 절을 처음 쓸 때는 llama-server가 데워진 런을 썼고, 그래서 translate가 1082ms·57%로 나왔다. 서버를 재시작하고 다시 재니 1294ms·62%다 — 결론(translate 바운드)은 같지만 숫자가 20% 틀렸다. §9 참조.

## 9. ⚠️ 방법론 — 측정이 다섯 번 틀렸다

### 잘못된 기준선을 한 번도 의심하지 않았다

§6의 4.4x는 **튜닝으로 얻은 게 아니라 잘못 핀된 서버를 고쳐서 나온 것**이다. 그 잘못된 구성 위에서 캡·`image_min_pixels`·양자화·동시성을 며칠에 걸쳐 재고, "vision이 전체의 89%"라는 진단까지 세웠다. 전부 vision 인코더가 GPU 밖에 있어서 생긴 그림이었다.

들킨 계기는 튜닝이 아니라 **사고**였다: `InaccessiblePaths`로 서비스를 망가뜨려 로그를 뒤지다 `prompt eval`의 절대값이 눈에 들어왔다. 0.9B 모델의 vision 인코딩이 229 토큰에 739ms(310 t/s)라는 건 그 자체로 이상한 값인데, **비교 대상이 없어서** 오래 정상으로 통했다.

> **교훈**: 배수(A/B)만 보지 말고 **절대값이 하드웨어에 비해 말이 되는지**를 따로 확인한다. 이 문서의 §3이 `~15 tok/s면 하드웨어 대비 비정상적으로 느리다`로 transformers의 병목을 잡아낸 것과 같은 감각을, llama.cpp 쪽에는 적용하지 않았다.

### llama-server 프롬프트 캐시가 측정을 세 번 망쳤다

벤치는 **같은 crop을 매번 다시 보낸다.** llama-server는 이미 본 프롬프트(이미지 포함)를 슬롯 캐시에서 돌려주므로, ~740ms짜리 vision prefill이 **~100ms로** 끝난다. 그 결과:

| 잘못 나온 값 | 실제 | 어떻게 들켰나 |
|---|---|---|
| 동시성 c=4 **11.047** crops/sec | 1.32 | 재현 불가 → A/B로 폐기 |
| box150k **1.920** | 1.250 | per-crop 37ms가 물리적으로 불가능(188토큰 × 3.2ms = 600ms+) |
| 4연속 스윕 **8.7~9.6** | 1.2~1.9 | med 89~114ms |

**막지 못한 것들:**
- 요청의 `cache_prompt: false` — **이 빌드가 무시한다**(실측).
- 사후 이상치 탐지 — **전부 캐시되면 중앙값이 같이 내려가 아무 이상치도 안 남는다.** 실제로 8.7x 런에서 경고가 침묵했다.

**해법은 캐시를 애초에 못 맞추게 하는 것**: `--pad-uncached`가 crop마다 몇 px(런마다 다른 값)을 덧대 모든 요청을 새것으로 만든다. 비용은 패치 한 열(~190개 중 1개). 정확도 런에는 쓰지 않는다(입력을 건드리므로).

> **교훈**: 속도만 보고 나머지를 확인하지 않은 게 공통 원인이다. 위 셋 중 첫 번째는 **출력을 아예 안 봤고**(`--no-reference`), 나머지 둘은 **per-crop 분포를 안 봤다.** 집계값 하나만 보면 안 된다.

### 스윕 둘을 가로질러 비교했다

§7의 캡 판정을 처음엔 **다른 두 런의 수치를 나란히 놓고** 냈다. 두 런은 조건이 달랐다:

| 런 | 캐시 | 같은 GPU 위의 다른 프로세스 |
|---|---|---|
| 1차(캡 스윕) | 전부 fresh | **transformers 참조 패스**가 PaddleOCR-VL을 9060 XT에 올린 채 |
| 2차(box 스윕) | `pow2_150k`·`nocap`은 1차에서 **캐시됨** | 없음 |

그래서 같은 `pow2_150k`가 4.689와 5.592로 나왔다 — 한쪽은 옆에 모델이 떠서 느렸고, 한쪽은 캐시로 빨랐다. 배수를 여기서 뽑으면 어느 쪽으로든 편향된다. **서버를 재시작하고 네 구성을 한 런에서 다시 재서** §7 표를 만들었다(결론은 유지, 대가는 5.7% → 7%로 정정).

> **교훈**: 런 하나 안에서 비교한다. 런을 넘겨 비교해야 하면 **캐시 상태와 같은 GPU를 쓰는 다른 프로세스**를 둘 다 맞춘다. 참조 패스(transformers)는 그 자체가 GPU 점유자라 llama.cpp 패스의 조건을 바꾼다.

## 남은 일

- **translate** — 이제 파이프라인의 **62%**다(§8). 다음 레버는 여기다. 2026-07-16 동시성 스윕의 **열 상한(동시성 4 = junction 100°C)은 그대로다** — 그때도 recognize는 CPU였고 MI50는 translate 전용이었으니 이번 변경은 MI50 부하를 안 건드린다. 바뀐 건 **공급**이다: 그 스윕이 "다음 병목은 CPU recognize 직렬화(lockwait 0→172→996ms)"로 끝났는데, 지금은 **lockwait이 0**이라 translate 슬롯이 처음으로 제대로 채워진다. 같은 동시성에서 더 나올 여지가 여기 있다.
- **detect** — 14.4%다(§8). CPU RT-DETR 302ms가 통째로 사라질 수 있으니, translate 다음으로는 [TODO](../../../README.md)의 detector 풀링·GPU detect다.
- **남은 전처리 노브** — §6의 미측정 목록 중 `image_min_pixels`·동시성(캡은 §7에서 완료).
- **llama-swap** — 유휴 언로드가 사라져 9060 XT가 상시 ~15W다(§5 대가). TTL 프록시로 회수 가능, 콜드스타트 ~1.9초.
- ~~**플러그인 재설치 경로**~~ — 해결. `/install_plugins/`·`/install_plugin_stream/`이 `force`를 받고, `/admin` 플러그인 탭의 **재설치** 버튼이 그걸 쓴다. 설치 여부 판정은 "import 되는가"라 **코드가 최신인지는 안 본다** — 서버 이미지를 다시 빌드해도 `/plugins` 볼륨의 플러그인은 그대로이므로, 플러그인 코드를 고쳤으면 이 버튼을 눌러야 반영된다.

## 관련

- flash(3.7x)·cap(1.66x)·멀티워커(W4·K2 1.5x) 전사(前史)는 [recognize-gpu-speed.md](recognize-gpu-speed.md).
- 배치(단일 forward에 N크롭)는 [recognize-crop-batching.md](recognize-crop-batching.md) — per-crop보다 느려 폐기.
- CPU 멀티워커(manga-ocr)는 [recognize-cpu-threads.md](recognize-cpu-threads.md).

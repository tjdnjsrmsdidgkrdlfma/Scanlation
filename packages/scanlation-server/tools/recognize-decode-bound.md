# recognize가 왜 느린가 — decode는 "런치 오버헤드" 바운드 (그리고 MI50는 torch recognize 불가)

작성 2026-07-25. 측정 도구: [bench_recognize_gpu_concurrency.py](bench_recognize_gpu_concurrency.py) `--profile-decode`. flash·cap·멀티워커를 다 짜낸([recognize-gpu-speed.md](recognize-gpu-speed.md)) 뒤 "그래도 recognize가 느리다 / per-crop을 더 내릴 레버가 뭐냐"를 끝까지 판 기록. **두 개의 확정 결론**: (1) MI50(gfx906)은 torch recognize에 못 쓴다, (2) cap을 켠 프로덕션 스택에서 recognize decode는 대역폭도 연산도 아니라 **커널 런치 오버헤드 바운드**다 — 이게 양자화·MI50 아이디어를 둘 다 죽이고, 남은 레버를 `torch.compile`/static cache로 좁힌다.

## 결론 먼저

| 레버 | 판정 | 근거 |
|---|---|---|
| **양자화 (weight-only INT8/INT4)** | ❌ 폐기 | decode의 weight read가 토큰당 ~5.6ms로 **전체의 ~9%**뿐. 작은 모델(0.9B)이라 줄일 여지 자체가 작다 |
| **MI50로 recognize** | ❌ 폐기 | torch rocm7.0 rocBLAS에 **gfx906 Tensile 라이브러리가 없음** — 로드는 되지만 첫 matmul에서 죽는다 |
| **동시성 W4·K2** | ✅ 채택(무료 1.5x) | 오버헤드 바운드 decode는 런치 사이 GPU가 놀아, 워커가 그 유휴를 채우는 게 정확히 이 상황 ([recognize-gpu-speed.md](recognize-gpu-speed.md) §동시성) |
| **decode 오버헤드 제거** (`torch.compile` / HIP graph) | ❌ **폐기(이 스택)** — 프로브로 3중 벽 확인(§4) | 런치 오버헤드는 회수 대상이 맞지만, 큰 이득(그래프 캡처)이 **동적 shape과 근본 충돌** + inductor는 Triton·C컴파일러 부재로 실패 |
| **manga-ocr로 전환/하이브리드** | 🔸 정확도 트레이드 대안 | 더 가벼운 recognizer. 단 오버헤드 바운드면 이득 메커니즘이 달라 자체 벤치 필요 |

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

## 3. decode는 대역폭도 연산도 아니라 "런치 오버헤드" 바운드

steady/token이 크롭 크기에 **안 늘어난다**는 건 vision-attention(KV 캐시 읽기)이 병목이 아니라는 결정적 증거다(그랬다면 큰 crop = 더 많은 vision 토큰 = 토큰당 느려야 함). 그럼 ~64ms의 정체를 분해하면:

| 성분 | 시간 | 비중 |
|---|---|---|
| weight read (1.8GB ÷ ~320 GB/s) | ~5.6ms | ~9% |
| compute (2×0.9B FLOP ÷ 수십 TFLOPS) | ~0.2ms | <1% |
| **커널 런치 / 디스패치 오버헤드** | **~58ms** | **~90%** |

0.9B 모델이 **~15.6 tok/s**면 이 하드웨어 능력 대비 비정상적으로 느리다 — B=1 eager decode가 스텝마다 수많은 작은 커널을 launch하고 그 런치 레이턴시·파이썬 루프가 지배하는 전형적 그림이다. **대역폭 바운드(내가 처음 말한 것)도, 연산 바운드도 아니고, 레이턴시/런치 오버헤드 바운드가 정답.**

이게 레버 랭킹을 바꾼다:
- **양자화**는 그 ~9% weight read만 줄인다 → 잘해야 5~7%. 작은 모델이라 상한 자체가 낮다. **폐기.**
- **동시성**은 오히려 딱 맞는다 — 런치 사이 GPU 유휴를 다른 워커가 채운다(문서의 "W=1에서 GPU가 ~76% 바쁨, ~24% 회수"가 바로 이 오버헤드의 유휴). **무료 1.5x, 채택.**
- **런치 오버헤드 자체를 접는 것**(static KV cache + `torch.compile`, 또는 HIP graph 캡처)이 정확도를 안 버리는 유일한 큰 per-crop 레버 — 였으나 **프로브 결과 이 스택에선 막혔다(§4)**.

## 4. torch.compile 프로브 — 3중 벽으로 폐기

decode가 런치 오버헤드 바운드니 `torch.compile`(런치 제거)이 이론상 정답이라, 플러그인 손대기 전 standalone 프로브([bench_recognize_compile.py](bench_recognize_compile.py))로 9060 XT(GPU 1)에서 재봤다. baseline은 **0.651 crops/sec / per-crop med 1343ms**(flash on, 정상). compile은 세 겹의 벽에 막혔다:

1. **inductor 백엔드 = Triton 필요, 이미지에 없음.** `torch._inductor.exc.TritonMissing`. → `/plugins`에 `pytorch-triton-rocm`(3.5.1) 설치로 넘김.
2. **Triton AMD 백엔드 = C 컴파일러 필요, slim 런타임 이미지에 없음.** Triton이 HIP util 모듈을 런타임에 빌드하는데 `RuntimeError: Failed to find C compiler`. → gcc를 이미지에 넣어야 넘어감(안 함).
3. **큰 이득(그래프 캡처)은 동적 shape과 근본 충돌.** `--backend cudagraphs`는 Triton·컴파일러 없이 HIP 그래프를 캡처하는데, 이 모델 forward의 처리 시퀀스 길이가 스텝마다 바뀌어(`size of tensor a (213) must match b (214)`) 캡처 그래프에 다음 호출을 못 흘려넣고 죽는다. static KV cache로 캐시는 고정해도 **처리 시퀀스가 가변**(dynamic-resolution)이라 replay가 불가능하다 — 설정이 아니라 그래프 캡처의 본질이라 못 고친다.

즉 런치 오버헤드를 통째로 없애는 유일한 길(그래프 캡처)이 dynamic-res와 원천 충돌하고, 남는 inductor-fusion 경로(부분 이득)마저 Triton + C컴파일러 두 겹의 이미지 부채를 요구하는 데다 mrope graph break로 fusion 이득이 쪼개질 리스크까지 있어, **이 스택에서 `torch.compile`은 값을 못 한다.** 프로브 스크립트는 이미지에 컴파일러+Triton이 생기면 재실행할 수 있게 남겨둔다.

## 남은 일

per-crop 레버(양자화·MI50·`torch.compile`)가 전부 이 스택에서 막혔다. per-crop은 현재 바닥(~1.34s med, flash+cap 적용)이고, "여러 장 동시"의 급성 문제는 동시성(W4·K2)으로 이미 풀렸다. 남은 선택지는 둘:

- **현상 유지** — 정확도(PaddleOCR-VL 88%, no weak category)를 지키고 per-crop 바닥을 받아들인다.
- **manga-ocr로 전환/하이브리드** — 훨씬 가벼운 recognizer로 per-crop을 크게 줄이되 정확도를 트레이드. 채택 전 [tools/compare](compare/)로 정확도, 자체 벤치로 속도를 실측.

## 관련

- flash(3.7x)·cap(1.66x)·멀티워커(W4·K2 1.5x) 전사(前史)는 [recognize-gpu-speed.md](recognize-gpu-speed.md).
- 배치(단일 forward에 N크롭)는 [recognize-crop-batching.md](recognize-crop-batching.md) — per-crop보다 느려 폐기.
- CPU 멀티워커(manga-ocr)는 [recognize-cpu-threads.md](recognize-cpu-threads.md).

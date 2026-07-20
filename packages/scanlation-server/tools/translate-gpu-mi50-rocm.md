# translate GPU — MI50에서 ROCm 재도전 경로 (gfx906 커널 부활)

조사 2026-07-20. [translate-gpu-mi50.md](translate-gpu-mi50.md)에서 **gfx906(MI50)은 ollama/HIP 모두 막혀 Vulkan 채택**으로 결론냈는데, 이후 웹 리서치로 **커뮤니티가 gfx906+ROCm을 실제로 풀었음**을 확인했다. 이 문서는 그 재도전 경로·근거·리스크·실험 계획을 모아둔 **참고 기록**이다. 배경·측정·복구 런북은 [translate-gpu-mi50.md](translate-gpu-mi50.md), recognize 쪽 GPU 사정은 [recognize-gpu-speed.md](recognize-gpu-speed.md).

> **상태 (2026-07-20): 조사만 완료, 실험 미착수.** 현재 진짜 블로커는 **냉각**(패시브 카드 과열 → amdgpu hang, 백엔드 무관)이라 지금은 ROCm 실험을 하지 않는다. **냉각 보강이 선행 조건.** 아래는 냉각 정리 후 착수할 재도전 런북의 밑그림이다. 확정 사실이 아니라 **웹 근거 + 우리 관측의 대조**이며, 미확인 항목은 그때그때 명시했다.

## TL;DR

- **"gfx906 = ROCm 불가"는 과한 결론이었다.** AMD가 **ROCm 6.3 이후 gfx906 커널을 드롭**했을 뿐이고, 커뮤니티는 그 커널을 **되공급**해서 돌린다.
- **우리가 겪은 "모델 로드 성공 → warmup segfault"는 알려진 버그일 가능성이 높다: `SOLVE_TRI` 커널을 ROCm 7.1 HIP 컴파일러가 gfx906에서 잘못된 기계어로 컴파일.** 모델 초기화 시에만 터지고, **원라인 패치**(해당 op를 CPU 폴백)로 우회. decode 성능 영향 0. (우리 로그와 대조 검증은 미실시 — §5.)
- **턴키 옵션 존재**: [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)이 gfx906용 **프리빌드 Docker 이미지**(llama.cpp/vLLM/PyTorch/ComfyUI, ROCm 6.3.3~7.2.4)를 활발히 유지 중.
- **우리 모델은 MoE**(`gemma-4-26B-A4B`)라 ROCm이 실이득일 수 있다: 벤치들이 **MoE·긴 컨텍스트·prompt processing에서 ROCm 우세, 짧은 dense decode에선 Vulkan 우세**라고 일관되게 말한다. 우리 약점이 정확히 **prefill**(47.3 vs decode 89.88)이고, 그게 **배칭 스케일 천장**의 주범이라(§translate-gpu-mi50의 동시성 스윕) 노릴 값이 크다.
- **순서**: 냉각 보강 → 콜드 부팅 → ROCm 빌드/이미지 확보 → warmup 통과 확인 → prefill·배칭 위주 A/B([bench_translate_concurrency.py](bench_translate_concurrency.py) 재사용).

## 1. 무엇이 막혔던 건가 — 커널 드롭, 그리고 되공급

우리 doc의 벽은 "ggml의 HIP 커널 fatbin에 gfx906 코드가 없다"(`invalid device function`)였다. 커뮤니티가 확인한 근본 원인과 해법:

- **AMD가 ROCm 6.3을 마지막으로 gfx906 사전컴파일 커널(rocBLAS/Tensile)을 드롭**했다. 신형 ROCm의 러너 번들엔 gfx906 코드가 없다 → ollama의 `invalid device function`이 이 증상.
- **되공급 레시피**: 신형 ROCm 런타임(7.x) 위에 **ROCm 6.3의 Tensile 커널(gfx906 rocBLAS 파일들, ~156개)** 을 얹고 `HSA_OVERRIDE_GFX_VERSION=9.0.6`(gfx906의 자기 버전)으로 인식시킨다. — [MTLoser/ollama-mi50-rocm71-build](https://github.com/MTLoser/ollama-mi50-rocm71-build)가 이 방식(ROCm 7.1 런타임 + ROCm 6.3 Tensile).

> **우리 호스트에 대한 중요한 단서.** [translate-gpu-mi50.md](translate-gpu-mi50.md)는 **우리 rocBLAS 7.1.1이 이미 gfx906 커널을 싣고 있음**을 확인했다(`/usr/lib64/rocblas/library/`에 `*_gfx906.*`). 즉 **우리 경우엔 Tensile 되공급 단계가 불필요할 수 있다** — rocBLAS 쪽 커널은 이미 있고, 남은 벽은 **ggml 자신의 HIP 커널**(§2)뿐이다. 이게 재도전 비용을 크게 낮춘다.

## 2. 우리 warmup segfault = `SOLVE_TRI` (유력 가설)

이번 조사의 최대 소득. MTLoser 빌드가 적용하는 패치 설명:

> ROCm 7.1의 HIP 컴파일러가 **gfx906에서 `SOLVE_TRI` 커널을 잘못된 GPU 기계어로 컴파일**한다. **모델 초기화 시에만** 발생하며, 토큰 생성 성능 영향은 0.

우리 관측과 정렬:

| 우리 관측 (translate-gpu-mi50.md §HIP 네이티브) | SOLVE_TRI 버그 |
|---|---|
| 모델이 MI50에 **완전히 로드**(`offloaded 43/43 layers`, KV on ROCm0) | 로드는 통과 |
| **그 직후 warmup에서 segfault** (`-fa off`로도 동일) | **모델 초기화 시에만** 터짐 |
| 호스트 **ROCm 7.1.1** | **ROCm 7.1** HIP 컴파일러 이슈 |

- **픽스**: `ggml-cuda.cu`에서 `SOLVE_TRI` 케이스를 `false` 반환시켜 **CPU 폴백** 강제(한 줄). 초기화 1회성이라 decode 속도엔 무영향.
- 관련 상류 이슈: [llama.cpp #10701 "ROCm error … AMD MI50/60: gfx906"](https://github.com/ggml-org/llama.cpp/issues/10701), [#19880 "ROCm support for newer Qwen models broken"](https://github.com/ggml-org/llama.cpp/issues/19880).
- **⚠ 미확인**: 우리 segfault가 *정확히* SOLVE_TRI인지는 우리 크래시 로그(백트레이스)와 대조해야 확정된다. 증상·버전 일치도는 높지만 단정 금지. **첫 실험 항목 = 이 패치로 warmup이 통과하는지**.

## 3. 사용 가능한 옵션 (우리 적합성 순)

우리 translator 플러그인은 **llama-server의 OpenAI 호환 `/v1/chat/completions`** 를 물고 있다. 따라서 드롭인은 **llama.cpp의 ROCm 빌드**다(ollama로 가면 엔드포인트·모델 태그 재배선 필요 — 불리).

| 옵션 | 무엇 | 우리 적합성 |
|---|---|---|
| **[mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)** | gfx906 프리빌드 **Docker**(llama.cpp/vLLM/PyTorch/ComfyUI), ROCm 6.3.3~7.2.4, 릴리스 28개·2026-07 유지 | **1순위.** 빌드 없이 `docker pull`. gfx906 Tensile 포함. `docker.io/mixa3607/llama.cpp-gfx906:<ver>-rocm-7.2.4` |
| **네이티브 HIP + SOLVE_TRI 패치** | 우리 호스트(ROCm 7.1.1, gfx906 rocBLAS 있음)에서 mainline llama.cpp를 gfx906 타깃 빌드 + §2 패치 | **2순위/검증용.** 되공급 불필요 가설(§1) 검증에 최적. 가장 가벼운 실험 |
| **[eslowney/llama.cpp-gfx906](https://github.com/eslowney/llama.cpp-gfx906)** · **[iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)** | gfx906 **flash attention** 포크 | **모델 head dim 의존.** eslowney FA는 **D=128 전용**(Qwen3-30B 등), 다른 D면 크래시. `gemma-4` head_dim 확인 선행(§6) |
| [MTLoser/ollama-mi50-rocm71-build](https://github.com/MTLoser/ollama-mi50-rocm71-build) · [xxDoman/ollama_mi50](https://github.com/xxDoman/ollama_mi50) | 커스텀 **ollama** 빌드(gfx906) | 참고용. 우린 llama.cpp 경로라 재배선 비용 |

**두 갈래 ROCm 버전 전략:**
- **Route A (안정 우선) — ROCm 6.4.x.** gfx906 커널이 아직 있고(6.3까지 정식), **7.1의 SOLVE_TRI 코드젠 버그 자체가 없다**. eslowney가 ROCm 6.4.1에서 검증. 대신 호스트 ROCm 다운그레이드(또는 6.x 기반 mixa3607 이미지) 필요.
- **Route B (현 호스트 유지) — ROCm 7.1.1 + SOLVE_TRI 패치.** 지금 설치 그대로. rocBLAS 7.1.1에 gfx906 커널이 이미 있으니 되공급 없이 **패치만**으로 될 가능성(§1 단서). 가장 덜 침습적.

## 4. ROCm이 우리에게 주는 이득 (모델 특정, 정직하게)

[MI50 ROCm 7 vs Vulkan 벤치(2026-03, r/LocalLLaMA 유래)](https://insights.marvin-42.com/articles/localllama-shares-mi50-rocm-7-vs-vulkan-benchmarks-for-llamacpp) 요지 — 하드 tok/s 표는 없고(취미 나이틀리 벤치) 방향만:

- **Vulkan 우세**: 짧은 컨텍스트(<16K) **dense** 모델의 prompt processing.
- **ROCm 우세**: **긴 컨텍스트(16K+)와 MoE.** **decode(generation) 차이는 "덜 극적"**, 큰 이득은 **prompt processing(prefill)**.
- 테스트 스택: ROCm 7.13.0a20260321(나이틀리), Vulkan 1.4.341.1, llama.cpp build 8467, 모델 Qwen 3.5(9B/27B dense, 122B MoE)·Nemotron Cascade 2, MI50 32GB(Proxmox).

**우리에게 걸리는 지점:**
- **우리 모델 = MoE**(`gemma-4-26B-A4B`, active 4B) → ROCm 우세 영역.
- **우리 서버 `-c 16384` = 16K** → Vulkan이 "떨어지기 시작"한다는 경계 바로 그 지점.
- **우리 약점 = prefill**(prompt eval 47.3 vs decode 89.88) → ROCm의 prompt-processing 이득이 정확히 여길 때린다.
- **배칭 천장의 주범이 prefill 고정비**였다(translate-gpu-mi50.md §보충 — 포화 스케일링). prefill이 빨라지면 **동시성 스케일 상한이 함께 오른다** — 단일 decode보다 훨씬 큰 잠재 이득.
- **단일 스트림 decode는 대역폭 바운드라 이득이 작다** — Vulkan 89.88이 이미 준수. ROCm 전환의 값은 decode가 아니라 **prefill·배칭·MoE**에 있다.

**참고 수치(우리 모델 아님, 직접 비교 불가):**
- eslowney 포크 Qwen3-30B on MI50: **pp512 1224 tok/s, pp4096 862 tok/s**(KV 양자화), FA가 pp에 +5~11%.
- MTLoser ollama qwen3.5:9b(dense): **49.5 tok/s** decode(MI50 16GB).
- Nemotron Cascade 2 Q4_1 @65K on **MI60**: pp ~726 tok/s(ROCm).

## 5. 제약·리스크·순서

- **진짜 현재 블로커는 냉각.** hang은 백엔드 무관(패시브 카드 과열 + GPU 작업 중 반복 종료 → amdgpu 컨텍스트 hang). **HIP 크래시는 오히려 GPU를 D-state·VRAM 미반납으로 더 나쁜 상태로 남긴다**(이미 경험) → Vulkan보다 취급 주의가 더 필요. **냉각 정리 전 ROCm 실험 금지.**
- **ROCm 나이틀리 불안정**: "weird behavior", prompt cache 할당 OOM 보고. **나이틀리 말고 핀된 조합**(ROCm 6.4.x, 또는 mixa3607의 태깅된 이미지)으로.
- **vLLM은 나중.** mixa3607이 gfx906 vLLM 이미지도 제공하고 배칭은 llama.cpp보다 우수하지만, **지금은 translate가 recognize(CPU 직렬) 병목이라 배칭 포화가 안 나** 당장 이득 없음. throughput 바운드가 됐을 때(9060 XT 재장착 후)의 카드.
- **역할 어휘 유지**: 이 실험은 translate 백엔드 교체다. detect/recognize는 무관(현재 CPU manga-ocr).

## 6. 착수 전 선행 확인

- **`gemma-4-26B-A4B`의 head_dim.** D=128이면 eslowney/iacopPBK FA 포크 직접 적용 가능, 아니면(예: 256) 그 FA는 못 쓰고 Route A/B의 비-FA 또는 다른 head dim 지원 포크로. → 모델 config(`config.json`/GGUF 메타)에서 확인.
- **우리 crash 로그 확보/대조**: 지난 warmup segfault 백트레이스가 남아 있으면 SOLVE_TRI 여부 사전 판별. 없으면 패치 전/후로 재현해 확인.
- **rocBLAS gfx906 커널 현존 재확인**: `/usr/lib64/rocblas/library/`에 `*_gfx906.*` 여전한지(Route B 되공급 생략 가능 여부).

## 7. 실험 런북 초안 (냉각 보강 후)

전제: §복구 런북(translate-gpu-mi50.md)대로 카드 건강 확인(diag_runaway probe ~90 t/s) + 냉각(GPU 온도 추종 팬 커브/직결 송풍 강화) 완료.

1. **백엔드 확보** (택1)
   - Route B: mainline llama.cpp를 `GGML_HIP=ON` + gfx906 타깃 빌드, §2 SOLVE_TRI 패치 적용.
   - 턴키: `docker pull mixa3607/llama.cpp-gfx906:<ver>-rocm-7.2.4` (또는 6.x 태그로 안정 우선).
2. **warmup 통과 확인** — 모델 로드 후 1요청. **여기서 segfault가 사라지면 SOLVE_TRI 가설 확정.** 실패 시 로그로 다른 op 추적.
3. **디바이스 핀 주의** — Vulkan 때의 index 함정과 별개로, ROCm은 `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`로 MI50만 노출(iGPU gfx1036 배제). `rocminfo`로 인덱스 확인 후.
4. **raw decode 스팟 체크** — 단일 요청 decode t/s. Vulkan 89.88 대비 동급 이상인지(하회면 즉시 중단·원인).
5. **prefill·배칭 A/B (핵심)** — [bench_translate_concurrency.py](bench_translate_concurrency.py)를 **같은 report·같은 프로토콜**(P별 개별 실행, junction 60°C 시작, probe 브래키팅)로 ROCm 백엔드에 던져 Vulkan 수치와 대조. 볼 값:
   - **요청당 시간의 prefill 분해**(prompt eval t/s) — ROCm 이득의 본진.
   - **P1/2/4 스케일** — prefill이 싸지면 스케일이 Vulkan(×1.22/×1.48)보다 오르는지.
   - **온도/전력** — `rocm-smi --showtemp --showpower --showclocks` 병행, junction<crit·mem<85°C 감시.
6. **판정 기준** — ROCm이 **prefill 또는 배칭 스케일에서 유의미(예: prefill ≥1.3x, 또는 conc4 포화가 crit 없이 Vulkan 대비 처리량↑)** + **안정성(장시간 무hang)** 둘 다 만족해야 채택. 하나라도 미달이면 **Vulkan 유지**(현 프로덕션이 이미 검증됨).

## 참고 링크 (Sources)

- **되공급 레시피 / SOLVE_TRI 패치**: [MTLoser/ollama-mi50-rocm71-build](https://github.com/MTLoser/ollama-mi50-rocm71-build) — ROCm 7.1 런타임 + ROCm 6.3 Tensile(156 gfx906 커널) + SOLVE_TRI 패치, `HSA_OVERRIDE_GFX_VERSION=9.0.6`, ollama v0.18.2.
- **턴키 Docker 이미지**: [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) — gfx906용 llama.cpp/vLLM/PyTorch/ComfyUI, ROCm 6.3.3~7.2.4, 2026-07 유지.
- **gfx906 flash attention 포크**: [eslowney/llama.cpp-gfx906](https://github.com/eslowney/llama.cpp-gfx906)(D=128 전용, ROCm 6.4.1 검증, pp512 1224 t/s) · [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)(Q8 타일 커널, Q4_0/Q4_1 벡터 로드).
- **커스텀 ollama(gfx906)**: [xxDoman/ollama_mi50](https://github.com/xxDoman/ollama_mi50)(v0.18.0).
- **ROCm 7 vs Vulkan on MI50 벤치**: [insights.marvin-42 요약](https://insights.marvin-42.com/articles/localllama-shares-mi50-rocm-7-vs-vulkan-benchmarks-for-llamacpp) · [aibytes 요약](https://aibytes.blog/comparisons/rocm-7-vs-vulkan-on-mi50-4-model-benchmark-results) (Vulkan=짧은 dense, ROCm=MoE·긴 컨텍스트·prefill).
- **상류 이슈/논의**: [llama.cpp #10701 (ROCm MI50/60 gfx906 에러)](https://github.com/ggml-org/llama.cpp/issues/10701) · [#19880 (newer Qwen ROCm broken)](https://github.com/ggml-org/llama.cpp/issues/19880) · [ROCm 성능 논의 #15021](https://github.com/ggml-org/llama.cpp/discussions/15021) · [Vulkan 성능 논의 #10879](https://github.com/ggml-org/llama.cpp/discussions/10879).
- **실사용 사례**: [Country Boy Computers — dual MI50-32GB로 self-built llama.cpp MoE 구동](https://countryboycomputersbg.com/dual-instinct-mi50-32gb-running-moe-models-with-self-built-llama-cpp-gpt-oss20b-qwen330b-and-gpt-oss120b/).

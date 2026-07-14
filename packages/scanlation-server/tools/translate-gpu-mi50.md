# translate GPU — MI50에서 LLM 번역 돌리기 (gfx906: ollama 불가 → llama.cpp Vulkan)

작성 2026-07-14. LLM translator를 **MI50(AMD Instinct, gfx906/Vega20, 32GB HBM2)** 에서 돌리기까지의 기록 + 재현 레시피 + 남은 일. 배경(GPU 역할 분리)은 [SCANLATION_DESIGN.md](../../../SCANLATION_DESIGN.md), recognize 쪽 GPU 사정은 [recognize-gpu-speed.md](recognize-gpu-speed.md).

## 배경 — 왜 MI50인가

recognizer를 PaddleOCR-VL로 바꾸면 파이프라인이 recognize-bound가 되고, LLM(translator)이 VRAM을 많이 먹어 한 카드에서 recognize와 경합한다. 그래서 **역할을 물리 분리**한다: recognize(PaddleOCR-VL) = 9060 XT 16GB, **translate(LLM) = MI50 32GB**. MI50는 HBM2(~1TB/s) + 넉넉한 VRAM이라 LLM 추론에 맞는다. 케이스 공간 문제로 **현재는 MI50만** 장착돼 있어, 이 문서는 "MI50에서 LLM translate가 도는가"를 먼저 검증한 기록이다.

## 결론 먼저

| 항목 | 결과 |
|---|---|
| **ollama on gfx906** | **불가** — ollama의 `libggml-hip.so`에 gfx906 커널이 컴파일돼 있지 않음(`invalid device function`). 마운트로 못 고침. |
| **HIP 네이티브 빌드 on gfx906** | **불안정** — 디바이스 init은 되지만 warmup에서 segfault(`-fa off`로도 동일). |
| **Vulkan(RADV) on gfx906** | **작동** — RADV가 런타임에 SPIR-V를 그 GPU용으로 컴파일해 arch 비의존. **채택.** |
| **모델** | ollama blob은 로드 불가(ollama-custom 포맷). **네이티브 GGUF** 필요. gemma-4는 llama.cpp가 지원(Google·unsloth가 GGUF 배포). |
| **실측** | `unsloth/gemma-4-26B-A4B-it-qat-GGUF`, MI50 Vulkan, **89.88 tok/s** decode. |

**한 줄 결론: 구세대 AMD(gfx906)에서 LLM은 ollama/ROCm-HIP이 아니라 llama.cpp의 Vulkan 백엔드 + 네이티브 GGUF로 돌린다.**

## 환경

- 호스트: Ryzen 7 9700X, ROCm **7.1.1**(dnf 설치, `/usr/lib64` — `/opt/rocm` 아님).
- GPU: **MI50**(gfx906, DID `0x66a1`, UUID `GPU-3b3210a17337ec1b`, 32GB) + CPU 내장 **iGPU**(gfx1036).
- 호스트 ROCm은 **gfx906를 완전 지원**한다: `rocminfo`가 gfx906 열거, `rocblas-7.1.1`이 `/usr/lib64/rocblas/library/`에 gfx906 커널(`Kernels.so-000-gfx906-xnack-.hsaco`, `TensileLibrary_*_gfx906.*`)을 싣고 있음. **막힌 건 하드웨어/호스트가 아니라 런타임(ollama/llama.cpp) 패키징이었다.**

## 왜 ollama가 안 되나

`docker run ... ollama/ollama:rocm`(0.20.7)로 MI50가 `library=ROCm compute=gfx906 total=32GiB`로 **잡히긴** 한다. 그런데 모델 로드 시 러너가 죽는다:

```
ggml_cuda_compute_forward: IM2COL failed
ROCm error: invalid device function   (ggml-cuda.cu)
```

- `IM2COL`은 rocBLAS가 아니라 **ggml 자체의 HIP 커널**이다. `invalid device function` = `libggml-hip.so` fatbin에 **gfx906 코드오브젝트가 없음**.
- ollama 번들 rocBLAS의 gfx 목록: `gfx908 gfx90a gfx942 gfx950 gfx1030…gfx1201` — **gfx906 없음**. ggml-hip도 같은 `AMDGPU_TARGETS`로 빌드되므로 마찬가지.
- **ollama 공식 문서도 gfx906 미지원 명시**(지원 최저 구형 타깃이 gfx908=MI100).

### 마운트로 못 고치는 이유 (막다른 길, 재시도 금지)

호스트 rocBLAS(7.1.1)에 gfx906 커널이 있으니 컨테이너에 얹어봤다:
- `ROCBLAS_TENSILE_LIBPATH` / `-v .../rocblas/library` 마운트 → **discovery는 통과**(더 진행됨)하지만 **여전히 IM2COL에서 죽음**.
- 이유: 벽은 **rocBLAS가 아니라 ggml의 컴파일된 커널**이다. rocBLAS를 아무리 맞춰도 `libggml-hip.so`는 안 바뀐다. `HSA_OVERRIDE_GFX_VERSION`도 무의미 — 번들에 gfx906와 ISA 호환되는 GCN5 arch(gfx900/906)가 하나도 없어 덮을 대상이 없다(gfx908/90a는 CDNA라 Vega에서 실행 불가).
- ollama의 Vulkan 백엔드(`OLLAMA_VULKAN=1`)도 이 이미지엔 **`libggml-vulkan.so`가 없어** discovery가 0개(RADV 드라이버는 컨테이너에 있는데 태울 백엔드가 없음).

## GGUF는 "컨테이너 포맷"이지 "실행 보장"이 아니다

같은 `.gguf`라도 아무 llama.cpp에서나 도는 게 아니다. 실행에는 **(1) 그 arch의 C++ 구현이 바이너리에 있고, (2) 그 버전이 파일의 스펙과 맞아야** 한다. 새 모델 = 새 arch 코드 = 더 최신 llama.cpp. (비유: `.exe`는 다 같은 형식이나 구형 OS에선 안 돈다.)

특히 **ollama는 신모델을 자체 엔진 + 자체 GGUF 레이아웃**으로 돌린다. 그래서 ollama가 받아둔 blob은 upstream llama.cpp가 못 읽는 경우가 많다. 실측:

| 모델 (ollama blob) | llama.cpp 로드 결과 |
|---|---|
| gemma4:e4b | `wrong number of tensors; expected 2131, got 720` (텐서 구성 상이) |
| qwen3.5:9b | `qwen35.rope.dimension_sections ... expected 4, got 3` (버전 상이) |
| gpt-oss:20b | `unknown model architecture: 'gptoss'` (구버전 이미지) |
| **bartowski/gemma-2-9b (네이티브 GGUF)** | **정상 로드** ✅ |

→ **llama.cpp에는 ollama blob이 아니라 네이티브(llama.cpp 변환) GGUF를 준다.**

## HIP 네이티브 빌드도 gfx906에서 불안정

호스트에 이미 있던 네이티브 빌드 [`/opt/llama/llama.cpp/build-hip`](file:///opt/llama/llama.cpp)(버전 7761, gfx906 타깃 포함)는 ollama와 달리 **모델을 MI50에 완전히 올린다**(`offloaded 43/43 layers`, KV cache on ROCm0). 그런데 그 직후 죽는다:

```
common_init_from_params: warming up the model with an empty run ...
→ Segmentation fault (core dumped)
```

`-fa off`(flash attention 끔)로도 동일하게 warmup에서 segfault. FA가 아니라 **forward 그래프의 gfx906 커널 자체**가 문제. (같은 모델·GPU에서 Vulkan은 멀쩡히 돈다.) → **HIP는 접고 Vulkan.** 최신 HIP로 재빌드하면 고쳐질 수도 있으나, Vulkan이 이미 되므로 추적 보류.

## Vulkan이 답 — 그리고 gemma-4는 지원된다

- **RADV(mesa Vulkan)** 는 SPIR-V 셰이더를 런타임에 그 GPU용으로 JIT 컴파일한다. **arch별 사전 컴파일이 불필요** → gfx906 커널 문제 자체가 사라진다. Vega20는 RADV 지원이 성숙하다. (llama.cpp의 llama-cpp translator plugin docstring도 "AMD엔 ROCm보다 Vulkan이 안정적"이라 명시.)
- **gemma-4는 llama.cpp가 지원**한다(2026 Google 모델). Google(`gemma-4-12B-it-qat-q4_0-gguf`)·unsloth(`gemma-4-26B-A4B-it-qat-GGUF`, `gemma-4-31B-it-qat-GGUF`)가 네이티브 GGUF를 배포 — 이들이 GGUF를 낸다는 것 자체가 arch 지원의 증거.
- **모델 선택: `unsloth/gemma-4-26B-A4B-it-qat-GGUF`.** 프로덕션에서 쓰던 `VladimirGav/gemma4-26b-16GB-VRAM`(ollama)과 **같은 26B-A4B 계열**. **MoE(active 4B)** 라 26B 총량인데도 토큰당 4B만 계산 → MI50에서 빠르고 32GB에 넉넉. **QAT quant**라 단순 Q4보다 품질 좋음.

## 실측

| 모델 | 백엔드 | decode |
|---|---|---|
| gemma-2-9b-it Q4_K_M (dense 9B) | Vulkan(RADV) | 53.6 tok/s |
| **unsloth/gemma-4-26B-A4B-it-qat (MoE, active 4B)** | **Vulkan(RADV)** | **89.88 tok/s** (prompt eval 47.3) |

26B-A4B가 dense 9B보다 빠른 건 active 파라미터가 4B라 그렇다. **89.88 tok/s는 MI50(discrete)에서만 나오는 수치** — CPU면 한 자릿수, iGPU/lavapipe도 10~15가 천장. 번역 품질도 양호.

## 작동 레시피 (재현)

호스트에 ROCm 7.1.1(gfx906 지원) + 네이티브 llama.cpp 소스가 있는 상태 기준.

```bash
# 1) Vulkan 빌드 의존성 (Fedora/el10 — 패키지명은 배포판 따라 확인)
dnf install -y vulkan-loader-devel vulkan-headers glslc \
  spirv-headers-devel spirv-tools-devel glslang-devel \
  mesa-vulkan-drivers cmake gcc-c++ git

# 2) 최신 소스로 Vulkan 빌드
cd /opt/llama/llama.cpp
git pull
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --config Release -t llama-server

# 3) MI50에서 서버 (네이티브 gemma-4 GGUF)
/opt/llama/llama.cpp/build-vulkan/bin/llama-server \
  -hf unsloth/gemma-4-26B-A4B-it-qat-GGUF -ngl 99 -c 8192 \
  --host 0.0.0.0 --port 8080
```

- 로드 로그에서 **어떤 Vulkan 디바이스를 잡았는지 확인**. MI50(RADV, VEGA20/gfx906)가 아니라 iGPU나 `llvmpipe`(CPU)를 잡으면 `GGML_VK_VISIBLE_DEVICES=<MI50인덱스>`로 핀. (기본값이 discrete를 우선하는 편.)
- 포트 충돌 주의: 같은 8080에 다른 서버(도커 등)가 떠 있으면 `couldn't bind ... port 8080`으로 즉사하니 먼저 비운다.

## 남은 일 (TODO)

1. **스캔레이션 서버 연결.** translate를 llama.cpp 플러그인으로: env `LLAMACPP_ENDPOINT=http://<MI50호스트>:8080`, `/admin`에서 llama.cpp 플러그인 설치 + 모델 `unsloth/gemma-4-26B-A4B-it-qat-GGUF` 선택. (plugin: [scanlation-llama-cpp](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py), OpenAI 호환 `/v1/chat/completions`.)
2. **벤치.** `run_report_20260710_111941.md`(전 CPU 실행)와 **같은 챕터 이미지**로 [run_report.py](run_report.py) → `translate_ms` 비교 = "MI50가 CPU 대비 얼마나 빠른가"의 end-to-end 답. (CPU 기준은 `VladimirGav/gemma4-26b`, 이번은 같은 26B-A4B 계열 unsloth QAT — 근사 동일 모델.)
3. **영속화.** 지금은 `&` 백그라운드라 셸/재부팅에 죽는다. `systemd` 유닛으로 굳혀 재부팅 생존.
4. **MI50 디바이스 핀 확정.** Vulkan이 iGPU/lavapipe를 안 잡도록 `GGML_VK_VISIBLE_DEVICES` 고정 여부 결정.
5. **최종 토폴로지(9060 XT 재장착 후).** detect=CPU / recognize=9060 XT / **translate=MI50** 물리 병렬. translate는 파이프라인상 이미 gate 밖이라 배포만으로 병렬 활성(코드 변경 불필요, [recognize-gpu-speed.md](recognize-gpu-speed.md) 참조).
6. **하드코딩 회피 점검.** 엔드포인트·모델·포트 등 조절값은 env 기본 + `/admin` 노출 원칙을 따른다(신규 값 생기면).

## 관련

- recognize 쪽 GPU 속도(9060 XT, PaddleOCR-VL) — [recognize-gpu-speed.md](recognize-gpu-speed.md)
- GPU 역할 분리 설계 — [SCANLATION_DESIGN.md](../../../SCANLATION_DESIGN.md)
- llama.cpp translator plugin(OpenAI 호환) — [scanlation-llama-cpp/plugin.py](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py)

# translate GPU — MI50에서 LLM 번역 돌리기 (gfx906: ollama 불가 → llama.cpp Vulkan)

작성 2026-07-14. LLM translator를 **MI50(AMD Instinct, gfx906/Vega20, 32GB HBM2)** 에서 돌리기까지의 기록 + 재현 레시피 + 남은 일. 배경(GPU 역할 분리)은 [SCANLATION_DESIGN.md](../../../SCANLATION_DESIGN.md), recognize 쪽 GPU 사정은 [recognize-gpu-speed.md](recognize-gpu-speed.md).

> **상태 (2026-07-15):** **작동·검증 완료.** raw 모델 MI50 확인(gemma-4 89 tok/s) → 파이프라인 느림 원인 규명(grammar 아니라 **reasoning 과다 thinking**, 조치 `--reasoning-budget 0`) → **end-to-end 벤치: 이전 GPU 대비 translate 1.62x**(§3). 로드타임 ~84초는 Vulkan 특성(디스크 아님, §5), GPU hang(§4)은 콜드 부팅으로 복구. 남은 건 **systemd 영속화**([deploy/llama-server.service.example](../../../deploy/llama-server.service.example))와 **어드민 `think` 옵션 서버 검증**(§3). 상세는 §[파이프라인 통합 시도](#파이프라인-통합-시도-2026-07-14--미완--gpu-hang-사고).

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

## 파이프라인 통합 시도 (2026-07-14) — 미완 + GPU hang 사고

raw 모델 검증(위) 이후 실제 파이프라인(`run_report` → 서버 → llama.cpp)에 물리려다 막힌 기록. **셋업 결함이 아니라 (a) translate 느림 원인 미확정 + (b) 운영 사고**다.

### 1. 플러그인 설치 — 권한 문제 (해결)
- `/admin`에서 llama.cpp 플러그인 설치가 `PermissionError: [Errno 13] ... 'torchfrtrace'`로 실패.
- 원인: 서버 컨테이너는 `app` 유저로 pip 설치하는데, `/plugins` 볼륨에 **root 소유 torch 잔재**(PaddleOCR-VL 테스트하며 수동 `pip install --target /plugins`를 root로 돌린 흔적)가 있어 `--upgrade` 재설치 중 그 파일을 못 지움.
- 해결: `docker exec -u 0 scanlation-server chown -R app:app /plugins` → 재설치 성공.

### 2. 엔드포인트는 이미 배선돼 있음
[docker-compose.yml](../../../docker-compose.yml)에 `LLAMACPP_ENDPOINT: http://host.docker.internal:8080`가 이미 있다. 컨테이너가 host-gateway로 호스트 llama.cpp:8080에 도달 → **별도 엔드포인트 설정 불필요.** `/admin`에서 translator=llama.cpp + 모델 선택만 하면 배선 끝.

### 3. translate가 느림 — 원인 미확정 (재개 시 최우선)
- 파이프라인 translate가 페이지당 **>10초, 심하면 >5분**. `httpx.ReadTimeout: timed out` → 말풍선 단위 순차 폴백 → 그것도 타임아웃 → `regions=0`, 500.
- 두 겹으로 의심:
  - **HTTP 타임아웃 10초(`SCANLATION_HTTP_TIMEOUT`)가 26B엔 짧다.** "think-off ~1s 백엔드"를 가정한 기본값이라, 무거운 GPU LLM엔 부족.
  - **grammar-constrained JSON + gemma 256k vocab.** 배치 translate는 `response_format: json_schema, strict`라 **매 토큰 grammar로 256k 어휘를 필터** → 자유생성(89 t/s)이 크게 느려질 수 있다(gemma vocab은 32k급의 8배).
- **깨끗한 측정을 못 냈다.** 죽인 `run_report` 런들의 요청이 llama.cpp 큐에 남아 계속 처리(클라 타임아웃해도 서버는 안 멈춤) → 후속 측정이 백로그 뒤에 밀려 오염. grammar 가설을 확정할 **스키마 유/무 curl 비교**를 결국 못 냈다.
- **→ 재개 시 1순위: fresh 서버에서 스키마-없는 자유생성 vs 스키마-있는(grammar) curl 시간 비교.** 후자가 수 배~수십 배 느리면 grammar+vocab 확정.

**해결 (2026-07-15) — reasoning 과다생성이 범인. grammar는 무죄.**
- 콜드 부팅 복구 후 fresh 서버(idle, 큐 깨끗)에서 스키마 유/무 curl: **①6.0s vs ②6.9s (~1s 차)** → **grammar 기각.**
- 대신 응답 JSON: `completion_tokens: 313` + `reasoning_content` 필드가 가득. gemma-4-26B-A4B는 **reasoning 모델이고 thinking이 켜져 있어**, "2단어 번역"에도 번역 옵션(존댓말/반말/…)을 **300토큰씩 따진 뒤** 짧은 답을 낸다. `timings`: prompt 44ms(113 t/s)·decode 83 t/s로 **prefill·decode 둘 다 정상** → MI50는 빠르고 **모델이 말이 많았을 뿐**. 배치로 가면 말풍선마다 300토큰 → 페이지당 수천 토큰 → 타임아웃 폭발.
- **플러그인 `strip_think=True`는 생성 뒤 잘라내는 것이라 속도엔 무효** — 생성 자체를 막아야 한다.
- **조치**: llama-server에 `--reasoning-budget 0`으로 thinking 억제(서버 플래그라 코드 변경 없이 전역 적용). 모델 템플릿이 안 먹으면 요청에 `chat_template_kwargs:{enable_thinking:false}` 또는 `reasoning_effort:none`. **확인법**: 같은 요청의 `completion_tokens`가 300+ → ~5로 떨어지는지.
- 부산물: 이걸로 **HTTP 타임아웃(10초)·큐 오염 문제도 대부분 해소** — 요청이 빨라지니 타임아웃에 안 걸리고 큐가 안 쌓인다.

**⚠ 디바이스 핀 함정.** 재기동 때 `GGML_VK_VISIBLE_DEVICES=0`을 걸었더니 **MI50가 아니라 lavapipe(CPU Vulkan)/iGPU**를 잡아 **10배 느림**(decode 83→8.5 t/s). llama.cpp Vulkan 열거의 index 0이 MI50가 아니다. **핀 없이 두면 기본 로직이 discrete(MI50)를 자동 선택**한다 → **핀 걸지 말 것**(굳이 걸려면 로그의 디바이스 목록에서 MI50 인덱스 확인 후).

**파이프라인 벤치 (2026-07-15) — 이전 GPU 대비 translate 1.62x.** reasoning off + MI50(핀 없이) 확정 후 같은 21장 챕터로 `run_report`:

| translate | 이전 GPU (ollama, gemma4-26b) | MI50 (llama.cpp, gemma-4-26B-A4B) |
|---|---|---|
| 평균 ms | 1509.1 | **932.9** (1.62x) |
| 중앙값 | 1513.0 | 886.3 (1.71x) |
| 최대(무거운 페이지) | 2874.8 | 1431.0 (2.0x) |
| total 평균 | 2505.3 | 1687.8 (1.48x) |

21/21 성공, translate 최대 1431ms(타임아웃 여유 충분). detect+recognize는 양쪽 CPU manga-ocr로 동일(recognize 468→457ms) → **translate 백엔드 교체 효과만 분리**. **HBM2 대역폭이 그대로 숫자로** 나온다 — "decode는 대역폭 바운드"([recognize-gpu-speed.md](recognize-gpu-speed.md))가 translate에서도 재현, 무거운 페이지일수록 이득(최대 2x). 속도만이 아니라 **recognize GPU(9060 XT) 해방**이라는 아키텍처 목표도 달성.

**어드민 reasoning 제어 — 플러그인에 `think` 옵션 추가 (2026-07-15).** [llama.cpp 플러그인](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py)의 `strip_think`은 출력을 잘라낼 뿐이라, 생성 자체를 막는 **`think`(bool, 기본 False)** 옵션을 추가했다(ollama의 `think`와 대칭). 요청 body에 `chat_template_kwargs:{enable_thinking: think}`로 전달. **단 모델 chat 템플릿이 그 kwarg를 존중해야 실효** — gemma-4에서 실제 먹는지는 **서버 검증 필요**(플러그인 재설치 후 /admin에서 토글, `completion_tokens` 변하나 확인). 안 먹으면 서버 `--reasoning-budget 0`이 전역 폴백이라 무해. 단위테스트(body shape)는 green.

### 4. GPU hang 사고 (미복구)
- llama-server를 **GPU 작업 중 `pkill -9`로 반복 종료** → **amdgpu 컨텍스트 hang**. 프로세스가 D 상태(uninterruptible)로 안 죽고, **VRAM 16.8GB가 프로세스 킬 후에도 반납 안 됨**(`rocm-smi`).
- warm `reboot` 시도 → 부팅이 amdgpu init에서 멈춘 정황(SSH가 TCP는 받되 검은 화면, 웹 도메인 Cloudflare 521→523). **원격에 전원 제어 수단(스마트 플러그·IPMI)이 없어 콜드 부팅 불가 → 머신 다운 상태로 방치.** (WOL은 켜진 hang 머신을 리셋하지 못함.)
- **교훈: llama-server를 GPU 작업 중 `kill -9` 금지.** 하드킬이 amdgpu 컨텍스트를 꼬아 **콜드 전원 순환 전까지 GPU가 안 풀린다.** 반드시 systemd로 올려 `systemctl stop`(graceful)만 쓸 것.

### 5. 로드타임 ~84초 — Vulkan 특성(1회성), 디스크 아님
- 서버 기동 시 모델 로드가 **~84초**(로그 타임스탬프 1.5s→1:24가 통째로 텐서 로드 구간).
- **디스크가 범인 아님**: 모델은 **NVMe SSD**(`/`=cs-root, Solidigm 1.9TB)에 있어 raw 16GB 읽기면 ~3~5초. `lsblk`상 sda(TOSHIBA 14.6T)는 HDD지만 그건 별개 스토리지고, cs-root는 nvme.
- **범인 = Vulkan 백엔드 오버헤드**: 가중치를 **스테이징 버퍼 경유로 VRAM 업로드**(ROCm/CUDA 직접 memcpy보다 느림) + 첫 로드 **셰이더/파이프라인 컴파일**. gfx906에서 ROCm이 안 돼 Vulkan을 쓰는 대가의 일부. 멀티모달 projector(`mmproj-BF16.gguf`)도 같이 로드돼 시간 일부 차지(텍스트 번역만이면 스킵 여지, 부차).
- **핵심: per-request가 아니라 1회성 로드 비용.** 추론은 83 t/s로 빠르다. **systemd 상주면 부팅 때 한 번만** 낸다 → 파이프라인 처리량엔 무영향. (줄이려면: Vulkan 파이프라인 캐시 영속 + mmproj 스킵 — 둘 다 부차.)

### 복구 런북 (다운 상태에서 재개)
1. **콜드 부팅**(warm reboot 아님): 전원 완전 차단 후 재투입 — amdgpu hang은 콜드 리셋만 확실히 푼다.
2. `rocm-smi --showmeminfo vram`로 VRAM used ~0.2GB(깨끗) 확인.
3. llama-server를 **systemd 유닛**으로 (graceful stop + MI50 핀). ← TODO 3과 통합, 이걸 먼저.
4. fresh 서버에서 **스키마 유/무 curl 시간 비교**(위 §3) → translate 느림 원인 확정.
5. 원인별 조치: 타임아웃만 올리면 되는지(`SCANLATION_HTTP_TIMEOUT`↑, 코드 규칙상 `/admin` 노출도 검토) vs grammar 회피(플러그인 코드) vs 더 작은 모델(gemma-4-12B 등).
6. 그다음 `run_report` 벤치 — **중간에 죽이지 말 것**(큐 오염).

## 남은 일 (TODO)

> 즉시 재개는 위 §복구 런북을 따른다. 아래는 그 외 남은 항목.

1. **스캔레이션 서버 연결.** 플러그인 설치·엔드포인트 배선은 됨(§통합 시도 1·2). 남은 건 §3의 **translate 느림 해결** — 그게 풀려야 실제 연결 완료. (plugin: [scanlation-llama-cpp](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py), OpenAI 호환 `/v1/chat/completions`, 모델 `unsloth/gemma-4-26B-A4B-it-qat-GGUF`.)
2. ~~**벤치.**~~ **완료 (2026-07-15)** — 이전 GPU 대비 translate **1.62x**(§3 파이프라인 벤치). 기준 `run_report_20260710_111941.md`(ollama, 이전 GPU) vs `run_report_mi50_translate.md`(llama.cpp, MI50). 참고: 20260710은 CPU가 아니라 **이전 GPU** 실행이었다.
3. **영속화.** 유닛 파일 작성됨 → [deploy/llama-server.service.example](../../../deploy/llama-server.service.example)(`--reasoning-budget 0` 박음 + 핀 없음 + SIGTERM graceful). 서버에 복사 + `systemctl enable --now`만 하면 됨. **아직 미배포 — 현재는 `&` 백그라운드.**
4. ~~**MI50 디바이스 핀 확정.**~~ **결정 (2026-07-15): 핀 없이 자동.** `GGML_VK_VISIBLE_DEVICES=0`이 오히려 lavapipe/iGPU를 잡아 10x 느렸다(§3 함정) — 기본 자동선택이 discrete(MI50)를 고른다.
5. **최종 토폴로지(9060 XT 재장착 후).** detect=CPU / recognize=9060 XT / **translate=MI50** 물리 병렬. translate는 파이프라인상 이미 gate 밖이라 배포만으로 병렬 활성(코드 변경 불필요, [recognize-gpu-speed.md](recognize-gpu-speed.md) 참조).
6. **하드코딩 회피 점검.** 엔드포인트·모델·포트 등 조절값은 env 기본 + `/admin` 노출 원칙을 따른다(신규 값 생기면).

## 관련

- recognize 쪽 GPU 속도(9060 XT, PaddleOCR-VL) — [recognize-gpu-speed.md](recognize-gpu-speed.md)
- GPU 역할 분리 설계 — [SCANLATION_DESIGN.md](../../../SCANLATION_DESIGN.md)
- llama.cpp translator plugin(OpenAI 호환) — [scanlation-llama-cpp/plugin.py](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py)

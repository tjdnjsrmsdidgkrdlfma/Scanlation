# translate GPU — MI50에서 LLM 번역 돌리기 (gfx906: ollama 불가 → llama.cpp Vulkan)

작성 2026-07-14. LLM translator를 **MI50(AMD Instinct, gfx906/Vega20, 32GB HBM2)** 에서 돌리기까지의 기록 + 재현 레시피 + 남은 일. 배경(GPU 역할 분리)은 [SCANLATION_DESIGN.md](../../../SCANLATION_DESIGN.md), recognize 쪽 GPU 사정은 [recognize-gpu-speed.md](recognize-gpu-speed.md).

> **상태 (2026-07-15 저녁):** 2차 hang은 **콜드 부팅(전원 버튼 강제 차단)으로 복구 완료** — 카드 생존(gfx906 정상 열거·모델 로드 정상, 유닛 자동 기동 확인). `/var/log/messages` 포렌식으로 **2차 사고 인과 규명**: hang의 실체는 SIGKILL이 아니라 **지속 부하 중 카드가 PCIe 버스에서 이탈한 하드웨어 크래시**(과열 추정, 로그는 §4 2차). 남은 순서: **(1) 쿨링+전력 캡 확보(§MI50 쿨링) → (2) 폭주 원인 확정**(요청당 ~3300토큰, §복구 런북 4) → (3) 동시성 벤치.
> **그 외는 작동·검증 완료.** raw 모델 MI50 확인(gemma-4 89 tok/s) → 파이프라인 느림 원인 규명(grammar 아니라 **reasoning 과다 thinking**) → **end-to-end 벤치: 이전 GPU 대비 translate 1.62x**(§3). 로드타임 ~84초는 Vulkan 특성(디스크 아님, §5). **reasoning 제어는 Option B** — 서버 `--reasoning-budget 0` 대신 플러그인 `enable_thinking`(기본 off)으로 제어해 /admin 토글이 실제로 작동(§3). systemd 영속화·플러그인 재설치 완료. 상세는 §[파이프라인 통합 시도](#파이프라인-통합-시도-2026-07-14--미완--gpu-hang-사고).

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

## MI50 쿨링 — 능동 냉각 필수

MI50는 **팬이 없는 패시브 서버 카드**(TDP 300W)로, 서버 섀시의 강제 공기흐름을 전제로 설계됐다. 데스크톱에서 **카드 전용 팬(블로워+슈라우드) 없이, 케이스 팬까지 끄면 지속 부하에서 과열**한다 — hang·열 스로틀링·(장기적으로) 하드웨어 손상 위험. 팬 전력은 개당 1~3W라 300W 카드 앞에선 무시할 수준이니 **끄지 말 것**(소음이면 GPU 온도 연동 PWM 커브).

- **2차 사고로 실증됨 (2026-07-15)**: 무냉각 지속 부하에서 카드가 **PCIe 버스 이탈(하드웨어 크래시)**까지 갔다(§4 2차). 콜드 부팅 후 **idle에서도 junction 68°C** 실측(모델만 로드한 저전력 상태) — 공기 흐름이 사실상 없다는 뜻. 쿨링은 재부하 전 **전제조건**이다.
- **온보드 부저**: MI50는 카드에 경보 부저가 실장돼 있어 **과열·전원 이상 시 삐 소리**를 낸다([Level1Techs 스레드](https://forum.level1techs.com/t/amd-instinct-mi50-beeping-constantly/232359)). 2차 사고에서 크래시 후 fault 상태로 **~3시간 연속 경보** 실측(호스트·SSH는 내내 정상 — 죽은 건 카드뿐). 유일한 하드웨어 경보 수단이니 부저를 제거하지 말 것.
- **임계 온도(amdgpu 노출값)**: junction crit 100°C·emerg 105°C, **HBM(mem) crit 94°C·emerg 99°C** — HBM 한계가 더 낮아 부하 중 중단 기준은 **mem ~85°C**를 먼저 본다. 무풍 idle 68°C 실측 기준 풀로드 +30~40°C면 바로 crit 영역 — 2차 크래시(§4)와 정량적으로 정합.
- **직접 송풍 장착 (2026-07-15 저녁, 사고 후)**: NF-A4x10 PWM(40mm)을 카드 후단 나사홀에 장착 — 슈라우드 터널로 밀어 넣어 브래킷으로 배기하는 방향. 장착만으로 idle 68 → 48°C. **사고 당시엔 이 팬이 없었다**(위 무풍 서술은 사고 시점 기준 그대로 유효). 40mm 풍량이 지속 부하(150W 캡)에 충분한지는 전력 캡 스윕에서 mem 온도로 검증. 팬 커브는 BIOS 제어(온도 소스 CPU) — 보드는 GPU 온도를 못 보므로 **GPU 단독 부하는 커브가 추적하지 못한다**. 최저 회전을 깔아두는 게 방어선이고, 부하 실험 중엔 `rocm-smi --showtemp` 병행 감시.
- **확인**: 부하 중 `rocm-smi --showtemp --showpower --showclocks`(`--showtemp`가 edge·junction·mem 셋 다 표시). idle 온도는 무의미 — 추론을 돌리며 junction·mem 온도와 클럭을 본다. 클럭이 떨어지면 스로틀링.
- **열린 관측 → 열 쪽 유력**: 벤치 중 decode가 §실측의 89 → 33 t/s 부근으로 낮게 관측된 적이 있는데(배칭 per-slot rate vs 열 스로틀링), 2차 사고의 폭주 요청이 크래시 직전 **tg=32 t/s**로 실측되며 열 스로틀링 쪽 증거가 강해졌다(§4 2차). 쿨링 확보 후 온도 보며 최종 확인.

### 전력 캡 — 발열을 원천에서 줄인다 (쿨링 보완이지 대체 아님)

LLM decode는 대역폭 바운드라 코어 전력을 낮춰도 속도 손실이 작다 → **300W 풀파워가 필요 없다.** 캡을 걸면 발열·소음·전원 배선 부담(8핀 데이지체인)이 함께 내려간다.

```bash
rocm-smi -d 0 --showmaxpower           # 현재 한도 확인 (MI50 = GPU 0)
rocm-smi -d 0 --setpoweroverdrive 150  # 150W로 제한 (root)
rocm-smi -d 0 --showpower              # 적용 확인
```

- **재부팅 시 리셋**되므로 영속화는 llama.cpp 유닛의 `ExecStartPre`로([deploy/llama.cpp.service.example](../../../deploy/llama.cpp.service.example)).
- 값은 150W에서 시작해 **decode tok/s 실측으로 조정**(손실이 크면 170~200W). 캡을 걸어도 **팬의 대체는 아니다** — idle 68°C가 말해주듯 공기 흐름이 없으면 150W도 쌓인다.

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

**어드민 reasoning 제어 — 플러그인에 `think` 옵션 추가 (2026-07-15).** [llama.cpp 플러그인](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py)의 `strip_think`은 출력을 잘라낼 뿐이라, 생성 자체를 막는 **`think`(bool, 기본 False)** 옵션을 추가했다(ollama의 `think`와 대칭). 요청 body에 `chat_template_kwargs:{enable_thinking: think}`로 전달. 단위테스트(body shape) green.

**검증 (2026-07-15) — 토글은 작동한다. 중간에 낸 "no-op" 결론은 오판이었고, 아래가 최종.**

처음엔 `--reasoning-budget 0`을 켠 채 `enable_thinking`을 테스트해서 "gemma-4가 kwarg를 무시한다"고 잘못 결론냈다. 실은 **`--reasoning-budget 0`(하드 캡)이 per-request `enable_thinking`을 덮어써서** 가려진 것. **왜 ollama는 `think` 하나로 됐나**: ollama는 native `think` 필드를 각 모델 방식으로 내부 매핑(추상화)한다. llama.cpp는 raw **chat 템플릿 변수**(`enable_thinking`)를 그대로 노출하므로 모델 규약이 맞아야 한다(Qwen3·gemma=`enable_thinking`, Granite=`thinking`…). gemma-4는 `enable_thinking`을 쓴다([Google 문서](https://ai.google.dev/gemma/docs/capabilities/thinking): 기본 off).

budget 플래그를 빼고 재검증(실측):

| 요청 | reasoning? | tokens |
|---|---|---|
| free 프롬프트 · `enable_thinking:false` | False | 373 (본문 수다) |
| free 프롬프트 · `enable_thinking:true` | **True** | 702 (+330 reasoning 블록) |
| **schema · `enable_thinking:false`** | False | **27** (terse) |

→ **gemma-4도 `enable_thinking`을 정상 존중한다.** free 프롬프트의 373은 reasoning이 아니라 본문 수다(스키마·시스템프롬프트 없어서)고, **파이프라인이 쓰는 schema 경로는 27토큰**으로 terse.

**→ 채택: Option B.** 서버 `--reasoning-budget 0`(하드 캡, /admin 토글을 죽임)을 빼고, **플러그인 `think`(기본 False → `enable_thinking:false`)에 제어를 맡긴다.** 그러면 (1) **/admin 토글이 실제로 작동**, (2) 파이프라인은 스키마로 terse, (3) `enable_thinking` 쓰는 다른 모델도 대응. systemd 유닛에서 `--reasoning-budget 0` 제거함. **플러그인 재설치 필요** — `think` 옵션(=`enable_thinking:false` 명시 전송)이 든 최신 버전이라야 확실히 off(옛 플러그인은 kwarg 미전송).

**배포 검증 (2026-07-15) — end-to-end 작동 확인.** 플러그인 재설치(+ `docker restart scanlation-server`) + budget 제거 후 파이프라인(00.jpg)으로 /admin 토글 확인: **`think`=False → translate 957.9ms, `think`=True → 26641ms(26.6초, 말풍선마다 reasoning)**. **~28배 차이 = 토글이 /admin에서 실제로 제어됨.** 프로덕션은 `think`=False(fast). 옛 플러그인이 깔려 있으면 llama.cpp만 삭제 후 재설치: `docker exec -u 0 scanlation-server rm -rf /plugins/scanlation_llama_cpp*` → /admin 재설치 → `docker restart scanlation-server`.

**`strip_think` 제거 (2026-07-15).** reasoning 제어가 `think` 토글(생성 단계)로 옮겨졌고, llama.cpp는 reasoning을 `reasoning_content`로 분리해 `content`가 이미 깨끗하므로 사후 `<think>` 스트립(`strip_think`)이 no-op이라 옵션을 제거했다(생성을 못 막는 사후 청소는 무의미). 다른 OpenAI 서버가 `<think>`를 inline으로 흘리는 엣지케이스 방어는 잃지만, 실사용은 llama.cpp 하나라 순손실 없음.

### 4. GPU hang 사고 (2회) — 1차: SIGKILL이 amdgpu를 꼰다 · 2차: 과열발 하드웨어 크래시

**1차 (kill -9 직접).**
- llama-server를 **GPU 작업 중 `pkill -9`로 반복 종료** → **amdgpu 컨텍스트 hang**. 프로세스가 D 상태(uninterruptible)로 안 죽고, **VRAM 16.8GB가 프로세스 킬 후에도 반납 안 됨**(`rocm-smi`).
- warm `reboot` 시도 → 부팅이 amdgpu init에서 멈춘 정황(SSH가 TCP는 받되 검은 화면, 웹 도메인 Cloudflare 521→523). **원격에 전원 제어 수단(스마트 플러그·IPMI)이 없어 콜드 부팅 불가 → 머신 다운 상태로 방치.** (WOL은 켜진 hang 머신을 리셋하지 못함.) 이후 콜드 부팅으로 복구.
- **교훈: GPU 작업 중 `kill -9` 금지.** 하드킬이 amdgpu 컨텍스트를 꼬아 **콜드 전원 순환 전까지 GPU가 안 풀린다.**

**2차 (2026-07-15) — 실체는 카드의 PCIe 버스 이탈(하드웨어 크래시, 과열 추정). SIGKILL은 원인이 아니라 증상.** 당일엔 "systemctl restart의 SIGKILL이 hang을 냈다"고 결론냈으나, 다음 날 `/var/log/messages` 포렌식으로 뒤집혔다 — **GPU는 SIGKILL 12분 전에 이미 하드웨어 레벨로 죽어 있었다.** (journald가 volatile이라 `journalctl -b -1`은 유실됐고, rsyslog의 `/var/log/messages`가 살렸다 — persistent journal을 켜둘 것, §복구 런북 3.)
- **연쇄(교정판)**: 번역 **폭주**(요청당 ~3300토큰·~120초, 아래 §폭주 재발) → 무냉각 카드에 분 단위 지속 부하 → 과열 정황(폭주 요청 decode **32 t/s** = 정상 83~89의 절반 이하, 스로틀링 시그니처) → **컴퓨트 링 timeout → 링 리셋 실패 → GPU 리셋도 실패: 카드가 PCIe 버스에서 이탈**(`device lost from bus`, -19=ENODEV) → llama-server는 죽은 GPU를 기다리는 D-state → `systemctl restart`의 SIGTERM에 못 빠짐 → 90초 뒤 systemd가 SIGKILL 승격.
- **로그 — 커널: 크래시가 먼저다** (`/var/log/messages`, 15:25:45):
  ```
  15:25:33 llama: task 4324 | n_decoded = 3294, tg = 32.13 t/s   ← 폭주(정상 ~27토큰) + 스로틀 정황
  15:25:45 kernel: amdgpu 0000:03:00.0: ring comp_1.2.0 timeout, signaled seq=140118, emitted seq=140120
           kernel: amdgpu:  Process llama-server pid 51590
           kernel: amdgpu: Starting comp_1.2.0 ring reset
           kernel: amdgpu: Ring comp_1.2.0 reset failed
           kernel: amdgpu: GPU reset begin!. Source:  1
           kernel: amdgpu: device lost from bus!                 ← 카드가 PCIe에서 사라짐
           kernel: amdgpu: GPU reset end with ret = -19           ← -ENODEV(장치 없음)
           kernel: amdgpu: GPU Recovery Failed: -19
  (이후 SMU 응답 전부 0xffffffff = 죽은 장치 읽기. powerplay 클럭 조회 실패 반복,
   ring page1 timeout도 같은 -19, kworker "blocked for more than 122 seconds".)
  ```
- **로그 — systemd: SIGKILL은 그 12분 뒤** (같은 파일):
  ```
  15:14:12 systemd: llama.cpp restart — SIGTERM에 즉시 graceful 종료   ← 사고 전 마지막 정상 지점
  15:37:38 systemd: Stopping llama.cpp.service...                     ← systemctl restart = SIGTERM
  15:39:08 systemd: State 'stop-sigterm' timed out. Killing.          ← 90초(DefaultTimeoutStopSec)
  15:39:08 systemd: Killing process 51590 (llama-server) with signal SIGKILL.
  15:39:09 systemd: llama.cpp.service: Failed with result 'timeout'.
  15:39:09 systemd: Started llama.cpp.service                         ← restart의 start 단계(죽은 GPU 상대로 기동)
  15:46:58 systemd: Deactivated successfully.                         ← 그 인스턴스는 stop에 3초 만에 graceful 종료
  ```
  15:46:58이 대조군이다: **GPU에 안 물린 llama-server는 SIGTERM에 즉시 빠진다.** SIGTERM이 안 통한 건 백로그 때문이 아니라 죽은 GPU 대기 때문.
- **VRAM 미반납·`--gpureset` 무효의 재해석**: 장치가 버스에 없으니(-19) 커널 부킹만 남아 `rocm-smi`가 **VRAM 16.87GB**(죽은 프로세스 할당량과 byte 일치)를 계속 보고했고, gpureset은 리셋할 장치 자체가 없었다(`Successfully reset`을 찍어도 무효 — `rocm_smi_lib` #85 *"reset will not always work, depending on the manner in which the GPU is hung"*, 링크는 §관련). **콜드 부팅(재-POST)만이 해결** — 기존 결론 유지, 이유가 명확해졌다. 카드가 영구 고장난 건 아니다: 콜드 부팅 후 정상 열거·모델 로드.
- **부저**: 크래시 후 카드가 fault 상태로 **~3시간 연속 경보**(§MI50 쿨링). 그동안 호스트·SSH는 정상 — 죽은 건 카드뿐이라, 원격에선 웹/SSH 생존만 보고 있으면 모른다.
- **교훈(재교정)**:
  1. **무냉각 지속 부하는 카드를 하드웨어 크래시까지 몰고 간다** — 쿨링+전력 캡이 재부하 전 전제조건(§MI50 쿨링).
  2. 1차 교훈(GPU 작업 중 `kill -9` 금지)은 유효하다. 단 **2차의 SIGKILL은 원인이 아니었다** — 죽은 GPU에 물린 프로세스는 어떤 graceful로도 안 빠지므로 SIGKILL 승격은 필연이었다(`TimeoutStopSec`을 올려도 죽은 GPU 앞에선 결과가 같아 미채택).
  3. **폭주 제거가 소프트웨어 쪽 절반** — 요청이 짧으면 분 단위 지속 부하 자체가 없어 과열이 쌓일 시간이 없다.

**§폭주 재발 (2026-07-15) — 원인 미확정.** §3에서 reasoning으로 규명·해결했다고 봤으나, 이날 동시성 벤치 준비 중 **schema 경로인데도 요청당 ~3300토큰으로 다시 폭주**했다. `/admin` think 토글은 off였다. 원인 후보 셋인데 **매 시도가 백로그/hang이라 clean 측정을 못 해 확정 못 함**: (a) thinking이 실제로 안 꺼짐 — 플러그인이 `enable_thinking:false`를 안 보내거나 모델이 무시(reasoning은 grammar **밖**에서 생성되므로 schema 경로여도 못 막는다), (b) `response_format`(schema/grammar)이 요청에 안 실림/미적용, (c) 문자열 필드 안 **반복 루프** — grammar는 JSON 구조만 강제하고 문자열 내용은 못 잡으므로 schema가 정상 적용돼도 폭주와 모순이 아니다. **폭주의 내용물은 아직 아무도 못 봤다** — 파이프라인은 최종 결과만 보이고, 안 끝나는 요청은 클라이언트에 아무것도 안 남긴다. 그래서 내용물 분류가 진단 1순위다([tools/diag_runaway.py](diag_runaway.py), §진단 방법론 Phase 1). (Option B로 `--reasoning-budget 0` 하드 캡을 뺀 뒤라 per-request 억제가 안 먹으면 막을 게 없다 — **배포 유닛에 budget 플래그 없음은 사고 후 `systemctl cat`으로 확인**, 즉 사고 당시 방어선은 플러그인의 `enable_thinking:false` 하나였다.) **복구 후 최우선 확인**(아래 런북 4번).

**폭주 격리 — HTTP 타임아웃으론 못 막는다, 서버측 `max_tokens`라야.** `SCANLATION_HTTP_TIMEOUT`은 **클라만 포기시키지 llama-server의 생성을 안 멈춘다**(서버는 EOS나 슬롯 컨텍스트까지 계속 태움 — 클라 타임아웃 뒤에도 GPU는 요청당 ~120초를 태워 백로그가 쌓인다). 그래서 타임아웃 값(10초든 120초든)으로는 폭주의 GPU 소모·백로그를 격리 못 한다. 실제로 서버가 멈추는 건 둘뿐이다: **llama-server 전역 캡 `--n-predict N`**(요청이 max_tokens를 안 보내면 기본값이 되고, 더 큰 값을 요구해도 서버가 클램프 — 유닛 ExecStart에 걸 수 있어 **plugin 상태와 무관하게 작동**, [deploy/llama.cpp.service.example](../../../deploy/llama.cpp.service.example)에 `--n-predict 1024`로 반영), 그리고 요청의 **`max_tokens`**(현재 plugin 미전송, TODO 7). 단 이건 **`n_ctx_slot`보다 낮아야** 의미가 있다 — `-c 16384 --parallel 4`면 슬롯당 4096이라 **4096 캡은 무효**(어차피 4096까지 태움), **1024급**이라야 GPU 시간을 실제로 줄인다(약 32초). 대가는 텍스트 많은 정상 페이지 truncation이므로 **폭주 제거가 본선, 낮은 `max_tokens`는 보조 방어선**(넣으면 하드코딩 말고 env 기본+`/admin` — TODO 7). 한국어 출력은 대략 1토큰 ≈ 1.5~2.5글자라 1024 ≈ 2000자 수준(실측 권장: `completion_tokens ÷ 글자수`).

**hang/reset은 커널(amdgpu) 소관 — 커널·펌웨어 업글은 우선순위 낮다.** hang·리셋·VRAM 미해제는 ROCm 유저스페이스나 llama.cpp가 아니라 **커널 amdgpu 드라이버** 층이고, Vulkan(RADV)도 amdgpu를 거치므로 이 문제를 피하지 못한다(`rocm-smi`는 리셋을 amdgpu에 부탁하는 도구일 뿐). 2차처럼 카드가 **버스에서 이탈**하면 커널 리셋도 -ENODEV로 즉시 실패한다 — 소프트웨어 층 전체가 무력. 커널/`linux-firmware` 업글이 리셋 신뢰성을 **높일 수는** 있으나 (a) 개선이지 보장 아님(하드 hang은 여전히 전원순환), (b) gfx906은 ROCm 지원 **공식 종료(EOL)**라 새 펌웨어 기대값 낮음, (c) 되던 gfx906+Vulkan 조합이 깨질 리스크. → **예방(SIGKILL·폭주·열 제거)이 커널 업글보다 확실·저위험.** 커널은 "예방 다 했는데도 트리거 없이 hang"일 때의 다음 카드.

### 5. 로드타임 ~84초 — Vulkan 특성(1회성), 디스크 아님
- 서버 기동 시 모델 로드가 **~84초**(로그 타임스탬프 1.5s→1:24가 통째로 텐서 로드 구간).
- **디스크가 범인 아님**: 모델은 **NVMe SSD**(`/`=cs-root, Solidigm 1.9TB)에 있어 raw 16GB 읽기면 ~3~5초. `lsblk`상 sda(TOSHIBA 14.6T)는 HDD지만 그건 별개 스토리지고, cs-root는 nvme.
- **범인 = Vulkan 백엔드 오버헤드**: 가중치를 **스테이징 버퍼 경유로 VRAM 업로드**(ROCm/CUDA 직접 memcpy보다 느림) + 첫 로드 **셰이더/파이프라인 컴파일**. gfx906에서 ROCm이 안 돼 Vulkan을 쓰는 대가의 일부. 멀티모달 projector(`mmproj-BF16.gguf`)도 같이 로드돼 시간 일부 차지(텍스트 번역만이면 스킵 여지, 부차).
- **핵심: per-request가 아니라 1회성 로드 비용.** 추론은 83 t/s로 빠르다. **systemd 상주면 부팅 때 한 번만** 낸다 → 파이프라인 처리량엔 무영향. (줄이려면: Vulkan 파이프라인 캐시 영속 + mmproj 스킵 — 둘 다 부차.)

### 복구 런북 (hang 상태에서 재개)
0. **소프트 리셋 먼저**(무해): `systemctl stop llama.cpp` → `pgrep -a llama-server`(잔여 없음 확인) → `rocm-smi --gpureset -d 0` → `rocm-smi --showmeminfo vram`. VRAM이 ~0.2GB로 떨어지면 복구, **그대로면**(2차 사례가 이것) 1번.
1. **콜드 부팅**(warm reboot 아님): 전원 완전 차단 후 재투입 — 전원 버튼 강제 차단도 유효(2차 실증). amdgpu hang·버스 이탈은 콜드 리셋(재-POST)만 확실히 푼다. ⚠ 원격이면 **강제 전원 차단 수단(IPMI·스마트플러그·물리 접근)을 먼저 확인** — warm `reboot`은 amdgpu init에서 멈춰 머신을 원격 다운시킬 수 있다(1차 사례).
2. `rocm-smi --showmeminfo vram`로 VRAM used 확인(유닛 자동 기동이면 모델 ~17GB가 정상, 유닛 내리면 ~0.2GB).
   - ⚠ **쿨링+전력 캡부터 확보** — 재부하 전 케이스 팬 켜고/카드 팬 확인(§MI50 쿨링). 과열이 2차 크래시의 유력 원인이라, 냉각 없이 다시 부하 주면 원인 규명이 오염되고 카드도 위험하다.
3. **재기동 전 안전장치**: (a) **persistent journal 켜기** — `mkdir -p /var/log/journal && systemctl restart systemd-journald`(2차 포렌식은 rsyslog `/var/log/messages` 덕에 가능했다 — `journalctl -b -1`도 남게 해둔다). (b) **전역 토큰 캡** — 유닛 ExecStart에 `--n-predict 1024`(§폭주 격리): 폭주가 남아 있어도 요청당 GPU 소모가 최대 ~30초로 잘려 백로그·과열이 못 쌓인다. `--reasoning-budget 0`(reasoning 하드 캡)은 /admin think 토글을 무력화하므로 **4번에서 원인을 가른 뒤** 결정.
4. **폭주 원인 확정 (최우선)** — 큐 빈 fresh 서버에서 **schema + `enable_thinking:false` 프로브 하나**: [tools/diag_runaway.py](diag_runaway.py)(스트리밍 + `max_tokens` 캡 내장이라 안 매달리고, 폭주 내용물까지 분류):
   - `completion_tokens` 수십 + `reasoning_content` 빔 → 모델·스키마 정상 → **파이프라인(플러그인)이 kwarg/스키마를 안 보내는 것** → 플러그인 재설치(§배포 검증).
   - `completion_tokens`가 캡에 걸림 + `reasoning_content` 참 → 모델이 `enable_thinking:false` 무시 → `--reasoning-budget 0` 하드 캡.
   - content 안에서 같은 구절 반복(루프) → 샘플링 문제 → repeat/presence penalty 조절(§진단 방법론 2c).
5. 폭주 잡힘 확인(llama 로그의 `n_decoded`가 수십대) 후에야 `run_report` 벤치 — **중간에 죽이지 말 것**(큐 오염). 동시성 스윕은 `run_report --concurrency 1/2/4`.

## 진단 방법론 — 폭주(runaway) 잡기

이 사태의 뿌리는 둘이다 — **번역 폭주(요청당 ~3300토큰, 정상 ~27 · 소프트웨어)** 와 **무냉각(하드웨어)**. 폭주가 분 단위 지속 부하를 만들고, 무냉각이 그걸 버스 이탈 크래시(§4 2차)로 키웠다. 백로그·SIGKILL·hang은 전부 downstream이다. 냉각은 §MI50 쿨링으로 잡고, **이 절은 폭주를 잡는다** — 폭주가 사라지면 지속 부하도, 연쇄도 통째로 사라진다. 여태 진단이 안 된 유일한 이유는 **측정 환경이 매번 오염**(백로그/hang)됐기 때문 — 그래서 방법론의 절반은 "깨끗한 측정 확보"다.

**관통 원칙 (어기면 또 헤맨다):**
1. **한 번에 한 변수** — 폭주 살아있는 채로 동시성·벤치를 건드리지 않는다.
2. **매 측정 전 큐 비우기** — 백로그 뒤에 밀린 측정은 무효.
3. **GPU 작업 중 kill/restart 금지** — 드레인하거나 기다린다(hang 방아쇠 제거, §4).
4. **모든 진단 요청은 bound** — `max_tokens` 캡 + `--max-time`로 프로브 자체가 hang 안 되게.
5. **레이어 이진탐색** — 모델(curl 직타) ↔ 파이프라인(plugin)을 갈라 어느 층인지 특정한다.

**Phase 0 — 깨끗·안전한 베이스라인.** §복구 런북 그대로(콜드 부팅 → 쿨링 → fresh 기동 → `/health`=ok → `TimeoutStopSec=300`). 진단 중엔 **동시성 1**(한 요청씩).

**Phase 1 — 폭주 격리 (핵심).** 먼저 **배포 drift 확인**(문서 검증 땐 terse였는데 지금 폭주 → 뭔가 바뀐 게 1순위 용의자): 실행 중 ExecStart에 `--reasoning-budget 0` 유무(**사고 후 확인: 없음, Option B 일치**), 설치된 plugin이 `think` 옵션 든 **신버전**인지(옛버전은 `enable_thinking` kwarg 미전송 — 아직 미확인: `docker exec scanlation-server sh -c "grep -rn enable_thinking /plugins/scanlation_llama_cpp/plugin.py"`). 그 다음 **이진탐색 2측정:**
- **A (모델 층)** — [tools/diag_runaway.py](diag_runaway.py)로 llama-server 직타(플러그인과 동일 body를 조립, `--body`로 캡처한 body 재생도 됨). **미완성 응답도 내용이 보인다**: ① `max_tokens` 캡이 "안 끝나는 요청"을 "잘렸지만 완료"(`finish_reason:length`)로 바꿔 부분 출력을 통째로 반환하고, ② `stream:true`라 생성되는 토큰이 실시간 출력된다(중단돼도 받은 만큼 남음). 스크립트가 내용물을 생각(`reasoning_content`)/반복(문자열 루프)/schema 미적용(content가 JSON 아님)으로 분류해 준다.
- **B (파이프라인 층)** — `SCANLATION_LOG_LEVEL=DEBUG` + `run_report`로 **이미지 1장**(동시성1) → 플러그인이 실제 보내는 body + llama 로그 `n_decoded`.

| A(모델) | B(파이프라인) | 결론 |
|---|---|---|
| terse | 폭주 | **플러그인 문제** — kwarg/schema 미전송 → 2a |
| 폭주 | 폭주 | **모델/샘플링 층** — 내용물로 가름: reasoning→2b, 반복→2c |

A가 폭주면 내용물로 다시 가른다: `reasoning_content`가 크면 **생각**(→2b), content 안에서 같은 구절이 돌면 **반복**(→2c), content가 스키마 JSON 형태가 아니면 **schema 미적용**(→2a).

**Phase 2 — 층에 맞게 고침.**
- **2a 플러그인**: 신버전 재설치(`think`=`enable_thinking:false` 명시 전송) → `docker restart`(§배포 검증).
- **2b 모델/템플릿**: `--reasoning-budget 0` 하드 캡(단 /admin 토글 죽음 = Option A 수용).
- **2c 반복 루프**: repeat/presence/frequency penalty(또는 DRY 샘플러)를 플러그인 옵션으로 노출(env 기본 + `/admin` — 하드코딩 금지). 과한 값은 grammar의 구조 토큰(따옴표·쉼표)까지 벌점을 먹여 JSON을 깨니 보수적으로.
- **종료 조건**(공통): 실제 파이프라인 이미지 1장에서 **llama 로그 `n_decoded`가 수십대** = 폭주 죽음 확인.

**Phase 3 — 폭주 잡힌 뒤에야 벤치.** 요청이 짧아지면 백로그·SIGKILL 위험이 사라진다 → llama `--parallel 4` + /admin 동시성 4 + `run_report --concurrency 1/2/4`, 부하 중 `rocm-smi --showtemp --showclocks`로 열·클럭 관찰(89 vs 33 t/s 의문도 여기서 갈림).

## 남은 일 (TODO)

> 즉시 재개는 위 §복구 런북을 따른다. 아래는 그 외 남은 항목.

1. **스캔레이션 서버 연결.** 플러그인 설치·엔드포인트 배선은 됨(§통합 시도 1·2). 남은 건 **폭주 원인 확정**(§복구 런북 4) — 그게 풀려야 안정 운영. (plugin: [scanlation-llama-cpp](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py), OpenAI 호환 `/v1/chat/completions`, 모델 `unsloth/gemma-4-26B-A4B-it-qat-GGUF`.)
2. ~~**벤치.**~~ **완료 (2026-07-15)** — 이전 GPU 대비 translate **1.62x**(§3 파이프라인 벤치). 기준 `run_report_20260710_111941.md`(ollama, 이전 GPU) vs `run_report_mi50_translate.md`(llama.cpp, MI50). 참고: 20260710은 CPU가 아니라 **이전 GPU** 실행이었다.
3. ~~**영속화.**~~ **완료 — 배포·enable 확인 (2026-07-15).** 콜드 부팅에서 유닛 자동 기동 실측(`Started llama.cpp.service`). 배포 유닛은 Option B대로 budget 플래그 없음, `-c 16384 --parallel 4`. 예시 파일([deploy/llama.cpp.service.example](../../../deploy/llama.cpp.service.example))을 배포본과 동기화 + 전력 캡 `ExecStartPre` + 전역 토큰 캡 `--n-predict 1024` 추가 — **서버 유닛에 반영은 남음.**
4. ~~**MI50 디바이스 핀 확정.**~~ **결정 (2026-07-15): 핀 없이 자동.** `GGML_VK_VISIBLE_DEVICES=0`이 오히려 lavapipe/iGPU를 잡아 10x 느렸다(§3 함정) — 기본 자동선택이 discrete(MI50)를 고른다.
5. **최종 토폴로지(9060 XT 재장착 후).** detect=CPU / recognize=9060 XT / **translate=MI50** 물리 병렬. translate는 파이프라인상 이미 gate 밖이라 배포만으로 병렬 활성(코드 변경 불필요, [recognize-gpu-speed.md](recognize-gpu-speed.md) 참조).
6. **하드코딩 회피 점검.** 엔드포인트·모델·포트 등 조절값은 env 기본 + `/admin` 노출 원칙을 따른다(신규 값 생기면).
7. **폭주 방어선.** 전역 캡은 유닛 `--n-predict 1024`로 적용(§폭주 격리 — plugin 상태와 무관하게 작동). 요청 단위 세밀 제어가 필요해지면 plugin이 `max_tokens`를 싣도록 + env 기본 + `/admin` 노출(하드코딩 금지). 값은 `n_ctx_slot`보다 낮게(1024급). **폭주 근본 원인 확정·수정이 우선**이고 이건 보조 방어선.
8. **전력 캡 적용·실측.** `rocm-smi -d 0 --setpoweroverdrive 150` 후 decode tok/s 실측으로 값 확정(§MI50 쿨링 — 대역폭 바운드라 손실 작을 것) → 유닛 `ExecStartPre`로 영속화. 부수 효과: 발열·소음·8핀 데이지체인 부담 감소.

## 관련

- recognize 쪽 GPU 속도(9060 XT, PaddleOCR-VL) — [recognize-gpu-speed.md](recognize-gpu-speed.md)
- GPU 역할 분리 설계 — [SCANLATION_DESIGN.md](../../../SCANLATION_DESIGN.md)
- llama.cpp translator plugin(OpenAI 호환) — [scanlation-llama-cpp/plugin.py](../../scanlation-llama-cpp/scanlation_llama_cpp/plugin.py)
- MI50/gfx906 hang·reset 외부 근거 — [ROCm `rocm_smi_lib` #85 (gpureset이 항상 안 됨, 리부트 필요)](https://github.com/ROCm/rocm_smi_lib/issues/85), [FreeBSD MI50 ring timeout + GPU reset 스레드](https://forums.freebsd.org/threads/opencl-crashes-the-gpu-amd-mi50.102074/)
- MI50 온보드 부저(과열·전원 경보) 보고 — [Level1Techs: MI50 beeping constantly](https://forum.level1techs.com/t/amd-instinct-mi50-beeping-constantly/232359)
- gfx906 전용 llama.cpp **성능** 포크(HIP 기반, D=128 모델용) — [eslowney/llama.cpp-gfx906](https://github.com/eslowney/llama.cpp-gfx906) ※ 안정성/hang은 안 다룸. HIP라 Vulkan 회귀 리스크 → **안정화 뒤 속도 실험 후보**로만.

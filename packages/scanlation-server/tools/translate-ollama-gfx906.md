# translate 백엔드 — ollama를 gfx906에 올려보고 되돌린 기록

작성 2026-07-26. **결론: 기각. translate는 llama.cpp Vulkan을 유지한다.** ollama는 gfx906에서 빌드해 돌리는 것까지 성공했지만, **두 번째 GPU에서 도는 recognize를 죽인다** — 파이프라인이 양쪽 카드를 동시에 쓰면 9060 XT가 `vk::DeviceLostError`로 넘어간다(§기각 사유). ROCm이 gfx906에서 되는지에 대한 규명은 [translate-gpu-mi50-rocm.md](translate-gpu-mi50-rocm.md), MI50 도입·폭주·냉각 전반은 [translate-gpu-mi50.md](translate-gpu-mi50.md).

이 문서를 남기는 이유: **ollama를 gfx906에서 빌드·구동하는 방법은 유효한 자산**이고(공식 이미지는 여전히 안 된다), 재시도할 사람이 같은 벽을 다시 만나지 않게 하기 위해서다.

## 왜 옮기려 했나 — 성능이 아니라 운영 기능

성능만 보면 옮길 이유가 없었다. 같은 모델·같은 조건에서:

| 백엔드 | decode |
|---|---|
| llama.cpp Vulkan (직전 프로덕션) | 89.3 t/s |
| llama.cpp HIP | 88.1 t/s |
| **ollama (ROCm/gfx906)** | **83.4 t/s** (**-6.6%**) |

옮긴 이유는 **llama-server가 구조적으로 못 주는 것** 둘이다.

- **idle 언로드.** translator는 서버 프로세스 밖의 HTTP 백엔드라 [registry.py](../app/registry.py)의 `idle_candidates`가 구조적으로 배제한다(`LOCAL_ROLES = ("detector", "recognizer")`). 즉 `model_idle_unload_minutes`는 torch 엔진 전용이고, translate LLM의 VRAM은 **서버가 회수할 수단이 없다.** systemd로 상주하는 llama-server는 MI50의 ~16GB를 24/7 붙잡는다. ollama는 `OLLAMA_KEEP_ALIVE`로 이 일을 대신한다.
- **모델 스왑.** llama-server는 런치 시 `-hf`로 한 모델만 올리고 요청의 `model` 필드를 무시한다 — `/admin`의 모델 드롭다운이 사실상 장식이었다. ollama는 `/api/tags`가 설치된 전 모델을 반환하고 요청의 `model`이 실제로 스왑을 일으킨다.

**단, 첫 번째 동기는 사후에 무너졌다** — idle 언로드는 **socket activation으로 llama.cpp에서도 얻을 수 있다**(§recognize에서 실증). 프로세스째 내려가니 오히려 상위집합이다. 그리고 MI50는 카드 특성상 어차피 D3에 못 가므로, translate 쪽에서 얻을 수 있었던 건 VRAM 회수뿐이었고 그 값이 아래 리스크를 정당화하지 못했다.

## 기각 사유 — ollama가 옆 카드의 recognize를 죽인다

증상: 파이프라인을 연속으로 돌리면 **9060 XT가 커널 레벨에서 리셋**되고 recognize llama-server가 죽는다(`Restart=on-failure`로 재기동되지만 그 요청은 실패). 하루에 26회 관측.

```
amdgpu 0000:09:00.0: ring comp_1.1.0 timeout, signaled seq=2113, emitted seq=2115
amdgpu: reset compute queue (1:1:0)
[drm] device wedged, but recovered through reset
→ llama-server: terminate called after throwing 'vk::DeviceLostError'
```

변수를 하나씩 갈라 얻은 표. **ollama와 파이프라인이 함께 돌 때만 터진다**:

| 구성 | 결과 |
|---|---|
| ollama 실행 + recognize만(`--no-translate`) | ✅ 3/3, DeviceLost 0 |
| ollama 정지 + 전체 파이프라인 | ✅ recognize 정상(translate만 백엔드 부재로 실패), DeviceLost 0 |
| **ollama 실행 + 전체 파이프라인** | ❌ DeviceLost |
| **llama.cpp translate + 전체 파이프라인** | ✅ **5/5, DeviceLost 0, 0.437 p/s** (기준선 0.43 p/s 재현) |

**env 핀으로 막지 못한다.** 두 네임스페이스를 모두 MI50로 묶어도(`ROCR_VISIBLE_DEVICES=<UUID>` + `GGML_VK_VISIBLE_DEVICES=1`) 터졌고, ollama를 **Vulkan 백엔드로 돌려 양쪽 카드를 모두 Vulkan으로 통일해도** 터졌다(`OLLAMA_LLM_LIBRARY=vulkan`). ollama가 모델을 올리는 카드는 로그상 MI50 하나였다(`using device Vulkan0 … 0000:03:00.0`, `offloaded 31/31`) — 그런데도 옆 카드가 죽는다. 주기적 GPU 재탐색처럼 **ollama가 자기 것이 아닌 카드를 계속 건드리는 동작**이 의심되지만, env로 닿는 범위를 넘었다.

**재시도한다면** 컨테이너로 **디바이스 자체를 격리**하는 방향이다 — MI50의 render node만 통과시키면 ollama가 9060 XT를 물리적으로 볼 수 없다. 공식 이미지는 gfx906에서 못 쓰므로 우리 빌드를 담은 이미지를 만들어야 한다.

## 공식 이미지는 여전히 안 된다 — 직접 빌드해야 했다

호스트에 `ollama/ollama:rocm`(0.20.7) 컨테이너가 이미 돌고 있었고, GPU 탐지까지는 된다:

```
inference compute library=ROCm compute=gfx906 name=ROCm0 total="32.0 GiB"
```

그런데 모델을 올리면 러너가 죽는다 — `llama runner terminated, exit status 2`. [translate-gpu-mi50.md](translate-gpu-mi50.md)가 기록한 `invalid device function`과 같은 벽이고, 번들 `libggml-hip.so`에 gfx906 코드오브젝트가 없어서다. 호스트 rocBLAS를 컨테이너에 마운트하는 우회는 그 문서가 이미 막다른 길로 판정했다(벽은 rocBLAS가 아니라 ggml의 컴파일된 커널이다).

**빌드**(Go 1.26 + cmake + hipcc 7.1 필요):

```bash
git clone --depth 1 https://github.com/ollama/ollama.git /opt/ollama-src
cd /opt/ollama-src
cmake -B build . -DOLLAMA_LLAMA_BACKENDS="vulkan;rocm_v7_2" -DCMAKE_HIP_ARCHITECTURES=gfx906
cmake --build build --parallel 8
go build -o /tmp/ollama-bin .
cmake --install build --prefix /opt/ollama-dist
```

- `OLLAMA_LLAMA_BACKENDS`는 세미콜론 목록이라 **Vulkan과 ROCm을 함께** 넣을 수 있다. `rocm_v7_2`가 호스트 rocBLAS 7.2.0과 맞고, `vulkan`은 이 이미지에 없어서 못 썼던 `libggml-vulkan.so`를 만들어 준다 — 나중에 백엔드를 바꿔볼 여지를 남긴다.
- 검증은 **바이너리에 gfx906이 들어갔는지**로 한다(이게 과거 실패의 전부였다): `strings build/llama-server-rocm_v7_2/bin/libggml-hip.so | grep -oE 'gfx[0-9]+' | sort -u` → `gfx906`.
- 결과: `/opt/ollama-dist/lib/ollama/{vulkan,rocm_v7_2}/` 두 백엔드 + CPU 변종들.

**우리 빌드는 두 카드를 각각 맞는 API로 잡고 iGPU는 스스로 배제한다:**

```
inference compute id=0 library=ROCm   compute=gfx906  libdirs=ollama,rocm_v7_2  total="32.0 GiB"   ← MI50
inference compute id=1 library=Vulkan (RADV GFX1200)  libdirs=ollama,vulkan     total="15.9 GiB"   ← 9060 XT
dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1                                      ← iGPU
```

## 모델은 그대로 옮긴다 — quant를 바꾸지 않는다

ollama 저장소에 남아 있던 옛 quant(`VladimirGav/gemma4-26b-16GB-VRAM`)는 **-26% 느리다**([translate-gpu-mi50.md](translate-gpu-mi50.md) §모델 A/B). 그래서 프로덕션과 **같은 GGUF 파일**을 Modelfile로 들인다:

```bash
printf 'FROM %s\n' /root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/*/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf > /tmp/Modelfile
ollama create gemma-4-26b-a4b-qat -f /tmp/Modelfile   # 34초, 13.27GB
```

- **커뮤니티 업로드 ollama blob은 평범한 GGUF다** — 매직 바이트가 `GGUF`이고 `-m`으로 llama.cpp가 바로 읽는다. [translate-gpu-mi50.md](translate-gpu-mi50.md) §GGUF의 로드 실패 사례는 ollama **공식** 신형 레이아웃 모델에 해당하며, 사용자 업로드에는 적용되지 않는다.
- **ollama의 기본 컨텍스트는 위험하게 크다** — 우리 카드에서 `n_ctx = 262144`(256K)를 잡아 VRAM 32GB를 꽉 채웠다. 플러그인이 `num_ctx`(기본 2048)를 명시로 보내므로 실사용에선 문제없지만, 수동 테스트 때는 직접 지정해야 한다. 참고로 **decode 속도는 컨텍스트와 무관했다**(2048/16384/262144 모두 82.9~83.6 t/s) — llama.cpp Vulkan에서 관측된 "`-c`가 크면 느려진다"는 병리가 ROCm 경로에선 재현되지 않았다.

## 폭주 방어선을 다시 세운다 (가장 중요)

llama.cpp에서 ollama로 오면 **방어선 두 겹이 사라진다.** 과거 GPU hang 사고의 원인이 정확히 이 부재였으므로 대체가 필수다.

| 잃는 것 | 대체 |
|---|---|
| 플러그인 `dry_multiplier` 0.8 (DRY, 품질 보존 우수) | ollama에 DRY가 없다 → **`frequency_penalty`** (반복 횟수에 비례해 깎임). ⚠ **기본값 0 = 방어 꺼짐**이라 반드시 명시해야 한다 |
| 유닛 `--n-predict 1024` (플러그인 상태와 무관한 서버측 캡) | ollama 데몬엔 대응 플래그가 없다 → 플러그인에 **`num_predict`(기본 1024)** 를 추가했다 |

실증: 과거 사고를 일으킨 그 텍스트를 `frequency_penalty 1.0` + `num_predict 1024`로 돌렸다.

| 원문 | 결과 |
|---|---|
| `イ．．．くぅぅぅ〜〜〜んっっ` | `아... 가버렷, 응으으윽~~~!` — **루프 없이 종료** |

## 배선 — 코드 변경 0

엔진 발견이 entry_points 기반이라 코어에 하드코딩된 엔진 맵이 없다([registry.py](../app/registry.py)). 그래서 전환에 필요한 것은:

1. **엔드포인트** — `OLLAMA_ENDPOINT`가 [docker-compose.yml](../../../docker-compose.yml)에 **이미 배선돼 있다**(수정 불필요).
2. **`/admin`** — 플러그인 설치 → translator = Ollama → 모델 선택. 옵션은 **role-scoped 키**(`translator:Ollama`)로 저장되므로, 옛 포맷(`Ollama`)에 남은 값은 무시된다 — 전환 시 옵션을 다시 넣어야 한다.
3. **되돌리기는 무손실** — `translator:llama.cpp` 옵션이 state에 그대로 남는다.

**systemd 유닛** — 깨진 공식 컨테이너는 `docker stop`으로 내린다(`restart: unless-stopped`라 재부팅 후에도 안 올라온다):

```ini
[Service]
Environment="HOME=/root"                                  # systemd는 $HOME을 안 준다 -> ollama가 즉시 죽는다
Environment="ROCR_VISIBLE_DEVICES=GPU-3b3210a17337ec1b"   # MI50를 UUID로 (인덱스는 흔들린다)
Environment="OLLAMA_MODELS=/opt/ollama/models"            # 옛 컨테이너와 같은 저장소 재사용
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=5m"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=4"                       # /admin의 translate 동시성과 같은 값으로
ExecStart=/opt/ollama-dist/bin/ollama serve
```

recognize는 별 네임스페이스(`GGML_VK_VISIBLE_DEVICES`)로 9060 XT에 핀돼 있어 **영향받지 않는다** — 다른 env·다른 포트(8090)·다른 role 키다.

## ollama의 idle 언로드는 작동했다 (그러나 대체 가능했다)

```
t+280s  loaded_models=1  vram=15.5GB
t+300s  loaded_models=0  vram=0.01GB    ← KEEP_ALIVE 만료
```

기능 자체는 실측으로 확인됐다. **다만 systemd socket activation이 같은 일을 더 확실히 한다** — 모델만 내리는 게 아니라 프로세스째 내려가므로 상위집합이고, ollama 없이 llama.cpp에 그대로 걸 수 있다(§아래). 그래서 이 기능은 ollama를 유지할 근거가 되지 못했다.

## recognize도 유휴에 놓게 한다 — socket activation

[recognize-decode-bound.md §5](recognize-decode-bound.md)가 llama.cpp 교체의 대가로 이미 지목한 항목이다: "llama-server는 프로세스 수명 동안 모델을 붙들어 9060 XT가 D3hot으로 못 내려가고 상시 ~15W를 먹는다 — 회수하려면 TTL 프록시가 필요하다(콜드스타트 ~1.9초라 싸다)." **그 처방을 실행한 기록**이 이 절이다. recognize를 ollama로 옮겨 해결하는 길은 막혀 있다(§아래).

llama-swap 대신 **systemd socket activation**을 골랐다 — recognize는 모델이 하나뿐이라 llama-swap의 강점(모델 스왑)을 쓸 데가 없고, 남는 이득인 health-wait는 `ExecStartPost` 한 줄로 대체되므로 새 의존성을 유지할 값을 하지 못한다. 유닛 셋:

```ini
# …-PaddleOCR-VL-For-Manga.socket   — 공개 포트를 항상 열어둔다(요청이 connection-refused를 안 본다)
[Socket]
ListenStream=0.0.0.0:8090
Service=llama.cpp-PaddleOCR-VL-For-Manga-proxy.service   # ← 필수

# …-proxy.service — 유휴에 스스로 빠지고, 그러면 아래 서버도 "불필요"가 된다
[Unit]
Requires=llama.cpp-PaddleOCR-VL-For-Manga.service
After=llama.cpp-PaddleOCR-VL-For-Manga.service
[Service]
ExecStart=/usr/lib/systemd/systemd-socket-proxyd --exit-idle-time=5min 127.0.0.1:8091

# …-PaddleOCR-VL-For-Manga.service — 실제 서버는 8091로 내려간다
[Unit]
StopWhenUnneeded=yes
[Service]
ExecStart=… --host 127.0.0.1 --port 8091
ExecStartPost=/bin/sh -c 'for i in $(seq 1 120); do curl -sf -o /dev/null http://127.0.0.1:8091/health && exit 0; sleep 1; done; exit 1'
```

- **`Service=`를 빼면 안 된다.** socket은 기본적으로 **같은 이름의 service**를 활성화하므로, 그러면 프록시를 건너뛰고 실제 서버가 직접 깨어난다 — 그리고 아무도 그걸 `Requires`하지 않으니 `StopWhenUnneeded`가 즉시 죽여서 start-limit까지 간다(실제로 밟았다).
- **`ExecStartPost`의 health 폴링이 llama-server의 socket-activation 미지원을 메운다.** llama-server는 전달된 fd를 안 받고 자기가 bind하므로 프록시가 먼저 붙으면 connection refused가 난다. 준비될 때까지 systemd가 "시작 완료"로 안 보게 막는다.
- 공개 포트가 8090으로 유지되므로 **플러그인 설정(`LLAMACPP_RECOGNIZE_ENDPOINT`)은 그대로**다.
- `systemctl disable` 실제 서버 + `systemctl enable` socket — 부팅 시엔 socket만 뜬다.

**실측**: 유휴 3.74GB → **0.06GB**, 콜드 스타트 **2.0초**(모델이 498MB Q8_0 + mmproj라 가볍다), 파이프라인 정상(recognize 479ms).

## 최종 절전 상태

| | idle 언로드 | PCI D3 |
|---|---|---|
| **MI50** (translate) | ❌ **적용 안 함** — llama-server 상주, ~16GB 점유 | **불가** — 카드 특성상 런타임 PM이 안 된다(`power/control=on`), 그래서 VRAM만 회수해도 절전 이득이 없다 |
| **9060 XT** (recognize) | ✅ 프록시 유휴 종료 → 3.74GB → 0.06GB | ✅ `ctrl=auto`, **D3cold 실측** |

**translate에 언로드를 걸지 않는 이유**: MI50가 D3에 못 가므로 VRAM을 놓아도 전력·발열이 그대로다. 유휴 후 첫 요청에 모델 재로드(~5초)만 새로 생긴다 — 얻는 것 없이 비용만 붙는다. 절전이 실이득인 쪽은 **D3가 되는 9060 XT**이고, 그쪽엔 적용했다.

## 함정 둘 (이번에 밟은 것)

**① `/plugins` 볼륨의 구버전 코드가 `git pull`을 무력화한다.** 컨테이너를 재빌드해도 플러그인은 `/plugins` 볼륨에서 임포트되므로 **옛 코드가 계속 살아 있다.** 이번엔 그 때문에 recognize가 계속 실패했고, 증상이 원인을 가렸다 — 워커가 HTTP 에러를 던지려다 `TypeError: HTTPStatusError.__init__() missing 2 required keyword-only arguments`로 죽어 `BrokenProcessPool`이 됐고, 진짜 에러는 안 보였다. **판별**: 설치본과 리포지토리의 해시를 비교한다.

```bash
docker exec scanlation-server md5sum /plugins/scanlation_llama_cpp/recognizer.py
md5sum packages/scanlation-llama-cpp/scanlation_llama_cpp/recognizer.py
```

**해결**: `POST /install_plugins/ {"plugins":{...},"force":true}` → **서버 재시작**(임포트된 모듈은 재시작 없이 안 바뀐다).

**② 상관관계를 인과로 읽지 않는다 — D3cold는 무죄였다.** ①의 증상을 처음 봤을 때 마침 9060 XT가 D3cold였고(`power/control=auto`라 유휴 시 거기까지 내려간다), "잠든 카드가 상주 서버의 Vulkan 컨텍스트를 깨뜨렸다"고 진단했다. **실측으로 반증됐다** — `server=active` + `card=D3cold` 상태에 실제 인식 요청을 넣으니 카드가 깨어나 정상 처리되고(recognize 1418ms) 서버는 살아남았다. 그때 저널에 있던 core-dump는 요청 때문이 아니라 **유닛을 편집하고 `daemon-reload`한 창에서 난 재기동 실패**였다. 원인은 ①뿐이었다.

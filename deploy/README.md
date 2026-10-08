# 호스트 배포 — 유닛과 모델 레이아웃

컨테이너 밖(호스트)에서 도는 것들의 설정 예시. 서버 컨테이너 자체는 [docker-compose.yml](../docker-compose.yml), 성능 결론은 [BENCHMARKS.md](../BENCHMARKS.md).

## 유닛

| 파일 | 역할 | enable |
|---|---|---|
| `llama.cpp-gemma-4-26B-A4B.service.example` | translate llama-server (:8080, MI50) | ✅ |
| `llama.cpp-PaddleOCR-VL-For-Manga.socket.example` | recognize 진입점 (:8090) | ✅ **이것만** |
| `llama.cpp-PaddleOCR-VL-For-Manga-proxy.service.example` | 유휴 종료 프록시 | ✗ (socket이 띄움) |
| `llama.cpp-PaddleOCR-VL-For-Manga.service.example` | recognize llama-server (:8091) | ✗ (proxy가 띄움) |
| `llama.cpp-PaddleOCR-VL-For-Manga-watchdog.service.example` | 9060 XT 컴퓨트 링 행 시 llama-server SIGKILL | ✗ (recognize 서버가 띄움) |
| `GPU-powercap.service.example` | 부팅 시 전력 캡 | ✅ |
| `nct6687-load.service.example` | MI50 팬 제어용 `nct6687` 로드(재시도) | ✅ |
| `mi50-fan.service.example` + `mi50-fan.example` | MI50 junction 온도로 팬(`pwm4`) 제어 | ✅ |
| `ollama.service.example` | **미배포** — 기각 기록용 | ✗ |

**MI50 팬**은 세 조각이 다 있어야 동작한다. 하나라도 빠지면 보드 EC가 팬을 되찾아가 **CPU 온도로** 돌린다 — 카드가 시원해도 시끄럽고, 카드가 뜨거워도 안 돈다.

1. [`mi50-fan.example`](mi50-fan.example) → `/usr/local/sbin/mi50-fan`(755), [`mi50-fan.service.example`](mi50-fan.service.example) enable — junction 온도에 매핑한 커브. 유닛이 2번을 `Requires=`/`After=`로 건다
2. [`nct6687-load.service.example`](nct6687-load.service.example) enable — 부팅 초기엔 EC가 응답하지 않아 `systemd-modules-load`로는 **항상 실패**한다(`EC base I/O port unconfigured`). `/etc/modules-load.d/nct6687.conf`는 두지 말 것
3. `/etc/modprobe.d/nct6687.conf`에 `options nct6687 msi_fan_brute_force=1` — 없으면 EC가 1초 안에 `pwm4`를 덮어쓴다

커브 근거와 실측은 `mi50-fan.example` 머리 주석과 [cooling-mi50-fans.md](../packages/scanlation-server/tools/cooling-mi50-fans.md).

`nct6687`은 커널에 없는 out-of-tree 모듈([Fred78290/nct6687d](https://github.com/Fred78290/nct6687d))이라 **커널마다 빌드해야 한다.** DKMS에 등록하지 않았다면 커널을 올리기 전에 새 커널용으로 빌드해 둘 것 — 없으면 1번이 `pwm4`를 못 찾고 EC가 팬을 가져간다.

**lm-sensors `fancontrol`은 쓰지 않는다.** 시작할 때, 종료할 때, 온도를 한 번 못 읽을 때마다 팬을 최대(약 15,000 rpm)로 돌려놓고 나간다. `mi50-fan`은 어떤 경로로도 `MAXPWM`(74, 약 5,000 rpm)을 넘지 않고, 멈출 때 카드가 식어 있으면 최저 duty로 둔다.

**recognize는 온디맨드**다(socket activation): 유휴 5분에 프로세스가 내려가 VRAM을 놓고 카드가 D3cold까지 간다(amdgpu가 이 카드엔 BOCO 런타임 PM을 켠다). 콜드 스타트 ~2초. 세 유닛이 필요한 이유와 함정은 각 파일 주석과 [translate-ollama-gfx906.md](../packages/scanlation-server/tools/translate-ollama-gfx906.md)에 있다.

**translate는 상주**다. amdgpu가 그 카드(MI50)엔 런타임 PM을 안 켜므로(`control=on` — auto는 Vega20의 BACO를 대상에서 뺀다) VRAM을 놓아도 절전 이득이 없고, 첫 요청 재로드(~5초)만 붙는다. **`amdgpu.runpm=1`로 BACO를 강제하지 않는다** — 카드는 잠들지만(잠들 때 `psp gfx command UNLOAD_TA failed (0x117)`) 잠든 뒤 첫 요청에 서버 전체가 멈췄고(내장 그래픽 화면이 초록색으로 굳었다) 전원을 끊어야 살아났다(2026-10-08, 커널 6.12.0-233.el10). 자세한 근거는 [translate-ollama-gfx906.md](../packages/scanlation-server/tools/translate-ollama-gfx906.md) §최종 절전 상태.

## 모델 레이아웃 — `/opt/models` 한 뿌리

```
/opt/models/
├── hf/                       HF_HOME. `-hf`로 받는 것 전부 (translate)
│   └── hub/models--…
└── gguf/                     손으로 넣거나 변환한 GGUF
    └── paddleocr-vl/         recognize (모델 + mmproj)
```

**한 뿌리로 두는 이유**: 백업 제외가 `/opt/models/**` 한 줄로 끝난다. 예전엔 `/opt/llama/hf-cache`·`/opt/llama/models`·`/root/.cache/huggingface`·`/opt/ollama/models`로 흩어져 있어서 timeshift 제외 목록이 실제 상태를 따라가지 못했다(모델이 스냅샷에 들어갔다).

**ollama를 되살린다면** `OLLAMA_MODELS=/opt/models/ollama`로 둔다 — ollama의 저장소는 content-addressed blob이라 GGUF 폴더를 가리킬 수 없어 자기 디렉토리가 필요하다.

## 백업(timeshift) 제외

모델·빌드·캐시는 전부 재생성 가능하므로 스냅샷에서 뺀다. **빼면 안 되는 건 `scanlation_data/_data/data/`** — `state.json`(=`/admin` 설정)과 sqlite 캐시가 거기 있고, 합쳐서 1MB 남짓이다.

```
/opt/models/**
/opt/llama/llama.cpp/**                                   # git clone + cmake로 재생성
/var/lib/docker/volumes/scanlation_plugins/**             # /admin 재설치로 복구
/var/lib/docker/volumes/scanlation_data/_data/hf/**
/var/lib/docker/volumes/scanlation_data/_data/models/**
/var/lib/docker/volumes/scanlation_data/_data/miopen/**
/var/lib/docker/volumes/scanlation_data/_data/torch-kernels/**
```

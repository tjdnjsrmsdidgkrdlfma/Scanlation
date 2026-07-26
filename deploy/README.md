# 호스트 배포 — 유닛과 모델 레이아웃

컨테이너 밖(호스트)에서 도는 것들의 설정 예시. 서버 컨테이너 자체는 [docker-compose.yml](../docker-compose.yml), 성능 결론은 [BENCHMARKS.md](../BENCHMARKS.md).

## 유닛

| 파일 | 역할 | enable |
|---|---|---|
| `llama.cpp-gemma-4-26B-A4B.service.example` | translate llama-server (:8080, MI50) | ✅ |
| `llama.cpp-PaddleOCR-VL-For-Manga.socket.example` | recognize 진입점 (:8090) | ✅ **이것만** |
| `llama.cpp-PaddleOCR-VL-For-Manga-proxy.service.example` | 유휴 종료 프록시 | ✗ (socket이 띄움) |
| `llama.cpp-PaddleOCR-VL-For-Manga.service.example` | recognize llama-server (:8091) | ✗ (proxy가 띄움) |
| `GPU-powercap.service.example` | 부팅 시 전력 캡 | ✅ |
| `MI50-fan-duty.service.example` | 부팅 시 MI50 팬 duty 고정 (fancontrol 전 임시) | ✅ |
| `ollama.service.example` | **미배포** — 기각 기록용 | ✗ |

**recognize는 온디맨드**다(socket activation): 유휴 5분에 프로세스가 내려가 VRAM을 놓고 카드가 D3까지 간다. 콜드 스타트 ~2초. 세 유닛이 필요한 이유와 함정은 각 파일 주석과 [translate-ollama-gfx906.md](../packages/scanlation-server/tools/translate-ollama-gfx906.md)에 있다.

**translate는 상주**다. 그 카드(MI50)는 런타임 PM을 못 하므로 VRAM을 놓아도 절전 이득이 없고, 첫 요청 재로드(~5초)만 붙는다.

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

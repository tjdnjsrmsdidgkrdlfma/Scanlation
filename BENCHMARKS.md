# 벤치마크 종합 — detector · recognizer · translator

흩어진 측정 문서들의 **결론만** 파이프라인 순서로 모은 색인이다. 수치의 경위·반증·재현 커맨드는 각 절이 가리키는 **1차 자료**에 있고, 여기서 반복하지 않는다. 어떤 레버를 이미 재봤는지, 무엇이 채택·기각됐는지를 한 곳에서 확인하는 용도다.

## 지금의 파이프라인 (기준선)

같은 21장 챕터를 [run_report.py](packages/scanlation-server/tools/run_report.py) serial로 돌린 값. **양 llama-server 재시작 직후**(프롬프트 캐시 없음), recognizer 옵션 `box` / 300k. 측정 2026-07-26, 출처 [recognize-decode-bound.md §8](packages/scanlation-server/tools/recognize-decode-bound.md).

| 스테이지 | 엔진 | 디바이스 | 중앙값 | 비중 |
|---|---|---|---|---|
| decode | — | CPU | 11.9ms | 0.6% |
| **detect** | comic-text-and-bubble-detector (RT-DETR-v2) | CPU | 302.1ms | 14.4% |
| **recognize** | PaddleOCR-VL-For-Manga (llama.cpp/Vulkan) | 9060 XT | 502.8ms | 23.9% |
| **translate** | gemma-4-26B-A4B (llama.cpp/Vulkan) | MI50 | 1293.8ms | 61.6% |

> 재확인 (2026-07-26): recognize를 socket activation으로 온디맨드화한 뒤 같은 구성으로 5페이지 serial — **0.437 p/s**(위 0.43과 일치), detect ~295 / recognize 377~682 / translate 958~1392ms. 유휴 후 첫 페이지만 콜드스타트(detect 3072 · recognize 2690ms)를 낸다.
| 페이지 total | | | **2101.6ms** | serial 0.43 p/s · 동시성 2에서 21장 32.8초 |

lockwait / semwait 모두 0 — 서버 측 직렬화는 남아 있지 않다. **다음 레버는 translate(62%), 그 다음이 detect(14.4%)다.**

## 문서 지도

| 문서 | 역할 | 다루는 것 | 상태 |
|---|---|---|---|
| [recognize-cpu-threads.md](packages/scanlation-server/tools/recognize-cpu-threads.md) | recognizer | manga-ocr CPU 스레드/워커 배치 | 종결 |
| [recognize-crop-batching.md](packages/scanlation-server/tools/recognize-crop-batching.md) | recognizer | crop 배치(단일 forward에 N크롭) | 종결(기각) |
| [recognize-gpu-speed.md](packages/scanlation-server/tools/recognize-gpu-speed.md) | recognizer | flash · 해상도 캡 · 멀티워커(W·K) | 종결 |
| [recognize-decode-bound.md](packages/scanlation-server/tools/recognize-decode-bound.md) | recognizer | decode 병목 규명 → llama.cpp 교체 | 현행 기준 |
| [translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md) | translator | MI50 도입·폭주·동시성·서버 설정 | 현행 기준 |
| [translate-gpu-mi50-rocm.md](packages/scanlation-server/tools/translate-gpu-mi50-rocm.md) | translator | gfx906 ROCm 재도전 경로 | 종결(된다·동률) |
| [translate-ollama-gfx906.md](packages/scanlation-server/tools/translate-ollama-gfx906.md) | translator | ollama를 gfx906에 올려보고 되돌린 기록 + recognize socket activation | 종결(기각) |
| [cooling-mi50-fans.md](packages/scanlation-server/tools/cooling-mi50-fans.md) | translator(인프라) | 팬·쉬라우드·fancontrol | 하드웨어 대기 |
| `compare_out/`(로컬) | detector · recognizer | 모델 대결 원본 출력 | gitignore, 개발 PC에만 |

detector 쪽만 커밋된 1차 자료가 없다 — 대결 산출물이 [.gitignore](.gitignore)된 `packages/scanlation-server/compare_out/`(`_compare.md`, `_compare_box.html`, `_compare_crops.md`)에 있어서다. 그래서 아래 detector 절과 recognizer 정확도 표는 **이 문서가 그 결론의 유일한 커밋본**이다.

---

# 1. detector — comic-text-and-bubble-detector

## 결론

| 항목 | 값 |
|---|---|
| 채택 모델 | **`ogkalu/comic-text-and-bubble-detector`** (RT-DETR-v2). 기본값은 [config.py](packages/scanlation-server/app/config.py)의 `default_detector` |
| 이긴 상대 | comic-text-detector(구 기본), kiuyha YOLO26, ogkalu yolov8m, kitsumed seg |
| 확정 튜닝 | `conf=0.6`, `nms_iou=0.6`, `contain_thresh=0.85`, 클래스 `text_bubble`+`text_free` |
| 후처리 | IoU + **IoS**(intersection-over-smaller) dedup — RT-DETR은 NMS-free인데도 중첩·내포 박스를 낸다 |
| 대가 | **축 정렬 박스만** — 회전 quad를 주던 comic-text-detector의 이점을 포기했다 |
| 현재 비용 | CPU 302ms/페이지 = 파이프라인의 **14.4%** |

## 대결 방법

[compare_models.py](packages/scanlation-server/tools/compare_models.py)(연구용 하네스)로 같은 만화 페이지에 여러 HF 검출기를 돌려 `compare_out/<분류>/<이미지>/<모델>.png`로 비교했다. 표본은 **17페이지를 5분류**(노이즈 / 많은 글씨 / 말풍선X / 손글씨 / 평범)로 나눈 것이고, 같은 표본이 아래 recognizer 대결의 크롭 공급원이기도 하다(106 크롭).

**모델별 성격**(눈으로 판정):

- **comic-text-and-bubble-detector** — 텍스트 커버리지 최상, 클래스 인지, GPU에서 빠름. 축 정렬 박스.
- **comic-text-detector** — 텍스트를 다 잡고 회전 quad를 주지만, 노이즈를 과검출하고 인접 영역을 합쳐 버리며 가려진 글자에서 실패.
- **kiuyha YOLO26** — 말풍선이 깨끗할 때만 좋고, 아니면 0 박스.
- **ogkalu yolov8m** — 결함은 없으나 평범.
- **kitsumed seg** — 말풍선 전용이지만 말풍선만큼은 탁월 → **하이브리드 후보**(마스크/인페인트는 kitsumed, 텍스트는 RT-DETR).

## 튜닝에서 확정된 것

- **클래스 필터가 먼저다.** 컨테이너인 `bubble` 박스가 내부 텍스트와 겹쳐 중복처럼 보였다 — 텍스트 클래스만 남긴다.
- **conf 0.3 → 0.6**이 저신뢰 노이즈를 실제 텍스트 손실 없이 잘라냈다.
- **conf로 못 잡는 잔여물이 있다.** 고신뢰 노이즈(예: 노이즈/133914778)는 어떤 conf에서도 살아남는다 — 필요한 건 conf가 아니라 **크기 하한**(min_area/min_side)이다. 아직 없다.

## 남은 레버

- **CPU 302ms를 통째로 없앨 여지** — detect는 1-worker 풀에서 **설계상 직렬**이다(페이지당 1회라 팬아웃할 게 없고, 워커를 늘리면 모델 사본만 는다). translate 다음 순번의 레버가 여기다. 후보는 검출기 풀링 · GPU detect · RT-DETR ONNX Runtime([PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md) Tier 4).
- **크기 하한 옵션** — 고신뢰 노이즈용.
- **kitsumed 하이브리드** — 말풍선 마스크가 필요해지면.

---

# 2. recognizer — PaddleOCR-VL-For-Manga

## 결론

| 항목 | 값 |
|---|---|
| 채택 모델 | **`jzhang533/PaddleOCR-VL-For-Manga`** — 인간 블라인드 채점 **88%**, 약한 분류가 없다 |
| 런타임 | **llama.cpp**(GGUF, Vulkan). transformers 경로도 `/admin`에 폴백으로 남아 있다 |
| 속도 | **4.28 crops/sec**(9060 XT, `box`/300k) — transformers 순차 0.557 대비 7.7x |
| GPU 핀 | **`GGML_VK_VISIBLE_DEVICES` 필수**. `--device`로 걸면 4.4x를 잃는다 |
| 전처리 | `downscale_mode=box`, `max_pixels=300000` |
| 폴백 | manga-ocr — 74%지만 CPU 171ms/crop으로 돌아간다 |

## 정확도 — 인간 블라인드 채점 (106 크롭, 분류별 수용률 %)

같은 17페이지에서 나온 106 크롭을 통합 HTML에서 셀 클릭 투표로 채점했다(= "이 판독은 수용 가능"). 시점 2026-07-04.

| 분류 | 크롭 | manga-ocr | qwen3vl | dots_ocr | PaddleOCR-VL | **PaddleOCR-VL-For-Manga** | mit_48px_ctc | mit_48px |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 노이즈 | 15 | 93 | 60 | 13 | 53 | **93** | 60 | 80 |
| 많은 글씨 | 40 | 65 | 73 | 35 | **88** | **88** | 43 | 70 |
| 말풍선X | 17 | 82 | 53 | 6 | 59 | **88** | 18 | 35 |
| 손글씨 | 27 | 67 | 41 | 22 | 30 | **81** | 22 | 48 |
| 평범 | 7 | 86 | 57 | 71 | 71 | **100** | 86 | 100 |
| **합계** | 106 | 74 | 58 | 26 | 62 | **88** | 39 | 62 |

- **승인은 최고점이 아니라 "약한 분류가 없음"이다.** 모든 분류에서 1위 또는 공동 1위.
- **파인튜닝이 산 것**은 만화 고유의 어려움뿐이다 — 베이스 PaddleOCR-VL 대비 손글씨 **+51**, 노이즈 +40, 말풍선X +29, 평범 +29, **많은 글씨 +0**(베이스가 이미 88%).
- **manga-ocr의 약점은 많은 글씨 65%** — 숫자·영문·기호 계열이고, 어려운 크롭에서 가끔 환각한다. 거꾸로 순수 VLM(qwen3vl 등)은 기호는 잘 읽고 **일본어 본문이 약하다**(qwen 평범 57%).
- **dots_ocr는 탈락**(말풍선X 6% = 17개 중 1개), mit_48px_ctc는 전 분류에서 mit_48px에 뒤졌다.

## 속도 변천사 — 같은 42 크롭 기준

| # | 단계 | crops/sec | 직전 | 누적 | 출처 |
|---|---|---|---|---|---|
| 0 | CPU | ~0.017 | — | **1x** | 플러그인 docstring `~60s/crop`(어림) |
| 1 | GPU, flash OFF | 0.094 | ~5.6x | ~5.6x | [gpu-speed](packages/scanlation-server/tools/recognize-gpu-speed.md) |
| 2 | **+ AOTriton flash attention** | 0.345 | **3.7x** | ~21x | 〃 — env 한 줄 |
| 3 | + 해상도 캡 150k / `pow2` | 0.58 | 1.7x | ~35x | 〃 |
| 4 | + 멀티워커 W4·K2 | ~0.77 | 1.3x | ~46x | 〃 |
| 5 | **llama.cpp로 런타임 교체** | 1.274 | 1.7x | ~76x | [decode-bound §5](packages/scanlation-server/tools/recognize-decode-bound.md) |
| 6 | **+ GPU 핀을 env로** | **5.585** | **4.4x** | **~340x** | 〃 §6 |
| 7 | 현재 — `box` / 300k | 4.277 | 0.77x | **~260x** | 〃 §7 — 절단되던 크롭을 되사는 의도적 후퇴 |

**누적에 `~`가 붙은 이유**: 0번의 `~60s/crop`은 잰 값이 아니라 어림이라 유효숫자가 없다. 배수를 정밀하게 인용해야 하면 실측된 첫 점인 **1번을 기준으로 쓴다 — 6번 59x, 현재 46x.** 크기만 말하면 크롭 하나에 1분 → **1초에 4개**다.

**가장 큰 두 도약(2·6번)은 튜닝이 아니라 잘못된 설정을 고친 것이다.** 캡·멀티워커를 다 짜낸 3·4번의 합이 2.2배인데, 런타임 교체+핀(5·6번)이 7.3배다.

## 레버 판정

| 레버 | 판정 | 근거 |
|---|---|---|
| **AOTriton flash attention** | ✅ 채택 3.7x | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — sdpa가 math 폴백 대신 flash 커널을 탄다. [docker-compose.rocm.yml](docker-compose.rocm.yml)에 반영 |
| **해상도 캡** | ✅ 채택 | `box` 300k. `pow2`는 배율이 2의 거듭제곱뿐이라 예산의 최대 4배를 버린다 — 302k 크롭이 150k 캡에서 75k로 접혀 4줄이 1줄로 잘렸다 |
| **llama.cpp 런타임 교체** | ✅ 채택 10x | per-crop 1745→168ms, VRAM 4.3배 절감, 꼬리 지연 16x. 정확도는 실질 동등(42개 중 35개 동일/표기차, 개선 3 대 악화 4) |
| **GPU 핀을 env로** | ✅ 필수 4.4x | `--device`는 LM 레이어만 제한한다. vision 인코더(mtmd)는 제한을 안 받고 스스로 옆 GPU를 고른다 |
| **멀티워커 W4·K2** | 🔸 transformers 한정 1.5x | K(크로스이미지 오버랩)가 없으면 크롭 천장에 걸려 W만 올려도 1.11x뿐. llama.cpp 구성에선 **미측정** |
| **crop 배치** | ❌ 폐기 | 공정 측정(같은 크롭·캡 대칭·`no_grad`)에서 per-crop보다 **1.3x 느리다**. B≥8에선 출력까지 조용히 깨진다 |
| **양자화(weight-only INT8/INT4)** | ❌ 폐기 | decode가 읽는 건 LM(893MB)뿐이라 weight read가 토큰당 ~2.8ms = 전체의 **~4%**. 다 없애도 3%다 |
| **`torch.compile` / HIP graph** | ❌ 실측 폐기 | 벽을 다 넘어도 1.11x뿐 + 출력 2/8 오독 + shape당 ~11s 컴파일이라 동적 해상도에서 상각 불가 |
| **MI50로 recognize** | ❌ 폐기(torch 한정) | torch rocm7.0 rocBLAS에 gfx906 Tensile 라이브러리가 없어 첫 matmul에서 죽는다. llama.cpp는 gfx906에서 돈다 |
| vLLM / SGLang / FastDeploy / ONNX / OpenVINO | ❌ 조사 단계 배제 | RDNA4 vLLM은 기동 버그 + 커널 부재(커뮤니티 실측 llama.cpp Vulkan이 29% 빠름), FastDeploy AMD 미지원, ONNX export 경로 없음, OpenVINO는 Intel 전용 |

## 왜 튜닝으로는 안 됐나 (진단)

캡을 켠 transformers 스택에서 per-crop의 **~95%가 decode**였고, 그 decode는 대역폭도 연산도 아닌 **호스트 측 오버헤드**였다 — steady/token이 크롭 크기와 무관하게 flat ~64ms인데 그중 weight read 4% + compute <1%. **카드 대역폭의 5%만 쓰고 있었다.** 그래서 torch 안의 레버(양자화·`torch.compile`·더 큰 카드)가 전부 죽고, 답이 런타임 교체가 됐다.

llama.cpp로 옮긴 뒤 구성이 뒤집혀 **이제 vision prefill이 per-crop의 2/3**다.

## CPU 경로 (manga-ocr) — 별도 결론

- **싱글스레드 워커 풀이 이득**: `8w×1t`가 `1w×전코어` 대비 **1.88x**(8C16T 기준).
- **물리 코어가 단위**, 하이퍼스레드는 손해(`8w×1t` > `8w×2t` > `16w×1t`).
- **CPU 핀 고정은 무효**(±2.5% = 노이즈) → 병목이 코어 배치가 아니라 메모리임을 역으로 확정. 최선 ~1.9x가 대역폭 천장이다.

## 남은 일

- **llama.cpp 구성의 동시성 미측정** — W·K 수치는 전부 transformers 시절 값이다.
- **`image_min_pixels`** 등 남은 전처리 노브.
- **llama-swap** — llama-server가 모델을 상시 붙들어 9060 XT가 유휴에도 ~15W다. TTL 프록시로 회수 가능(콜드스타트 ~1.9초).
- **모델 배포가 서버 관리자 몫**이 됐다(GGUF 교체·GPU 선택 = `llama-server` 커맨드라인).

---

# 3. translator — gemma-4-26B-A4B on MI50

## 결론

| 항목 | 값 |
|---|---|
| 백엔드 | **llama.cpp + Vulkan(RADV)**. ROCm도 되지만(§ROCm 재도전) 빠르지 않고, **ollama는 시도 후 기각**했다 — 옆 카드의 recognize를 죽인다([translate-ollama-gfx906.md](packages/scanlation-server/tools/translate-ollama-gfx906.md)) |
| 모델 | **`unsloth/gemma-4-26B-A4B-it-qat-GGUF`** — MoE(active 4B)라 26B인데도 빠르고 32GB에 넉넉, QAT quant |
| 실측 | decode **89.3 t/s**(`--parallel 4` 기준선. 초기 raw 검증 89.88) · 대안 백엔드는 HIP 88.1 / ollama 83.4 |
| idle 언로드 | **적용 안 함** — MI50는 카드 특성상 D3에 못 가므로 VRAM을 놓아도 전력 이득이 없고, 첫 요청 재로드(~5초)만 붙는다. 절전이 실이득인 recognize(9060 XT)에는 socket activation으로 적용했다 |
| 파이프라인 효과 | 이전 GPU(ollama) 대비 translate **1.62x**(1509 → 933ms 평균) |
| thinking | **off가 필수** — 플러그인 `think` 기본 False. 켜면 같은 페이지가 958ms → 26.6초 |
| 폭주 방어 | 플러그인 `dry_multiplier` 0.8(근본) + 유닛 `--n-predict 1024`(백스톱) |
| 전력 캡 | **150W** — decode 88.8~95.0 t/s로 225W와 동일, 비용 ≈ 0 |
| 서버 설정 | `-c 16384 --parallel 4` — 슬롯 비용은 4→8 사이가 절벽(-17%)이고 1~4는 평평하다 |
| GPU 핀 | 두 llama-server를 각각 **`Environment=GGML_VK_VISIBLE_DEVICES=<N>`** 로. `--device`는 쓰지 않는다(mtmd 비전 인코더가 다른 카드로 샌다) |
| 운영 동시성 | **2**(현 냉각 기준) |

**한 줄: gfx906에서 막힌 건 하드웨어가 아니라 런타임 패키징이었다.** ollama·llama.cpp 모두 **그 arch를 타깃으로 직접 빌드하면 ROCm으로 돈다**(`GPU_TARGETS`/`CMAKE_HIP_ARCHITECTURES=gfx906`). 배포 바이너리에 gfx906 코드가 없었을 뿐이고, 시스템 rocBLAS는 EPEL 7.2.0에서도 gfx906 커널 156개를 싣고 있어 되공급도 불필요하다. Vulkan(RADV)은 arch 비의존이라 지금도 유효한 대안이다 — 실제로 recognize는 Vulkan을 쓴다.

## 느렸던 원인은 grammar가 아니라 reasoning이었다

파이프라인 translate가 페이지당 10초를 넘던 시절, 의심은 grammar-constrained JSON + gemma 256k vocab이었다. **스키마 유/무 curl 비교로 기각**됐다(6.0s vs 6.9s, ~1s 차). 실제 범인은 gemma-4가 reasoning 모델이고 thinking이 켜져 있어 "2단어 번역"에도 300토큰씩 따지던 것이다.

- **`strip_think`처럼 생성 뒤 잘라내는 건 속도에 무효** — 생성 자체를 막아야 한다. 그래서 옵션을 제거하고 `think`(=`enable_thinking`) 토글로 옮겼다.
- **서버 `--reasoning-budget 0`은 채택하지 않았다** — 하드 캡이라 per-request `enable_thinking`을 덮어써 `/admin` 토글을 죽인다. 제어는 플러그인 쪽에 둔다(Option B).

## 동시성과 열

**파이프라인 동시성 스윕**(21장 챕터, DRY·150W 캡·케이스 팬 상태):

| 동시성 | wall-clock | 처리량 | translate 평균 | max junction |
|---|---|---|---|---|
| 1 | 42.5s | 0.494 p/s | 1002ms | 85°C |
| **2** | 25.2s | **0.832 p/s (×1.68)** | 1367ms | 92°C |
| 4 | 20.0s | 1.05 p/s (×2.13) | 1694ms | **100°C(crit)** |

**동시성 4는 20초 만에 junction crit에 닿는다** — 현 냉각으론 지속 운용 부적합. 냉각 보강 후 재평가한다.

**공급을 포화시킨 격리 측정**(백로그 무한 공급 가정, [bench_translate_concurrency.py](packages/scanlation-server/tools/bench_translate_concurrency.py))에서는 P4가 P2 대비 **+22%**가 실재한다(1.23 → 1.49 req/s). 수확체감의 절반은 요청당 유니크 프롬프트의 **prefill 고정비**라 슬롯을 늘려도 안 줄어든다.

**서버 설정 비용** — `-c`도 `--parallel`도 필요 최소로 둔다.

- **`-c`는 워크로드가 요구하는 최소로.** MI50 Vulkan(비-FA 경로)은 디코드 스텝마다 어텐션이 **할당된 KV 폭 전체**를 상대해, 시퀀스가 짧아도 `-c`에 비례해 느려진다(16k→32k = **-42%**).
- **`--parallel`도 최소로.** 슬롯 자체가 고정비라 활성이 4개여도 8슬롯이면 느리다. 건강 카드에서 슬롯 수만 스윕한 단일 스트림 decode: **P4 89.3 / P2 88.9 / P1 87.4 / P8 73.7 t/s** — 1~4는 평평하고 **4→8이 절벽(-17%)**이다. `translate_concurrency`가 4이므로 8슬롯은 절반이 놀면서 비용만 낸다. **`--parallel 4` 확정.**

**모델 A/B** — `--parallel 4` 고정, 600토큰 정상상태 decode:

| 모델 | decode |
|---|---|
| **`unsloth/gemma-4-26B-A4B-it-qat` UD-Q4_K_XL (프로덕션)** | **89.3 t/s** |
| 같은 repo의 이전 리비전 blob | 89.3 t/s (차이 없음) |
| `VladimirGav/gemma4-26b-16GB-VRAM` (ollama 시절 quant) | 66.3 t/s (**-26%**) |

**`-hf`는 리비전을 조용히 따라간다** — repo가 같은 파일명·같은 크기로 재업로드하면 캐시에 스냅샷이 둘 생기고 `refs/main`이 새 쪽을 가리킨다. 성능 이상을 볼 땐 blob 해시부터 확인한다.

## 냉각 — 지금의 진짜 블로커

MI50는 패시브 서버 카드라 **능동 공랭이 필수**다. 무냉각 지속 부하가 과열 → PCIe 버스 이탈 → amdgpu hang으로 이어진 실측이 있고, 포화 벤치 연타 뒤엔 **성능 latch**도 관측됐다 — 클럭·PCIe·전력 캡이 전부 정상인데 단일요청 decode가 89~95 → **62~64 t/s로 잠기고 idle로는 안 풀린다**(재부팅으로 즉시 복구). 판별기는 [diag_runaway.py](packages/scanlation-server/tools/diag_runaway.py) 1발이고, 판독은 `--parallel 4` 기준 3단계다: **~90 = 건강 / ~80~85 = 건강+열**(고온 누설전류가 클럭을 깎는 가역 효과) **/ ~62~64 = latch → 재부팅.**

**단, 판독 전에 슬롯 수부터 본다.** 오래 미해결로 남아 벤치를 중단시켰던 "상한 ~74"는 카드가 아니라 **유닛이 `--parallel 8`로 드리프트한 것**이었다 — 콜드 부팅으로도 안 풀렸고 열·클럭·캡·PCIe가 전부 정상이었으며, 슬롯을 4로 되돌리자 89.3으로 복귀했다. 서멀 인터페이스 열화 가설은 기각됐고 벤치는 재개 가능하다.

- **보드는 GPU 온도를 모른다** — SYS_FAN 헤더는 CPU/보드 온도로 돈다. BIOS 팬커브로는 해결 불가, `fancontrol`로 `amdgpu` 온도에 매핑해야 한다.
- **판단 기준은 `junction`(temp2)과 `mem`(temp3)** — `edge`는 느긋하다. Vega20는 HBM2가 먼저 병목되는 경우가 많다.
- 확정 하드웨어: ARCTIC S4028-15K + 3D 프린팅 쉬라우드, **소음 상한 8,000~9,000 rpm**(냉각 설계의 하드 제약). 2팬 vs 3팬은 실물 A/B로 결정 — **팬·쉬라우드 도착 대기 중**이라 실측 태스크가 하드웨어 게이트다.

## ROCm 재도전 — 실험 완료: 된다, 그러나 빠르지는 않다

"gfx906 = ROCm 불가"는 과한 결론이었다. 실측 결과([translate-gpu-mi50-rocm.md](packages/scanlation-server/tools/translate-gpu-mi50-rocm.md)):

- **된다. 패치도 되공급도 필요 없었다** — `-DGGML_HIP=ON -DGPU_TARGETS=gfx906` 빌드로 warmup이 그냥 통과했다. 이 문서가 유력하게 봤던 **`SOLVE_TRI` 오컴파일 가설은 불필요**했고, 과거 "HIP는 segfault"의 원인은 **빌드 타깃이 `gfx1200`이었던 것**으로 보인다(MI50용 코드가 없는 바이너리로 시험한 셈).
- **성능은 동률** — decode 88.1 vs Vulkan 89.3, prefill 412.5 vs 410.3. 기대했던 **prefill 우위는 1.13x에 그쳐** 채택 기준(1.3x) 미달이다.
- **배칭 스케일은 측정 불가** — P2·P4 포화는 62~65°C에서 시작해도 각각 94·96°C에 닿아 중단된다. **현 냉각으로 완주 가능한 건 P1뿐**이고, 그마저 43초 연속 부하로 101°C까지 간다. 냉각 보강이 선행 조건.
- **그래서 백엔드는 Vulkan을 유지한다.** ROCm이 열어준 선택지(ollama)도 시도했지만 옆 카드의 recognize를 죽여 기각했다([translate-ollama-gfx906.md](packages/scanlation-server/tools/translate-ollama-gfx906.md)).

## 남은 일

- **다음 레버가 여기다**(파이프라인의 62%). 2026-07-16 스윕의 열 상한은 그대로지만 **공급이 바뀌었다** — 그때는 "다음 병목은 CPU recognize 직렬화"로 끝났는데 지금은 lockwait이 0이라 translate 슬롯이 처음으로 제대로 채워진다. 같은 동시성에서 더 나올 여지가 있다.
- **translate에 idle 언로드는 두지 않는다** — MI50가 D3에 못 가서 VRAM 회수의 절전 이득이 0이고, 첫 요청 재로드(~5초)만 붙는다. 카드가 바뀌면(D3 되는 GPU) 다시 볼 항목이다.
- 냉각 보강 → 동시성 4 재평가(+22% 회수). **P2조차 포화에서 94°C**라 현 냉각으로 측정 가능한 건 P1뿐이다.
- ~~**recognize를 온디맨드로**~~ **완료** — systemd socket activation(socket + `systemd-socket-proxyd --exit-idle-time=5min` + `StopWhenUnneeded`)으로 8090을 온디맨드화했다. 유휴 시 3.74GB → 0.06GB, 콜드 스타트 2.0초. 공개 포트가 그대로라 플러그인 설정은 안 바뀐다. ※ **recognize를 ollama로 옮기는 건 불가** — ollama 변환기가 `PaddleOCRVLForConditionalGeneration`을 지원하지 않고, 이미 있는 mmproj GGUF를 붙일 Modelfile 지시자도 없다(bare GGUF는 `model does not support multimodal requests`).

---

# 4. 교차 교훈 — 측정 방법론

이 저장소의 측정은 여러 번 틀렸고, 틀린 방식이 반복된다. 새 벤치를 짜기 전에 읽을 것.

1. **플래그 A/B는 OFF 기준선을 따로 잰다.** AOTriton을 "무효"로 판정했던 근거가 **플래그를 켠 채 두 번 돌려 같은 값이 나온 것**이었다(비교 기준이던 이전 baseline도 켜진 값). 가진 숫자가 전부 ON이라 정보가 0이었고, 그 오판으로 이후 측정이 3.7배 느린 상태에서 진행됐다.
2. **프롬프트 캐시를 맞춘다.** llama-server가 데워진 런과 재시작 직후 런은 다른 값을 낸다 — 같은 결론(translate 바운드)이라도 숫자가 20% 틀렸다(1082ms·57% → 1294ms·62%).
3. **스윕끼리 가로질러 비교하지 않는다.** 다른 날·다른 설정의 표를 나란히 놓으면 baseline이 달라진 것을 레버 효과로 읽는다(배치의 2.04x가 stale이 된 경위).
4. **열 상태를 측정 조건에 넣는다.** 연속 실행하면 마지막 패스는 열 손해, 직전 패스가 많은 패스는 캐시 이득 — 서로 반대 방향이라 순서 역전까지 만든다(첫 시도에서 P2가 P4를 이기는 가짜 결과). **P별 개별 실행 + 동일 냉각 시작점(junction 60°C) + 자체 워밍업**이 공정 프로토콜이다.
5. **비교군의 설정을 대칭으로 맞춘다.** 배치 측정은 네 결함을 동시에 안고 있었다 — stale baseline, 한쪽만 캡 적용, 서로 다른 크롭 표본, `no_grad` 누락. 걷어내니 결론이 뒤집혔다.
6. **작은 표본에서 개별 항목으로 튜닝하지 않는다.** 다운스케일 모드 비교에서 1픽셀 차이(281×533 vs 280×532)가 개별 크롭의 성패를 갈랐다. 24개 표본에서 믿을 신호는 **계열의 서열**뿐이다.
7. **하드웨어를 의심하기 전에 설정 드리프트를 확인한다.** 한 주 넘게 벤치를 세워둔 "카드가 ~74 t/s에 잠겼다"는 실제로 유닛의 `--parallel 8`이었다. 이 저장소에서 가장 큰 배수들(flash 3.7x, GPU 핀 4.4x)도 전부 튜닝이 아니라 **잘못 걸린 설정을 고친 것**이다 — 이상한 숫자를 보면 먼저 실행 중인 커맨드라인과 env를 그대로 출력해 본다.
8. **도는 코드가 리포지토리의 코드라고 가정하지 않는다.** 플러그인은 `/plugins` 볼륨에서 임포트되므로 `git pull` + 컨테이너 재빌드로도 **옛 코드가 그대로 살아 있다.** 이번엔 그게 recognize를 계속 실패시켰고, 워커가 에러를 던지려다 죽어 `BrokenProcessPool`이 되면서 진짜 원인까지 가렸다. 판별은 해시 비교(`docker exec … md5sum /plugins/…` vs 리포지토리), 해결은 `force: true` 재설치 **+ 서버 재시작**이다.
9. **측정이 중간에 끊긴 값을 사실로 쓰지 않는다.** "P1은 68°C까지만 오른다"고 판단해 온도 가드를 뺐는데, 그 68°C는 **일찍 잘린 실행의 피크**였다. 완주하면 43초 연속 부하로 101°C까지 간다 — 잘린 런의 최대값은 그 런이 도달할 수 있었던 값이 아니다.

---

# 5. 도구 색인

| 도구 | 무엇을 재나 |
|---|---|
| [run_report.py](packages/scanlation-server/tools/run_report.py) | 파이프라인 end-to-end, 스테이지별 분해(`--parallel` / `--no-translate`) |
| [compare_models.py](packages/scanlation-server/tools/compare_models.py) | 검출·인식 모델 대결 하네스(`ocrbatch`/`consolidate`, 채점 HTML) |
| [bench_recognize_threads.py](packages/scanlation-server/tools/bench_recognize_threads.py) | CPU recognize 워커×스레드 스윕 |
| [bench_recognize_batch.py](packages/scanlation-server/tools/bench_recognize_batch.py) | crop 배치 스윕 |
| [bench_recognize_gpu_concurrency.py](packages/scanlation-server/tools/bench_recognize_gpu_concurrency.py) | GPU 멀티워커·캡·`--profile-decode` |
| [bench_recognize_llamacpp.py](packages/scanlation-server/tools/bench_recognize_llamacpp.py) | llama.cpp recognize per-crop |
| [bench_recognize_compile.py](packages/scanlation-server/tools/bench_recognize_compile.py) | `torch.compile` 프로브 |
| [bench_translate_concurrency.py](packages/scanlation-server/tools/bench_translate_concurrency.py) | 포화 공급 translate 스케일링 |
| [bench_occupancy.py](packages/scanlation-server/tools/bench_occupancy.py) | 워커 풀 점유율 |
| [diag_batch.py](packages/scanlation-server/tools/diag_batch.py) · [diag_runaway.py](packages/scanlation-server/tools/diag_runaway.py) | 배치 공정 비교 · 폭주 관측 |

## 관련 문서

- 남은 최적화 후보의 위험·규모별 분류 — [PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md)
- 설계(역할 분리·플러그인 구조) — [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md)
- 운영·배포 — [README.md](README.md) · [deploy/](deploy/)

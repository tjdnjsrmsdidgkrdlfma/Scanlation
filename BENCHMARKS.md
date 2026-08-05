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
- **conf로 못 잡는 잔여물이 있다.** 고신뢰 노이즈(예: 노이즈/133914778)는 어떤 conf에서도 살아남는다 — 필요한 건 conf가 아니라 **크기 하한**이다. `min_area`·`min_side`로 구현돼 `/admin`에 있고(env `SCANLATION_DETECT_MIN_AREA`/`_MIN_SIDE`), **기본은 0 = 끔**이다. 켤 값은 실사용 박스 분포를 보고 정한다 — 아직 안 정했다.

## 남은 레버

- **torch 안의 CPU 레버는 소진됐다 (2026-08-05 실측)** — 단독 실행 기준 per-page 237ms인데 **forward가 98%**다(전처리 4.4ms + 후처리 0.4ms = 2%). 스레드는 **8이 최적이고 이미 그 값**이며(1→892 / 4→303 / **8→223** / 12→257 / 16→253ms) 12·16이 느려지는 것도 recognize의 "물리 코어가 단위" 결론과 같다. `torch.inference_mode`는 223.9 vs 225.8ms로 **노이즈**다([PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md) Tier 1-C는 이걸로 닫힌다). **입력이 640×640 고정**이라(2400×1800 페이지가 비율 무시하고 리사이즈된다) 해상도 노브도 없다.
- **남은 건 런타임 교체뿐이고, ONNX Runtime이 2.33x다** — 같은 21페이지에서 torch 237.4 → **ORT 102.0ms**([bench_detect_runtime.py](packages/scanlation-server/tools/bench_detect_runtime.py), 전처리 텐서 공유·스레드 8 동일). 모델 저장소가 safetensors 옆에 ONNX를 같이 배포하는데 현재 설치는 그걸 건너뛴다. **INT8은 오히려 느리다**(154.9ms) — 이 크기에선 양자화 오버헤드가 이득을 못 넘는다. **단 출력이 완전히 같지는 않다**: 42 vs 41 박스(21페이지 중 1장에서 ORT가 하나를 놓침), 좌표는 20페이지가 1.5~10.4px에 1페이지만 35.7px. 채택은 **그 차이가 실제 손해인지 눈으로 판정한 뒤**다(`--html`이 두 런타임 박스를 겹쳐 그린 페이지를 낸다).
- **GPU detect** — 아직 미측정. 컨테이너 torch가 보는 카드가 `HIP_VISIBLE_DEVICES=0`(= MI50, translate가 상주하는 카드)이므로 옮기려면 그 배정부터 정리해야 한다.
- **크기 하한 값 정하기** — 옵션(`min_area`·`min_side`)은 있고 기본 0(끔)이다. 실제 노이즈 박스의 크기 분포를 봐야 켤 값이 나온다.
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
| **멀티워커 W4·K2** | 🔸 transformers 한정 1.5x | K(크로스이미지 오버랩)가 없으면 크롭 천장에 걸려 W만 올려도 1.11x뿐 |
| **동시성 (llama.cpp 슬롯)** | ❌ 실측 폐기 — 이득 0 | 슬롯 4개가 균등히 쓰이는데 처리량이 안 늘고(5.76 → 5.53 → 5.26) 요청당 디코드만 3.66 → 51.59 ms/token으로 나뉜다. **c=1에서 이미 카드 포화** |
| **컨텍스트 폭 `-c`** | 🔸 속도 무관 | 슬롯당 32,768 → 2,048(16배)에 변화 없음. `-c 8192`는 속도가 아니라 "워크로드 최소"로 채택(실제 요청 ~250 토큰) |
| **`image_min_pixels` 하향** | ❌ 실측 폐기 — 출력이 바뀐다 | +20~25%(188→64→32 토큰)가 실재하고 min이 반토막인데 max는 그대로다. 그런데 42 크롭 중 **20~21개에서 문자가 달라지고**, 64와 32의 오독 집합이 거의 같아 값을 올려 타협할 수 없다. CLI `--image-min-tokens`는 첫 요청에서 GPU가 죽어 아예 못 쓴다 |
| **crop 배치** | ❌ 폐기 | 공정 측정(같은 크롭·캡 대칭·`no_grad`)에서 per-crop보다 **1.3x 느리다**. B≥8에선 출력까지 조용히 깨진다 |
| **LM 양자화 Q8_0** | ✅ 채택 — BF16 대비 **1.19~1.22x** | per-crop max 315 → 239ms. 출력은 **42개 중 41개가 BF16과 바이트 동일**(남은 1개도 2자 차)이고 VRAM도 417MB 덜 쓴다. 세 후보 중 **유일하게 대가가 없다.** 프로덕션이 2026-07-25부터 이 값이다 |
| **LM을 Q4까지 내리기** | ❌ 실측 폐기 — 이득 **0** | 가중치를 40% 더 줄여도 0.6%(런 간 편차 이하)다 — decode가 Q8_0에서 **이미 대역폭 바운드를 벗어났다.** 그런데 출력은 16/42가 바뀐다 |
| **mmproj 양자화** | ❌ 실측 폐기 — 이득 **0** | 만들어서 재봤다(882→597MB). 순차 3회 평균 5.969 → 6.009 crops/sec로 **+0.7%인데 구간이 겹치고**, 출력은 41/42 동일 — 무해하지만 무익. prefill은 가중치를 한 번 읽고 400토큰을 계산하므로 **읽기를 0으로 만들어도 상한이 2.6%**다. ※ 변환하려면 파인튜닝 저장소의 `vision_config.architectures`를 `SiglipVisionModel` → `PaddleOCRVisionModel`로 고쳐야 한다(mmproj 재생성 경로이기도 하다) |
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

- ~~**`image_min_pixels`**~~ **측정 완료 → 기각 (2026-08-05)** — 속도는 실재했다(+20~25%, mmproj 바닥 147,384 → 50,176 → 25,088). 대가가 크다: 42 크롭 중 프로덕션과 바이트 일치가 12개뿐이고, 표기 흔들림을 정규화해도 **20~21개에서 문자가 다르다.** 64토큰과 32토큰의 오독 집합이 거의 같아 **중간값 타협이 안 된다** — 문제는 얼마나 낮췄나가 아니라 바닥을 건드렸는가다. 되살리려면 106 크롭 인간 채점이 선행돼야 한다. [decode-bound §10](packages/scanlation-server/tools/recognize-decode-bound.md)
- **recognize에서 살 수 있는 속도는 남지 않았다** — 캡·동시성·`-c`·`image_min_pixels`·양자화가 전부 닫혔다(2026-08-05). prefill은 per-crop의 2/3인데 연산이고, decode는 Q8_0에서 이미 대역폭 바운드를 벗어났다. **다음 이득은 recognize 안이 아니라 translate(62%)·detect(14.4%)에 있다.**
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
| 운영 동시성 | **4** — 서버 게이트 `translate_concurrency`. 피크 74°C로 완주하고 포화 천장의 80%를 낸다 |

**한 줄: gfx906에서 막힌 건 하드웨어가 아니라 런타임 패키징이었다.** ollama·llama.cpp 모두 **그 arch를 타깃으로 직접 빌드하면 ROCm으로 돈다**(`GPU_TARGETS`/`CMAKE_HIP_ARCHITECTURES=gfx906`). 배포 바이너리에 gfx906 코드가 없었을 뿐이고, 시스템 rocBLAS는 EPEL 7.2.0에서도 gfx906 커널 156개를 싣고 있어 되공급도 불필요하다. Vulkan(RADV)은 arch 비의존이라 지금도 유효한 대안이다 — 실제로 recognize는 Vulkan을 쓴다.

## 느렸던 원인은 grammar가 아니라 reasoning이었다

파이프라인 translate가 페이지당 10초를 넘던 시절, 의심은 grammar-constrained JSON + gemma 256k vocab이었다. **스키마 유/무 curl 비교로 기각**됐다(6.0s vs 6.9s, ~1s 차). 실제 범인은 gemma-4가 reasoning 모델이고 thinking이 켜져 있어 "2단어 번역"에도 300토큰씩 따지던 것이다.

- **`strip_think`처럼 생성 뒤 잘라내는 건 속도에 무효** — 생성 자체를 막아야 한다. 그래서 옵션을 제거하고 `think`(=`enable_thinking`) 토글로 옮겼다.
- **서버 `--reasoning-budget 0`은 채택하지 않았다** — 하드 캡이라 per-request `enable_thinking`을 덮어써 `/admin` 토글을 죽인다. 제어는 플러그인 쪽에 둔다(Option B).

## 대역폭 지붕 — decode는 peak의 19%다

토큰당 읽는 바이트를 **프로덕션 GGUF의 텐서 합**에서 직접 구한 값(추정 아님, 658 텐서). 128 전문가 중 8개 활성이라 총 14.25GB 중 15%만 읽는다.

| | 값 |
|---|---|
| 토큰당 (dense 1.39GB + 활성 전문가 0.80GB) | **2.19 GB** |
| MI50 peak | 1024 GB/s |
| **이론 천장** | **~468 t/s** |
| 실측 / 실효 대역폭 | 89.3 t/s / ~196 GB/s |
| **peak 대비** | **19%** |

**같은 llama.cpp가 9060 XT에서는 65%를 낸다** — B=1 decode가 현실적으로 닿는 영역이 그쯤이므로 translate는 3배 이상을 흘리고 있다.

**그런데 그 격차는 대역폭 효율이 아니다.** KV(할당 4096 전폭 0.88GB)를 전부 낭비로 치고 대역폭을 이론 peak로 쳐도 메모리는 토큰당 11.2ms 중 **3.0ms(27%)**뿐이다 — 나머지 73%는 메모리를 읽는 시간이 아니다. 소거하면 **연산 바운드 아님**(150W 캡 비용 0)이고 **백엔드 문제도 아니다**(HIP 88.1 ≈ Vulkan 89.3). 남는 건 커널 디스패치·동기화 지연이고, 이 모델이 유독 디스패치 밀도가 높다(레이어당 norm/scale 텐서 10개, dense FFN과 routed expert 병존, hidden 2816). **⚠️ 여기까지 중 "디스패치가 범인"은 가설이다** — 확증은 `rocprof`/`GGML_VULKAN_PERF_LOGGER`로 커널을 직접 세는 것이고 아직 안 했다.

| 레버 | 회수 가능분 |
|---|---|
| `-c` 하향 (실측 컨텍스트 122~347 토큰인데 슬롯당 4096 할당) | 스텝의 **8%** |
| 나머지 73% | 설정으로 불가 |

**translate decode를 완전히 고쳐도 파이프라인 상한은 1.7배**(페이지 2101.6 → ~1240ms)다. 유도·재현은 [translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md).

## 동시성과 열

**파이프라인 동시성 스윕**(21장 챕터, DRY·150W 캡, ARCTIC S4028-15K 맨팬 1개 ~6,500rpm, recognize는 9060 XT. 2026-07-26, 회차별 개별 실행):

| 동시성 | wall-clock | 처리량 | detect+recognize 평균 | translate 평균 | max junction |
|---|---|---|---|---|---|
| 1 | 33.4s | 0.628 p/s | 551ms | 1020ms | 75°C |
| 2 | 24.8s | 0.847 p/s (×1.35) | 623ms | 1592ms | 77°C |
| **4** | 16.0s | **1.309 p/s (×2.08)** | 794ms | 2103ms | 74°C |

**동시성 4가 crit 없이 완주하고, 그게 운영값이다.** 처리량은 translate 포화 천장(1.64 req/s)의 80%고 남은 격차는 prefill 고정비 + d+r 직렬 구간의 몫이다. 이 스윕은 실운영 게이트 그대로를 통과했다(`translate_concurrency` 4; 활성 recognizer(llama.cpp)는 풀 override가 없어 d+r **직렬** — state의 2/4 override는 비활성 torch 엔진 몫) — **실사용도 같은 수준으로 흐르고, 바꿀 운영값이 없다.** 첫 conc2는 0.715로 나왔는데 실행 순서 캐시 편향(교훈 4)이었고, 표준은 웜 재실행 값이다.

**단, 동시성 4는 냉각에 걸려 있다.** 케이스 팬만 쓰던 구성에서는 같은 스윕의 conc4가 20초 만에 junction crit(100°C)에 닿아 스로틀이 개입했고, 그래서 운영값이 2였다. 팬이 빠지거나 duty가 떨어지면 4는 다시 안전하지 않다([cooling-mi50-fans.md](packages/scanlation-server/tools/cooling-mi50-fans.md)).

**공급을 포화시킨 격리 측정**(백로그 무한 공급 가정, [bench_translate_concurrency.py](packages/scanlation-server/tools/bench_translate_concurrency.py))에서는 P4가 P2 대비 **+22%**가 실재한다(1.23 → 1.49 req/s). 수확체감의 절반은 요청당 유니크 프롬프트의 **prefill 고정비**라 슬롯을 늘려도 안 줄어든다. 냉각 1차 보강 후 재실측(2026-07-26, 맨팬 ~6,500rpm)은 **스로틀 없이 P1/P2/P4 완주 — 1.03/1.39/1.64 req/s(피크 81/89/78°C)**. 구 수치는 부분 스로틀 속의 값이었고, 천장은 **1.64 req/s**다.

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

**단, 판독 전에 슬롯 수부터 본다.** 오래 미해결로 남아 벤치를 중단시켰던 "상한 ~74"는 카드가 아니라 **유닛이 `--parallel 8`로 드리프트한 것**이었다 — 콜드 부팅으로도 안 풀렸고 열·클럭·캡·PCIe가 전부 정상이었으며, 슬롯을 4로 되돌리자 89.3으로 복귀했다. 서멀 인터페이스 열화 가설은 기각됐고 벤치는 재개 가능하다. 이 드리프트는 재발한다 — 2026-07-26에도 프로덕션 유닛이 8인 채 돌던 걸 열 벤치의 73.6 t/s가 다시 잡아 유닛과 deploy 예시를 4로 고정했다(89.7 복귀).

- **보드는 GPU 온도를 모른다** — SYS_FAN 헤더는 CPU/보드 온도로 돈다. BIOS 팬커브로는 해결 불가, `fancontrol`로 `amdgpu` 온도에 매핑해야 한다.
- **판단 기준은 `junction`(temp2)과 `mem`(temp3)** — `edge`는 느긋하다. Vega20는 HBM2가 먼저 병목되는 경우가 많다.
- 확정 하드웨어: ARCTIC S4028-15K + 3D 프린팅 쉬라우드, **소음 상한 8,000~9,000 rpm**(냉각 설계의 하드 제약). **당분간 운영은 맨팬 1개**(SYS_FAN2, `nct6687` 드라이버, 30% duty ≈ 6,100rpm 고정, 부팅 영속 `MI50-fan-duty.service`) — 이 상태로 포화 P1/P2/P4·파이프라인 conc4가 완주한다. 고정 duty라 온도 연동이 없지만 **온도 알람(Task 5)은 보류**했다 — 부하가 버스트성이고 유휴 복귀가 빨라(10초에 수십 °C 하강) 고정 duty로 충분하다고 봤다. 검증된 건 16~42초 버스트까지라, **수 분 지속 부하를 돌리게 되면 그때 온도를 함께 본다.** 쉬라우드 A/B·`fancontrol`(Task 2~4)은 장착 시점으로 이월([cooling-mi50-fans.md](packages/scanlation-server/tools/cooling-mi50-fans.md)).

## ROCm 재도전 — 실험 완료: 된다, 그러나 빠르지는 않다

"gfx906 = ROCm 불가"는 과한 결론이었다. 실측 결과([translate-gpu-mi50-rocm.md](packages/scanlation-server/tools/translate-gpu-mi50-rocm.md)):

- **된다. 패치도 되공급도 필요 없었다** — `-DGGML_HIP=ON -DGPU_TARGETS=gfx906` 빌드로 warmup이 그냥 통과했다. 이 문서가 유력하게 봤던 **`SOLVE_TRI` 오컴파일 가설은 불필요**했고, 과거 "HIP는 segfault"의 원인은 **빌드 타깃이 `gfx1200`이었던 것**으로 보인다(MI50용 코드가 없는 바이너리로 시험한 셈).
- **성능은 동률** — decode 88.1 vs Vulkan 89.3, prefill 412.5 vs 410.3. 기대했던 **prefill 우위는 1.13x에 그쳐** 채택 기준(1.3x) 미달이다.
- **배칭 스케일은 당시 측정 불가였다** — P2·P4 포화가 62~65°C에서 시작해도 각각 94·96°C에 닿아 중단됐고, 완주 가능한 건 P1뿐(그마저 43초 연속 부하로 101°C). 냉각 1차 보강 후 완주 실측은 §번역 GPU의 포화 절 참조.
- **그래서 백엔드는 Vulkan을 유지한다.** ROCm이 열어준 선택지(ollama)도 시도했지만 옆 카드의 recognize를 죽여 기각했다([translate-ollama-gfx906.md](packages/scanlation-server/tools/translate-ollama-gfx906.md)).

## 남은 일

- ~~다음 레버(파이프라인의 62%)~~ **실측 완료 (2026-07-26)** — 공급 교체(recognize GPU)의 몫이 파이프라인에서 확인됐다: conc1 0.494 → 0.628, conc4 1.05 → **1.309 p/s**. 남은 격차(천장 1.64의 20%)는 prefill 고정비와 d+r 직렬 구간의 몫이다.
- **`-c` 하향 스윕** — 실측 컨텍스트가 122~347 토큰인데 슬롯당 4096을 잡는다. `-c 8192`는 안전한 드롭인이고, 더 내리려면 `--n-predict`(현 1024, 실제 생성 34~116)를 먼저 낮춘다. **다만 스텝의 8%짜리다** — 대역폭 지붕이 남은 73%는 KV가 아니라고 말한다.
- **디스패치 프로파일 (가설 확증)** — 스텝의 73%가 메모리도 연산도 아닌 시간인데, 범인을 커널 디스패치로 지목한 건 아직 가설이다. `rocprof`/`GGML_VULKAN_PERF_LOGGER`로 토큰당 커널 수와 커널당 시간을 세면 끝난다. **이게 확증돼야 "모델을 바꿔야 하나"가 답이 있는 질문이 된다.**
- **translate에 idle 언로드는 두지 않는다** — MI50가 D3에 못 가서 VRAM 회수의 절전 이득이 0이고, 첫 요청 재로드(~5초)만 붙는다. 카드가 바뀌면(D3 되는 GPU) 다시 볼 항목이다.
- ~~냉각 보강 → 포화·파이프라인 동시성 재평가~~ **완료 (2026-07-26)** — 맨팬 1개(~6,500rpm)로 포화 P1/P2/P4 완주(천장 1.49 → **1.64 req/s**), 파이프라인 conc4 **1.309 p/s**(피크 74°C, crit 해소). 남은 것: 쉬라우드 A/B·`fancontrol`([cooling-mi50-fans.md](packages/scanlation-server/tools/cooling-mi50-fans.md) Task 2~5), **수 분 지속 부하 검증**(지금까지는 16~42초 버스트).
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
9. **계산으로 낸 예측을 결론으로 쓰지 않는다.** Q4의 이득을 "per-crop 15%"로 적어둔 근거는 대역폭 계산이었는데 실측은 **0%**였다. 같은 계산이 Q8_0에 대해서는 맞았다(예측한 폭 그대로 1.22x). 모델이 맞는 구간과 깨지는 구간이 있고 **어디서 깨지는지는 재봐야 안다** — 여기선 Q8_0 지점에서 decode가 대역폭 바운드를 벗어난 게 경계였다.
10. **측정이 중간에 끊긴 값을 사실로 쓰지 않는다.** "P1은 68°C까지만 오른다"고 판단해 온도 가드를 뺐는데, 그 68°C는 **일찍 잘린 실행의 피크**였다. 완주하면 43초 연속 부하로 101°C까지 간다 — 잘린 런의 최대값은 그 런이 도달할 수 있었던 값이 아니다.

---

# 5. 도구 색인

| 도구 | 무엇을 재나 |
|---|---|
| [run_report.py](packages/scanlation-server/tools/run_report.py) | 파이프라인 end-to-end, 스테이지별 분해(`--parallel` / `--no-translate`) |
| [compare_models.py](packages/scanlation-server/tools/compare_models.py) | 검출·인식 모델 대결 하네스(`ocrbatch`/`consolidate`, 채점 HTML) |
| [bench_detect_runtime.py](packages/scanlation-server/tools/bench_detect_runtime.py) | detect 런타임 A/B(torch vs ONNX Runtime) — 속도 + 박스 일치, `--html`로 두 박스를 겹쳐 그린 판정 페이지 |
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

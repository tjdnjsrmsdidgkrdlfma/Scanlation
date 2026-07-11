# recognize crop-batching — 설계 분석 (manga-ocr vs PaddleOCR-VL)

작성 2026-07-07. 설계 분석으로 시작해 같은 날 [bench_recognize_batch.py](bench_recognize_batch.py)로 **실측까지 했고, 실측이 설계 가정을 두 군데 뒤집었다** — 확정된 수치·판단은 아래 **[실측](#실측-2026-07-07)** 절과 갱신된 **[판단](#판단-실측-반영)** 절을 따른다. 이 서두부터 §gpu_lock까지는 실측 전 a-priori 분석으로 남겨둔다(추론 경위).

## 배경 — 왜 이 문서

[recognize-cpu-threads.md](recognize-cpu-threads.md)의 "실용 판단"이, 싱글스레드 워커 풀보다 **나을 수 있는 대안**으로 **crop들을 한 forward에 배치**를 지목했다: weight를 한 번 읽어 N crop에 재사용 → arithmetic intensity를 올려 **대역폭 병목을 직접 깎는다**(그 문서가 확정한 병목이 "반쯤 메모리 바운드"였다). 단 "manga-ocr API가 1장씩이라 배치 지원을 손봐야 함"이라고만 남겼다.

이 문서는 그 대안을 실제로 파고든다:

1. 지금 두 recognize 엔진에서 **배치가 되는가**, 되면 얼마나 깨끗한가.
2. **어떤 함정**이 있고, 그 함정이 common case인가 tail인가.
3. **어디를 손봐야** 하나(통합 지점), 그리고 이게 `gpu_lock` 개편과 얽히나.

**결론 선점 (a-priori)**: 배치의 실질 대상은 **manga-ocr 한쪽**이다 — 고정 해상도·CPU 병목·straggler-면역 인코더 세 박자가 맞아 깔끔한 승리. PaddleOCR-VL은 배치가 correctness 문제는 아니지만 **payoff÷effort가 나쁘다**(동적 해상도 ragged + 스텝 단가 + GPU 기본이라 동기 약함). 그리고 배치는 **`gpu_lock`과 무관한 [pipeline.py](../app/pipeline.py) 국소 리팩터**로 들어간다.

> **실측이 이 선점을 깎았다.** manga-ocr 배치는 되지만 "깔끔한 승리"는 아니다 — 실제 페이지에서 **~1.27x(B=8)**에 그치고, 같은 베이스라인의 멀티워커(~1.8x)에 진다. "straggler-면역 인코더 은행"은 정확히 뒤집혔다: CPU에서 **인코더는 배치가 거의 안 먹히고**(c_8/c_1≈7), amortize되는 건 디코더인데 하필 거기 straggler가 산다. `pipeline.py` 국소 리팩터라는 점만 그대로. 자세히는 아래 [실측](#실측-2026-07-07).

## 두 엔진의 구조 차이 (핵심)

| | manga-ocr | PaddleOCR-VL |
|---|---|---|
| 구조 | `ViTImageProcessor`(고정 224) + `VisionEncoderDecoderModel` | 0.9B autoregressive VLM |
| 입력 해상도 | **고정 224²** → 크롭 전부 동일 텐서 | `smart_resize` **동적**(min 384²~max 1536²) |
| vision 토큰/크롭 | 항상 256 (16×16) | **크롭마다 다름** (~196 ~ ~3000) |
| 배치 텐서 | `(N,C,H,W)` 한 방, **ragged 없음** | ragged → pad + mask + position_ids 필요 |
| 출력 캡 | `max_length=300` | `max_new_tokens=1024`, `do_sample=False` |
| 기본 디바이스 | CPU (배치 원 동기가 여기) | **cuda** (~1s/crop; CPU는 ~60s/crop 사실상 불가) |
| 가중치 | ~400MB | ~1.8GB (bf16) |

- **manga-ocr**: 래퍼 `MangaOcr.__call__`이 1장씩만 노출하지만([manga_ocr/ocr.py](../../../venv/Lib/site-packages/manga_ocr/ocr.py) `__call__`), 밑의 `.processor`/`.model`/`.tokenizer`는 다 접근 가능. `processor([c1..cN])` → `model.generate` → `tokenizer.batch_decode`로 우회하면 끝. `VisionEncoderDecoderModel.generate`는 배치 `pixel_values`를 native 지원.
- **PaddleOCR-VL**: `smart_resize`(Qwen2-VL 계열)가 종횡비 유지 + 픽셀 예산 + patch14·merge2 배수 반올림으로 이미지를 리사이즈 → **크롭마다 vision 토큰 수가 다르다**. 이게 배치의 모든 골칫거리의 뿌리.

## straggler — 둘 다 있다, degree 차이

기계적으로: **stock transformers는 continuous batching을 안 한다.** 한 행이 EOS를 뱉으면 `unfinished_sequences` 마스크로 "끝남" 표시만 하고 **행을 배치에서 빼지 않는다** — 배치가 전부 끝날(또는 max에 닿을) 때까지 매 스텝 forward를 계속 태운다. 즉 디코드 루프의 벽시계 = **배치 내 최장 출력**이 정한다. manga-ocr도 autoregressive라 **똑같이 있다.**

그런데 크기는 이렇게 잡힌다:

- **max cap은 천장일 뿐, 보장이 아니다.** greedy 디코더는 텍스트가 끝나면 EOS로 멈춘다 — 12글자 말풍선은 ~12토큰에서 끝나지 300/1024 근처도 안 간다. 그래서 **정상 배치에선 cap이 안 물린다.** cap 차이(300 vs 1024)는 정상 케이스 차이가 아니라 **폭주 하나가 터졌을 때 그 하나의 blast radius** 차이다.
- **페이지 내 크롭은 길이가 대체로 비슷** → 정상 배치의 straggler 분산은 작다. 키우는 건 **outlier**뿐: 진짜 긴 캡션, 또는 반복 루프(하필 SFX·장식 크롭에 몰림).

엔진 차이는 **kind가 아니라 degree**다:

1. **낭비 스텝 단가.** straggler가 배치를 N스텝 더 돌릴 때 그 헛스텝 하나가 manga-ocr(작은 모델)은 싸고 PaddleOCR-VL(0.9B)은 비싸다. 상수배지만 무시 못 함.
2. **manga-ocr은 straggler-면역 인코더를 은행처럼 챙긴다.** straggler는 **디코드 루프**만의 문제다. `(N,C,H,W)` 인코더 forward는 비-autoregressive라 한 방 → 순수 이득. manga-ocr은 인코더(ViT over 224²)가 비용의 상당 부분이라 그만큼 straggler 무관하게 확보. PaddleOCR-VL은 0.9B LLM 디코드가 비용을 지배해 이 완충이 얇다.
3. **(약함) VLM 폭주 성향** — 일반 VLM이 textless 크롭에서 반복 루프에 더 잘 빠짐. 단 우리가 실측한 폭주는 gemma4 **translate** 쪽 SFX 루프라, recognizer에 대해선 추측.

**순이득 산수 예시**(인코더 은행이 straggler를 이기는 이유): B=8, 7개 15토큰, 1개 150토큰. 배치-8 스텝 단가 c8 ≈ 2·c1(8배 아님, 대역폭 바운드)이라 치면 —

- 디코드만: 순차 = 7×15+150 = 255(×c1) vs 배치 = 150(×c8=2c1) = 300 → **디코드만 보면 straggler 탓에 배치가 짐.**
- 인코더: 순차 8회 vs 배치 1회 → **6 절약**(면역).
- 합: +6 − 1.7 → **순이득 유지.** straggler 없으면(다 15토큰) 디코드도 30 vs 120으로 압승.

> **c8 ≈ 2·c1은 추정치다.** 실제 배치 단가는 안 재봤다. 이 문서의 유일한 정량 미지수이고, 아래 벤치가 이걸 확정한다.
> → **실측: c_8/c_1 = 3.47**(추정 2보다 큼). 인코더가 안 amortize돼서다 — 이 예시가 깔던 "인코더 은행" 전제 자체가 틀렸다. [실측](#실측-2026-07-07) 참고.

## ragged / left-pad / VRAM — PaddleOCR-VL 쪽 실제 비용

### left-pad 자체는 쉽다

생성은 오른쪽 끝에 새 토큰을 붙이고 맨 끝 hidden state로 다음을 예측하므로, pad는 **왼쪽**에 둬야 모든 시퀀스의 생성 프런티어가 오른쪽 끝에 정렬된다(right-pad면 다음 토큰이 pad 뒤에서 시작 = 깨짐). 코드로는:

```python
tokenizer.padding_side = "left"
inputs = processor(text=[...], images=[...], padding=True, return_tensors="pt")
model.generate(**inputs)   # attention_mask 자동
```

**한 줄 설정.** 어려운 건 left-pad가 아니다.

### 어려운 건 dynamic-res의 ragged vision 토큰

크롭마다 vision 토큰 수가 달라 배치할 때 **`position_ids`(M-RoPE/2D rope)를 pad·이미지블록에 맞춰 재구성**해야 한다. transformers의 `PaddleOcrVLProcessor`가 배치 멀티이미지(`images=[c1..cN], text=[t1..tN]`)를 패딩까지 제대로 지원하면 쉽고, 아니면 position_ids를 손으로 짜야 한다. 그리고 이 조합은 **"에러 없이 조용히 틀리는"** 대표 부류 — 비용은 타이핑이 아니라 **정합성 검증**에 있다.

### VRAM — tail에서만 문제

가중치는 bf16 × 0.9B ≈ **1.8GB**. 나머지는 **텍스트 KV 캐시가 아니다**(OCR 출력 짧음) — **vision 토큰의 prefill**이 지배한다:

- 큰 패널 크롭(~1536²): patch14·merge2 → **~3000 vision 토큰** → prefill 활성값+KV 수백 MB.
- **전형적 말풍선 크롭은 작아서 `min_pixels=384²` 바닥에 걸림** → ~196 토큰 → per-crop 수십 MB로 **작다.**

그래서 VRAM이 B를 조이는 건 **큰 패널이 섞일 때**뿐이고, 전형적 말풍선 배치에선 per-crop 메모리가 작아 B가 꽤 커질 수 있다.

### 정직한 재조정

VRAM·straggler·패딩 낭비 셋은 **전부 tail 얘기**다. common case(비슷한 작은 말풍선)에선 두 엔진의 배치 난이도 차가 생각보다 작다. **남는 진짜 차이**는:

- (a) VLM ragged/position_ids **일회성 엔지니어링**(되면 끝나는 비용),
- (b) 스텝 단가 **상수배**(0.9B vs 작은 모델),
- (c) 배치의 원 동기가 **CPU 대역폭 병목 = manga-ocr 얘기**였다는 점. PaddleOCR-VL은 GPU 기본 ~1s/crop이라 그 동기가 얘 것이 아니다(배치하면 GPU 처리량은 오르나 다른 층위).

## 어디를 손봐야 하나 — 통합 지점

지금 [pipeline.py `detect_and_recognize`](../app/pipeline.py#L43-L80)는 recognize를 **crop마다 순차 호출**한다:

```python
for region in regions:
    crop = deskew_crop(img, region)
    text = recognizer.recognize(crop, region, opt_recognize).strip()
```

배치는 이걸 **"detect로 crop 전부 → deskew 전부 → `recognize_batch(crops)` 한 번"**으로 바꾸는 것이다. 페이지 하나가 boundary라 배치 정합성/트랜잭션 걱정이 없다.

- SDK 계약에 `recognize_batch(crops, regions, options) -> list[str]`를 추가하고, **없는 엔진은 지금처럼 per-crop 루프로 폴백** — 이미 [`_translate_all`](../app/pipeline.py#L83-L90)이 `translate_batch` 유무로 분기하는 그 패턴 그대로.
- manga-ocr에 `recognize_batch` 구현(processor 리스트 입력 + `generate` + `batch_decode`). PaddleOCR-VL은 폴백에 둔다(위 이유).
- **`gpu_lock`을 건드리지 않는다.** 이건 pipeline.py 국소 변경이다.

## gpu_lock / detect·recognize 분리 (인접 설계 질문)

배치와 별개로 "`gpu_lock`을 없앨까 / detect·recognize를 쪼갤까"가 나왔다. 정리:

1. **`gpu_lock`은 두 가지를 한 자물쇠로 용접**한다([orchestrator.py `_read_sync`](../app/orchestrator.py#L76-L93) 주석): **① `registry.get` thread-safety**(`_instances` check-then-set + 모델 로드 직렬화)와 **② compute 직렬화**. 그냥 없애면 compute만 풀리는 게 아니라 **`_instances` 레이스**가 같이 터진다. 어떤 lock 변경이든 **1단계는 registry 안전성 분리**(전용 작은 lock, 또는 selection 시점 preload로 get을 순수 read화). translator를 lock 안에서 resolve하는 것도([orchestrator.py](../app/orchestrator.py#L88)) 이때 같이 챙긴다.
2. **lock 제거만으로 end-to-end 이득은 지금 없다.** 파이프라인이 **translate-bound**라 detect+recognize를 이미지 간 겹쳐도 병목이 **`lockwait` → `semwait`로 자리만 옮긴다**(아티팩트 시뮬레이션 결론). 실측 레버는 **K(`translate_sem`)**이다. recognize 병렬화 1.85배가 end-to-end에 드러나려면 translate가 병목에서 빠져야 하고, 그건 K를 크게 올려야 하는데 `OLLAMA_NUM_PARALLEL`에 묶여 있다.
3. **값어치 있는 "분리"는 위의 페이지 내 detect→batch-recognize다** — lock 무관, manga-ocr 배치 이득이 실제로 꽂히는 자리. detect/recognize를 아예 별도 stage(각자 pool)로 쪼개는 큰 개편은 그 뒤 로그의 `lockwait`/`semwait`을 보고 결정해도 늦지 않다.

## 실측 (2026-07-07)

[bench_recognize_batch.py](bench_recognize_batch.py)로 잰다 — 실제 페이지(Pixiv 한 챕터 21장)에서 detector로 크롭 42개를 잘라(detect + deskew, 파이프라인과 동일) manga-ocr CPU에서 batch 1~16 스윕. 셋으로 쪼개 잰다: **인코더 단독**(straggler-면역), **고정길이 생성**(L=32, straggler 인위 제거 = 이상적 상한), **자연 생성**(실제 길이 = straggler 포함). PaddleOCR-VL은 GPU가 필요해 여기선 빠지고, 아래 **§PaddleOCR-VL GPU 배치**에서 ROCm으로 따로 잰다(2026-07-07 추가).

| batch | encoder speedup | fixed-length speedup | natural speedup |
|---|---|---|---|
| 1 | 1.00x | 1.00x | 1.00x |
| 2 | 1.11x | 1.45x | 1.09x |
| 4 | 1.15x | 1.99x | 1.23x |
| 8 | 1.15x | **2.30x** | **1.27x** |
| 16 | 1.04x | 2.40x | 1.16x |

(speedup = 그 batch의 crops/sec ÷ B=1. **핵심 한 줄: 이상적 상한이 2.3x, 실제는 1.27x(B=8).**)

### 뒤집힌 것 1 — 인코더는 CPU에서 "은행"이 아니다

설계는 "인코더가 straggler-면역이라 공짜로 배치되는 은행"이라 봤는데(§straggler, §순이득 예시) **정반대**다. fixed-length의 ms/batch를 인코더/디코더로 쪼개면:

| 부분 | c_8/c_1 | 배치가 먹히나 |
|---|---|---|
| 인코더 (ViT 224²) | **6.97** (≈8, 거의 선형) | ❌ B=1에서 이미 CPU 대역폭 포화 |
| 디코더 (생성 32스텝) | **1.69** | ✅ 잘 amortize |

인코더는 배치로 못 줄이는 **고정세**처럼 작동해 전체 c_8/c_1을 3.47로, 이상적 상한을 **2.3x**로 묶는다. amortize되는 건 디코더 — 문서가 straggler 위험으로 지목했던 바로 그 부분이다.

### 뒤집힌 것 2 — straggler가 이상적 2.3x를 실제 1.27x로 깎는다

배치가 잘 먹히는 디코더가 하필 straggler가 사는 곳이다. 실제 출력 길이는 **median 26 / p90 49 / max 55 토큰**으로 편차가 있고, stock transformers는 continuous batching을 안 해 배치 전체가 최장까지 돈다. 그래서 straggler-free 2.3x가 **자연 생성에선 1.27x(B=8)**로 붕괴한다(처리량 −54%: 13.0 → 6.0 crops/sec). **B=16은 1.16x로 오히려 후퇴** — B=8이 유일한 스윗스팟.

- **정확도 OK**: 배치 경로가 실제 일본어를 per-crop과 동일하게 읽는다(`MangaOcr.__call__`의 `convert("L")` grayscale + `post_process` 복제 확인). 조용한 회귀 없음.

### 배치 vs 멀티워커 — 곱해지지 않는다

둘 다 "1 크롭 × 전 코어 × 직렬"([pipeline.py](../app/pipeline.py) 현재 동작)을 같은 베이스라인으로 잰 값이다:

| 방식 | 실측 | 성격 |
|---|---|---|
| crop-batching | ~1.27x | intra-page, pipeline.py 국소 |
| 멀티워커 ([스레드 벤치](recognize-cpu-threads.md)) | ~1.8x | 동시성, gpu_lock→semaphore 필요 |

**멀티워커가 배치보다 낫고, 둘은 곱해지지 않는다** — 같은 CPU 메모리 대역폭 여유를 두 방식이 나눠 쓰는 것이라, 멀티워커가 이미 대역폭 천장(스레드 벤치가 확인한 flattening) 근처다. 배치는 디코더 weight 읽기를 줄여 천장을 살짝 밀 뿐이고, 순진하게 얹으면(16스레드 배치 × N워커) 오버서브스크립션으로 오히려 느려진다. 합쳐도 ~1.8–2.0x가 상한이지 1.27×1.8=2.3x는 안 나온다.

### PaddleOCR-VL GPU 배치 (2026-07-07, ROCm 실측 — ⚠ stale, 하단 §판단의 2026-07-12 최종 실측 참조)

문서의 [§다음](#다음--실측-항목) 4번(GPU torch 세팅 후 재실행)을 채운다. 환경: 서버 9060 XT(gfx1200/RDNA4) + ROCm 7.1, `torch 2.x+rocm7.0`(컨테이너, MIOpen 커널 캐시를 `/data`에 영속). 같은 Pixiv 챕터 크롭으로 batch 1~16 스윕. **커널 캐시가 warm된 2회차 값만 유효**하다 — 1회차는 shape별 MIOpen JIT가 지배(baseline 12.7s→2.9s/crop, 캐시 히트 시 4.4x). dynamic-res라 새 crop 크기마다 커널을 컴파일하지만, 픽셀 단위가 아니라 **patch14·merge2=28px 그리드 버킷**이 바뀔 때만 재컴파일한다.

| batch | match | crops/sec | speedup |
|---|---|---|---|
| 1 (baseline) | — | 0.34 | 1.00x |
| 2 | 2/2 | 0.70 | **2.04x** |
| 4 | 4/4 | 0.60 | 1.75x |
| 8 | **7/8** | 0.21 | 0.60x |
| 16 | **15/16** | 0.21 | 0.60x |

셋이 드러났다:

1. **스윗스팟 B=2(2.04x), B≥8은 후퇴(0.60x).** manga-ocr(B=8 스윗스팟)과 반대다 — 0.9B VLM이라 작은 배치에서 이득이 크고 빨리 포화·후퇴한다(padding 낭비 + straggler가 큰 배치를 잡음).
2. **⚠️ B≥8에서 correctness가 깨진다.** B=2·4는 per-crop과 완전 일치인데 B=8·16에서 crop #5만 갈린다: want=`ん...っ♥♥\n♥おっ♥` vs got=`ん...コ♥♥\n✓おっく`. [§ragged](#어려운-건-dynamic-res의-ragged-vision-토큰)가 예언한 "position_ids/padding이 조용히 틀림(silently wrong)"이 실측됐다 — 하필 ♥ 반복 SFX형 crop(폭주 성향)이 left-pad가 많아지는 큰 배치에서 취약하다. **배치가 무조건 output-preserving은 아니다.**
3. **절대속도가 manga-ocr CPU에 못 미친다.** warm B=2가 0.70 crops/sec(≈1.4s/crop)인데 manga-ocr CPU는 natural ~6 crops/sec(≈0.17s) — **manga-ocr CPU가 8배 빠르다.** PaddleOCR-VL은 정확도(88%) 프리미엄이지만 GPU에 배치를 얹어도 그 격차는 안 메워지고, dynamic-res 커널 JIT의 롱테일(캐시 미스 시 ~7s)까지 남는다.

**단, "B=2·4 완전 일치"를 안전으로 읽으면 안 된다** — 벤치가 크롭을 고정 순서로 배치해([bench](bench_recognize_batch.py) `crops[i % len]`) 깨지는 crop #5는 B≥8에서야 배치에 들어간다. B=2·4는 그 크롭을 안 넣은 더 쉬운 부분집합을 잰 것이라, 작은 배치가 correctness를 보존한다는 증거가 아니라 **미측정**이다. 근본 원인은 배치 "크기"가 아니라 배치 내 **길이 편차**(ragged vision 토큰 → left-pad → position_ids 오프셋 오류)라, B=2라도 [짧은 크롭 + 긴 SFX]를 묶으면 똑같이 깨질 수 있다. 안전 조건은 "작은 배치"가 아니라 "배치 내 길이 균일"(길이 버킷팅)이다.

**결론: PaddleOCR-VL 배치는 실패다.** B=2~4에서만 2x인데 그마저 절대속도가 manga-ocr CPU의 1/8이고, B≥8은 느려지고 틀린다.

### 후속 — 세 레버 재측정, 그리고 AOTriton 오판 정정 (2026-07-08)

4070 Ti Super(~1s/crop) 대 서버 9060 XT(2.9s/crop)의 ~3배 차가 셋업 탓인지 보려고 세 레버를 걸어 재측정했는데, **당시 AOTriton 판정이 틀렸다**(경위·정정은 [recognize-gpu-speed.md](recognize-gpu-speed.md)).

- **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`(RDNA4 flash/mem attention) — 실은 3.7x 레버다.** "무효"라 판정한 근거가 **플래그를 켠 채 두 번 돌려 같은 값이 나온 것**이었다(비교 기준이던 이전 baseline도 켜진 값). OFF 기준선을 한 번도 안 쟀던 것. 뒤에 같은 스크립트로 A/B: **OFF 0.094 / ON 0.345 crops/sec.** → [docker-compose.rocm.yml](../../../docker-compose.rocm.yml)에 반영 완료.
- **`attn_implementation="sdpa"` — 명시해도 변화 없음.** 모델이 **이미 sdpa를 기본으로** 쓰기 때문(`ValueError` 없이 로드; [BENCH_ATTN 노브](bench_recognize_batch.py), 커밋 b896376). AOTriton은 그 sdpa가 느린 math 폴백 대신 **flash 커널**을 타게 하는 스위치다.
- **`PYTORCH_KERNEL_CACHE_PATH` — 경고만 옮겼다.** torch JIT 커널 캐시(MIOpen과 별개)가 `/root/.cache`→`/data/torchkernels` 어디서도 "could not be created"로 꺼진다(app 유저가 볼륨에 못 씀 = §부채 4번과 같은 권한 뿌리). 단 per-crop 수엔 무영향 — 한 프로세스 안에선 컴파일 커널이 메모리에 남아 반복 shape는 1회만 컴파일, 디스크 캐시는 새 프로세스 cold start만 돕는다.

**따라서 warm B=1 = 0.34 crops/sec(≈2.9s/crop)는 flash-ON의 정상값이다.** 4070 Ti Super 대비 ~3배 차는 하드웨어 체급 + 대역폭(672 vs ~320 GB/s)으로 설명되는 범위다. flash를 끄면 같은 crop이 0.094 crops/sec로 떨어진다 — 병목은 vision attention이고, AOTriton이 그 스위치다.

### GPU recognize 실투입 셋업 부채

이 문서의 GPU 실측은 컨테이너에 수동 `-e`로 우회해 돌렸다. PaddleOCR-VL을 실제 파이프라인의 GPU recognizer로 붙이려면 코드/설정 네 군데를 고쳐야 한다 — `accelerate` 의존성, AMD torch가 PyPI의 CUDA 빌드로 새는 문제, 기본 rocm 인덱스, entrypoint의 `HOME`. **증상·원인·고칠 것·진행 상태는 [recognize-gpu-speed.md](recognize-gpu-speed.md)의 §실투입 — 셋업 부채**에 있다(AOTriton env는 반영 완료).

## 판단 (실측 반영)

- **manga-ocr `recognize_batch` — 보류 쪽으로 기움.** 실이득 1.27x(B=8)로 작고 멀티워커(1.8x)보다 못하다. 저위험 국소 변경이라 값이 0은 아니나, **translate-bound면 end-to-end엔 안 드러난다** → 구현 전 recognize의 end-to-end 비중부터 로그로 확인(§다음).
- **기각 — PaddleOCR-VL 배치.** GPU(ROCm)로 측정 완료(§PaddleOCR-VL GPU 배치): B=2 2.04x가 최대고 절대속도가 manga-ocr CPU의 1/8, B≥8은 후퇴 + correctness 깨짐. ragged/position_ids 우려가 실측으로 확인됐다. 후속(2026-07-08): **AOTriton flash는 3.7x 레버**(당시 "무효" 판정은 오류 — §후속), 2.9s/crop은 그 flash-ON 값이다. 작은 배치의 correctness는 측정 착시라 여전히 미보장.

> **최종 실측 (2026-07-12) — 배치 폐기 확정.** 위 2026-07-07 표의 2.04x/1.75x는 **stale**이고, 파이프라인에 실제로 배선해 재보니 PaddleOCR-VL 배치는 per-crop보다 **느리다**. 확정까지 네 개의 측정 결함을 걷어냈다:
> 1. **옛 수치가 stale** — 2026-07-07 이후 AOTriton flash(07-08)·해상도 캡(07-10)이 per-crop을 1.71x 빠르게(0.34→0.58 crops/sec) 만들어 배치의 *상대* 이득이 사라졌다. 배치 절대속도는 그대로(0.70→0.71 crops/sec)인데 baseline만 좋아진 것.
> 2. **bench probe의 캡 비대칭** — baseline은 `rec.recognize`(내부 캡)인데 batch probe는 `proc(원본 크롭)`이라 캡 미적용. baseline만 빨라 speedup이 부풀었다.
> 3. **bench의 크롭 셋 불공정** — baseline은 큰 크롭 24개 평균, 배치는 작은 앞 b개. 다른 표본이라 배치가 나아 보였다.
> 4. **`no_grad` 누락** — 파이프라인 `recognize_batch`와 첫 [diag_batch.py](diag_batch.py)가 `generate`를 `torch.no_grad()`로 안 감싸 배치 forward가 autograd 그래프를 쌓았다(단일 크롭엔 안 보이지만 배치는 activation N배라 폭발). 이게 파이프라인 5.5x·첫 diag 10.5x의 정체였다.
>
> **공정 측정**(같은 크롭 · 캡 대칭 · no_grad, [diag_batch.py](diag_batch.py) `--n 4`): per-crop 합 4184ms vs 배치 5346ms = **배치 1.3x 느림(0.78x)**. `input_ids(4,223)` 진짜 배치 + `pixel_values` concat(패딩 없음)이라 패딩은 무관 — 순수 **straggler**다. 배치 forward는 gen-step당 오히려 약간 빠르나(53.5 vs 59.8ms), 짧은 `ぶくっ`(4토큰)이 배치 최장 25토큰까지 헛돌아 gen-step이 100 vs 70으로 는다. vision-prefill 지배라 배치 효율 이득이 애초에 작고(1.12x), straggler(1.43x)가 그걸 넘는다.
>
> **→ 배치 폐기 확정.** recognize 처리량 레버는 vision을 직접 줄이는 flash(3.7x)·캡(1.66x)과, straggler가 원천 없는 **멀티워커([recognize-gpu-speed.md](recognize-gpu-speed.md), 1.38x)**다. correctness(B≥8 깨짐)까지 고려하면 배치는 어느 축으로도 못 이긴다. (대조: 번역기 gemma는 메모리-바운드 텍스트 LLM이라 `translate_batch`가 유효 — 배치가 안 통하는 건 recognize의 VLM 아키텍처(짧은 출력 + vision-prefill 지배) 때문이지 배치 일반이 아니다.)
- **완료 — registry 안전성을 `gpu_lock`에서 분리.** 커밋 2c7e38e; 자체 lock으로 `gpu_lock`과 무관하게 thread-safe. (§gpu_lock 1번이 요구한 전제.)
- **CPU recognize를 정말 빠르게 할 거면 배치보다 멀티워커(동시성)가 우선** — 1.8x > 1.27x 실측. 단 `gpu_lock`→semaphore 필요하고, 역시 recognize 비중 확인이 선행. translate-bound면 둘 다 가려진다.
- **방향 전환 — 처리량은 배치가 아니다. 단 동시성도 GPU 엔진엔 약하다.** 세 recognize 경로 다 배치가 신통찮다(manga-ocr 1.27x, PaddleOCR-VL 기각). manga-ocr은 CPU 멀티워커가 레버(1.8x)지만 **GPU 멀티워커는 실측 천장 1.31x**다 — ROCm엔 MPS가 없어 커널이 타임슬라이스되고, 벌 수 있는 건 W=1의 GPU 유휴(~24%)뿐이다([recognize-gpu-speed.md](recognize-gpu-speed.md)). **GPU 쪽 진짜 레버는 flash attention(AOTriton, 3.7x)이었다.** 하드웨어 확장(9060 XT + MI50) 시 **GPU별 역할 분리** — 한 GPU는 recognize(PaddleOCR-VL), 다른 GPU는 translate(Gemma) — 는 한 GPU를 나눠 쓰는 게 아니라 물리적 분리라 이 천장과 무관하고, `gpu_lock` 경합을 없애는 다음 설계 후보로 유효하다(별도 검토).

## 다음 — 실측 항목

1. ~~manga-ocr **배치 forward 단가**(c8/c1)~~ — **완료**([실측](#실측-2026-07-07)): c_8/c_1=3.47, 이상적 2.3x.
2. ~~실제 페이지 크롭의 **출력 길이 분포**(straggler)~~ — **완료**: median 26/p90 49/max 55 → 실이득 1.27x(B=8).
3. **결정타 — recognize의 end-to-end 비중.** [orchestrator.py `run_page`](../app/orchestrator.py#L163-L169)가 페이지마다 찍는 `detect+recognize=…ms` vs `translate=…ms`를 실제 처리 로그에서 확인. translate가 지배하면 배치든 멀티워커든 값어치가 marginal → 이게 구현 여부를 최종 판정한다. **이걸 먼저.**
4. ~~PaddleOCR-VL — GPU torch(ROCm)로 배치 프로브 재실행~~ — **완료**(§PaddleOCR-VL GPU 배치): 배치 **기각**. 멀티이미지 배치가 되긴 하나 B≥8에서 correctness가 깨지고(silently-wrong 실측) 절대속도가 manga-ocr CPU의 1/8이다.

## 관련

- 스레드 배치(intra-op vs 멀티워커) 실측은 [recognize-cpu-threads.md](recognize-cpu-threads.md). 이 문서는 그 "대안 후보(crop 배치)"를 이어받은 것.
- 동시성·`gpu_lock`·translate 배치의 그림은 아티팩트 [동시성과 번역 배치](https://claude.ai/code/artifact/543ff4c0-d2be-4d4f-9d70-fc35fac17c1f).
- 통합 지점: [pipeline.py `detect_and_recognize`](../app/pipeline.py) · [orchestrator.py `run_page`](../app/orchestrator.py) · 엔진 [manga-ocr plugin](../../scanlation-manga-ocr/scanlation_manga_ocr/plugin.py) / [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py).

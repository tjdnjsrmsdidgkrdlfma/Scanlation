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

[bench_recognize_batch.py](bench_recognize_batch.py)로 잰다 — 실제 페이지(Pixiv 한 챕터 21장)에서 detector로 크롭 42개를 잘라(detect + deskew, 파이프라인과 동일) manga-ocr CPU에서 batch 1~16 스윕. 셋으로 쪼개 잰다: **인코더 단독**(straggler-면역), **고정길이 생성**(L=32, straggler 인위 제거 = 이상적 상한), **자연 생성**(실제 길이 = straggler 포함). PaddleOCR-VL은 컨테이너 torch가 CPU 빌드라 `cuda.is_available()`=False로 이번엔 스킵.

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

## 판단 (실측 반영)

- **manga-ocr `recognize_batch` — 보류 쪽으로 기움.** 실이득 1.27x(B=8)로 작고 멀티워커(1.8x)보다 못하다. 저위험 국소 변경이라 값이 0은 아니나, **translate-bound면 end-to-end엔 안 드러난다** → 구현 전 recognize의 end-to-end 비중부터 로그로 확인(§다음).
- **보류 — PaddleOCR-VL 배치.** ragged/position_ids + 스텝 단가 + GPU 기본이라 payoff÷effort 나쁨(그대로). 아직 미측정 — GPU torch 세팅 후 프로브 재실행.
- **완료 — registry 안전성을 `gpu_lock`에서 분리.** 커밋 2c7e38e; 자체 lock으로 `gpu_lock`과 무관하게 thread-safe. (§gpu_lock 1번이 요구한 전제.)
- **CPU recognize를 정말 빠르게 할 거면 배치보다 멀티워커(동시성)가 우선** — 1.8x > 1.27x 실측. 단 `gpu_lock`→semaphore 필요하고, 역시 recognize 비중 확인이 선행. translate-bound면 둘 다 가려진다.

## 다음 — 실측 항목

1. ~~manga-ocr **배치 forward 단가**(c8/c1)~~ — **완료**([실측](#실측-2026-07-07)): c_8/c_1=3.47, 이상적 2.3x.
2. ~~실제 페이지 크롭의 **출력 길이 분포**(straggler)~~ — **완료**: median 26/p90 49/max 55 → 실이득 1.27x(B=8).
3. **결정타 — recognize의 end-to-end 비중.** [orchestrator.py `run_page`](../app/orchestrator.py#L163-L169)가 페이지마다 찍는 `detect+recognize=…ms` vs `translate=…ms`를 실제 처리 로그에서 확인. translate가 지배하면 배치든 멀티워커든 값어치가 marginal → 이게 구현 여부를 최종 판정한다. **이걸 먼저.**
4. (조건부) PaddleOCR-VL — GPU torch(ROCm)로 바꾼 뒤 벤치의 배치 프로브 재실행. `PaddleOcrVLProcessor`가 배치 멀티이미지를 패딩까지 지원해 출력이 per-crop과 일치하는지(correctness gate). 통과하면 재평가.

## 관련

- 스레드 배치(intra-op vs 멀티워커) 실측은 [recognize-cpu-threads.md](recognize-cpu-threads.md). 이 문서는 그 "대안 후보(crop 배치)"를 이어받은 것.
- 동시성·`gpu_lock`·translate 배치의 그림은 아티팩트 [동시성과 번역 배치](https://claude.ai/code/artifact/543ff4c0-d2be-4d4f-9d70-fc35fac17c1f).
- 통합 지점: [pipeline.py `detect_and_recognize`](../app/pipeline.py) · [orchestrator.py `run_page`](../app/orchestrator.py) · 엔진 [manga-ocr plugin](../../scanlation-manga-ocr/scanlation_manga_ocr/plugin.py) / [PaddleOCR-VL plugin](../../scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py).
순서	작업	위험	값어치
0	registry 안전성 분리	낮음	동시성 전제 + 하이진
1	동시성 ~4-way (lock→Semaphore)	중(torch 스레드 결정)	K=8 목표 해결
2	crop-batching	낮음	단일-페이지 지연(선택)

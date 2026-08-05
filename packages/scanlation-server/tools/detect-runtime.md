# detect 런타임 — CPU 레버는 소진, ONNX 2.33x, GPU는 카드가 못 버틴다

측정 2026-08-05. detect는 파이프라인의 **14.4%**(302ms/페이지)이고 translate 다음 순번의 레버다. 이 문서는 그 302ms를 어디까지 줄일 수 있는지 잰 기록이다. 도구는 [bench_detect_runtime.py](bench_detect_runtime.py), 표본은 같은 챕터 21페이지(2400×1800).

## 결론 먼저

| 경로 | per-page(단독) | 파이프라인 detect | 박스 | 판정 |
|---|--:|--:|---|---|
| **torch CPU (현행)** | 237.4ms | 302ms | 기준 | — |
| **ONNX Runtime CPU** | **102.0ms** | 미측정 | 42 vs **41**, 최악 35.7px | 🔸 **유력 — 박스 1개 차이만 판정하면 채택** |
| ONNX Runtime CPU (int8) | 154.9ms | — | 동일 | ❌ fp32보다 느리다 |
| torch GPU (`cuda:1`) | 98.8ms | **151.5ms** | torch와 동일 | ❌ **GPU hang — 못 쓴다** |

**속도만 보면 GPU가 최고이고 실제로 파이프라인 detect를 302 → 151ms로 정확히 반토막 냈다. 그런데 같은 런에서 GPU가 두 번 hang했다.** recognize와 카드를 공유하는 한 쓸 수 없고, 옮길 카드도 없다(MI50은 torch가 안 돈다). **ONNX가 카드를 전혀 안 건드리면서 GPU와 같은 속도를 낸다** — 그래서 답은 ONNX다.

## 1. 비용 구조 — forward가 98%다

단독 실행(스레드 8), 스테이지별 중앙값:

| 스테이지 | ms | 비중 |
|---|--:|--:|
| 전처리(`RTDetrImageProcessor`) | 4.4 | 2% |
| **forward** | **218.1** | **98%** |
| 후처리(post_process + dedup) | 0.4 | 0.2% |

**입력은 640×640 고정이다.** `preprocessor_config.json`이 `do_resize: true` · `size: {640, 640}` · `do_pad: false`라 2400×1800 페이지가 **비율을 무시하고** 정사각형으로 들어간다. 그래서 페이지 크기와 무관하게 forward 비용이 일정하고, **해상도를 낮춰 시간을 사는 노브가 없다.**

전처리·후처리를 합쳐도 2%라 **줄일 수 있는 건 forward뿐**이다.

## 2. torch 안의 CPU 레버는 이미 최적이다

**스레드 수** (9700X = 8물리코어/16스레드):

| threads | 1 | 2 | 4 | 6 | **8** | 12 | 16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| total ms | 891.9 | 425.2 | 303.3 | 244.4 | **222.9** | 256.9 | 253.3 |

**8이 최적이고 이미 그 값이다** — torch가 물리코어를 감지해 잡는 기본값이라 설정한 적도 없다. 12·16이 되레 느려지는 것은 recognize CPU 경로에서 나온 **"물리 코어가 단위, 하이퍼스레드는 손해"** 와 같은 결론이다.

**`torch.inference_mode`**: 223.9ms vs `no_grad` 225.8ms = **0.8%, 노이즈**. forward가 98%인 구조에서 autograd 부기는 잴 수 있는 몫이 아니다. [PERFORMANCE_PLAN.md](../../../PERFORMANCE_PLAN.md) Tier 1-C는 이걸로 닫힌다.

## 3. ONNX Runtime — 2.33x, 단 같은 모델이 아니다

모델 저장소(`ogkalu/comic-text-and-bubble-detector`)는 safetensors 옆에 ONNX를 함께 배포하는데, 현재 설치는 그걸 **일부러 건너뛴다**(`_download()`의 `WEIGHT_PATTERNS`). 21페이지 A/B:

| 런타임 | per-page med | 대비 |
|---|--:|--:|
| torch fp32 | 237.4ms | — |
| **ORT fp32** | **102.0ms** | **×2.33** |
| ORT int8 (43.8MB) | 154.9ms | ×1.51 |

**INT8이 fp32보다 느리다.** 이 크기(168MB)에서는 양자화·역양자화 오버헤드가 이득을 넘지 못한다 — recognize에서 Q4가 0이었던 것과 같은 종류의 결과다.

**ORT도 스레드 8이 최적**이다(1→471 / 4→149 / **8→102** / 12→134 / 16→127ms). 곡선 모양이 torch와 같다.

**출력은 완전히 같지 않다.** 21페이지에서 torch 42박스 vs ORT 41박스(**19.jpg에서 하나 누락**), 좌표는 20페이지가 1.5~10.4px에 **13.jpg만 35.7px**. 전처리 텐서를 동일하게 넣고도 그대로라 **export된 ONNX가 safetensors와 미세하게 다른 모델**이라는 뜻이다. `--html`이 두 런타임의 박스를 같은 이미지에 겹쳐 그린 판정 페이지를 낸다(불일치 페이지가 맨 위).

## 4. GPU detect — 빠르지만 카드가 못 버틴다

`/admin`에서 detector device를 `cuda:1`로 바꾸면 끝이다. compose 수정도 컨테이너 재시작도 필요 없다 — `.env`가 이미 `HIP_VISIBLE_DEVICES=0,1`이라 torch는 두 카드를 다 보고 있다.

> **⚠ `cuda:0`은 MI50이다.** torch는 gfx906에서 첫 matmul에 죽는다(rocBLAS에 gfx906 Tensile 라이브러리 없음). 반드시 `cuda:1`이어야 하고, **이 인덱스는 재부팅·PCIe 재열거로 움직일 수 있다.**
>
> | torch device | 카드 | arch | VRAM |
> |---|---|---|--:|
> | `cuda:0` | MI50 | `gfx906:sramecc+:xnack-` | 32.0GB |
> | `cuda:1` | 9060 XT | `gfx1200` | 15.9GB |

**속도는 실제로 좋다.** 단독 98.8ms(CPU 237.4 대비 2.40x)이고, `run_report.py --no-translate` 파이프라인에서 **detect 중앙값 151.5ms**(CPU 시절 302ms의 정확히 절반), recognize 중앙값도 588.2ms로 정상 범위였다.

**그런데 같은 런에서 GPU가 두 번 hang했다.**

```
17:47:46 amdgpu 0000:09:00.0: ring comp_1.0.1 timeout, signaled seq=339656, emitted seq=339657
17:47:46 amdgpu 0000:09:00.0: [drm] device wedged, but recovered through reset
```

시간순으로 **00.jpg가 HTTP 500 `timed out`으로 실패 → GPU 리셋 → 01.jpg가 복구를 기다리며 61.9초** → 이후 19페이지는 정상(159~1126ms). 평균이 3747.8ms로 튄 건 그 이상치 하나 때문이고 **중앙값은 745.5ms로 오히려 기준선(805ms)보다 낫다** — 즉 성능 문제가 아니라 **안정성 문제**다.

**옮길 카드도 없다.** recognize를 MI50으로 보내면 translate와 경합하고, 애초에 torch recognize는 gfx906에서 안 돈다. detect만 다른 카드로 보낼 여지도 없다(카드가 둘뿐이고 하나는 translate 상주).

## 5. GPU hang은 온도도 VRAM도 아니다

2026-08-05 하루에 **7번** 났다. 전부 `0000:09:00.0`(9060 XT)이고 **MI50은 한 건도 없다** — 같은 날 내내 translate를 돌렸는데도.

| 시각 | 상황 | 그때 VRAM |
|---|---|--:|
| 15:51 | llama-server 2개 (`--image-min-tokens`) | ~7GB |
| **16:06** | **llama-server 1개** (`--image-min-tokens`) | ~3.7GB |
| 16:14 | llama-server 3개 | ~11GB |
| 16:57 ×2 | llama-server 2개 (Q8/BF16 A/B) | ~3.5GB |
| 17:46·17:47 | torch detect + llama-server | **~2.6GB** |

- **온도 아님** — dmesg에 thermal·throttle·overtemp 이벤트가 **0건**이고, 카드는 하루 종일 36~45°C였다.
- **VRAM 아님** — 16GB 카드에서 2.6GB일 때도 났다. 사고별 사용량이 제각각이고 항상 여유가 있었다.
- **찍히는 건 항상 같다** — `ring comp_*.* timeout`에 `signaled`와 `emitted`의 seq 차이가 **1~2**다. 큐가 밀린 게 아니라 **제출된 작업 하나가 완료 신호를 안 보내서** 드라이버 watchdog이 링을 리셋한다.
- **프로세스가 하나여도 난다**(16:06). 그러니 "동시 실행"이 필수 조건은 아니다.

공통점은 **평소와 다른 형태의 작업이 GPU에 들어갈 때**다 — 비정상 이미지 그리드(`--image-min-tokens`), 컨텍스트를 갈아타는 여러 프로세스, HIP와 Vulkan의 혼용. 반대로 **평범한 recognize 단독 부하에서는 하루 종일 한 번도 안 났다.**

> **⚠️ 여기까지 중 "gfx1200에서 amdgpu가 미성숙하다"는 가설이다.** 정황(성숙한 gfx906은 같은 부하에서 멀쩡, 커널 6.12에 RDNA4 지원이 들어온 지 얼마 안 됨)이 그쪽을 가리키지만 확증은 `devcoredump`를 떠서 어느 커널에서 멈췄는지 보는 것이고, 아직 안 했다.

**운영 규칙: 9060 XT에는 컴퓨트 소비자를 한 번에 하나만 둔다.** 변형 비교도 서버 2개까지고(3개는 첫 요청에서 죽었다), detect를 GPU로 올리는 것은 recognize가 같은 카드에 있는 한 하지 않는다.

## 6. 함정 (재현할 때)

- **`orig_target_sizes`는 (width, height)다.** transformers의 `target_sizes`는 (height, width)라 그대로 넘기면 **모든 박스가 조용히 잘못된 배율로 스케일된다.** 그런데 **박스 개수는 멀쩡히 일치**해서 개수만 확인하면 성공으로 보인다. 처음에 이걸로 200~586px 오차가 났다.
- **전처리를 공유해야 런타임만 비교된다.** ORT 쪽에서 PIL로 직접 리사이즈하면 "런타임 차이 + 리사이즈 구현 차이"가 섞인다. `RTDetrImageProcessor`가 만든 텐서를 그대로 넣는다.
- **`/plugins`의 플러그인이 저장소 코드보다 오래됐을 수 있다.** 이 도구는 `filter_small`을 `getattr`로 옵셔널 참조한다 — 런타임 비교하자고 플러그인 버전을 맞출 이유는 없어서다.
- **onnxruntime은 프로젝트 의존성이 아니다.** 임시 경로에 설치하고 측정 후 지운다(설치·실행·정리 순서는 [bench_detect_runtime.py](bench_detect_runtime.py) docstring).

## 남은 일

- **ONNX 채택 판정** — `detect_ab.html`에서 19.jpg의 누락 박스와 13.jpg의 35.7px 어긋남이 실제 손해인지 눈으로 본다. 채택하면 플러그인에 ONNX 경로를 붙이는 구현이 남는다(엔진을 나눌지 같은 엔진의 옵션으로 둘지가 설계 판단이고, onnxruntime 의존성이 붙는 대신 **torch 없이 detect가 도는** 이득이 있다).
- **표본 확대** — 벤치 21페이지는 한 작품이다. 다른 화풍·손글씨 많은 페이지에서도 박스 차이가 1개 수준인지 봐야 계열의 문제인지 표본의 우연인지 갈린다.
- **GPU hang 확증** — `devcoredump`로 어느 커널이 멈췄는지. 이게 풀리면 GPU detect가 되살아날 수도 있다(파이프라인 detect 151ms는 ONNX보다도 나은 값이다).
- **검출기 풀링(W>1)** — detect는 1-worker 풀에서 설계상 직렬이다. 페이지당 1회라 팬아웃할 게 없고 워커마다 모델 사본이 붙으므로, **이미지가 겹치는 워크로드에서 detect가 실제 병목으로 측정될 때만**.

## 관련

- 파이프라인 전체 결론 — [BENCHMARKS.md](../../../BENCHMARKS.md)
- recognize 쪽 런타임 교체 기록(같은 종류의 결론) — [recognize-decode-bound.md](recognize-decode-bound.md)
- 위험·규모별 최적화 후보 — [PERFORMANCE_PLAN.md](../../../PERFORMANCE_PLAN.md)

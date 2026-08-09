# 번역 품질 비교 — Gemma 4 vs Qwen3.6

detector·recognizer 때와 같은 방식으로, **사람이 눈으로 채점**한다. 번역엔 정답 문자열이 없어서 자동 지표(BLEU 류)가 만화 대사의 어투·의역·SFX를 못 잡는다. 그래서 하네스가 하는 일은 "같은 조건을 만들어 주고 표로 늘어놓는 것"까지고, 승자는 클릭 투표로 정한다.

## 무엇을 통제하는가

- **원문이 동일하다.** 페이지를 다시 인식하지 않고 recognizer 배치가 남긴 `compare_out/<분류>/<이미지>/ocr.json`의 텍스트를 읽는다(기본 `paddleocr_manga`). recognizer가 틀린 곳은 모든 모델에게 똑같이 틀린 채로 간다.
- **요청 모양이 프로덕션과 동일하다.** 어댑터가 실제 플러그인(`scanlation_llama_cpp`)을 그대로 호출한다 — 같은 시스템 프롬프트, 페이지 단위 배치 JSON 스키마, DRY 반복 브레이크, 실패 시 per-text 폴백. 점수가 모델의 것이지 하네스가 다시 짠 프롬프트의 것이 아니다.
- **서버 플래그가 배포와 동일하다.** `-ngl 99 -c 16384 --parallel 4 --n-predict 1024` — [deploy/llama.cpp-gemma-4-26B-A4B.service.example](../../../deploy/llama.cpp-gemma-4-26B-A4B.service.example)와 같다.

## 후보

| id | 레포 (`-hf …:UD-Q4_K_XL`) | 구조 | Q4 크기 |
|---|---|---|---|
| `gemma-4-26B-A4B` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` | MoE, 4B active | ~15GB |
| `gemma-4-31B` | `unsloth/gemma-4-31B-it-qat-GGUF` | dense | ~18GB |
| `Qwen3.6-27B` | `unsloth/Qwen3.6-27B-GGUF` | dense | ~17GB |
| `Qwen3.6-35B-A3B` | `unsloth/Qwen3.6-35B-A3B-GGUF` | MoE, 3B active | ~20GB |

Gemma 4는 QAT(4비트로 학습된) 빌드가 있어 그걸 쓰고, QAT가 없는 Qwen3.6은 unsloth의 dynamic UD 양자화를 쓴다 — 각 진영에서 Q4로 실제 배포할 물건끼리 붙인다. 넷 다 MI50 32GB 한 장에 들어가지만 **동시에는 한 개만** 올라간다.

## 돌리는 법

llama-server는 `-hf`로 띄운 모델 하나만 서빙하므로 **모델당 한 패스**다. 어댑터가 `/v1/models`로 지금 뭐가 올라와 있는지 물어보고 자기 것일 때만 돌기 때문에, 명령은 매번 똑같고 llama-server만 바꿔 재시작하면 된다. 결과는 페이지별로 **누적**되므로 나중에 모델을 하나 더 붙여도(예: Qwen3.8-27B 공개 후) 앞의 것들을 다시 돌리지 않는다.

```sh
# 0) 원문 준비 — 아직 없다면 recognizer 배치부터 (compare_out/**/ocr.json 생성)
../../venv/Scripts/python tools/compare_models.py ocrbatch --only paddleocr_manga

# 1) 모델 하나 띄우고
llama-server -hf unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL \
  -ngl 99 -c 16384 --parallel 4 --n-predict 1024 --host 0.0.0.0 --port 8080

# 2) 번역 (원격이면 LLAMACPP_ENDPOINT=http://<서버>:8080)
../../venv/Scripts/python tools/compare_models.py translate

# 3) 1~2를 모델 수만큼 반복 (llama-server만 다음 -hf로 재시작)

# 4) 채점 페이지
../../venv/Scripts/python tools/compare_models.py translatehtml
```

`list`는 후보마다 지금 돌 수 있는지와, 못 돈다면 **띄워야 할 정확한 llama-server 커맨드**를 같이 찍는다.

## 결과물

- `compare_out/<분류>/<이미지>/translate.json` — 원문 + 모델별 번역·소요 ms·배치 폴백 횟수
- `compare_out/_translate_summary.md` — 모델별 총 시간·ms/page·ms/text·폴백 수 (품질 아닌 **비용** 쪽)
- `compare_out/_compare_translate.html` — 크롭 이미지 + 원문 + 모델별 번역을 한 줄로 놓고 칸을 클릭해 투표. 분류별 채택률이 실시간 집계된다. 투표는 `trsel:` 네임스페이스라 recognizer(`ocrsel:`)·detector(`boxsel:`) 채점과 섞이지 않는다.

번역끼리는 diff 하이라이팅을 하지 않는다 — 잘된 번역 둘이 글자를 거의 안 공유하는 게 정상이라 빨간 칠이 정보가 아니라 소음이 된다.

## 읽을 때 주의

- **배치 폴백 수는 품질 신호다.** 배치 JSON이 깨졌다는 뜻이고, 대개 SFX·신음에서 반복 루프에 빠진 경우다. 문장이 아무리 좋아도 이게 잦은 모델은 프로덕션에서 페이지마다 per-text로 되돌아가 느려진다.
- **속도는 같은 조건에서만 비교된다.** MI50 한 장 기준이고 서버는 warm 상태여야 한다. 콜드 로드(~80초)가 첫 페이지에 섞이면 ms/page가 통째로 오염된다.
- `--timeout`은 기본 180초다. SDK 프로덕션 기본값(10초)으로 두면 큰 모델의 페이지 배치가 타임아웃에 걸려 폴백 경로를 채점하게 된다.

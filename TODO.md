# TODO

미뤄둔 작업 모음. translate/MI50 관련 상세·완료분은 [translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md), 설계는 [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md).

## llama.cpp recognizer — 로컬 실행 시 재설치

새 PC에서 로컬로 띄워 볼 때: venv의 `scanlation-llama-cpp` dist-info에 `scanlation.recognizers`
entry point가 없으면(설치가 recognizer 추가 이전) llama.cpp가 인식기 후보로 안 뜬다 → 재설치.

## recognizer 어드민 표면 — PaddleOCR-VL도 정리할지 (미결)

`91b8047`은 llama.cpp recognizer만 손봤다. 인프로세스
[PaddleOCR-VL](packages/scanlation-paddleocr-vl-for-manga/scanlation_paddleocr_vl_for_manga/plugin.py)은
여전히 `do_sample`·`temperature`·`top_p`·`max_new_tokens`를 `/admin`에 노출한다 — 같은 기준
("인식기의 어드민 표면은 크롭을 어떻게 읽느냐")이면 빠져야 할 것들이다. llama.cpp 전환으로 생긴 게
아니라 원래 그랬던 거라 이번 범위 밖에 뒀다.

- [ ] 같은 기준으로 정리할지 결정. 정리한다면 `max_pixels`·`downscale_mode`만 남기고 나머지는
  env 접근자로(=`91b8047`과 같은 모양).

## llama-swap — /admin 다중 모델 스왑 (선택, 필요해지면)

ollama처럼 /admin에서 여러 모델을 오가고 싶어지면 [llama-swap](https://github.com/mostlygeek/llama-swap)(Go, 오픈소스 프록시)을 `LLAMACPP_ENDPOINT` 앞에 둔다. 요청의 `model`을 읽어 해당 upstream llama-server를 띄우고 스왑하며, `/v1/models`가 설정된 모델 전부를 반환 → /admin 드롭다운이 ollama처럼 여러 모델을 담고, 고른 값이 실제로 반영된다(지금은 한 항목뿐이고 서버가 그 값을 무시한다).

- [ ] 필요해지면 llama-swap YAML(모델→실행 커맨드) + systemd 구성. 대가: 스왑 시 ~80초 콜드 로드, 기본은 한 번에 한 모델.

## recognize 게이트 — 크롭 예산 admission (선택, 분포가 요구하면)

게이트 폭은 이제 **풀 크기를 그대로 따른다**(K 다이얼 삭제, 2026-07-26 — 폭>풀은 큐만 늘고 폭<풀은 풀을 굶겨서, 조절할 값이 아니었다). 이로써 예전에 걱정하던 **"K<W 공급 퇴화"(1크롭 페이지에서 W4K2 = 사실상 W4K1)는 소멸** — 페이지당 크롭이 1개뿐이어도 이미지 W장이 들어와 풀 W개가 찬다. 남은 이론적 개선은 admission을 이미지 수가 아니라 **in-flight 크롭 수**로 세는 크롭 예산 게이트뿐인데, 이득은 "크롭 많은 페이지가 들어올 때 불필요하게 많은 이미지를 들여보내지 않음"(메모리) 정도로 줄었다.

- [ ] 크롭 예산 게이트는 **실사용 크롭 분포가 메모리 문제를 실제로 만들 때만** 구현(그 전엔 과설계).

## 참고 — translate/MI50 남은 배포

[translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md)의 "남은 일"/"복구 런북" 참조:

- [x] ~~systemd 상주 전환~~ **완료 (2026-07-15)** — `llama.cpp.service` active·enabled(재부팅 생존) + budget 플래그 없음(Option B). [deploy/llama.cpp-gemma-4-26B-A4B.service.example](deploy/llama.cpp-gemma-4-26B-A4B.service.example)
- [x] ~~MI50 최종 토폴로지~~ **완료 (2026-07-24)** — recognize=9060 XT(`HIP_VISIBLE_DEVICES=0,1` + `/admin` cuda:1) ∥ translate=MI50(llama-server `--device Vulkan1`) 물리 분리. translate는 gate 밖이라 배포만으로 활성.

## 로컬 LLM 웹 텍스트 번역 (아이디어)

상시 상주 중인 MI50 translator(gemma-4)를 임의 웹 페이지 텍스트 번역에 재사용하는 별도 프로젝트. 지금은 구글 번역 확장으로 대체 중. 재사용 자산이 크다 — translator 엔드포인트(llama.cpp/ollama) + SDK [`http_translator`](packages/scanlation-sdk/scanlation_sdk/http_translator.py)·배칭·동시성, MV2 확장 뼈대(popup·content script·DOM 주입). **스코프는 온디맨드·선택영역**으로 가는 게 맞다: 실시간 전체 페이지 자동번역은 구글이 우위(즉시성·언어 커버리지·인플레이스 UX)이고 MI50 한 장으론 못 따라간다(웹 한 장 수천 토큰 → 약 89 t/s로도 통째 번역엔 수~수십 초). LLM의 이점은 속도가 아니라 **품질**(문맥·뉘앙스·튜닝된 레지스터)·**프라이버시**(외부 미전송). 진짜 난도는 번역이 아니라 **임의 페이지 DOM 텍스트 추출·재삽입**(인라인 태그 보존, SPA·iframe·shadow DOM, 수천 노드 청킹/재주입)이고, 만화 파이프라인과 같은 MI50를 공유하므로 동시 사용 시 경합한다.

- [ ] **선행 — 기존 확장으로 대체 검토([KISS Translator](https://github.com/fishjar/kiss-translator)).** 오픈소스·Firefox·선택영역/웹페이지/입력창/자막 번역, 커스텀 OpenAI 호환 API(Ollama 포함) 지원 → MI50 엔드포인트만 꽂으면 됨. 트리거는 우클릭이 아니라 핫키 토글(Alt+Q)·호버·사이트별 자동번역 규칙. 제일 어려운 DOM 추출·재삽입을 안 짜고 품질·프라이버시 이점만 취함. 직접 구현은 만화 파이프라인과 UI 통합 등 기존 확장이 못 주는 요구가 생길 때로 미룬다.
  - 단점: 선택 번역 팝업 UI가 별로. (전체 토글·자동 규칙 위주로 쓰면 팝업은 덜 탐)
- [ ] MVP(직접 구현 시) — 우클릭 "선택 영역을 로컬 LLM으로 번역". 확장 뼈대+translator 재사용, 전체-DOM 문제 우회, 품질 이점만 취함. 만족 시 글(article) 단위 → 전체 페이지 순 확대.
- [ ] 레포 위치 결정 — 별도 레포 vs 이 모노레포에 얹기.

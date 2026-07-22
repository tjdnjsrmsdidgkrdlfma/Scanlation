# TODO

미뤄둔 작업 모음. translate/MI50 관련 상세·완료분은 [translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md), 설계는 [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md).

## /admin — llama.cpp translator가 "단일 모델"임을 반영

llama.cpp `llama-server`는 **런치 시 `-hf`로 한 모델만** 올리고 요청의 `model` 필드는 무시한다 — ollama처럼 on-demand 스왑이 안 된다. 지금 /admin은 ollama(다중 모델) 전제라 이 차이를 안 드러낸다.

- [ ] **한국어 i18n 설명 보강** — llama.cpp 플러그인 설명에 "모델은 서버 런치 시 고정, /admin에서 못 바꾼다(바꾸려면 `-hf` 수정 후 llama-server 재기동)"를 추가. catalog description의 "model selected in /admin"도 오해 소지라 같이 손볼 것.
- [ ] **모델 선택 UI** — 여러 모델 전제의 드롭다운이 단일 모델엔 안 맞는다. 단일 값 읽기전용 표시나 별도 컴포넌트 고려(ollama는 드롭다운 유지). ※ llama-swap 도입 시엔 다시 드롭다운이 맞음(아래).

## /admin — "모델 유휴 언로드" 라벨 스코프 명확화

"모델 유휴 언로드 (분)" 라벨의 "모델"이 두루뭉술해 번역 LLM으로 오해될 수 있다. 실제론 로컬 torch **탐지·인식** 엔진 전용이고([idle_unload.py](packages/scanlation-server/app/idle_unload.py), [registry.py](packages/scanlation-server/app/registry.py) `idle_candidates`), 번역기(ollama/llama.cpp)는 외부 프로세스라 무관(ollama=`OLLAMA_KEEP_ALIVE`, llama.cpp=상시 상주).

- [ ] 라벨을 "인식·탐지 모델 유휴 언로드 (분)" 등으로 좁혀 스코프를 라벨 자체에서 드러낸다. i18n ko/en **라벨만**(`behavior.idleUnload.label`) 손봄 — 설명문(`.desc`)은 이미 "ollama 번역기는 별도(OLLAMA_KEEP_ALIVE)"라 정확.

## llama-swap — /admin 다중 모델 스왑 (선택, 필요해지면)

ollama처럼 /admin에서 여러 모델을 오가고 싶어지면 [llama-swap](https://github.com/mostlygeek/llama-swap)(Go, 오픈소스 프록시)을 `LLAMACPP_ENDPOINT` 앞에 둔다. 요청의 `model`을 읽어 해당 upstream llama-server를 띄우고 스왑하며, `/v1/models`가 설정된 모델 전부를 반환 → /admin 드롭다운이 ollama처럼 부활한다.

- [ ] 필요해지면 llama-swap YAML(모델→실행 커맨드) + systemd 구성. 대가: 스왑 시 ~80초 콜드 로드, 기본은 한 번에 한 모델.

## recognize 게이트 — 이미지 수(K)가 아니라 크롭 예산으로 (선택, 분포가 요구하면)

recognize 게이트는 지금 **이미지 K장**을 들여보내는데(프로덕션 W=4·K=2, [recognize-gpu-speed.md](packages/scanlation-server/tools/recognize-gpu-speed.md) §크로스이미지 오버랩), 워커 풀을 채우는 실제 단위는 **크롭**이다. 그래서 공급 = K × 페이지당 크롭 수가 되어, **1크롭 페이지 위주 자료에선 K2 공급(2크롭)이 W4의 절반 = 사실상 W4K1**로 퇴화한다(경계 겹침 몇 %만 남음). 근본 형태는 detect가 낸 크롭을 큐에 넣고 **in-flight 크롭 수가 W에 찰 때까지 이미지를 들여보내는 크롭 예산 게이트** — K 다이얼이 없어지고 크롭 분포에 자동 적응한다. (풀 작업 큐 자체는 이미 크롭 단위 공유 FIFO — 부족한 건 상류 admission이 이미지 수를 세는 것뿐.)

- [ ] 당장은 대응 노브 존재: 1크롭 위주 자료면 /admin에서 K를 `W ÷ 평균 크롭`(=4)으로. 크롭 예산 게이트는 **그런 자료가 실사용에 실재할 때** 구현(그 전엔 과설계) — 9060 XT 재장착 후 실사용 크롭 분포 확인이 선행.

## 참고 — translate/MI50 남은 배포

[translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md)의 "남은 일"/"복구 런북" 참조:

- [x] ~~systemd 상주 전환~~ **완료 (2026-07-15)** — `llama.cpp.service` active·enabled(재부팅 생존) + budget 플래그 없음(Option B). [deploy/llama.cpp.service.example](deploy/llama.cpp.service.example)
- [ ] MI50 최종 토폴로지 — 9060 XT 재장착 후 recognize(9060)∥translate(MI50) 물리 병렬(translate는 이미 gate 밖이라 배포만으로 활성).

## 로컬 LLM 웹 텍스트 번역 (아이디어)

상시 상주 중인 MI50 translator(gemma-4)를 임의 웹 페이지 텍스트 번역에 재사용하는 별도 프로젝트. 지금은 구글 번역 확장으로 대체 중. 재사용 자산이 크다 — translator 엔드포인트(llama.cpp/ollama) + SDK [`http_translator`](packages/scanlation-sdk/scanlation_sdk/http_translator.py)·배칭·동시성, MV2 확장 뼈대(popup·content script·DOM 주입). **스코프는 온디맨드·선택영역**으로 가는 게 맞다: 실시간 전체 페이지 자동번역은 구글이 우위(즉시성·언어 커버리지·인플레이스 UX)이고 MI50 한 장으론 못 따라간다(웹 한 장 수천 토큰 → 약 89 t/s로도 통째 번역엔 수~수십 초). LLM의 이점은 속도가 아니라 **품질**(문맥·뉘앙스·튜닝된 레지스터)·**프라이버시**(외부 미전송). 진짜 난도는 번역이 아니라 **임의 페이지 DOM 텍스트 추출·재삽입**(인라인 태그 보존, SPA·iframe·shadow DOM, 수천 노드 청킹/재주입)이고, 만화 파이프라인과 같은 MI50를 공유하므로 동시 사용 시 경합한다.

- [ ] **선행 — 기존 확장으로 대체 검토([KISS Translator](https://github.com/fishjar/kiss-translator)).** 오픈소스·Firefox·선택영역/웹페이지/입력창/자막 번역, 커스텀 OpenAI 호환 API(Ollama 포함) 지원 → MI50 엔드포인트만 꽂으면 됨. 트리거는 우클릭이 아니라 핫키 토글(Alt+Q)·호버·사이트별 자동번역 규칙. 제일 어려운 DOM 추출·재삽입을 안 짜고 품질·프라이버시 이점만 취함. 직접 구현은 만화 파이프라인과 UI 통합 등 기존 확장이 못 주는 요구가 생길 때로 미룬다.
  - 단점: 선택 번역 팝업 UI가 별로. (전체 토글·자동 규칙 위주로 쓰면 팝업은 덜 탐)
- [ ] MVP(직접 구현 시) — 우클릭 "선택 영역을 로컬 LLM으로 번역". 확장 뼈대+translator 재사용, 전체-DOM 문제 우회, 품질 이점만 취함. 만족 시 글(article) 단위 → 전체 페이지 순 확대.
- [ ] 레포 위치 결정 — 별도 레포 vs 이 모노레포에 얹기.

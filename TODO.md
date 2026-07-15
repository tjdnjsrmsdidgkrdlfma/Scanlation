# TODO

미뤄둔 작업 모음. translate/MI50 관련 상세·완료분은 [translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md), 설계는 [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md).

## /admin — llama.cpp translator가 "단일 모델"임을 반영

llama.cpp `llama-server`는 **런치 시 `-hf`로 한 모델만** 올리고 요청의 `model` 필드는 무시한다 — ollama처럼 on-demand 스왑이 안 된다. 지금 /admin은 ollama(다중 모델) 전제라 이 차이를 안 드러낸다.

- [ ] **한국어 i18n 설명 보강** — llama.cpp 플러그인 설명에 "모델은 서버 런치 시 고정, /admin에서 못 바꾼다(바꾸려면 `-hf` 수정 후 llama-server 재기동)"를 추가. catalog description의 "model selected in /admin"도 오해 소지라 같이 손볼 것.
- [ ] **모델 선택 UI** — 여러 모델 전제의 드롭다운이 단일 모델엔 안 맞는다. 단일 값 읽기전용 표시나 별도 컴포넌트 고려(ollama는 드롭다운 유지). ※ llama-swap 도입 시엔 다시 드롭다운이 맞음(아래).

## llama-swap — /admin 다중 모델 스왑 (선택, 필요해지면)

ollama처럼 /admin에서 여러 모델을 오가고 싶어지면 [llama-swap](https://github.com/mostlygeek/llama-swap)(Go, 오픈소스 프록시)을 `LLAMACPP_ENDPOINT` 앞에 둔다. 요청의 `model`을 읽어 해당 upstream llama-server를 띄우고 스왑하며, `/v1/models`가 설정된 모델 전부를 반환 → /admin 드롭다운이 ollama처럼 부활한다.

- [ ] 필요해지면 llama-swap YAML(모델→실행 커맨드) + systemd 구성. 대가: 스왑 시 ~80초 콜드 로드, 기본은 한 번에 한 모델.

## 참고 — translate/MI50 남은 배포

[translate-gpu-mi50.md](packages/scanlation-server/tools/translate-gpu-mi50.md)의 "남은 일"/"복구 런북" 참조:

- [x] ~~systemd 상주 전환~~ **완료 (2026-07-15)** — `llama.cpp.service` active·enabled(재부팅 생존) + budget 플래그 없음(Option B). [deploy/llama.cpp.service.example](deploy/llama.cpp.service.example)
- [ ] MI50 최종 토폴로지 — 9060 XT 재장착 후 recognize(9060)∥translate(MI50) 물리 병렬(translate는 이미 gate 밖이라 배포만으로 활성).

# CLAUDE.md

이 저장소에서 Claude Code의 **대화·응답 방식**과 **항상 지켜야 할 코드 규칙·커밋 규칙** 지침. (PC 간 공유를 위해 커밋해 둔다. 상세한 프로젝트 사실·설계·워크플로는 [README.md](README.md) / [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md) / 코드를 따른다.)

## 커뮤니케이션
- 모든 출력은 **한국어**로 작성한다 — 채팅 답변뿐 아니라 계획·요약 등 내가 쓰는 파일 내용까지. 코드·식별자·CLI 명령은 영어를 유지하고, 인라인 주석은 주변 스타일을 따른다(이 트리 주석은 영어).
- **존댓말**을 쓰고 반말은 쓰지 않는다. 사용자가 반말로 말해도 그 말투를 따라가지 않는다.
- 2인칭 호칭으로 **"당신"을 쓰지 않는다.** 한국어는 2인칭을 자연스럽게 생략하므로 빼거나 다시 표현한다.

## 수정할 때
- 요청한 변경만 담백하게 한다. 부수적 서술을 반사적으로 덧붙이지 않는다 — 특히 (1) **특정 환경 관측치를 일반 사실처럼 박기**(예: 한 셋업의 "14GB" VRAM 값), (2) **회고적 이력 주석**("이전엔 X였다", "⚠ 이후 변경됨", 구 목표 폐기 서술). 현재의 일반적 진실만 현재형으로 쓴다. 그 디테일이 정말 load-bearing일 때만 예외로 남긴다.

## 코드 규칙
- **하드코딩 금지 → `/admin` 노출.** 동작을 좌우하는 매직 값(임계값·크기·개수 등)을 코드에 리터럴로 박지 않는다. 새 조절값은 (1) env 기본값 + `state.json` 영속 서버 설정, (2) `/admin` UI 필드로 노출(i18n 포함), (3) 확장 동작에 영향 주는 값이면 handshake(`GET /`) 응답에 실어 popup→storage→content로 전달. 기존 하드코딩을 건드리게 되면 이 규칙대로 빼내는 걸 우선한다.
- **역할 어휘.** 역할은 `detector`/`recognizer`/`translator`, 결과 아이템 키는 `{bounds, source, destination}`, 내부 옵션은 `opt_detect`/`opt_recognize`/`opt_translate`. 역할 라벨로 **BOX/OCR/TSL은 코드·로그·주석·문서·와이어 어디에도 쓰지 않는다.** (예외: 제품명 manga-ocr/PaddleOCR의 일반 "OCR", 기하학적 "bounding box"는 그대로.)
- **plugin vs engine.** `plugin` = 설치 단위(pip 패키지·plugins 볼륨·설치 액션·catalog·admin 플러그인 탭), `engine` = 런타임(registry·엔진 선택·파이프라인 실행·engine_meta·cache). 새 코드·주석·이름은 이 구분을 따른다. 의도적 예외(리네임 금지): env `SCANLATION_ENGINE_REPO`/`_REF`와 `engine_repo()`/`engine_ref()`는 설치 소스 설정이라 유지, admin의 "model"은 LLM 모델 태그라 유지.
- **내부 패키지 버전 bump 금지.** 모노레포의 내부 패키지(`scanlation-sdk`·엔진 플러그인)는 항상 같은 `@main` 커밋에서 함께 설치되고(`pip --upgrade`가 재fetch) 버전 문자열이 의존 해석에 관여하지 않는다. 새 SDK API를 추가·사용해도 `version`이나 의존 하한(`scanlation-sdk>=…`)을 올리지 않는다 — `/admin` 재설치는 버전과 무관하게 새 코드를 집는다.

## 커밋·푸시
- **커밋 메시지 언어: 제목은 영어, 본문은 한국어.** 제목(subject)은 `type(scope): English subject`(conventional commits) — 위 커뮤니케이션의 "모든 출력 한국어"에 대한 **예외**다(제목은 영어라 grep·스캔에 무난). 한국어 상세 서술은 **본문(body)에만** 둔다.
- 커밋하면 **같은 흐름에서 바로 `git push origin main`까지** 한다. "푸시할까요?"를 매번 따로 묻지 않는다. 커밋 지시나 계획 승인 후 구현이 끝나면 테스트 green 확인 → 커밋 → push → 결과(`old..new` 해시) 보고. force push·히스토리 재작성 등 되돌리기 어려운 경우만 예외적으로 먼저 확인. (솔로 프로젝트라 main 직접 커밋+푸시가 기본.)

## 메모리
- `.claude` 자동 메모리는 이 기기의 `~/.claude`에만 저장돼 **회사 PC·집 PC·리눅스 서버 간 동기화되지 않는다.** 지속 지침은 커밋되는 파일에 둔다: 응답 방식 → 이 CLAUDE.md, 프로젝트 사실·설계·워크플로 → [README.md](README.md) / [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md). `.claude/.../memory/`에 새로 저장하지 않는다.

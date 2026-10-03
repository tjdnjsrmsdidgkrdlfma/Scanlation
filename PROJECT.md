# Scanlation

이 저장소에서만 통하는 것들. 어느 프로젝트에서나 참인 것은 [CLAUDE.md](CLAUDE.md)에 있다. 프로젝트 사실·설계·워크플로는 [README.md](README.md) / [SCANLATION_DESIGN.md](SCANLATION_DESIGN.md) / 코드를 따른다.

## 말투와 언어

- 대화는 **존댓말**로 한다. 사용자가 반말로 말해도 그 말투를 따라가지 않는다.
- 2인칭 호칭으로 **「당신」을 쓰지 않는다.** 한국어는 2인칭을 자연스럽게 생략하므로 빼거나 다시 표현한다.
- 코드·식별자·CLI 명령은 영어를 유지한다. 인라인 주석은 주변 스타일을 따른다(이 트리 주석은 영어).

## 수정할 때

- 요청한 변경만 담백하게 한다. 부수적 서술을 반사적으로 덧붙이지 않는다.
- **특정 환경 관측치를 일반 사실처럼 박지 않는다**(예: 한 셋업의 「14GB」 VRAM 값). 그 디테일이 정말 load-bearing일 때만 남긴다.

## 코드 규칙

- **하드코딩 금지 → `/admin` 노출.** 동작을 좌우하는 매직 값(임계값·크기·개수 등)을 코드에 리터럴로 박지 않는다. 새 조절값은 (1) env 기본값 + `state.json` 영속 서버 설정, (2) `/admin` UI 필드로 노출(i18n 포함), (3) 확장 동작에 영향 주는 값이면 handshake(`GET /`) 응답에 실어 popup→storage→content로 전달. 기존 하드코딩을 건드리게 되면 이 규칙대로 빼내는 걸 우선한다.
- **역할 어휘.** 역할은 `detector`/`recognizer`/`translator`, 결과 아이템 키는 `{bounds, source, destination}`, 내부 옵션은 `opt_detect`/`opt_recognize`/`opt_translate`. 역할 라벨로 **BOX/OCR/TSL은 코드·로그·주석·문서·와이어 어디에도 쓰지 않는다.** (예외: 제품명 manga-ocr/PaddleOCR의 일반 "OCR", 기하학적 "bounding box"는 그대로.)
- **plugin vs engine.** `plugin` = 설치 단위(pip 패키지·plugins 볼륨·설치 액션·catalog·admin 플러그인 탭), `engine` = 런타임(registry·엔진 선택·파이프라인 실행·engine_meta·cache). 새 코드·주석·이름은 이 구분을 따른다. 의도적 예외(리네임 금지): env `SCANLATION_ENGINE_REPO`/`_REF`와 `engine_repo()`/`engine_ref()`는 설치 소스 설정이라 유지, admin의 "model"은 LLM 모델 태그라 유지.
- **내부 패키지 버전 bump 금지.** 모노레포의 내부 패키지(`scanlation-sdk`·엔진 플러그인)는 항상 같은 `@main` 커밋에서 함께 설치되고(`pip --upgrade`가 재fetch) 버전 문자열이 의존 해석에 관여하지 않는다. 새 SDK API를 추가·사용해도 `version`이나 의존 하한(`scanlation-sdk>=…`)을 올리지 않는다 — `/admin` 재설치는 버전과 무관하게 새 코드를 집는다.

## 커밋

- **커밋 메시지는 전부 영어.** 제목은 `type(scope): English subject`(conventional commits), 본문도 영어로 쓴다. [CLAUDE.md](CLAUDE.md)의 「모든 출력은 한국어」는 **커밋 메시지엔 적용하지 않는다.**
- **`Co-Authored-By` 트레일러를 메시지에 직접 쓴다.** `Co-Authored-By: Claude <모델명> <noreply@anthropic.com>`(예: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`). 기기에 따라 자동으로 안 붙을 수 있으니 설정에 기대지 않는다.

## 배포 서버

- 리눅스 배포 서버(translate=MI50 / recognize=9060 XT)의 접속 정보는 `~/.ssh/hosts.txt`의 `[scanlation-deploy]` 블록에 있다. 접속 방법은 [CLAUDE.md](CLAUDE.md) 「서버 접속」.
- **GPU를 오래 돌리는 작업에는 온도 가드를 넣는다.** MI50는 온도 스로틀이 없어 완충이 없다 — junction(`temp2_input`, hwmon은 PCI `1002:66A1`로 판별)을 폴링해 95°C에 닿으면 작업을 중단한다.
- 서버에서 무거운 측정을 할 때는 **프로덕션 유닛을 쓴다**(자원 경쟁·변인 오염 없음). 별도 인스턴스를 같은 카드에 얹으면 VRAM이 모자라 시스템 메모리로 밀려나고 PCIe 대역폭에 묶여 측정이 무의미해진다. 끝나면 원복은 `trap`으로 보장한다.

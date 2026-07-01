# CLAUDE.md

이 저장소에서 Claude Code가 따라야 할 지침. (PC 간 공유를 위해 로컬 메모리 대신 커밋되는 이 파일에 둔다.)

## 커뮤니케이션
- 모든 출력은 **한국어**로 작성한다 — 채팅 답변뿐 아니라 계획·요약 등 내가 쓰는 파일 내용까지. 코드·식별자·CLI 명령은 영어를 유지하고, 인라인 주석은 주변 스타일을 따른다(이 트리 주석은 영어).
- **존댓말**을 쓰고 반말은 쓰지 않는다. 사용자가 반말로 말해도 따라가지 않는다.
- 2인칭 호칭으로 **"당신"을 쓰지 않는다.** 한국어는 2인칭을 자연스럽게 생략하므로 빼거나 다시 표현한다.

## Git 워크플로
- "push" 요청 시 `main`에 **직접** 커밋·푸시한다. 피처 브랜치·PR 금지(솔로 개인 저장소, 선형 히스토리).
- 절차: `git add` → 커밋(Co-Authored-By 트레일러 포함) → `git push origin main`. `a..b main -> main` 줄로 성공 확인.

## 개발 환경
- 레포 루트 가상환경 이름은 **`venv`**(절대 `.venv` 아님). venv 참조를 바꿀 땐 트리 전체를 grep.
- 의존성은 그 `venv`에만 — **전역 `pip install` 금지**. 설치: `pip install -e "./server[ctd,mangaocr]"`.

## 테스트 — pytest 미사용, 자체 핸드롤 러너
- **pytest를 쓰지 않는다.** 각 테스트 파일은 plain `test_*` 함수 + `TESTS = [...]` 목록 + `helpers.run(TESTS, title)` 실행 규약을 따른다(ComprehensiveScraper와 동일 스타일).
- 실행(반드시 `server/`에서): 빠른 전체 `python -m tests`, 개별 파일 `python -m tests.test_routes`.
- 모델 필요한 스모크(`test_ctd`, `test_mangaocr`)는 빠른 스위트에서 제외하고 개별 실행하며, 가중치/패키지 없으면 `"SKIP: ..."` 문자열을 반환해 스스로 건너뛴다.
- 공용 헬퍼는 [server/tests/helpers.py](server/tests/helpers.py)(TestClient·payload·run), 데이터 격리는 [server/tests/__init__.py](server/tests/__init__.py).

## 엔진 역할 어휘
- 역할 이름은 끝단까지 **detector / recognizer / translator** 한 어휘로 통일한다(BOX/OCR/TSL·매핑 레이어 없음). 구 `ocr_extension` 호환은 지원하지 않는다.
- 결과 아이템 키 `{ocr, tsl, box}`는 역할이 아니라 데이터 필드이므로 그대로 둔다.

## 설계 원칙
- 상세 설계 근거는 [YOMU_DESIGN.md](YOMU_DESIGN.md), 현재 상태·배포는 [README.md](README.md). 발명하지 말고 이 문서들을 먼저 읽는다.
- **모노레포 유지** — 서버(`server/`)+확장(`extension/`)이 와이어 계약을 공동 진화시키므로 한 리포에 둔다. 독립 배포/라이브러리 공개 같은 구체적 트리거 전까지 분리하지 않는다(분리 시 `git subtree split`으로 히스토리 보존).
- 정확도 최우선. 검출이 병목이므로 회전 기하(폴리곤/각도/마스크)를 계약이 품는다.

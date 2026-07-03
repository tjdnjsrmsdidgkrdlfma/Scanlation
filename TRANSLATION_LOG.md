# 번역 로그(TM, `translations` 테이블) — 현재 상황

> 결정 대기 메모. 캐시의 `translations` 테이블이 **지금은 아무도 안 읽는 write-only 상태**라
> 어떻게 할지(제거 / 제한적 재사용 / 보존) 정리해 둔다. 현재 상태·용법은
> [README.md](README.md)를 우선한다.

## 캐시는 두 종류

서버 캐시 DB(`data/scanlation.sqlite`, [cache.py](packages/scanlation-server/app/cache.py)):

| 테이블 | 키 | 내용 | 상태 |
|---|---|---|---|
| `ocr_runs` | (md5, src, dst, engines, opt_hash) | **페이지 통째 결과** — lazy 흐름의 핵심 | ✅ 활발히 사용 |
| **`translations`** | (원문, src_lang, dst_lang, model) | **말풍선 하나하나의 번역**(원문→번역문) | ⚠️ 아무도 안 읽음 |

## `translations`(번역 로그)의 지금 동작

- **쓰기 O** — `/run_pipeline/`이 돌 때마다 `translate_regions → _translate_all → put_translation`으로
  각 말풍선 번역이 계속 쌓인다 ([pipeline.py](packages/scanlation-server/app/pipeline.py)).
- **읽기 X** — 이 테이블을 읽어서 재사용/조회하는 코드가 **없다.** `clear_cache`가 비우기만 한다.
- 원래 용도는 구 `ocr_extension`의 **번역 메모리(TM)** — "이 텍스트의 과거 번역 조회". 클린룸 확장은 안 쓴다.

→ **결론: 순수 write-only dead weight.** (PK가 `(원문,언어,model)`이라 같은 원문은 덮어써서 무한정
커지진 않고, 서로 다른 원문 수만큼 쌓인다.)

## "같은 텍스트면 가져다 쓰기"가 왜 없나 (의도된 것)

텍스트 단위 재사용은 **구현돼 있지 않고, 그게 설계와 맞는다:**

- 번역은 **이미지 단위 전체 배치** — 한 이미지의 말풍선들을 **한 LLM 호출로 상호 문맥과 함께** 번역해
  일관성·정확도를 높인다(정확도 최우선 원칙).
- 그래서 **같은 원문도 이미지(문맥)가 다르면 다르게 번역되는 게 정상**이다(예: `はい` → 네/그래/응…).
- 게다가 `translations` 키에는 **문맥이 없다**(원문+언어+model뿐). 여기서 가져다 쓰면 **문맥 무시 번역**을
  재사용하는 셈이라 배치 설계와 충돌한다.
- 반면 **이미지(md5) 캐시(`ocr_runs`)는 정상 동작** — 같은 페이지 재방문은 즉시(전체 결과 재사용).

## 선택지

1. **제거(추천)** — `put_translation` 호출 + `translations` 테이블 + `get_translations`/`put_translation` +
   `clear()`의 관련 줄 + 이미 orphan인 `translate_text` 함수 제거, `test_clear_cache`를 `ocr_runs`만
   검증하도록 조정. 정확도 최우선 프로젝트라 문맥 무시 재사용은 안 맞고, 이미지 캐시가 이미 큰 이득을 준다.
2. **제한적 재사용** — 문맥 영향이 적은 **짧은/독립 텍스트(SFX·단발 단어)에만** md5 미스 시 TM 폴백.
   GPU 절약 ↔ 약간의 오역 리스크 tradeoff.
3. **보존** — 나중에 "이 말풍선의 다른 번역 후보 보기" 기능 여지로 남긴다.

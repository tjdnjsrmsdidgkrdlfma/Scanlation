# 리팩토링 — 완료 아카이브 + 남은 것

트리 전체를 네 축(**벤치 통합 · 어휘 정리 · 대형 파일 분할 · 하드코딩→`/admin`**)으로 훑어 R1~R9를, 그 뒤 자라난 부분을 R10 재스윕으로 정리했다 — **구조 부채는 소진.** 완료 항목별 상세·커밋은 git 로그에 있다(이 문서는 더 나열하지 않는다). 원칙은 **동작 보존**; 코드·커밋 규칙은 [CLAUDE.md](CLAUDE.md).

이 문서에 남긴 건 셋뿐이다 — **아직 열린 항목**, **재제안 금지(의도적으로 안 한 것)**, **다른 PC 인수인계용 venv 진단**.

---

## 남은 것 (열린 항목)

측정 장비/소유자 판단이 필요해 미뤄둔 것뿐. (일반 미뤄둔 작업은 [TODO.md](TODO.md).)

- **B5·B6 — 벤치 크롭 세트 불일치 (GPU 호스트에서만).** `_bench_common.deskewed_crops`(batch/gpuconc, deskew)와 [bench_recognize_threads.py](packages/scanlation-server/tools/bench_recognize_threads.py)의 `_raw_bbox_crop_files`(threads, raw bbox)는 **크롭 픽셀이 다르다** → [recognize-crop-batching.md](packages/scanlation-server/tools/recognize-crop-batching.md)의 "배칭 1.27x가 멀티워커 1.8x에 진다"는 apples-to-apples가 아니다. 리포트 산문의 `"best ~1.8x over base"`(B6)도 측정값과 무관하게 하드코딩돼 있다. 고치는 것: threads 크롭을 deskew로 바꾸고 재측정 → `tools/*.md` 결론 갱신.
- **LICENSE 파일 부재 (소유자 보류).** `tools/vendored/`에 manga-image-translator GPL 사본(`_mit_ocr_48px.py`·`_mit_ocr_ctc.py`, ~1,003줄)과 MIT `_mit_xpos.py`가 있다 — **research-only·tools 전용·wheel/Docker 미포함**(제품/런타임엔 GPL 없음; deskew는 독립 재구현). bake-off 하네스라 **유지 결정**(2026-07-09). 남은 건 저장소에 `LICENSE`/`COPYING`이 없다는 것 — 공개 배포하려면 upstream COPYING + `_mit_xpos.py`용 MIT 고지가 필요하다.

---

## 재제안 금지 (의도적으로 안 한 것)

동작 보존·범위 제약으로 일부러 두었다. 다음 감사가 다시 파헤치지 않도록:

- **TEMP occupancy 계측 유지** — `recognize_pool`의 4-tuple 반환·`_OCC`·`_split_occ`·occupancy 헬퍼 3개, `admin`의 `/bench_occupancy(_reset)/`, `tools/bench_occupancy.py`("revert via git" 라벨). 9060 XT 재장착 후 크롭 분포 재측정에 재사용하려 소유자가 남김. 결론은 [recognize-gpu-speed.md](packages/scanlation-server/tools/recognize-gpu-speed.md)에 기록됨.
- **compare/ 역할 라벨** — `ocrsel:`/`boxsel:` localStorage 키, `BOX compare` 제목, 서브커맨드 의미는 **데이터·계약**이라 개명 금지(저장된 투표가 고아가 된다).
- **확장 타이포 노브** — 박스 채움 `0.8`·최소폰트 `7px`·디바운스 `150ms`는 content-script 내부 휴리스틱, `/admin` 노출은 저가치.
- **계약 표면 유지** — `SUPPORTED_SRC/DST`(소비하면 언어 필터로 동작이 바뀌고, 삭제하면 플러그인 정보 손실), `Region.mask/angle`(deskew·future inpaint), `Recognizer.recognize(region=)`(엔진 계약)는 소비 여부가 설계 결정이라 유지.
- **A등급 내부 상수** — `geometry.py`의 eps/min_size/패딩색, `cache.py`의 `[:16]`, pip 에러 tail 길이 등은 알고리즘·포맷 상수라 `/admin`에 올리면 노이즈. `prompt.py`의 "OCR error" 문구는 LLM 시스템 프롬프트 본문이라 바꾸면 모델 동작이 바뀐다.
- **벤치 warm()/add_data_args/sweep_baseline 통합, 확장 fetch 타임아웃** — 측정 조건·CLI가 바뀌거나 없던 동작 신설이라 범위 밖.
- **엔트리포인트 이름·플러그인 attr 이름** — `Ollama`/`llama.cpp`/`PaddleOCR-VL-For-Manga` 등은 업스트림 고유명(kebab 강제 통일 금지, `state.json`·캐시 키의 일부), `MODEL_REPO`/`REPO`/`self._m` 등 drift는 공유 표면 없는 cosmetic.
- **내부 패키지 버전 bump** — `@main` 동시 설치라 의존 해석에 무의미(CLAUDE.md).

---

## 다른 PC 인수인계 — venv 설치 메타데이터

`git pull` 후, 저장소에 담기지 않는 것: **editable 설치의 dist-info.** 패키지/entry-point 이름이 바뀐 뒤 재설치를 안 했다면 `importlib.metadata`가 낡은 이름을 본다 → `registry`와 `catalog`가 같은 엔진을 다른 이름으로 부른다. 유령 entry_point는 `_discover`가 삼켜 `/admin`엔 안 뜨고 서버 로그에 `scanlation.registry` warning으로만 남는다.

```bash
# 유령 entry_point 점검 — 로드 실패가 있으면 낡은 설치다
venv/Scripts/python -c "
from importlib.metadata import entry_points
for g in ('scanlation.detectors','scanlation.recognizers','scanlation.translators'):
    for ep in entry_points(group=g):
        try: ep.load(); s='ok'
        except Exception as e: s=type(e).__name__
        print(g, ep.name, s)"

# 고치기: 사라진 패키지를 지우고 남은 것을 editable 재설치
venv/Scripts/python -m pip uninstall -y <구-패키지명>
venv/Scripts/python -m pip install -e packages/scanlation-sdk -e packages/scanlation-server --no-deps
```

엔진 가중치도 저장소 밖이다 — `manga-ocr`/`comic-text-and-bubble-detector`/`paddleocr` 스모크는 가중치가 없으면 auto-skip한다(코어 단위는 모델 없이 돈다). 엔진 본문을 건드린 변경은 가중치 있는 머신에서 검증한다.

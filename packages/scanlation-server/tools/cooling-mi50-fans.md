# MI50 팬·냉각 설정 (ARCTIC S4028-15K + 3D 쉬라우드)

작성 2026-07-20. MI50(패시브 서버 카드)를 데스크톱 섀시에서 능동 공랭하기 위한 팬/쉬라우드/`fancontrol` 설정 계획 + 실행 런북. 배경과 기존 쿨링 관측(무냉각 크래시·전력 캡·온도 임계값)은 [translate-gpu-mi50.md §MI50 쿨링](translate-gpu-mi50.md#mi50-쿨링--능동-냉각-필수)에 있고, 여기서 반복하지 않는다.

> **상태 (2026-08-06):** 운영은 맨팬 1개, **`fancontrol`이 MI50 junction 온도로 `pwm4`를 제어한다.** 구성: SYS_FAN2 헤더, 커브 `MINTEMP 60 / MAXTEMP 80 / MINPWM 26(1,587rpm) / MAXPWM 74(4,983rpm)`, 부팅 영속은 `fancontrol.service` + `/etc/modprobe.d/nct6687.conf`([deploy/](../../../deploy/README.md)). **Task 1·2·4 완료**(결과 아래). **Task 3(2팬 vs 3팬 A/B)은 쉬라우드 장착 대기.** **Task 5(온도 알람)는 보류** — 유휴 복귀가 빠르고(10초에 수십 °C 하강 실측) 커브가 junction을 직접 보므로.
>
> **지속 부하는 맨팬으로 못 잡는다 (2026-08-06 실측).** 3분 연속 풀로드에서 junction이 50초 만에 96°C에 닿는다. 팬 상한 5,208rpm은 46초에 97°C, 7,075rpm은 50초에 96°C — **36% 빠른 팬이 1°C를 샀다.** 팬 rpm은 이 문제의 레버가 아니다. 쉬라우드 없는 정압 누설이 원인이므로 Task 3(쉬라우드) 또는 전력 캡 150→120W로 풀어야 한다. 실사용 패턴인 16~42초 버스트는 현재 커브로 여유가 있다(포화 P1/P2/P4·파이프라인 conc4 전부 완주, 피크 ≤89°C — [translate-gpu-mi50.md §신냉각 1차 실측](translate-gpu-mi50.md)).

## 하드웨어 확정 사항 (조사·실측 완료, 재논의 불필요)

| 항목 | 값 |
|---|---|
| GPU | AMD Instinct MI50 32GB (Vega20/gfx906), 패시브. 최종 **2~3장** 다장착 예정 |
| 전력 캡 | **150W** (vBIOS 기본 225W). 토큰 생성 속도차 ≈ 0. 필요 시 120W까지 여지 |
| 팬 | **ARCTIC S4028-15K** — 40×40×28, PWM 4핀, 1,400~15,000 rpm, 0.47A/5.6W 풀로드, 듀얼볼베어링, 정압 26.5 mmH₂O |
| 냉각 방식 | 카드 끝단 흡입구에 깔때기형 3D 프린팅 쉬라우드로 강제 공랭. **쉬라우드 2종(2팬·3팬) 모두 보유** |
| 팬 결선 | 보드 팬 헤더 + **tach 한쪽만 연결된 Y스플리터**(2팬 구성 시). 허브 안 씀. **카드당 헤더 1개 → 카드별 독립 제어** |
| 팬 실측 | PWM 50% = 8,000 rpm (선형, 100%≈15,000 예상). 바람 방향·PWM 제어·바인딩 정상 |
| **소음 상한** | **8,000~9,000 rpm.** 그 위(9,000~15,000)는 실사용 안 함 — 냉각 설계의 하드 제약 |

## 미확정 — 이 작업으로 결정할 것

- **2팬 vs 3팬.** 분석상 2팬 우세(병렬 팬 이득 2~6%뿐, 2팬이 ~1.2 dB 조용, 허브 불필요). 단 카드 끝단 흡입 개구부 높이가 커서 2팬이 못 덮으면 3팬이 유리. **양쪽 쉬라우드 실물이 있으므로 추측 말고 Task 3의 실측 A/B로 결정.**

## OS / 스택

- Linux, llama.cpp Vulkan 백엔드. GPU는 `amdgpu` 드라이버.

## 핵심 원칙 (설정 시 반드시 지킬 것)

1. **보드는 GPU 온도를 모른다.** SYS_FAN 헤더는 기본 CPU/보드 온도로 팬을 돌린다 → BIOS 팬커브로는 절대 해결 안 됨. 반드시 소프트웨어(`fancontrol`)로 `amdgpu` 온도에 매핑. (기존 관측: CPU 커브는 GPU 단독 부하를 못 추적해 junction 105°C까지 간 실측 있음 — [translate-gpu-mi50.md:38](translate-gpu-mi50.md#L38).)
2. **판단 기준 온도는 `junction`(temp2)과 `mem`(temp3).** `edge`(temp1)는 느긋해서 쓰지 말 것. Vega20/MI50은 **HBM2(mem)가 먼저 병목**되는 경우가 많으니 mem을 반드시 같이 감시.
3. **온도 상한:** junction ~95°C, **mem ~90°C 목표 상한**(mem 95°C가 실질 물리 한계, amdgpu crit=94°C). 스로틀은 junction ~100°C(crit) 부근.
4. **팬 고장은 tach로 감지 불가**(Y스플리터가 한쪽 tach만 리턴). **감시는 팬 rpm이 아니라 GPU 온도로 한다.**

## Task 1 — 센서 확인

- `sensors` 출력에서 `amdgpu` hwmon을 찾고, temp1/temp2/temp3이 각각 edge/junction/mem 중 무엇인지 매핑 확정 후 보고(라벨은 드라이버 버전마다 다를 수 있으니 추측 금지, 실제 출력으로 확인).
- 보드 팬 제어 칩(`nct6775` 등) hwmon과 `pwmX` 경로도 확인.
- 참고: 기존 관측상 `rocm-smi --showtemp`가 edge·junction·mem 셋 다 노출하고, amdgpu crit은 junction 100/emerg 105, mem crit 94/emerg 99 — Task 1은 hwmon `tempN` 라벨을 이 값에 대응시켜 최종 확정만 하면 됨.

**결과 (2026-07-26) — 완료.**

- **MI50** = PCI `0000:03:00.0`, amdgpu hwmon의 `temp1/2/3` 라벨 = **edge/junction/mem** (라벨 실측 확인).
- **보드 팬 칩 = NCT6687D**(MSI PRO B850M-A WIFI). EL10 커널엔 드라이버가 없어(`nct6683` 모듈 미탑재) [Fred78290/nct6687d](https://github.com/Fred78290/nct6687d)를 빌드해 `/lib/modules/$(uname -r)/extra/` + `modules-load.d`로 영구 로드했다. **Secure Boot는 무서명 모듈을 거부하므로 꺼야 한다**(BIOS에서 Disabled 처리됨). **커널 업데이트 시 재빌드 필요.**
- **ARCTIC 팬 채널 = `nct6687` hwmon의 ch4**(라벨 "System Fan #2"). 수동 제어: `pwm4_enable`에 1 → `pwm4`에 0~255.
- **EC 특성:**
  - **`msi_fan_brute_force=1`이 필수다.** 없으면 EC가 `pwm4`에 쓴 값을 1초 안에 자기 커브 값으로 되돌리고(readback이 목표와 다름), **duty 바닥이 강제**돼 pwm 0에도 ~4,360rpm 아래로 안 내려간다. 켜면 커브 7포인트를 전부 써서 EC를 밀어내므로 쓴 값이 그대로 유지되고 바닥도 사라진다(pwm 0 = 1,046rpm). `/etc/modprobe.d/nct6687.conf`에 고정한다 — dmesg의 `MSI fan brute force mode: enabled`로 확인.
  - 값을 **내리는 방향은 EC가 수 초~수십 초에 걸쳐 수렴**시킨다(올리는 방향은 즉시). `fancontrol`은 `INTERVAL`마다 다시 쓰므로 결국 수렴한다.
  - **자동 모드(`pwm4_enable=2`)는 CPU 온도 커브다.** MI50이 49°C로 놀아도 CPU가 48→75°C로 오르면 팬이 3,846→7,042rpm까지 따라 올라가고, 반대로 CPU가 놀고 MI50만 뜨거우면 팬이 안 올라간다 — §핵심 원칙 1의 실증이다.

## Task 2 — PWM↔rpm 매핑 측정 (새 ARCTIC 팬 장착 후)

- 팬이 물린 `pwmX`에 값을 직접 write(수동 제어 활성화)하며 실제 rpm 측정, 표 작성:
  - PWM 듀티 **20/30/40/50/60/70/80/100%** → 실측 rpm
- 목적: `fancontrol` 커브의 MINPWM/MAXPWM을 실측으로 넣기 위함. 특히 **8,500 rpm에 해당하는 PWM 듀티값**을 정확히 확보(MAXPWM 캡에 사용).

**결과 (2026-08-06) — 완료** (맨팬 1개, `msi_fan_brute_force=1`, `pwm4_enable=1`):

| pwm | 0 | 26 | 51 | 77 | 102 | 128 |
|---|---|---|---|---|---|---|
| rpm | 1,046 | 1,587 | 3,432 | 5,154 | 7,075 | 8,823 |

- **소음 체감**: 5,000rpm부터 들리기 시작하고 6,400rpm이면 꽤 시끄럽다. 7,075rpm은 GPU 부하 중에만 나온다면 감수 가능한 수준.
- **MINPWM 하한은 26**(1,587rpm)으로 잡는다 — pwm 0의 1,046rpm은 팬 정격(1,400~15,000rpm) 밖이라 회전이 불안정할 수 있다.
- 쉬라우드 장착 후에는 정압이 달라지므로 이 표를 다시 뜬다.

## Task 3 — 쉬라우드 A/B 온도 비교 ★ (쉬라우드 도착 후에만)

- **쉬라우드가 없으면 이 Task 건너뛰고 "쉬라우드 대기 중" 명시 보고.** 맨 팬 상태 온도는 정압 누설로 무의미하므로 측정 금지.
- **절차** — 같은 카드·같은 부하·같은 rpm으로 **2팬 쉬라우드와 3팬 쉬라우드를 번갈아** 테스트해 직접 비교:
  1. **rpm 고정 8,500** (Task 2에서 구한 듀티로 고정, 온도 피드백 끄고 순수 비교).
  2. llama.cpp 실부하(추론) 건 상태로 정상상태까지 대기 후 junction·mem을 각각 **최소 10분 로깅**.
  3. 2팬 → 쉬라우드/팬 교체 → 3팬 동일 반복.
  4. 두 구성의 정상상태 junction·mem을 나란히 보고.
- **측정 전 육안 확인:** 3팬 쉬라우드에 팬 3개 다 꽂혔는지 / 2팬 쉬라우드에 빈 개구부 누설 없는지. 뚫려 있으면 그 데이터는 신뢰하지 말 것. 2팬(80mm)이 카드 끝단 흡입 개구부를 상하단까지 다 덮는지 관찰.
- **판정 로직:**
  - **둘 다 junction ≤ 95 && mem ≤ 90** → **2팬 채택**(더 조용·배선 단순), 3팬은 여분.
  - **2팬만 초과, 3팬은 통과** → 개구부를 2팬이 못 덮는 것. **3팬 채택.**
  - **둘 다 초과** → 소음 상한으로 rpm을 더 못 올림 = 팬으로 답 없음. **파워캡 150→120W로 내리고 재측정** 제안. 팬 개수로 해결 시도하지 말 것.
  - **mem만 튀고 junction은 여유**(구성 무관) → 쉬라우드 상하단 데드존 의심. 해당 쉬라우드 재출력(팬 커버리지 조정) 검토를 보고.
- **채택 구성을 명확히 보고**하고, Task 4는 그 구성 기준으로 진행.

## Task 4 — fancontrol 영구 설정 (채택 구성 기준)

- `lm-sensors`, `fancontrol` 설치 → `sensors-detect` → `pwmconfig`.
- **`/etc/fancontrol` 수동 편집**하여 `FCTEMPS`를 `amdgpu` junction(temp2)에 매핑. `pwmconfig` 자동 생성값(보드 센서만 제안)을 그대로 쓰지 말 것.
- 커브 값은 Task 2/3 실측 기반. **MAXPWM을 8,500 rpm 듀티로 캡**(15,000 rpm/100%로 두지 말 것 — 소음 상한). MAXTEMP는 junction ~85 부근.
- **hwmon 번호는 재부팅마다 변동.** `fancontrol`이 생성하는 `DEVPATH`/`DEVNAME` 라인 유지(불일치 시 서비스 시작 거부 → 조용한 오작동보다 안전).
- `systemctl enable --now fancontrol` 후 **재부팅 테스트까지** 확인.

**결과 (2026-08-06) — 완료** (맨팬 기준. 쉬라우드 장착 시 Task 2/3 재측정 후 커브를 다시 잡는다). `/etc/fancontrol` 핵심:

```
FCTEMPS=hwmon6/pwm4=hwmon1/temp2_input     # MI50 junction
INTERVAL=5
MINTEMP=60   MAXTEMP=80
MINPWM=26    MAXPWM=74
MINSTART=40  MINSTOP=26
```

| junction | ≤60°C | 65°C | 70°C | 75°C | ≥80°C |
|---|---|---|---|---|---|
| rpm | 1,587 | ~2,700 | ~3,700 | ~4,300 | 4,983 |

- **`MINTEMP 60`** — 유휴·경부하를 1,587rpm에 평탄하게 눕히려는 것이다. 소음이 들리기 시작하는 5,000rpm 구간을 junction 80°C 위로 밀어냈다.
- **`MAXPWM 74`(~4,983rpm)는 청취 기준으로 고른 값이지 냉각 여유로 고른 값이 아니다.** 상한을 7,075rpm까지 올려도 지속 부하 온도가 1°C밖에 안 내려가므로(§상태) 소음만 손해다.
- **`MI50-fan-duty.service`는 이 설정이 대체하므로 제거했다.** oneshot 고정 duty는 EC가 되찾아가 CPU 온도를 따라 진동했고, GPU 온도를 아예 보지 못했다.
- hwmon 번호는 부팅마다 바뀌므로 `DEVPATH`/`DEVNAME` 줄을 유지한다.

**부팅 함정 — `nct6687`은 `modules-load.d`로 못 올린다 (2026-08-06 실측).** 부팅 초기엔 Super I/O EC가 응답하지 않아 모듈이 `EC base I/O port unconfigured` → `Failed to insert module 'nct6687': No such device`로 죽는다. 같은 `modprobe`가 수십 초 뒤엔 성공하므로 **경합이 아니라 재시도로 풀어야 한다** — [`nct6687-load.service`](../../../deploy/nct6687-load.service.example)가 2초 간격 15회 재시도하고, `fancontrol`은 drop-in으로 이 유닛을 `Requires=`/`After=`에 건다. 이게 없으면 `fancontrol`이 `Device path of hwmonN has changed`로 종료하고 **팬이 조용히 EC 커브(CPU 온도)로 되돌아간다** — 재부팅 직후 MI50이 48°C인데 CPU 72°C를 따라 7,075rpm으로 돌던 상태가 그것이다.

## Task 5 — 온도 알람 (하드웨어 무관, 지금도 배포 가능)

- junction 95°C 또는 mem 90°C 초과 시 경고하는 감시 스크립트/서비스 작성(로그 또는 알림). **rpm이 아니라 온도 감시.**

## 다장착(2~3카드) 시 추가

- 채택된 쉬라우드 구성을 카드마다 동일 적용. 카드마다 헤더+스플리터 분리 → `fancontrol`에서 **카드별 junction을 카드별 pwm에 독립 매핑**. 가장 뜨거운 카드(보통 가운데)만 더 돌리는 구성이 목표.
- 카드가 여러 개면 `amdgpu` hwmon도 여러 개. Task 1에서 **카드↔hwmon 대응을 PCI 주소 기준**으로 명확히 해둘 것.

## 하지 말 것

- 맨 팬 상태 온도 측정 금지(정압 누설로 무의미).
- MAXPWM을 100%/15,000 rpm으로 두지 말 것(소음 상한 위반).
- `edge` 온도 기준으로 커브 짜지 말 것.
- BIOS 팬커브에 의존하지 말 것.
- 팬 브랜드·정압 재검토 금지(닫힌 결정). **단 2 vs 3팬은 Task 3 실측으로 결정하는 열린 항목**이므로 임의로 하나를 전제하지 말 것.

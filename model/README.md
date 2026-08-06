# model/ — npz2 음소 디코딩 파이프라인

11음소 / 자음-모음 디코딩 모델. 결과 정리는 [`../SUMMARY.md`](../SUMMARY.md),
시간순 상세 기록은 [`../RESULTS.md`](../RESULTS.md) 11절, 대체된 스크립트 지도는
[`../SCRIPTS.md`](../SCRIPTS.md).

## 성적 (ph11, 우연 0.091)

| | 값 |
|---|---|
한 번도 만지지 않은 최종 test셋 | **0.306** (SE 0.018, 선택 편향 +0.000) |
교차검증 15폴드 | 0.305 |
EEGNet 베이스라인 | 0.225 (동일 폴드 15/15 패, p=2e-8) |
라벨 순열 귀무 | 0.088 (우연 0.091) |
미학습 피험자 (LOSO) | 0.139 — 고전·딥 동률, **동작하지 않음** |

C/V(우연 0.5): 0.753 / EEGNet 0.685 / 미학습 피험자 0.583.

## 구조

```
X (61채널 × 256샘플, 0–1.0초, 256Hz)
 │
 ├ 1단  필터 : 신호대잡음 최대화 공간 투영  61 → 14
 │    S_phon  = Σ_s Σ_c (M_sc − M_s)(M_sc − M_s)ᵀ        피험자 내 클래스 간 = 신호
 │    S_noise = Σ_s Σ_c Σ_i (x_i − M_sc)(x_i − M_sc)ᵀ    피험자·클래스 내 잔차 = 잡음
 │    W = eigh(S_phon, S_noise + εI) 상위 14열
 │
 ├ 2단  베이스 모델 : 시행당 SPD 하나 (변 36 = 14 + 14 + 8)
 │    A(t) = [ Z(t) ; Z(t−20) ; φ₀(t) … φ₇(t) ]      φ = 직교 DCT
 │    C = A Aᵀ / T      중심화 안 함 (φ행이 유발 파형을 붙잡게)
 │       Z×Z        공간 공분산
 │       Z×Z(t−τ)   선형 교차지연 = 동역학
 │       Z×φ_k      각 모드 파형의 k번째 DCT 성분 = 시간분해 유발반응
 │    → log-Euclidean 접선 → 666차원
 │
 └ 3단  판독 : 접선공간 가우시안, 피험자별 결정규칙
```

2단이 이 설계의 핵심이며, 세 단 중 **주장한 메커니즘이 통제를 통과한 유일한 부품**이다.
행 수를 고정한 채 시간 구조만 파괴하면 전부 크게 떨어진다: `shuffled` −0.054,
`gauss` −0.066, `dct_high` −0.101, 행 없음 −0.052 (모두 p<1e-4).

## 파일

| 파일 | 역할 |
|---|---|
**`sdcore.py`** | 공유 라이브러리. 로딩·폴드·SPD 연산·공간 필터·표현·판독·보고. 나머지는 전부 여기서만 import하며 서로를 import하지 않는다 |
**`final_model.py`** | 생산 모델(위 3단) + `--study perm/split/den/adapt/readout` |
**`holdout.py`** | 미접촉 최종 test셋(시행 20%, seed 12345). 선택 편향 측정 |
`synth.py` | 밴드별 표현 비교, `--split trial/loso` |
`onemodel.py` | 통합 객체와 두피 라플라시안 리저버, 대조군 arm 8종 |
`arch.py` | 구조 감사. `--study filter` 필터 2×2 ablation, `--study phi` 시간 기저 귀속 통제 |
`nonlin.py` | 비선형 지연 행(2차항)과 동일 행 수 통제 4종 |
`geom.py` | 라플라시안 밴드와 기저 대조군 (`lap`/`pca`/`chan`/`rand`) |
`hier.py` | 계층 LDA, `--split trial/token/time` |
`hier2.py` | 2파라미터 계층 (λ 노이즈 공유, γ 클래스 공유) 표면 |
`contrast.py` | 대비별 판별방향 공유도 + PERM/ZERO 통제 |
`budget.py` | 자기 시행 vs 타인 시행 교환비 곡선 |
`eegnet.py` | EEGNet-8,2 베이스라인 4 arm, `--split trial/loso`, 라벨없는 BN 적응 |

## 실행

데이터는 저장소 루트의 `npz2/`에서 읽는다. 경로가 `__file__` 기준으로 해석되므로
저장소 루트에서든 `model/` 안에서든 동작한다.

```bash
python model/final_model.py --task ph11 --study split     # 네 분할 전부
python model/final_model.py --task ph11 --study perm      # 라벨 순열 귀무
python model/holdout.py     --task ph11                   # 미접촉 최종 test셋
python model/arch.py        --task ph11 --study phi       # 시간 기저 귀속 통제
python model/arch.py        --task ph11 --study filter    # 필터 2×2 ablation
python model/nonlin.py      --task ph11                   # 비선형 지연 통제
python model/contrast.py                                  # 대비별 공유도
python model/budget.py      --task ph11                   # 교환비 곡선
python model/synth.py       --task ph11 --seeds 3 --comb 1
python model/synth.py       --task ph11 --split loso --comb 1
python model/eegnet.py      --task ph11 --seeds 3
python model/eegnet.py      --task ph11 --split loso
```

## 평가 프로토콜 — 반드시 구분할 것

| | 나누는 기준 | test 피험자 | train∩test |
|---|---|---|---|
**시행 단위** | 시행 무작위 5분할, (피험자,클래스) 층화 | 16명 전원 | **피험자 전원 겹침** (시행은 0개) |
**LOSO** | 피험자 1명씩, 16회 | 1명 | 없음 |
토큰 | 자극 녹음 1개씩 보류 | 16명 전원 | 피험자 겹침 |
시간 블록 | 각자 세션의 연속 1/5 | 16명 전원 | 피험자 겹침 |

시행 단위 분할에서 **시행은 하나도 겹치지 않지만 피험자는 전원 겹친다.** 그래서 0.306은
"그 사람의 보정 데이터가 있을 때 그 사람의 새 시행"에 대한 값이고, "새 사람에게 통한다"는
주장은 LOSO 0.139로만 할 수 있다. 두 숫자를 섞어 쓰지 말 것.

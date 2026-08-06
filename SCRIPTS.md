# 스크립트 지도

저장소 루트에 `.py`가 140개쯤 있고 대부분은 이전 세션의 탐색 흔적이다. 이 문서는 **현행**과
**대체됨**을 구분한다. 정리본은 `SUMMARY.md`, 시간순 상세 기록은 `RESULTS.md`에 있고, 1–10절이 예전 파이프라인(`npz`, 0–0.5s),
11절이 현재 파이프라인(`npz2`, 0–1.0s)이다.

## 현행 — `model/` (npz2, RESULTS.md 11절, 정리본 `SUMMARY.md`)

코드는 `model/`에 있고 자체 README를 가진다: [`model/README.md`](model/README.md).

| 파일 | 역할 |
|---|---|
| **`model/sdcore.py`** | **공유 라이브러리.** 로딩·폴드·SPD 연산·공간 필터·표현·판독·보고. 아래 스크립트는 전부 여기서 import하며 서로를 import하지 않는다 |
| **`model/final_model.py`** | **생산 모델.** 3단(SNR 필터 → 통합 SPD → 개인 판독) + `--study perm/split/den/adapt` |
| `model/synth.py` | 밴드별 표현 비교 (`band_blocks` / `band_one` / concat / `global_one`), `--split trial/loso` |
| `model/onemodel.py` | 통합 객체와 두피 라플라시안 리저버, 대조군 arm 8종 |
| `model/arch.py` | 3단 구조 감사. `--study filter` 필터 2×2 ablation, `--study phi` 시간 기저 귀속 통제 |
| `model/nonlin.py` | 비선형 지연 행(2차항)과 동일 행 수 통제 4종 |
| `model/geom.py` | 라플라시안 밴드와 기저 대조군 (`lap`/`pca`/`chan`/`rand`) |
| `model/hier.py` | 계층 LDA (공유 Σ + γ-수축 평균), `--split trial/token/time` |
| `model/hier2.py` | 2파라미터 계층 (λ 노이즈 공유, γ 클래스 공유) 표면 |
| `model/contrast.py` | 대비별 판별방향 공유도 + PERM/ZERO 통제 |
| `model/budget.py` | 자기 시행 vs 타인 시행 교환비 곡선 |
| `model/eegnet.py` | EEGNet-8,2 베이스라인 4 arm, `--split trial/loso`, 라벨없는 BN 적응 |
| **`model/holdout.py`** | **미접촉 최종 test셋** (시행 20%, seed 12345). 선택 편향 측정 |

리팩터 후 모든 스크립트가 로그의 숫자를 자릿수까지 재현하는 것을 확인했다
(`hier.py --task cv`, `final_model.py --study den`, `arch.py --study phi`,
`onemodel.py --task place` 4종 + `sdcore` 대 원본 함수 36개 `max|diff| = 0`).

## `attic/` — 대체됨 (RESULTS.md 1–10절의 근거로 보존)

루트에 121개가 쌓여 있어서 주제별로 `attic/`으로 옮겼다. **삭제가 아니라 이동**이다.
1–10절의 숫자가 이 파일들에서 나왔고, 그중 14개(`shared_ceiling.py`, `stagewise.py`,
`spatial_bands.py`, `unified2.py`, `unified2b.py`, `pooled_unified.py`, `pooled_align.py`,
`pooled_factor.py`, `bayes_ard.py`, `mkl_riem.py`, `riem_kernel.py`, `align.py`, `fact.py`,
`best_model3.py`)는 **git에도 없는 유일본**이다.

| 디렉터리 | 개수 | 내용 |
|---|---|---|
`attic/decode/` | 29 | 표상·융합·전이 탐색 (ROCKET, Xdawn, 필터뱅크, task 통합 등) |
`attic/pooling/` | 16 | 피험자 통합·정렬 계보. `shared_ceiling.py`의 "공유 129%"는 11.4에서 반증 |
`attic/audit/` | 13 | 누수·confound·TMS 감사 |
`attic/speed/` | 12 | 고속 LDA/Riemannian, JAX 포팅, 검증 유틸 |
`attic/type1/` | 10 | Type I 기질 시험 (9절에서 종결) + `gstream_s1.npy`, `pstream.npy` |
`attic/reservoir/` | 9 | 리저버 6종 + 개념자. **W가 전부 랜덤이었고 전극 좌표를 쓴 적 없다** |
`attic/bestmodel/` | 8 | 최고 모델 계보 → `final_model.py`가 대체 |
`attic/prepro/` | 8 | 전처리·QC (+`prepro`) |
`attic/kernel/` | 6 | MKL·측도 커널. 블록 concat + LDA에 짐 |
`attic/dda/` | 6 | DDA 지연·카오스. a₃ 부재는 11.10에서 재확인 |
`attic/unified/` | 5 | 통합 SPD 초기형 → `onemodel.py`/`synth.py`가 시간 기저를 추가해 대체 |
`attic/misc/` | 나머지 | `eda.ipynb`, `final_results.csv` 등 |
`attic/_prerefactor/` | 11 | 리팩터 전 스크립트 원본 |
`attic/data_ml_ready/`, `attic/figs/`, `attic/figures/` | | 예전 산출물 |

예전 데이터셋(`npz`, 992MB)과 `attic/data_ml_ready/`(20MB)는 0–0.5s 창이라 11절 숫자와
직접 비교하면 안 된다.

| 묶음 | 파일 | 11절에서 대체한 것 |
|---|---|---|
| 통합 SPD 초기형 | `unified.py`, `unified2.py`, `unified2b.py`, `pooled_unified.py` | `onemodel.py` (시간 기저 추가), `synth.py` (밴드별) |
| 리저버 | `reservoir.py`, `reservoir_riemann.py`, `reservoir_verify.py`, `spd_reservoir.py`, `resonant_reservoir.py`, `tangent_into_reservoir.py`, `delay_geom_reservoir.py`, `neuron_types.py` | `onemodel.py`의 `montage_W` (전부 랜덤 W였음) |
| DDA | `dda_bank.py`, `dda_chaos.py`, `dda_delaysearch.py`, `dda_diagnostics.py`, `dda_reservoir.py`, `dda_vs_volt.py` | `nonlin.py` (모델 안에서 2차항 측정) |
| 피험자 통합 | `pooled_training.py`, `pooled_net.py`, `pooled_factor.py`, `pooled_align.py`, `hyperalign.py`, `hyperalign_check.py`, `xsubj_align.py`, `alldata_xsubj.py`, `riemann_xsubj.py`, `shared_ceiling.py`, `stagewise.py`, `bayes_ard.py`, `spatial_bands.py` | `hier.py`/`hier2.py`/`contrast.py`/`budget.py` — `shared_ceiling.py`의 "공유 129%"는 11.4에서 반증됨 |
| 커널 | `mkl_riem.py`, `mkl_test.py`, `riem_kernel.py`, `measure_kernel.py`, `conceptors.py` | 없음 (MKL 경로는 블록 concat + LDA에 짐) |
| 최고 모델 계보 | `best_model.py`, `best_model2.py`, `best_model3.py`, `mean_on_bm2.py`, `restore_mean.py`, `final.py`, `optimize_final.py` | `model/final_model.py` |
| 전처리·QC | `prepro`, `add_frequency.py`, `csd.py`, `csd_full_snr.py`, `qc2.py`, `erp_check.py`, `fetch_events.py`, `extract_pstream.py` | 유지 (npz2는 동료 전처리 산출물) |
| 누수 감사 | `leak.py`, `confound_audit.py`, `loto.py`, `syllable_audit.py`, `cvvc_diagnose.py`, `decode_confound.py`, `tms_confirm.py` | `hier.py --split token/time`, `final_model.py --study perm` |
| Type I 기질 | `eval_type1.py`, `type1_*.py`, `evolve_*.py`, `eval_embodied.py`, `ea_voltage_place.py` | 없음 (9절에서 종결) |
| 속도 | `fastlda*.py`, `fastriem.py`, `jax_pipeline.py`, `run_jax.py`, `verify_*.py`, `bench.py`, `tprof.py` | 유지 (검증 유틸) |

## 삭제됨

`p2.py`, `p3.py`, `patch_pu.py`, `patch_u2.py` — 다른 `.py`를 문자열 치환으로 고쳐 쓰는
일회성 도구였다. 결과물이 전부 파일로 남아 있어서 지웠다:
`patch_u2.py` → `unified2b.py`, `patch_pu.py`→`p2.py`→`p3.py` → 현재 `pooled_unified.py`
(ProcessPoolExecutor · `PER_SUBJECT_REF=0.270` · log-Euclidean · `--workers` 모두 반영,
헤더가 `logs/pu4.log`와 일치). `dbg_lda.py` — `fastlda` 대 sklearn 수치 비교 스크래치,
같은 역할의 `verify_lda.py`가 남아 있고 git 이력에도 있다.

`__pycache__/`, `._npz2` (macOS AppleDouble)도 삭제.

`.bak/`에 리팩터 전 11개 스크립트 원본이 있다. 세션의 새 파일은 git이 추적하지 않으므로
이것이 유일한 리팩터 전 사본이다. 리팩터는 함수 36개 `max|diff| = 0`과 로그 숫자 5종 정확
재현으로 검증됐으니, 커밋 후에는 지워도 된다.

## 재현 방법

```bash
python model/final_model.py --task ph11 --study split   # 네 분할 전부
python model/final_model.py --task ph11 --study perm    # 라벨 순열 귀무
python model/arch.py --task ph11 --study phi            # 시간 기저 귀속 통제
python model/arch.py --task ph11 --study filter         # 필터 단 2×2 ablation
python model/nonlin.py --task ph11                      # 비선형 지연 행 통제
python model/synth.py --task ph11 --split loso --comb 1  # 미학습 피험자
python model/eegnet.py --task ph11 --seeds 3            # 베이스라인
```

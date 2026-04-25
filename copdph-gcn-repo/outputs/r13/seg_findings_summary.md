# R13.2b — Segmentation-quality findings (categorized)

Three-way categorisation of the 80 cases flagged by
`R13_seg_quality_audit.py`:

- **34 REAL FAILURES** — at least one mask empty
  (lung/airway/artery/vein). These cases must be re-segmented or
  excluded from any downstream analysis. Per-case list below.

- **4 LUNG-COMPONENT ANOMALY** — lung mask has
  unusual component count (expected 1-2 lobes, observed >5 or 0).
  Inspect visually before re-segmenting.

- **42 VESSEL-FRAGMENTED ONLY** — only artery/vein
  flagged with >100 connected components. Vasculature is naturally
  fragmented (many vessel branches), so this is **likely a false
  positive** of the 100-component threshold. The R13 audit threshold
  for vessel components should be raised (e.g., to 1000+) in the
  next iteration.

## REAL FAILURES (re-segment or exclude)

| group | case_id | failure pattern |
|---|---|---|
| nonph_plain | `nonph_chenfuming_j05104579_friday_january_22_2021_000` | lung EMPTY |
| nonph_plain | `nonph_daijianmin_k01032552_monday_january_25_2021_000` | lung EMPTY |
| nonph_plain | `nonph_fanhongsheng_k01700477_friday_march_5_2021_000` | lung EMPTY |
| nonph_plain | `nonph_ganxiufang_l00829997_friday_august_16_2019_000` | lung EMPTY |
| nonph_refill | `nonph_gaoyuguo_9002006243_sunday_september_26_2021_000` | artery,vein EMPTY |
| nonph_refill | `nonph_guruping_f01898499_monday_october_14_2019_000` | artery,vein EMPTY |
| nonph_plain | `nonph_hepeigen_9001122928_monday_april_1_2019_000` | lung EMPTY |
| nonph_plain | `nonph_hezubian_9002016743_tuesday_october_19_2021_000` | lung EMPTY |
| nonph_refill | `nonph_huangjianxin_9001535968_monday_december_2_2019_000` | artery,vein EMPTY |
| nonph_plain | `nonph_huanglaifu_s08608794_monday_september_23_2019_000` | lung EMPTY |
| nonph_refill | `nonph_huangxiaokang_9001277040_thursday_october_17_2019_000` | artery,vein EMPTY |
| nonph_refill | `nonph_hujiaru_9001749049_thursday_november_19_2020_000` | all 4 EMPTY |
| nonph_plain | `nonph_hujuzhong_j02952601_monday_october_19_2020_000` | lung EMPTY |
| nonph_refill | `nonph_jiangyouxin_9001477095_tuesday_september_10_2019_000` | airway,artery,vein EMPTY |
| nonph_refill | `nonph_liuzhaowen_9001757218_tuesday_november_17_2020_000` | artery,vein EMPTY |
| nonph_refill | `nonph_liuzhishu_7100378918_saturday_april_24_2021_000` | artery,vein EMPTY |
| nonph_refill | `nonph_renchao_9001721397_tuesday_april_20_2021_000` | artery,vein EMPTY |
| nonph_refill | `nonph_ruanwenshang_9001311226_wednesday_april_3_2019_000` | artery,vein EMPTY |
| nonph_refill | `nonph_shenchengrong_9001646591_monday_july_20_2020_000` | artery,vein EMPTY |
| nonph_refill | `nonph_tengchunhai_h0372768x_friday_december_11_2020_000` | artery,vein EMPTY |
| nonph_refill | `nonph_wangyaoting_j01120061_wednesday_december_11_2019_000` | artery,vein EMPTY |
| nonph_refill | `nonph_xiaannian_b02866262_friday_july_26_2019_000` | artery,vein EMPTY |
| nonph_refill | `nonph_yangxindong_k01065776_monday_september_7_2020_000` | artery,vein EMPTY |
| nonph_plain | `nonph_yangxinfang_m01931704_wednesday_march_31_2021_000` | lung EMPTY |
| nonph_refill | `nonph_yangyong_9002255348_wednesday_july_27_2022_000` | artery,vein EMPTY |
| nonph_refill | `nonph_yuanqiurong_e03727573_monday_october_26_2020_000` | artery,vein EMPTY |
| nonph_refill | `nonph_yumingjun_9001629493_wednesday_june_10_2020_000` | artery,vein EMPTY |
| nonph_refill | `nonph_zhangshuisong_9001803404_tuesday_january_19_2021_000` | artery,vein EMPTY |
| nonph_refill | `nonph_zhangzengjian_k03290133_wednesday_june_26_2019_000` | artery,vein EMPTY |
| nonph_refill | `nonph_zhangzhenqi_5002270975_tuesday_june_16_2015_000` | artery,vein EMPTY |
| nonph_plain | `nonph_zhaominhua_m02606428_thursday_january_7_2021_000` | lung EMPTY |
| nonph_plain | `nonph_zhaozhufeng_m01964143_friday_october_16_2020_000` | lung EMPTY |
| nonph_refill | `nonph_zhengyaqin_m00763825_tuesday_december_3_2019_000` | artery,vein EMPTY |
| nonph_plain | `nonph_zhouminggen_9002272740_wednesday_august_10_2022_000` | lung EMPTY |

## LUNG-COMPONENT ANOMALY

| group | case_id | issue |
|---|---|---|
| nonph_plain | `nonph_baoxiaoping_9001193765_thursday_january_2_2020_000` | LUNG_COMPONENT_ANOMALY |
| nonph_plain | `nonph_lutaoxing_m17696948_tuesday_january_29_2019_000` | LUNG_COMPONENT_ANOMALY |
| nonph_plain | `nonph_yangguie_0800162433_saturday_september_15_2012_000` | LUNG_COMPONENT_ANOMALY |
| nonph_plain | `nonph_zhanglirong_9001737585_saturday_october_24_2020_000` | LUNG_COMPONENT_ANOMALY |

## Implications

- The legacy 282-cohort effective denominator drops by 34
  cases for any analysis requiring all 4 mask types. Most failures are
  in the `nonph_plain` stratum (pre-pipeline ingestion of plain-scan CT).
- This invalidates earlier R1-R12 within-nonPH analyses that assumed
  all 80 nonPH-plain were graphable. Effective n=80 may be closer to
  80 - (failures in nonph_plain) ≈ 68.
- For R14: re-segment failed cases via HiPaS-style unified pipeline
  (`reference_hipas_paper.md`) before claiming protocol-invariance.

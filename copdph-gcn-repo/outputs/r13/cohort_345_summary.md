# R13.1 — 345-cohort reconciliation + DCM-count audit

Source folders (post-user-prune):

| group | source | n_ok | n_failed |
|---|---|---|---|
| `ph_contrast` | PH contrast (post-prune) | 160 | 1 |
| `nonph_contrast` | nonPH contrast | 27 | 0 |
| `nonph_plain` | nonPH plain-scan (post-prune) | 58 | 1 |
| `nonph_refill` | nonPH plain refill (24 complete) | 24 | 0 |
| `nonph_new` | nonPH plain new | 76 | 0 |

**Total OK cases**: 345 | **Total failed**: 2 (see `cohort_audit_failures.md`)

## Diff vs legacy 282-cohort (`data/labels_expanded_282.csv`)

- legacy_ids: **282**
- 345_ids (status=ok): **345**
- common: **267**
- only in legacy (10-case PH overcount candidates): **15**
- only in 345 (newly ingestible): **78**

Cases only in legacy (likely duplicates/re-enrollments to be dropped):

- `nonph_mawenhu_g0265125x_wednesday_june_23_2021_000`
- `nonph_sunzegang_9001412676_monday_june_17_2019_000`
- `nonph_wangaie_9001779640_thursday_december_3_2020_000`
- `nonph_wangfurong_p04407432_saturday_october_17_2020_000`
- `nonph_wangjiongwei_h00974562_tuesday_november_17_2020_000`
- `ph_chenlinsheng_9001260682_saturday_october_13_2018_000`
- `ph_chenmeiying_l00103539_thursday_may_10_2018_000`
- `ph_chenshunming_9001485699_thursday_june_11_2020_000`
- `ph_chenyan_9001589206_tuesday_april_28_2020_000`
- `ph_hexujiang_9002569994_friday_march_24_2023_000`
- `ph_wanghaitao_9001792097_wednesday_january_13_2021_000`
- `ph_wangjunfei_9001172063_tuesday_june_29_2021_000`
- `ph_wangliuzhen_9001211738_saturday_january_23_2021_000`
- `ph_wangwei_9001044006_saturday_july_15_2017_000`
- `ph_zhaohongxia_9002303083_friday_september_9_2022_000`

Cases only in 345 (queued for ingestion):

- `nonph_baozuxian_l01210416_saturday_june_8_2019_000`
- `nonph_boqikang_g02801414_tuesday_july_26_2022_000`
- `nonph_chengangli_k06651686_saturday_july_3_2021_000`
- `nonph_chengshanju_9001559972_thursday_april_9_2020_000`
- `nonph_chenhonglian_9001350786_thursday_april_4_2019_000`
- `nonph_chenlaidong_9001959127_sunday_july_25_2021_000`
- `nonph_chenxuebin_m05271600_monday_july_15_2019_000`
- `nonph_chenyaoming_j06712779_thursday_november_7_2019_000`
- `nonph_chenyonghou_k03665679_monday_november_11_2019_000`
- `nonph_fengtianjun_0800201773_monday_may_27_2013_000`
- `nonph_heyuanzhu_7100176769_friday_june_14_2019_000`
- `nonph_jianghaihong_p05493088_monday_april_27_2020_000`
- `nonph_jiasuqin_e00082896_monday_march_18_2019_000`
- `nonph_kongderong_0800108397_monday_may_6_2019_000`
- `nonph_liujianbing_m06195806_friday_april_12_2019_000`
- `nonph_lushande_p10515253_wednesday_february_24_2021_000`
- `nonph_luyukang_p09392540_tuesday_april_21_2020_000`
- `nonph_mawenhu_g0265125x_wednesday_october_27_2021_000`
- `nonph_mawenjuan_j06837395_saturday_january_30_2021_000`
- `nonph_panzhonghua_m03257581_tuesday_february_2_2021_000`
- `nonph_qianrenye_m02629128_saturday_july_30_2022_000`
- `nonph_qiuchongqing_9001699338_wednesday_january_6_2021_000`
- `nonph_quanpufen_k01722430_monday_june_10_2019_000`
- `nonph_shaoronghua_9001214172_wednesday_august_12_2020_000`
- `nonph_shengjingen_k04979223_saturday_september_26_2020_000`
- `nonph_shimingying_h02310639_tuesday_february_23_2021_000`
- `nonph_sunye_j03752185_wednesday_april_1_2020_000`
- `nonph_sunzegang_9001412676_monday_june_29_2020_000`
- `nonph_tangjianping_q15418839_tuesday_february_25_2020_000`
- `nonph_tangjieliang_9001252248_saturday_september_5_2020_000`
- `nonph_tangjinbao_b01917216_friday_march_12_2021_000`
- `nonph_tangliming_n04335446_monday_june_24_2019_000`
- `nonph_tangmingfang_t1372018x_tuesday_october_25_2022_000`
- `nonph_wanggensong_j02396721_wednesday_january_20_2021_000`
- `nonph_wangguorong_f02419525_wednesday_january_13_2021_000`
- `nonph_wangjinhuo_9001714083_thursday_march_4_2021_000`
- `nonph_wangyixin_b01936265_wednesday_january_20_2021_000`
- `nonph_wangzhenyuan_9001827819_thursday_february_18_2021_000`
- `nonph_wangzhiguang_g02204513_friday_february_25_2022_000`
- `nonph_wengwenquan_t06534119_sunday_may_9_2021_000`
- `nonph_wuxinlai_h01672427_thursday_january_14_2021_000`
- `nonph_wuzhuxing_p10335519_friday_may_24_2019_000`
- `nonph_xiechangshou_9001761706_tuesday_january_12_2021_000`
- `nonph_xiguoliang_l02078840_saturday_june_20_2020_000`
- `nonph_xuhanming_9001668916_monday_august_3_2020_000`
- `nonph_xujinhong_j02306765_friday_july_2_2021_000`
- `nonph_xujinlong_j05425786_thursday_october_24_2019_000`
- `nonph_xumingao_9001294412_monday_july_8_2019_000`
- `nonph_xupeiyu_s00504587_wednesday_february_16_2022_000`
- `nonph_xurongqi_m02060453_friday_may_10_2019_000`
- ... and 28 more
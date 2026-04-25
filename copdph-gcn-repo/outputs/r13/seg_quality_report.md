# R13.2 — Segmentation-quality audit on 345-cohort source folders

Per user 2026-04-25: even when 00000001..5 DCM counts match, some
cases have visibly broken segmentations. This script scans each
case for NIfTI masks (lung/airway/artery/vein) and flags
EMPTY/ALL_FILLED/TOO_SMALL/TOO_FRAGMENTED/LUNG_COMPONENT_ANOMALY.

**nibabel available**: True  | **scipy.ndimage.label**: True

**Total audited**: 345
**No masks found in source folder**: 265 (DCM-only cases — masks pending segmentation pipeline)
**Cases with quality issues**: 80
**Cases passing audit**: 0

## Cases flagged with quality issues

| group | case_id | bad masks | first issue |
|---|---|---|---|
| nonph_plain | `nonph_baoxiaoping_9001193765_thursday_january_2_2020_000` | lung,artery,vein | lung has 9 components (expected 1-2) |
| nonph_plain | `nonph_baozhiding_k00748487_tuesday_june_18_2019_000` | artery,vein | 340 connected components (>100) |
| nonph_plain | `nonph_caizhenzhang_d00133633_saturday_january_10_2015_000` | artery,vein | 154 connected components (>100) |
| nonph_plain | `nonph_cenyufeng_7100333962_monday_april_12_2021_000` | artery,vein | 536 connected components (>100) |
| nonph_plain | `nonph_chaiguofeng_j00685132_monday_november_10_2014_000` | artery,vein | 131 connected components (>100) |
| nonph_plain | `nonph_changhairong_e01744812_wednesday_january_19_2022_000` | artery,vein | 380 connected components (>100) |
| nonph_plain | `nonph_chenfuming_j05104579_friday_january_22_2021_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_chenjinzhi_9001551084_friday_january_3_2020_000` | artery,vein | 255 connected components (>100) |
| nonph_plain | `nonph_chenqilong_j05095158_tuesday_december_3_2019_000` | artery,vein | 297 connected components (>100) |
| nonph_plain | `nonph_chenyun_p10610293_tuesday_january_19_2021_000` | artery,vein | 403 connected components (>100) |
| nonph_plain | `nonph_chenyunhua_9001865098_tuesday_july_27_2021_000` | artery,vein | 327 connected components (>100) |
| nonph_plain | `nonph_daijianmin_k01032552_monday_january_25_2021_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_dangshizhong_m0411251x_friday_november_6_2020_000` | artery,vein | 560 connected components (>100) |
| nonph_plain | `nonph_dingzhengde_s01699961_wednesday_december_9_2020_000` | artery,vein | 183 connected components (>100) |
| nonph_plain | `nonph_fanhongsheng_k01700477_friday_march_5_2021_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_fengxiaoguo_g03354804_monday_april_13_2020_000` | artery,vein | 200 connected components (>100) |
| nonph_plain | `nonph_ganxiufang_l00829997_friday_august_16_2019_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_gaodaxiang_9001646216_thursday_july_2_2020_000` | artery,vein | 245 connected components (>100) |
| nonph_plain | `nonph_gejianping_m03069169_monday_may_20_2019_000` | artery,vein | 555 connected components (>100) |
| nonph_plain | `nonph_guqunhua_9001612339_wednesday_may_20_2020_000` | artery,vein | 308 connected components (>100) |
| nonph_plain | `nonph_guyewu_9001534605_monday_december_9_2019_000` | artery,vein | 255 connected components (>100) |
| nonph_plain | `nonph_hanguangyin_9001531946_thursday_november_28_2019_000` | artery,vein | 298 connected components (>100) |
| nonph_plain | `nonph_hepeigen_9001122928_monday_april_1_2019_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_hezubian_9002016743_tuesday_october_19_2021_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_huangbingxing_p22647296_friday_december_27_2019_000` | artery,vein | 576 connected components (>100) |
| nonph_plain | `nonph_huanglaifu_s08608794_monday_september_23_2019_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_huangweixiong_p07189688_friday_march_6_2015_000` | artery,vein | 260 connected components (>100) |
| nonph_plain | `nonph_hujuzhong_j02952601_monday_october_19_2020_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_huyuzhong_h03327950_thursday_august_27_2020_000` | artery,vein | 558 connected components (>100) |
| nonph_plain | `nonph_jiangyuzhong_0800247938_thursday_april_10_2014_000` | artery,vein | 225 connected components (>100) |
| nonph_plain | `nonph_jixiaodong_9002047961_friday_august_5_2022_000` | artery,vein | 281 connected components (>100) |
| nonph_plain | `nonph_jizhensheng_v00496339_monday_june_24_2013_000` | artery,vein | 174 connected components (>100) |
| nonph_plain | `nonph_kuaidafeng_k03781354_monday_april_1_2019_000` | artery,vein | 257 connected components (>100) |
| nonph_plain | `nonph_liujianqiu_m508f6eb3_friday_march_19_2021_000` | artery,vein | 398 connected components (>100) |
| nonph_plain | `nonph_liwenbao_h02744058_thursday_may_14_2020_000` | artery,vein | 359 connected components (>100) |
| nonph_plain | `nonph_luolianping_k04909604_tuesday_july_13_2021_000` | artery,vein | 479 connected components (>100) |
| nonph_plain | `nonph_lutaoxing_m17696948_tuesday_january_29_2019_000` | lung,artery,vein | lung has 10 components (expected 1-2) |
| nonph_plain | `nonph_macaiqin_9002054354_tuesday_july_19_2022_000` | artery,vein | 337 connected components (>100) |
| nonph_plain | `nonph_mawei_9002005614_friday_september_17_2021_000` | artery,vein | 224 connected components (>100) |
| nonph_plain | `nonph_shiruifang_9002328043_friday_september_23_2022_000` | artery,vein | 517 connected components (>100) |
| nonph_plain | `nonph_tanzhangping_9001069198_monday_october_14_2019_000` | artery,vein | 250 connected components (>100) |
| nonph_plain | `nonph_wanggenyou_m03338605_thursday_january_9_2020_000` | artery,vein | 305 connected components (>100) |
| nonph_plain | `nonph_wanglianxi_g03297360_tuesday_august_18_2020_000` | artery,vein | 294 connected components (>100) |
| nonph_plain | `nonph_yangguie_0800162433_saturday_september_15_2012_000` | lung,artery,vein | lung has 6 components (expected 1-2) |
| nonph_plain | `nonph_yangxinfang_m01931704_wednesday_march_31_2021_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_yaoyixin_v00303373_thursday_may_20_2021_000` | artery,vein | 535 connected components (>100) |
| nonph_plain | `nonph_yuliming_m04381891_monday_july_18_2022_000` | artery,vein | 345 connected components (>100) |
| nonph_plain | `nonph_zhanglirong_9001737585_saturday_october_24_2020_000` | lung | lung has 8 components (expected 1-2) |
| nonph_plain | `nonph_zhangyayuan_9001487912_monday_september_14_2020_000` | artery,vein | 409 connected components (>100) |
| nonph_plain | `nonph_zhaoliang_p07004874_monday_january_25_2021_000` | artery,vein | 311 connected components (>100) |
| nonph_plain | `nonph_zhaominhua_m02606428_thursday_january_7_2021_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_zhaozhufeng_m01964143_friday_october_16_2020_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_zhengbaosheng_k06448872_wednesday_september_25_2019_000` | artery,vein | 220 connected components (>100) |
| nonph_plain | `nonph_zhouminggen_9002272740_wednesday_august_10_2022_000` | lung,airway,artery,vein | EMPTY |
| nonph_plain | `nonph_zhuguixiang_k07652613_wednesday_july_29_2020_000` | artery,vein | 462 connected components (>100) |
| nonph_plain | `nonph_zhuhuijun_m00044484_tuesday_august_13_2019_000` | artery,vein | 586 connected components (>100) |
| nonph_plain | `nonph_zhuhuiting_9001481772_tuesday_september_17_2019_000` | artery,vein | 188 connected components (>100) |
| nonph_plain | `nonph_ziyongqi_e03671020_monday_october_14_2019_000` | artery,vein | 182 connected components (>100) |
| nonph_refill | `nonph_gaoyuguo_9002006243_sunday_september_26_2021_000` | lung,airway,artery,vein | 359 connected components (>100) |
| nonph_refill | `nonph_guruping_f01898499_monday_october_14_2019_000` | lung,airway,artery,vein | 393 connected components (>100) |
| nonph_refill | `nonph_huangjianxin_9001535968_monday_december_2_2019_000` | lung,airway,artery,vein | 215 connected components (>100) |
| nonph_refill | `nonph_huangxiaokang_9001277040_thursday_october_17_2019_000` | lung,airway,artery,vein | 484 connected components (>100) |
| nonph_refill | `nonph_hujiaru_9001749049_thursday_november_19_2020_000` | lung,airway,artery,vein | EMPTY |
| nonph_refill | `nonph_jiangyouxin_9001477095_tuesday_september_10_2019_000` | lung,airway,artery,vein | 385 connected components (>100) |
| nonph_refill | `nonph_liuzhaowen_9001757218_tuesday_november_17_2020_000` | lung,airway,artery,vein | 279 connected components (>100) |
| nonph_refill | `nonph_liuzhishu_7100378918_saturday_april_24_2021_000` | lung,airway,artery,vein | 3186 connected components (>100) |
| nonph_refill | `nonph_renchao_9001721397_tuesday_april_20_2021_000` | lung,airway,artery,vein | 390 connected components (>100) |
| nonph_refill | `nonph_ruanwenshang_9001311226_wednesday_april_3_2019_000` | lung,airway,artery,vein | 404 connected components (>100) |
| nonph_refill | `nonph_shenchengrong_9001646591_monday_july_20_2020_000` | lung,airway,artery,vein | 204 connected components (>100) |
| nonph_refill | `nonph_tengchunhai_h0372768x_friday_december_11_2020_000` | lung,airway,artery,vein | 303 connected components (>100) |
| nonph_refill | `nonph_wangyaoting_j01120061_wednesday_december_11_2019_000` | lung,airway,artery,vein | lung has 98 components (expected 1-2) |
| nonph_refill | `nonph_xiaannian_b02866262_friday_july_26_2019_000` | lung,airway,artery,vein | 344 connected components (>100) |
| nonph_refill | `nonph_yangxindong_k01065776_monday_september_7_2020_000` | airway,artery,vein | 289 connected components (>100) |
| nonph_refill | `nonph_yangyong_9002255348_wednesday_july_27_2022_000` | lung,airway,artery,vein | 3318 connected components (>100) |
| nonph_refill | `nonph_yuanqiurong_e03727573_monday_october_26_2020_000` | lung,airway,artery,vein | 329 connected components (>100) |
| nonph_refill | `nonph_yumingjun_9001629493_wednesday_june_10_2020_000` | lung,airway,artery,vein | 1147 connected components (>100) |
| nonph_refill | `nonph_zhangshuisong_9001803404_tuesday_january_19_2021_000` | lung,airway,artery,vein | 416 connected components (>100) |
| nonph_refill | `nonph_zhangzengjian_k03290133_wednesday_june_26_2019_000` | lung,airway,artery,vein | 3482 connected components (>100) |
| nonph_refill | `nonph_zhangzhenqi_5002270975_tuesday_june_16_2015_000` | lung,airway,artery,vein | 155 connected components (>100) |
| nonph_refill | `nonph_zhengyaqin_m00763825_tuesday_december_3_2019_000` | lung,airway,artery,vein | 317 connected components (>100) |

## Cases without source-side masks (need pipeline ingestion)

Total: 265

(Listed in `seg_quality_report.json` under cases[].status='no_masks_in_source')
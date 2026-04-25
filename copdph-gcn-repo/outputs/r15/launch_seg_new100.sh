#!/bin/bash
# R15.1 segmentation launcher for 100 new plain-scan nonPH cases
set -e
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39 || conda activate HiPaS
cd "/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg"
mkdir -p /tmp/r15_seg_logs

run_one() {
  local case_id="$1" gpu="$2"
  local case_dir="/home/imss/cw/GCN copdnoph copdph/nii-new100/$case_id"
  local ct="$case_dir/ct.nii.gz"
  if [ ! -f "$ct" ]; then echo "[skip] $case_id: no ct.nii.gz"; return; fi
  if [ -f "$case_dir/lung.nii.gz" ] && [ -f "$case_dir/artery.nii.gz" ] && [ -f "$case_dir/vein.nii.gz" ]; then
    echo "[exists] $case_id"; return
  fi
  echo "[gpu$gpu] $case_id"
  CUDA_VISIBLE_DEVICES=$gpu python -c "
import sys; sys.path.insert(0, '/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg')
import nibabel as nib, numpy as np
from prediction import predict_zoomed
img = nib.load('$ct')
arr = img.get_fdata().astype('float32')
# Normalize HU [-1000, 600] -> [0, 1]
arr_n = np.clip((arr + 1000) / 1600.0, 0, 1).transpose(2, 0, 1)
artery, vein, lung = predict_zoomed(arr_n)
def save(mask, name):
    out = np.transpose(mask.astype('uint8'), (1, 2, 0))
    nib.save(nib.Nifti1Image(out, img.affine, img.header), '$case_dir/' + name)
save(artery, 'artery.nii.gz'); save(vein, 'vein.nii.gz'); save(lung, 'lung.nii.gz')
print('done')
" > /tmp/r15_seg_logs/${case_id}_gpu${gpu}.log 2>&1
}

# Two parallel queues, one per GPU
CASES=( "nonph_gaoyuguo_9002006243_sunday_september_26_2021_000" "nonph_guruping_f01898499_monday_october_14_2019_000" "nonph_huangjianxin_9001535968_monday_december_2_2019_000" "nonph_huangxiaokang_9001277040_thursday_october_17_2019_000" "nonph_hujiaru_9001749049_thursday_november_19_2020_000" "nonph_jiangyouxin_9001477095_tuesday_september_10_2019_000" "nonph_liuzhaowen_9001757218_tuesday_november_17_2020_000" "nonph_liuzhishu_7100378918_saturday_april_24_2021_000" "nonph_mawenhu_g0265125x_wednesday_october_27_2021_000" "nonph_renchao_9001721397_tuesday_april_20_2021_000" "nonph_ruanwenshang_9001311226_wednesday_april_3_2019_000" "nonph_shenchengrong_9001646591_monday_july_20_2020_000" "nonph_sunzegang_9001412676_monday_june_29_2020_000" "nonph_tengchunhai_h0372768x_friday_december_11_2020_000" "nonph_wangyaoting_j01120061_wednesday_december_11_2019_000" "nonph_xiaannian_b02866262_friday_july_26_2019_000" "nonph_yangxindong_k01065776_monday_september_7_2020_000" "nonph_yangyong_9002255348_wednesday_july_27_2022_000" "nonph_yuanqiurong_e03727573_monday_october_26_2020_000" "nonph_yumingjun_9001629493_wednesday_june_10_2020_000" "nonph_zhangshuisong_9001803404_tuesday_january_19_2021_000" "nonph_zhangzengjian_k03290133_wednesday_june_26_2019_000" "nonph_zhangzhenqi_5002270975_tuesday_june_16_2015_000" "nonph_zhengyaqin_m00763825_tuesday_december_3_2019_000" "nonph_baozuxian_l01210416_saturday_june_8_2019_000" "nonph_boqikang_g02801414_tuesday_july_26_2022_000" "nonph_chengangli_k06651686_saturday_july_3_2021_000" "nonph_chengshanju_9001559972_thursday_april_9_2020_000" "nonph_chenhonglian_9001350786_thursday_april_4_2019_000" "nonph_chenlaidong_9001959127_sunday_july_25_2021_000" "nonph_chenxuebin_m05271600_monday_july_15_2019_000" "nonph_chenyaoming_j06712779_thursday_november_7_2019_000" "nonph_chenyonghou_k03665679_monday_november_11_2019_000" "nonph_fengtianjun_0800201773_monday_may_27_2013_000" "nonph_heyuanzhu_7100176769_friday_june_14_2019_000" "nonph_jianghaihong_p05493088_monday_april_27_2020_000" "nonph_jiasuqin_e00082896_monday_march_18_2019_000" "nonph_kongderong_0800108397_monday_may_6_2019_000" "nonph_liujianbing_m06195806_friday_april_12_2019_000" "nonph_lushande_p10515253_wednesday_february_24_2021_000" "nonph_luyukang_p09392540_tuesday_april_21_2020_000" "nonph_mawenjuan_j06837395_saturday_january_30_2021_000" "nonph_panzhonghua_m03257581_tuesday_february_2_2021_000" "nonph_qianrenye_m02629128_saturday_july_30_2022_000" "nonph_qiuchongqing_9001699338_wednesday_january_6_2021_000" "nonph_quanpufen_k01722430_monday_june_10_2019_000" "nonph_shaoronghua_9001214172_wednesday_august_12_2020_000" "nonph_shengjingen_k04979223_saturday_september_26_2020_000" "nonph_shimingying_h02310639_tuesday_february_23_2021_000" "nonph_sunye_j03752185_wednesday_april_1_2020_000" "nonph_tangjianping_q15418839_tuesday_february_25_2020_000" "nonph_tangjieliang_9001252248_saturday_september_5_2020_000" "nonph_tangjinbao_b01917216_friday_march_12_2021_000" "nonph_tangliming_n04335446_monday_june_24_2019_000" "nonph_tangmingfang_t1372018x_tuesday_october_25_2022_000" "nonph_wanggensong_j02396721_wednesday_january_20_2021_000" "nonph_wangguorong_f02419525_wednesday_january_13_2021_000" "nonph_wangjinhuo_9001714083_thursday_march_4_2021_000" "nonph_wangyixin_b01936265_wednesday_january_20_2021_000" "nonph_wangzhenyuan_9001827819_thursday_february_18_2021_000" "nonph_wangzhiguang_g02204513_friday_february_25_2022_000" "nonph_wengwenquan_t06534119_sunday_may_9_2021_000" "nonph_wuxinlai_h01672427_thursday_january_14_2021_000" "nonph_wuzhuxing_p10335519_friday_may_24_2019_000" "nonph_xiechangshou_9001761706_tuesday_january_12_2021_000" "nonph_xiguoliang_l02078840_saturday_june_20_2020_000" "nonph_xuhanming_9001668916_monday_august_3_2020_000" "nonph_xujinhong_j02306765_friday_july_2_2021_000" "nonph_xujinlong_j05425786_thursday_october_24_2019_000" "nonph_xumingao_9001294412_monday_july_8_2019_000" "nonph_xupeiyu_s00504587_wednesday_february_16_2022_000" "nonph_xurongqi_m02060453_friday_may_10_2019_000" "nonph_yangguoqiang_b03360853_tuesday_september_22_2020_000" "nonph_yanghaohua_d00225976_monday_july_19_2021_000" "nonph_yangjianguo_k06692541_monday_march_18_2019_000" "nonph_yangzhengguo_v14583759_tuesday_march_9_2021_000" "nonph_yaofuming_h01819468_monday_february_8_2021_000" "nonph_yaojianguo_t14085456_wednesday_november_25_2020_000" "nonph_yaoyiqin_k00196643_tuesday_october_8_2019_000" "nonph_yerenyi_v00974759_tuesday_april_6_2021_000" "nonph_yinjianguo_9001412881_friday_june_21_2019_000" "nonph_yuguilin_r14640996_monday_october_14_2019_000" "nonph_yushiliang_f01491728_tuesday_june_29_2021_000" "nonph_zhangfengxia_9001830149_friday_february_5_2021_000" "nonph_zhangjianmin_f01675455_thursday_june_17_2021_000" "nonph_zhangxianguo_m01638612_wednesday_august_21_2019_000" "nonph_zhangxiuqin_t12334543_thursday_february_25_2021_000" "nonph_zhangxueliang_t00317375_wednesday_february_10_2021_000" "nonph_zhanhongfa_b02453337_friday_january_15_2021_000" "nonph_zhaoguoxiang_j06129794_tuesday_march_19_2019_000" "nonph_zhengyaorong_k01848113_monday_august_12_2019_000" "nonph_zhoujianhua_p29127429_wednesday_august_14_2019_000" "nonph_zhucaiguan_q03449689_tuesday_january_26_2021_000" "nonph_zhudeming_6000365916_thursday_april_25_2019_000" "nonph_zhuguanfa_m02277206_monday_november_2_2020_000" "nonph_zhuhuayong_9001419922_wednesday_june_26_2019_000" "nonph_zhujian_u01422414_friday_october_11_2019_000" "nonph_zhujieming_9001476476_monday_october_14_2019_000" "nonph_zhuliang_9001581280_friday_april_10_2020_000" "nonph_zhuyezhong_p06234811_monday_march_8_2021_000" )
N=${#CASES[@]}
HALF=$((N / 2))
(
  for ((i=0; i<HALF; i++)); do run_one "${CASES[i]}" 0; done
) &
P0=$!
(
  for ((i=HALF; i<N; i++)); do run_one "${CASES[i]}" 1; done
) &
P1=$!
wait $P0 $P1
echo "[done] all $N cases segmented"

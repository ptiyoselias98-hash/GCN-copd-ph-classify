"""Plan B prereq — is there any per-lobe segmentation in the data tree?

Looking for:
  - lobe.nii.gz / lobes.nii.gz / lung_lobes.nii.gz
  - upper_lobe_*.nii.gz / right_upper_lobe.nii.gz style
  - TotalSegmentator output dirs
  - anything with '叶' in filename
  - anatomical split in existing NIfTIs (lung.nii.gz with >2 labels would do)
"""
import paramiko

HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"
PROJ = "/home/imss/cw/GCN copdnoph copdph"

CMD = r'''
set -u
ROOTS=( "/home/imss/cw/Claude_COPDnonPH_COPD-PH_CT_nii/nii"
        "/home/imss/cw/nii-data/nii-expanded-85" )

for ROOT in "${ROOTS[@]}"; do
  echo "==================== $ROOT ===================="
  if [ ! -d "$ROOT" ]; then echo "(missing)"; continue; fi
  echo "--- ls top-level (first 5 case dirs) ---"
  ls -d "$ROOT"/*/ 2>/dev/null | head -5
  FIRST=$(ls -d "$ROOT"/*/ 2>/dev/null | head -1)
  if [ -n "$FIRST" ]; then
    echo "--- contents of first case: $FIRST ---"
    ls "$FIRST" 2>/dev/null | head -30 | sed 's/^/  /'
  fi
  echo "--- any *lobe* / *叶* / *total* in this tree (maxdepth 3) ---"
  find "$ROOT" -maxdepth 3 \( -iname '*lobe*' -o -name '*叶*' -o -iname '*total*' -o -iname '*seg_tot*' \) 2>/dev/null | head -10
  echo "--- distinct file-name stems (suffixes stripped) in first case ---"
  if [ -n "$FIRST" ]; then
    ls "$FIRST" 2>/dev/null | sed 's/\.nii\.gz$//;s/\.nii$//' | sort -u | head -30 | sed 's/^/  /'
  fi
done

echo
echo "=== try conda env python for nibabel + label-value probe ==="
source /home/imss/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate pulmonary_bv5_py39 2>/dev/null
python -c "
import nibabel as nib, numpy as np, os, glob
for root in ['/home/imss/cw/Claude_COPDnonPH_COPD-PH_CT_nii/nii',
             '/home/imss/cw/nii-data/nii-expanded-85']:
    cands = glob.glob(os.path.join(root, '*/lung*.nii.gz'))[:2]
    for c in cands:
        try:
            d = nib.load(c).get_fdata()
            u = np.unique(d.astype(int))
            print(c, ' labels=', u[:12], ' n_unique=', len(u))
        except Exception as e:
            print('fail', c, e)
"
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(CMD, timeout=60)
print(o.read().decode(errors="replace"))
err = e.read().decode(errors="replace")
if err.strip():
    print("--- STDERR ---")
    print(err)
c.close()

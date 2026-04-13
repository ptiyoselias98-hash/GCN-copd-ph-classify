"""Sprint 1.1 — Extract commercial CT radiomics matrix from xlsx.

Reads `copd-ph患者113例0331.xlsx` (Sheet1), filters to patients with
complete commercial CT segmentation data (i.e. `肺血管容积(ml)_y` not NaN),
and writes a clean CSV with:

    patient_id, label, + 45 radiomics features
    (26 global vascular + 15 parenchyma/airway + 4 derived ratios)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- planned feature column names (from COPD_PH_Implementation_Plan.md) ---
GLOBAL_VASC = [
    '肺血管容积(ml)_y', '肺血管平均密度(HU)', '肺血管最大密度(HU)', '肺血管最小密度(HU)',
    '肺血管血管分支数量_y', '肺血管弯曲度', '肺血管分形维度',
    '肺血管BV5(ml)_y', '肺血管BV10(ml)_y', '肺血管BV10+(ml)_y',
    '动脉容积(ml)', '动脉平均密度(HU)', '动脉弯曲度', '动脉分形维度',
    '动脉BV5(ml)_y', '动脉BV10(ml)_y', '动脉BV10+(ml)_y',
    '静脉容积(ml)_y', '静脉平均密度(HU)', '静脉弯曲度', '静脉分形维度',
    '静脉BV5(ml)_y', '静脉BV10(ml)_y', '静脉BV10+(ml)_y',
    '动脉血管分支数量_y', '静脉血管分支数量_y',
]
PARA_AIRWAY = [
    '左右肺容积(ml)', '左右肺LAA910(%)', '左右肺LAA950(%)',
    '左右肺平均密度(HU)', '左右肺密度标准差(HU)', '左右肺质量(g)',
    '左右肺支气管数量.1', '左右肺支气管长度(cm).1', '左右肺支气管体积(ml).1',
    '左右肺代', '左右肺支气管数量[D<2mm]', '左右肺支气管体积(ml)[D<2mm]',
    '左右肺Pi10(mm)', '左右肺弯曲度', '左右肺分形维度',
]


def resolve_columns(df: pd.DataFrame, wanted: list[str]) -> tuple[list[str], list[str]]:
    """Return (found, missing). Case-insensitive, strip NBSP/whitespace."""
    def norm(s: str) -> str:
        return str(s).replace("\u00a0", "").strip().lower()

    lookup = {norm(c): c for c in df.columns}
    found, missing = [], []
    for w in wanted:
        k = norm(w)
        if k in lookup:
            found.append(lookup[k])
        else:
            missing.append(w)
    return found, missing


def fuzzy_find(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    """Find columns containing all keywords (case-insensitive)."""
    hits = []
    for c in df.columns:
        cs = str(c)
        if all(k in cs for k in keywords):
            hits.append(c)
    return hits


def extract_id_and_label(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Locate patient id (pinyin) and PH label columns.

    PH column: '是' → 1 (COPD-PH), '/' → 0 (COPD without PH).
    """
    label_col = 'PH' if 'PH' in df.columns else None
    if label_col is None:
        raise RuntimeError("Cannot find 'PH' column.")

    # id column: prefer 拼音 / pinyin / name-based unique id
    id_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ('拼音', 'pinyin')):
            id_col = c
            break
    if id_col is None and 'name' in df.columns:
        id_col = 'name'
    if id_col is None:
        id_col = df.columns[0]

    labels = df[label_col].astype(str).map({'是': 1, '/': 0, '否': 0})
    ids = df[id_col].astype(str).str.strip()
    logger.info("id column: %r  label column: %r", id_col, label_col)
    return ids, labels


def main() -> int:
    root = Path(__file__).resolve().parent.parent  # project root
    xlsx_path = root / 'copd-ph患者113例0331.xlsx'
    out_dir = root / 'data'
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'copd_ph_radiomics.csv'

    logger.info("reading %s", xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name='Sheet1')
    logger.info("sheet shape: %s", df.shape)

    # ---- resolve planned columns
    vasc_found, vasc_missing = resolve_columns(df, GLOBAL_VASC)
    para_found, para_missing = resolve_columns(df, PARA_AIRWAY)
    logger.info("vascular: %d/%d found", len(vasc_found), len(GLOBAL_VASC))
    logger.info("parenchyma/airway: %d/%d found", len(para_found), len(PARA_AIRWAY))

    if vasc_missing:
        logger.warning("missing vascular columns: %s", vasc_missing)
    if para_missing:
        logger.warning("missing parenchyma/airway columns: %s", para_missing)

    # ---- filter 100-pt cohort using 肺血管容积(ml)_y
    filter_candidates = fuzzy_find(df, ['肺血管容积'])
    filter_candidates = [c for c in filter_candidates if str(c).endswith('_y')] or filter_candidates
    if not filter_candidates:
        raise RuntimeError("Cannot find 肺血管容积 column for cohort filter.")
    filter_col = filter_candidates[0]
    mask = df[filter_col].notna()
    logger.info("cohort filter on %r → %d patients", filter_col, mask.sum())

    # ---- id + label
    ids, labels = extract_id_and_label(df)

    sub = df.loc[mask].copy()
    sub_ids = ids.loc[mask].values
    sub_labels = labels.loc[mask].values

    # ---- build feature frame
    feats = pd.DataFrame({'patient_id': sub_ids, 'label': sub_labels})
    for c in vasc_found + para_found:
        feats[c] = pd.to_numeric(sub[c], errors='coerce').values

    # ---- derived ratios (4D)
    def safe_ratio(a: str, b: str) -> pd.Series:
        va = pd.to_numeric(sub.get(a), errors='coerce') if a in sub.columns else None
        vb = pd.to_numeric(sub.get(b), errors='coerce') if b in sub.columns else None
        if va is None or vb is None:
            return pd.Series(np.nan, index=sub.index)
        return (va / vb.replace(0, np.nan)).values

    # identify columns present for ratios
    def first_hit(keys: list[str]) -> str | None:
        for c in vasc_found + para_found:
            if all(k in str(c) for k in keys):
                return c
        return None

    col_total = first_hit(['肺血管容积'])
    col_bv5 = first_hit(['肺血管BV5'])
    col_art_vol = first_hit(['动脉容积'])
    col_vein_vol = first_hit(['静脉容积'])
    col_art_bv5 = first_hit(['动脉BV5'])
    col_vein_bv5 = first_hit(['静脉BV5'])
    col_art_br = first_hit(['动脉血管分支'])
    col_vein_br = first_hit(['静脉血管分支'])

    if col_bv5 and col_total:
        feats['bv5_ratio'] = feats[col_bv5] / feats[col_total].replace(0, np.nan)
    if col_art_vol and col_vein_vol:
        feats['artery_vein_vol_ratio'] = feats[col_art_vol] / feats[col_vein_vol].replace(0, np.nan)
    if col_art_bv5 and col_vein_bv5:
        feats['bv5_artery_vein_ratio'] = feats[col_art_bv5] / feats[col_vein_bv5].replace(0, np.nan)
    if col_art_br and col_vein_br:
        feats['branch_artery_vein_ratio'] = feats[col_art_br] / feats[col_vein_br].replace(0, np.nan)

    # drop rows with missing label
    feats = feats[feats['label'].notna()].copy()
    feats['label'] = feats['label'].astype(int)

    # ---- report
    n_feat = feats.shape[1] - 2  # minus patient_id, label
    logger.info("final shape: %s  (%d features)", feats.shape, n_feat)
    logger.info("class distribution:\n%s", feats['label'].value_counts().to_string())
    logger.info("NaN per feature (top 10):\n%s",
                feats.drop(columns=['patient_id', 'label']).isna().sum().sort_values(ascending=False).head(10).to_string())

    feats.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.info("wrote %s", out_csv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

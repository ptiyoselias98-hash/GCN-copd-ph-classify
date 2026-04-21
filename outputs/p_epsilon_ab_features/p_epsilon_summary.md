# P-ε v1 : A:B feature boost test (2026-04-21)

**n = 98** (radiomics×clinical intersection, PH=72, nonPH=26)
**Feature blocks**: radiomics=45, A:B=6  ·  CV: 5-fold × 3-rep stratified

| feats | clf | AUC | AUC_std | Accuracy | Sens | Spec | Prec | F1 |
|---|---|---|---|---|---|---|---|---|
| AB_only | logreg | 0.577 | 0.137 | 0.680 | 0.912 | 0.040 | 0.724 | 0.805 |
| AB_only | rf | 0.659 | 0.129 | 0.701 | 0.893 | 0.164 | 0.751 | 0.813 |
| radiomics+AB | logreg | 0.798 | 0.087 | 0.778 | 0.875 | 0.511 | 0.834 | 0.852 |
| radiomics+AB | rf | 0.774 | 0.096 | 0.810 | 0.921 | 0.498 | 0.838 | 0.876 |
| radiomics_only | logreg | 0.810 | 0.088 | 0.796 | 0.880 | 0.564 | 0.850 | 0.863 |
| radiomics_only | rf | 0.783 | 0.080 | 0.806 | 0.907 | 0.524 | 0.842 | 0.872 |

## ΔAUC from adding A:B features

| clf | radiomics_only | +A:B | ΔAUC |
|---|---|---|---|
| logreg | 0.810 | 0.798 | -0.012 |
| rf | 0.783 | 0.774 | -0.009 |

**Verdict**: adding P-β A:B features does not boost AUC over radiomics alone.
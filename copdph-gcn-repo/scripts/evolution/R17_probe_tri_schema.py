"""R17.0 — Probe cache_tri_v2 schema (run on remote)."""
import pickle, glob
fs = sorted(glob.glob('/home/imss/cw/GCN copdnoph copdph/tri_structure/cache_tri_v2/*_tri.pkl'))[:1]
print(f"n_pkls={len(fs)}")
with open(fs[0], 'rb') as f:
    d = pickle.load(f)
print(f"top keys: {list(d.keys())}")
for k in ('artery', 'vein', 'airway'):
    v = d.get(k)
    if v is None:
        print(f"  {k}: None")
        continue
    if hasattr(v, 'keys'):
        print(f"  {k} subkeys: {list(v.keys())}")
        g = v.get('graph')
        if g is not None and hasattr(g, 'x'):
            xs = tuple(g.x.shape)
            es = tuple(g.edge_index.shape)
            ea = tuple(g.edge_attr.shape) if hasattr(g, 'edge_attr') and g.edge_attr is not None else None
            print(f"    x.shape={xs} ei.shape={es} edge_attr.shape={ea}")
        # Look for descriptors / qc
        if 'descriptors' in v or 'desc' in v:
            d2 = v.get('descriptors') or v.get('desc')
            print(f"    descriptors: {type(d2).__name__} len={len(d2) if hasattr(d2,'__len__') else '?'}")

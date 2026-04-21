"""Peek one tri_structure cache pkl to confirm branch coords + diameters are available."""
import paramiko

HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

CMD = r'''
source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39
python -c "
import pickle, glob, os
root = '/home/imss/cw/GCN copdnoph copdph/cache_tri_converted'
if not os.path.isdir(root):
    root = '/home/imss/cw/GCN copdnoph copdph/cache_tri'
print('using cache:', root)
files = sorted(glob.glob(os.path.join(root, '*.pkl')))
print(f'n_pkl = {len(files)}')
if not files: raise SystemExit(0)
with open(files[0], 'rb') as f:
    d = pickle.load(f)
print('case_id:', os.path.basename(files[0]))
print('top-level keys:', list(d.keys()))
for struct in ['artery','vein','airway']:
    if struct in d and d[struct] is not None:
        s = d[struct]
        print(f'  {struct}: keys =', list(s.keys()) if isinstance(s, dict) else type(s))
        if isinstance(s, dict):
            for k in ['graph','branches','n_branches','branch_features','descriptors']:
                if k in s:
                    v = s[k]
                    if k == 'graph':
                        print(f'    graph.x.shape:', tuple(v.x.shape), 'edge_index.shape:', tuple(v.edge_index.shape))
                    elif k == 'branches':
                        print(f'    branches: type={type(v).__name__} len={len(v) if hasattr(v,\"__len__\") else \"?\"}')
                        if len(v) > 0:
                            b0 = v[0]
                            print(f'    branch[0] type:', type(b0).__name__, 'attrs/keys:',
                                  list(b0.keys()) if isinstance(b0, dict) else dir(b0)[:15])
                    elif k == 'branch_features':
                        print(f'    branch_features: len={len(v)}')
                        if len(v) > 0:
                            print(f'    bf[0] keys:', list(v[0].keys())[:20])
                    elif k == 'descriptors':
                        print(f'    descriptors keys:', list(v.keys())[:25])
                    else:
                        print(f'    {k} = {v}')
    else:
        print(f'  {struct}: MISSING')
"
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(CMD, timeout=60)
print(o.read().decode(errors="replace"))
err = e.read().decode(errors="replace")
if err.strip():
    print("--- STDERR ---")
    print(err[:1500])
c.close()

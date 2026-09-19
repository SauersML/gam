"""Model-bytes size and post-fit (re)decode cost vs n.  Usage: model_bytes.py N [N ...]"""
import sys, time, warnings
import numpy as np
import gamfit, pandas  # noqa
warnings.simplefilter("ignore")
R = gamfit._rust
for n in map(int, sys.argv[1:]):
    rng = np.random.default_rng(0); x = rng.uniform(0, 1, n); y = np.sin(2*np.pi*x) + rng.normal(0, .5, n)
    c = time.process_time(); m = gamfit.fit({"x": x, "y": y}, "y ~ s(x)"); cf = time.process_time() - c
    b = m._model_bytes
    c = time.process_time(); R.compile_model(b); cc = time.process_time() - c
    c = time.process_time(); R.inference_notes_from_model(b); cn = time.process_time() - c
    c = time.process_time(); m.summary(); cs = time.process_time() - c
    c = time.process_time(); s = m.dumps() if hasattr(m, "dumps") else b; cd = time.process_time() - c
    print(f"n={n} fit_cpu={cf:.3f}s model_bytes={len(b)/2**20:.2f}MB bytes_per_row={len(b)/n:.1f} compile_model={cc:.3f}s notes={cn:.3f}s summary={cs:.3f}s dumps={cd:.3f}s dumps_len={len(s)/2**20:.2f}MB")

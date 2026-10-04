"""Additional immutable architecture preflight for the GPU phase; no scoring change."""
import hashlib,subprocess,sys
from pathlib import Path
native=Path.home()/'mpd-data/vpd/t-9d2b8f02/model_config.yaml'
if hashlib.sha256(native.read_bytes()).hexdigest()!='9664c12d3492ee58520f89703e67ea2790ea13de1f88bf8e3c4594943e0cc59d':raise RuntimeError('original target architecture YAML changed')
result=subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),'evaluate'])
if result.returncode:raise RuntimeError('guarded baseline evaluation failed')

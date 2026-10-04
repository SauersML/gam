"""Read exact source-pinned report bytes from the published gzip archive."""
import gzip,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def load_report(name):
 entry=json.loads((HERE/'ARCHIVE.json').read_text())['reports'][name]
 packed=(HERE/entry['file']).read_bytes()
 assert hashlib.sha256(packed).hexdigest()==entry['sha256_gzip']
 data=gzip.decompress(packed)
 assert len(data)==entry['uncompressed_bytes']
 assert hashlib.sha256(data).hexdigest()==entry['sha256_uncompressed']
 return json.loads(data)
if __name__=='__main__':
 name=sys.argv[1] if len(sys.argv)>1 else 'full'
 report=load_report(name)
 print(json.dumps({'report':name,'passes':report['passes'],'episode_count':report['episode_count'],'spec_sha256':report['spec_sha256']}))

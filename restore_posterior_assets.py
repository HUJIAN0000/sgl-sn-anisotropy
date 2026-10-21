"""Verify and restore the five posterior Release assets; never run inference."""
from pathlib import Path
import argparse,json,hashlib,zipfile,shutil
ROOT=Path(__file__).resolve().parent
def digest_stream(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
    index=json.loads((ROOT/'provenance/posterior_release_index.json').read_text())
    restored=0
    for name,record in index['archives'].items():
        path=a.assets/name
        if not path.is_file():raise FileNotFoundError('Download all five posterior assets; missing '+str(path))
        with path.open('rb') as f:
            if digest_stream(f)!=record['sha256']:raise ValueError('Archive SHA-256 mismatch: '+name)
        with zipfile.ZipFile(path) as z:
            if set(z.namelist())!=set(record['members']):raise ValueError('Unexpected archive contents: '+name)
            for member,digest in record['members'].items():
                target=(ROOT/member).resolve()
                if not target.is_relative_to(ROOT.resolve()) or not member.startswith('results/production/'):raise ValueError(member)
                with z.open(member) as f:
                    if digest_stream(f)!=digest:raise ValueError('Member SHA-256 mismatch: '+member)
                if target.exists():
                    with target.open('rb') as f:
                        if digest_stream(f)!=digest:raise FileExistsError('Refusing to replace different file: '+str(target))
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(member) as f,target.open('wb') as out:shutil.copyfileobj(f,out)
                restored+=1
    source=ROOT/'reference_results/status/sample_buffered_task_status.json'
    target=ROOT/'results/production/sample_buffered_task_status.json'
    if target.exists() and target.read_bytes()!=source.read_bytes():raise FileExistsError(target)
    shutil.copy2(source,target)
    print('Verified and restored',restored,'files from five assets; no sampling was run.')
if __name__=='__main__':main()

"""Deterministic, recoverable adapters for unmodified learn2branch generators."""
from __future__ import annotations
import gzip, hashlib, importlib.util, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
from .config import DatasetConfig
from .manifest import ManifestRecord, parameters_json, read_manifest, write_manifest

def repository_root(start: Path|None=None)->Path:
    p=(start or Path(__file__)).resolve()
    for x in (p,*p.parents):
        if (x/'pyproject.toml').is_file(): return x
    raise RuntimeError('repository root not found')
def _module(root:Path):
    source=root/'third_party'/'learn2branch'/'01_generate_instances.py'; parent=str(source.parent)
    if parent not in sys.path: sys.path.insert(0,parent)
    spec=importlib.util.spec_from_file_location('learn2branch_generators',source)
    if spec is None or spec.loader is None: raise RuntimeError('cannot load generator')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def _commit(root:Path)->str:
    return subprocess.run(['git','-C',str(root/'third_party'/'learn2branch'),'rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip()
def _hash(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def _worker(task:dict)->dict:
    root=Path(task['root']); target=Path(task['target']); target.parent.mkdir(parents=True,exist_ok=True)
    mod=_module(root); kwargs=dict(task['parameters']); kwargs.pop('generator',None); kwargs.pop('assumptions',None)
    with tempfile.TemporaryDirectory(dir=target.parent) as d:
        lp=Path(d)/'instance.lp'; rng=np.random.RandomState(task['seed'])
        if task['problem']=='ca': mod.generate_cauctions(random=rng,filename=str(lp),**kwargs)
        else: mod.generate_setcover(filename=str(lp),rng=rng,**kwargs)
        temporary=target.with_name(target.name+'.tmp')
        try:
            with lp.open('rb') as src,temporary.open('wb') as dst:
                with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0) as gz: gz.writelines(src)
            os.replace(temporary,target)
        finally: temporary.unlink(missing_ok=True)
    return {'key':task['key'],'size':target.stat().st_size,'sha256':_hash(target)}
def generate_instances(config:DatasetConfig,*,problems:list[str],splits:list[str],counts:dict[str,int],output_root:Path,repo_root:Path|None=None,force:bool=False,compress:bool=True,workers:int=1)->list[ManifestRecord]:
    if not compress: raise ValueError('uncompressed LP output is disallowed')
    if workers<1: raise ValueError('workers must be >= 1')
    root=(repo_root or repository_root()).resolve(); out=output_root.resolve(); out.relative_to(root)
    commit=_commit(root)
    from .seeds import derive_seed
    manifest=out/'metadata'/'manifest.csv'; known={r.key:r for r in read_manifest(manifest)}; tasks=[]; created={}
    for problem in problems:
        params=dict(config.ca if problem=='ca' else config.sc)
        for split in splits:
            for index in range(counts[split]):
                key=(problem,split,index); seed=derive_seed(config.base_seed,problem,split,index); target=out/'data'/problem/split/f'{problem}_{split}_{index:04d}.lp.gz'
                old=known.get(key)
                if target.exists() and not force:
                    digest=_hash(target)
                    if old and old.sha256!=digest: raise ValueError(f'existing artifact hash differs from manifest: {target}')
                    if old: continue
                    created[key]=(target,seed,params,digest)
                else: tasks.append({'root':str(root),'target':str(target),'problem':problem,'parameters':params,'seed':seed,'key':key})
    results=[]
    if tasks:
        if workers==1: results=[_worker(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool: results=list(pool.map(_worker,tasks))
    for result in results:
        problem,split,index=result['key']; target=out/'data'/problem/split/f'{problem}_{split}_{index:04d}.lp.gz'; params=dict(config.ca if problem=='ca' else config.sc); created[result['key']]=(target,derive_seed(config.base_seed,problem,split,index),params,result['sha256'])
    for key,(target,seed,params,digest) in created.items():
        problem,split,index=key; known[key]=ManifestRecord(problem,split,index,seed,target.relative_to(out).as_posix(),str(params['generator']),commit,parameters_json(params),target.stat().st_size,digest,datetime.now(UTC).isoformat())
    write_manifest(known.values(),manifest); return list(known.values())
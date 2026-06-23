"""Adapters for the unmodified learn2branch generators."""
from __future__ import annotations
import gzip, hashlib, importlib.util, os, subprocess, sys, tempfile
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
from .config import DatasetConfig
from .manifest import ManifestRecord, parameters_json, read_manifest, write_manifest
def repository_root(start: Path|None=None) -> Path:
    p=(start or Path(__file__)).resolve()
    for x in (p,*p.parents):
        if (x/"pyproject.toml").is_file(): return x
    raise RuntimeError("repository root not found")
def _module(root:Path):
    source=root/"third_party"/"learn2branch"/"01_generate_instances.py"
    if not source.is_file(): raise FileNotFoundError(source)
    source_parent = str(source.parent)
    if source_parent not in sys.path: sys.path.insert(0, source_parent)
    spec=importlib.util.spec_from_file_location("learn2branch_generators",source)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load generator")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def _commit(root:Path) -> str:
    return subprocess.run(["git","-C",str(root/"third_party"/"learn2branch"),"rev-parse","HEAD"],check=True,text=True,capture_output=True).stdout.strip()
def _hash(p:Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def generate_instances(config:DatasetConfig, *, problems:list[str], splits:list[str], counts:dict[str,int], output_root:Path, repo_root:Path|None=None, force:bool=False, compress:bool=True) -> list[ManifestRecord]:
    
    root=(repo_root or repository_root()).resolve(); out=output_root.resolve()
    try: out.relative_to(root)
    except ValueError as e: raise ValueError("output_root must be under repository root") from e
    mod=_module(root); commit=_commit(root); manifest=out/"metadata"/"manifest.csv"; known={x.key:x for x in read_manifest(manifest)}
    from .seeds import derive_seed
    for problem in problems:
        params=dict(config.ca if problem=="ca" else config.sc)
        for split in splits:
            for index in range(counts[split]):
                seed=derive_seed(config.base_seed,problem,split,index); suffix = ".lp.gz" if compress else ".lp"; target=out/"data"/problem/split/f"{problem}_{split}_{index:04d}{suffix}"; target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or force:
                    with tempfile.TemporaryDirectory(dir=target.parent) as d:
                        lp=Path(d)/"instance.lp"; rng=np.random.RandomState(seed); kwargs=dict(params); kwargs.pop("generator",None); kwargs.pop("assumptions",None)
                        if problem=="ca": mod.generate_cauctions(random=rng,filename=str(lp),**kwargs)
                        else: mod.generate_setcover(filename=str(lp),rng=rng,**kwargs)
                        with lp.open("rb") as src,target.open("wb") as dst:
                            with gzip.GzipFile(filename="",mode="wb",fileobj=dst,mtime=0) as gz: gz.writelines(src)
                digest=_hash(target); key=(problem,split,index); old=known.get(key)
                if old and not force and old.sha256 != digest: raise ValueError(f"existing artifact hash differs from manifest: {target}")
                if not old or force:
                    known[key]=ManifestRecord(problem,split,index,seed,target.relative_to(out).as_posix(),str(params["generator"]),commit,parameters_json(params),target.stat().st_size,digest,datetime.now(UTC).isoformat())
    write_manifest(known.values(),manifest); return list(known.values())
"""Atomic CSV manifest storage."""
from __future__ import annotations
import csv, json, os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
MANIFEST_FIELDS = ("problem","split","index","seed","relative_path","generator","generator_commit","parameters_json","size_bytes","sha256","created_at_utc")
@dataclass(frozen=True)
class ManifestRecord:
    problem: str; split: str; index: int; seed: int; relative_path: str
    generator: str; generator_commit: str; parameters_json: str; size_bytes: int; sha256: str; created_at_utc: str
    @property
    def key(self): return (self.problem,self.split,self.index)
    def validate(self):
        if self.problem not in {"ca","sc"} or self.split not in {"train","valid","test"}: raise ValueError("invalid problem or split")
        if self.index < 0 or self.seed < 0 or self.size_bytes < 0: raise ValueError("negative numeric manifest field")
        if Path(self.relative_path).is_absolute() or "\\" in self.relative_path or not self.relative_path.endswith(".lp.gz"): raise ValueError("relative_path must be POSIX .lp.gz")
        if len(self.sha256)!=64 or any(c not in "0123456789abcdef" for c in self.sha256): raise ValueError("invalid sha256")
        json.loads(self.parameters_json)
def parameters_json(value: dict[str, object]) -> str: return json.dumps(value,sort_keys=True,separators=(",",":"))
def read_manifest(path: str|Path) -> list[ManifestRecord]:
    p=Path(path)
    if not p.exists(): return []
    with p.open(encoding="utf-8",newline="") as f:
        result=[]
        for r in csv.DictReader(f):
            x=ManifestRecord(r["problem"],r["split"],int(r["index"]),int(r["seed"]),r["relative_path"],r["generator"],r["generator_commit"],r["parameters_json"],int(r["size_bytes"]),r["sha256"],r["created_at_utc"]); x.validate(); result.append(x)
    return result
def write_manifest(records: Iterable[ManifestRecord], path: str|Path) -> None:
    p=Path(path); rows=sorted(records,key=lambda x:x.key)
    if len({x.key for x in rows})!=len(rows): raise ValueError("duplicate manifest instance key")
    for x in rows: x.validate()
    p.parent.mkdir(parents=True,exist_ok=True); temp=p.with_suffix(p.suffix+".tmp")
    with temp.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=MANIFEST_FIELDS); w.writeheader(); [w.writerow(asdict(x)) for x in rows]; f.flush(); os.fsync(f.fileno())
    os.replace(temp,p)
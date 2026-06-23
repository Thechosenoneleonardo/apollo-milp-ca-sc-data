import gzip
import hashlib
from pathlib import Path
import pytest
from milp_dataset.manifest import ManifestRecord, parameters_json, write_manifest
from milp_dataset.validation import validate_dataset

def dataset(tmp_path):
    path=tmp_path/"data"/"ca"/"train"/"one.lp.gz"; path.parent.mkdir(parents=True)
    lp_text = "maximize\nOBJ: +1 x1\n\nsubject to\nC1: +1 x1 <= 1\n\nbinary\n x1\n"
    path.write_bytes(gzip.compress(lp_text.encode("utf-8"), compresslevel=9, mtime=0))
    record=ManifestRecord("ca","train",0,1,"data/ca/train/one.lp.gz","test","a"*40,parameters_json({}),path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest(),"2026-01-01T00:00:00+00:00")
    write_manifest([record],tmp_path/"metadata"/"manifest.csv"); return path

def test_optional_reader_receives_closed_lp_and_temp_is_deleted(tmp_path):
    dataset(tmp_path); seen=[]
    def reader(path):
        seen.append(path); assert path.suffix==".lp"; assert path.is_file()
    warnings=[]; assert validate_dataset(tmp_path,solver_reader=reader,warnings=warnings)==[]; assert not warnings; assert len(seen)==1 and not seen[0].exists()

def test_solver_failure_is_warning_unless_strict(tmp_path):
    dataset(tmp_path)
    def failing(path): raise RuntimeError("required plugin was not found")
    warnings=[]; assert validate_dataset(tmp_path,solver_reader=failing,warnings=warnings)==[]; assert "required plugin" in warnings[0]
    assert validate_dataset(tmp_path,solver_reader=failing,strict_solver=True)

def test_missing_solver_is_skipped(tmp_path,monkeypatch):
    dataset(tmp_path); import milp_dataset.validation as validation
    monkeypatch.setattr(validation,"_installed_solver_reader",lambda:("solver",None)); warnings=[]
    assert validate_dataset(tmp_path,warnings=warnings)==[]; assert warnings==["solver check skipped: solver is not installed"]
"""Synthetic boundary controls, including optimized Python; no model or corpus."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from ambifix_hitl.contracts import ModelUnavailable
from ambifix_hitl.predictors import check_reference_gate
from ambifix_hitl.storage import AuditError, Store


BASE=Path(__file__).resolve().parents[1]


def valid_synthetic_gate():
    return {"schema_version":1,"status":"PASS_REFERENCE_REPLAY",
        "manifest_sha256":"a"*64,"checkpoint_fingerprint":"b"*64,
        "device":"cpu","class_order":[1,2,3,4,5,6],
        "reference_role":"validation_reference","rows":1,
        "prediction_mismatches":0,"maximum_probability_abs_error":0.0,
        "probability_tolerance":0.0001,
        "approved_reference_sha256":"c"*64,"actual_reference_sha256":"c"*64}


def test_well_formed_synthetic_gate_contract(tmp_path):
    path=tmp_path/"synthetic_gate.json";gate=valid_synthetic_gate()
    path.write_text(json.dumps(gate))
    assert check_reference_gate(path,"a"*64,"b"*64,"cpu")==gate


@pytest.mark.parametrize("key,value",[
    ("schema_version",True),("rows",True),("rows",0),
    ("prediction_mismatches",False),("prediction_mismatches",1),
    ("probability_tolerance",float("inf")),("probability_tolerance",float("nan")),
    ("probability_tolerance",True),("maximum_probability_abs_error",-1.0),
    ("maximum_probability_abs_error",float("nan")),
    ("maximum_probability_abs_error",0.1),("device","cuda"),
    ("class_order",[True,2,3,4,5,6]),("actual_reference_sha256","d"*64),
    ("approved_reference_sha256","not-a-digest"),
])
def test_invalid_gate_fields_fail_closed(tmp_path,key,value):
    gate=valid_synthetic_gate();gate[key]=value
    path=tmp_path/"synthetic_invalid_gate.json";path.write_text(json.dumps(gate))
    with pytest.raises(ModelUnavailable):check_reference_gate(path,"a"*64,"b"*64,"cpu")


def test_unknown_database_schema_fails_closed(tmp_path):
    path=tmp_path/"synthetic_unknown_schema.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES('schema_version','999')")
    with pytest.raises(AuditError,match="schema version"):Store(path)


@pytest.mark.parametrize("boundary",["reference_gate","database_schema"])
def test_boundaries_still_refuse_with_optimized_python(tmp_path,boundary):
    """-O removes assertions; explicit safety checks must still reject fixtures."""
    code='''
import json, sqlite3, sys
from pathlib import Path
from ambifix_hitl.contracts import ModelUnavailable
from ambifix_hitl.predictors import check_reference_gate
from ambifix_hitl.storage import AuditError, Store
root=Path(sys.argv[1]);boundary=sys.argv[2]
try:
    if boundary=="reference_gate":
        path=root/"synthetic_invalid_gate.json";path.write_text("{}")
        check_reference_gate(path,"a"*64,"b"*64,"cpu")
    else:
        path=root/"synthetic_unknown_schema.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("INSERT INTO metadata VALUES('schema_version','999')")
        Store(path)
except (ModelUnavailable,AuditError):
    print("REFUSED_SYNTHETIC_INVALID_BOUNDARY")
else:
    raise SystemExit("Unsafe boundary acceptance under optimized Python")
'''
    environment={**os.environ,"PYTHONPATH":str(BASE),"PYTHONDONTWRITEBYTECODE":"1"}
    result=subprocess.run([sys.executable,"-O","-c",code,str(tmp_path),boundary],
        cwd=BASE,env=environment,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=="REFUSED_SYNTHETIC_INVALID_BOUNDARY"

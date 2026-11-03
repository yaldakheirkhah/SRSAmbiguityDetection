"""Bind explicitly authorized LOCAL development inputs; never copy or publish rows.

No training, downloads, source split regeneration, or held-out access is performed.
The existing notebook repeats all integrity checks before its first ML import.
"""
from pathlib import Path, PurePosixPath
import argparse
from datetime import datetime, timezone
import hashlib
import json

FILES={
    "legacy":["data/train.csv","data/validation.csv","config/protocol.json","reports/QUALITY_GATE.json"],
    "lineage":["data/train_original.csv","data/validation.csv","data/train_A_Conservative_v1.csv",
        "data/train_B_Full_Provisional_v1.csv","config/training_config.json",
        "config/variant_A_class_weights.json","config/variant_B_class_weights.json",
        "reports/DATASET_VERIFICATION.json"],
}
EXPECTED_ORIGINAL={
    "data/train_original.csv":"c5d623c1738400170fe7b5feb23a82d4a3c15f071a77fe5c4bfba8c6c5cc70f8",
    "data/validation.csv":"9b5c7c76dad306bff4b306753d04a0169f84dd6f27eb8312bc63f5f3322bef42",
}


def local_file(root,relative):
    parts=PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        raise ValueError("Only fixed package-relative input names are allowed")
    if any(p.casefold()=="test.csv" for p in [*root.parts,*parts.parts]):
        raise ValueError("Held-out paths are prohibited before filesystem access")
    path=root.joinpath(*parts.parts)
    if any(root.joinpath(*parts.parts[:i]).is_symlink() for i in range(1,len(parts.parts)+1)):
        raise ValueError("Local input symlinks are prohibited")
    if not path.is_file():
        raise FileNotFoundError("Supply authorized development input or generated quality evidence: "+relative)
    return path


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):value.update(block)
    return value.hexdigest()


def prepare(root,profile,*,execute=False,authorized=False):
    if not execute or not authorized:
        raise PermissionError("Explicit execution and authorization for local development data are required; nothing accessed")
    if profile not in FILES:raise ValueError("Choose the legacy or lineage profile")
    root=Path(root).expanduser()
    if any(p.casefold()=="test.csv" for p in root.parts):raise ValueError("Held-out root prohibited")
    root=root.resolve(strict=True)
    destination=root/"package_manifest.json"
    if destination.exists() or destination.is_symlink():raise FileExistsError("Preserve existing manifest; never overwrite")
    inputs={name:local_file(root,name) for name in FILES[profile]}
    hashes={name:digest(path) for name,path in inputs.items()}
    gate_name="reports/QUALITY_GATE.json" if profile=="legacy" else "reports/DATASET_VERIFICATION.json"
    config_name="config/protocol.json" if profile=="legacy" else "config/training_config.json"
    gate=json.loads(inputs[gate_name].read_text());config=json.loads(inputs[config_name].read_text())
    checks=gate.get("checks")
    if (not isinstance(checks,dict) or not checks or not all(value is True for value in checks.values())
            or gate.get("approved_for_gpu_package") is not True):
        raise ValueError("Existing generated quality gate is missing or has unsuccessful checks")
    benchmark=config.get("benchmark_id")
    if not isinstance(benchmark,str) or not benchmark.strip() or gate.get("benchmark_id")!=benchmark:
        raise ValueError("Quality gate and configuration benchmark identities differ")
    if profile=="legacy":
        expected=local_file(root,"config/protocol.sha256").read_text().split()[0]
        if hashes[config_name]!=expected or gate.get("protocol_sha256")!=expected:
            raise ValueError("Predeclared legacy protocol identity differs")
        if (gate.get("status") not in {"PASS_PARTIAL_RECOVERY","PASS_GROUPED_EXPLORATORY"}
                or gate.get("eligible_for_original_frozen_test") is not False
                or config.get("eligible_for_original_frozen_test") is not False):
            raise ValueError("Legacy experiment must pass its gate and remain ineligible for the original Test")
    else:
        if (not str(gate.get("status","")).startswith("PASS")
                or gate.get("protected_test_accessed") is not False
                or gate.get("training_config_sha256")!=hashes[config_name]):
            raise ValueError("Lineage quality gate or training configuration binding differs")
        for name,expected in EXPECTED_ORIGINAL.items():
            if hashes[name]!=expected:raise ValueError("Frozen original development identity differs: "+name)
        for variant in ["A","B"]:
            data="data/train_A_Conservative_v1.csv" if variant=="A" else "data/train_B_Full_Provisional_v1.csv"
            weights=json.loads(inputs[f"config/variant_{variant}_class_weights.json"].read_text())
            if (weights.get("training_only") is not True or weights.get("training_data_path")!=data
                    or weights.get("training_data_sha256")!=hashes[data]
                    or gate.get("variants",{}).get(variant,{}).get("sha256")!=hashes[data]):
                raise ValueError("Variant training/weight/gate identities differ: "+variant)
    result={"manifest_version":1,"manifest_scope":"LOCAL_DEVELOPMENT_INPUT_IDENTITIES_ONLY",
        "status":"IDENTITIES_BOUND_NOTEBOOK_INTEGRITY_CHECKS_STILL_REQUIRED",
        "profile":profile,"benchmark_id":benchmark,"created_utc":datetime.now(timezone.utc).isoformat(),
        "copied_or_published_records":False,"training_performed":False,"held_out_accessed":False,
        "limitations":"Not independent provenance review or redistribution permission; the notebook must pass its unchanged input checks before training.",
        "files":[{"path":name,"sha256":hashes[name],"bytes":inputs[name].stat().st_size} for name in FILES[profile]]}
    with destination.open("x",encoding="utf-8") as handle:json.dump(result,handle,indent=2);handle.write("\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--profile",required=True,choices=FILES)
    parser.add_argument("--authorized-local-development-data",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.profile,execute=args.execute,
        authorized=args.authorized_local_development_data),indent=2))


if __name__=="__main__":main()

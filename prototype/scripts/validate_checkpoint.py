"""Compatibility + optional approved validation reference replay. Never trains."""
from pathlib import Path
import argparse,csv,json,math,platform,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ambifix_hitl.contracts import LABELS,utc_now,valid_label,valid_probabilities,validate_text
from ambifix_hitl.predictors import RealBERTPredictor,allowed_path,file_sha


def replay_reference(predictor,reference_path,approval):
    if approval.get("reference_role")!="validation_reference" or approval.get("class_order")!=[1,2,3,4,5,6]:
        raise ValueError("Only an explicitly approved six-class validation reference is allowed.")
    if predictor.manifest["model_id"]!=approval["model_id"]:raise ValueError("Checkpoint and approved reference model identities differ.")
    path=allowed_path(reference_path)
    actual_sha=file_sha(path)
    if actual_sha!=approval["reference_sha256"]:raise ValueError("Reference identity differs from the approved artifact; nothing evaluated.")
    c=approval["columns"]
    with path.open(newline="",encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
    if len(rows)!=approval["expected_rows"] or not rows:raise ValueError("Unexpected reference row count.")
    if len({r[c["source_id"]] for r in rows})!=len(rows):raise ValueError("Reference source IDs must be unique.")
    max_error=0.0;mismatches=0;cm=[[0]*6 for _ in range(6)]
    for start in range(0,len(rows),16):
        chunk=rows[start:start+16]
        outputs=predictor.predict_batch([validate_text(r[c["text"]]) for r in chunk])
        if len(outputs)!=len(chunk):raise ValueError("Predictor changed reference cardinality.")
        for row,out in zip(chunk,outputs):
            truth=valid_label(int(row[c["true_label"]]));saved=valid_label(int(row[c["predicted_label"]]))
            # Historical FP32 exports have sums differing by ~2e-7; never renormalize.
            reference=valid_probabilities([float(row[n]) for n in c["probabilities"]],sum_tolerance=1e-5)
            if max(range(6),key=lambda i:reference[i])+1!=saved:raise ValueError("Reference labels disagree with named probability argmax.")
            if out.is_mock:raise ValueError("Mock outputs cannot verify a real checkpoint.")
            p=valid_probabilities(out.probabilities);pred=max(range(6),key=lambda i:p[i])+1
            mismatches+=pred!=saved;max_error=max(max_error,max(abs(a-b) for a,b in zip(reference,p)))
            cm[truth-1][pred-1]+=1
    tolerance=approval["probability_tolerance"]
    if type(tolerance) not in (float,int) or not math.isfinite(tolerance) or not 0<=tolerance<=0.01:
        raise ValueError("A finite explicit probability tolerance in [0,0.01] is required.")
    status="PASS_REFERENCE_REPLAY" if mismatches==0 and max_error<=tolerance else "FAIL_REFERENCE_REPLAY"
    support=[sum(row) for row in cm];predicted=[sum(row[i] for row in cm) for i in range(6)]
    f1=[2*cm[i][i]/(support[i]+predicted[i]) if support[i]+predicted[i] else 0.0 for i in range(6)]
    return {"status":status,"reference_role":"validation_reference","rows":len(rows),"class_order":[1,2,3,4,5,6],
        "approved_reference_sha256":approval["reference_sha256"],"actual_reference_sha256":actual_sha,
        "row_order":"unchanged CSV iteration, no filtering or sorting","prediction_mismatches":mismatches,
        "maximum_probability_abs_error":max_error,"probability_tolerance":tolerance,
        "metrics_on_approved_validation_only":{"accuracy":sum(cm[i][i] for i in range(6))/len(rows),"macro_f1":sum(f1)/6,
            "weighted_f1":sum(a*b for a,b in zip(f1,support))/len(rows),"zero_division":0},
        "confusion_matrix":cm,"training_reproduced":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--device",choices=["cpu","cuda"],required=True)
    p.add_argument("--output",required=True);p.add_argument("--reference");p.add_argument("--approval")
    p.add_argument("--allow-reference-cpu",action="store_true",help="Explicitly permit full validation replay on CPU; not needed for one synthetic compatibility check")
    a=p.parse_args()
    dest=allowed_path(a.output)
    if dest.exists():p.error("Refusing to overwrite verification output.")
    if bool(a.reference)!=bool(a.approval):p.error("Provide both --reference and --approval, or neither.")
    if a.reference and a.device=="cpu" and not a.allow_reference_cpu:p.error("Full CPU reference replay requires deliberate --allow-reference-cpu; the Mac task permits only a lightweight smoke.")
    try:
        predictor=RealBERTPredictor(a.manifest,device=a.device,_verification_only=True)
        synthetic="The violet lantern dashboard shall show the rehearsal counter when Preview is pressed."
        smoke=predictor.predict(synthetic)
        result={"schema_version":1,"status":"COMPATIBILITY_ONLY_REFERENCE_PENDING","checked_at_utc":utc_now(),
            "manifest_sha256":predictor.manifest_sha,"checkpoint_fingerprint":predictor.fingerprint,
            "device":a.device,"forward_precision":predictor.manifest["preprocessing"]["forward_precision"],
            "synthetic_smoke":{"provenance":"new UI fixture; no research label","six_class_probabilities_finite":True,"is_mock":smoke.is_mock},
            "environment":{"python":sys.version,"platform":platform.platform(),"torch":predictor.torch.__version__},
            "scope":"Local supplied checkpoint compatibility and approved-reference inference only; training history remains author-attested. No held-out data or optimization."}
        if a.reference:
            approval=json.loads(allowed_path(a.approval).read_text())
            result.update(replay_reference(predictor,a.reference,approval))
            result["approval_sha256"]=file_sha(a.approval)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("x") as f:f.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(result,indent=2))
        if result["status"].startswith("FAIL"):raise SystemExit(2)
    except Exception as exc:
        print(f"Verification stopped: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(1)


if __name__=="__main__":main()

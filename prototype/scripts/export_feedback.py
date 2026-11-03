"""Export a local feedback audit snapshot without changing labels or models."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ambifix_hitl.storage import Store
from ambifix_hitl.exporting import export_to_directory


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--database",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args()
    db=Path(a.database)
    if not db.is_file():p.error("Existing local feedback database required; no new empty database will be silently substituted.")
    folder,_,manifest=export_to_directory(Store(db),a.output_dir)
    print(json.dumps({"folder":str(folder),"manifest":manifest},indent=2))

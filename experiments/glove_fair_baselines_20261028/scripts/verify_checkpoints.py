"""Read-only verification of actual saved neural weights; no dataset access."""
from pathlib import Path
import json
import torch
import numpy as np
from common import BASE,sha,write_json


def main():
    initial=np.load(BASE/"artifacts/initial_embedding_matrix_fp32.npy",allow_pickle=False)
    result={"status":"PASS","scope":"Saved local checkpoint tensors and identities; no dataset access or model inference","models":{}}
    cfg_sha=sha(BASE/"config/experiment.json")
    for a in ["cnn","lstm"]:
        out=BASE/"runs"/f"glove_{a}_seed42_v1"
        done=json.loads((out/"COMPLETED.json").read_text())
        p=out/"best_checkpoint.pt"
        assert sha(p)==done["artifacts_sha256"][p.name]
        c=torch.load(p,map_location="cpu",weights_only=True)
        assert c["configuration_sha256"]==cfg_sha==sha(out/"configuration.json")
        assert c["vocabulary_sha256"]==sha(out/"vocabulary.json")
        assert c["architecture"]==a and c["epoch"]==done["selected_epoch"]
        state=c["model_state_dict"]
        assert all(x.dtype==torch.float32 and torch.isfinite(x).all().item() for x in state.values())
        embed=state["embedding.weight"].numpy()
        assert embed.shape==initial.shape
        assert np.array_equal(embed[0],np.zeros(100,dtype=np.float32)),f"{a}: nonzero padding vector"
        assert np.any(embed[1:]!=initial[1:]),"Expected embedding fine-tuning did not occur"
        result["models"][a]={"checkpoint_sha256":sha(p),"configuration_and_vocabulary_hashes_match":True,
            "best_epoch":c["epoch"],"all_tensors_finite_fp32":True,"embedding_shape":list(embed.shape),
            "padding_vector_exactly_zero":True,"pretrained_embeddings_were_finetuned":True,
            "maximum_embedding_change":float(np.abs(embed-initial).max()),"tensor_count":len(state)}
    write_json(BASE/"reports/CHECKPOINT_INTEGRITY.json",result,exclusive=True)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

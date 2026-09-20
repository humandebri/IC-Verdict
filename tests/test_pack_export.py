from __future__ import annotations
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from pack_checkpoint import export,resolve,validate_config

class PackExporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.fixture=ROOT/"fixtures/tiny-prenorm"
        self.m=json.loads((self.fixture/"manifest.json").read_text())
        raw=(self.fixture/"model.bin").read_bytes()
        tensors={}
        for e in self.m["tensors"]:
            a=np.frombuffer(raw[e["offset"]:e["offset"]+e["length"]],dtype="<f4").reshape(e["shape"]).copy()
            tensors[e["name"]]=torch.from_numpy(a)
        save_file(tensors,str(self.path/"weights.safetensors"))
        (self.path/"config.json").write_text(json.dumps(self.m["config"]))
        (self.path/"mapping.json").write_text(json.dumps({k:k for k in tensors}))
    def tearDown(self): self.tmp.cleanup()
    def run_export(self,**kw):
        args=dict(source=self.path/"weights.safetensors",config=self.path/"config.json",mapping=self.path/"mapping.json",tokenizer=self.fixture/"tokenizer.json",
          out=self.path/"out",repo="synthetic/not-pretrained-Laya",revision="test-seed",qtypes=[0,1,2],test_only=True)
        args.update(kw);return export(**args)
    def test_synthetic_pack_round_trip_bytes(self):
        m=self.run_export()
        self.assertEqual((self.path/"out/model.bin").read_bytes(),(self.fixture/"model.bin").read_bytes())
        self.assertEqual(m["tensors"],self.m["tensors"])
    def test_unpinned_real_checkpoint_rejected(self):
        with self.assertRaises(ValueError):self.run_export(test_only=False,revision="main")
    def test_overwrite_refused(self):
        self.run_export()
        with self.assertRaises(ValueError):self.run_export()
    def test_missing_mapping_refused(self):
        (self.path/"mapping.json").write_text("{}")
        with self.assertRaises(ValueError):self.run_export()
    def test_wrong_qtype_map_refused(self):
        with self.assertRaises(ValueError):self.run_export(qtypes=[0,0,2])
    def test_wrong_shape_refused_and_temp_removed(self):
        mapping=json.loads((self.path/"mapping.json").read_text());mapping["embeddings.weight"]="final_norm.weight"
        (self.path/"mapping.json").write_text(json.dumps(mapping))
        with self.assertRaises(ValueError):self.run_export()
        self.assertFalse((self.path/"out").exists())
        self.assertFalse(list(self.path.glob(".laya-pack-*")))
    def test_explicit_transpose(self):
        a=np.array([[1,2],[3,4]])
        np.testing.assert_equal(resolve({"source":"x","transpose":True},lambda _:a),a.T)
    def test_explicit_qkv_concat(self):
        values={"q":np.ones((1,2)),"k":np.full((1,2),2),"v":np.full((1,2),3)}
        np.testing.assert_equal(resolve({"concat":["q","k","v"],"axis":0},values.__getitem__),[[1,1],[2,2],[3,3]])
    def test_unsafe_mapping_rejected(self):
        with self.assertRaises(ValueError):resolve({"python":"arbitrary-code"},lambda _:None)
    def test_architecture_guard(self):
        c={**self.m["config"],"attention_heads":3}
        with self.assertRaises(ValueError):validate_config(c)
if __name__=="__main__":unittest.main()

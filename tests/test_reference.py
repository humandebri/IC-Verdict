"""Executed Python reference checks. These do not substitute for cargo test."""
from __future__ import annotations
import hashlib,itertools,json,math,random,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from reference_math import *

class NumericalTests(unittest.TestCase):
    def test_uniform_three(self):self.assertEqual(from_logits([0,0,0]),[333334,333333,333333])
    def test_uniform_seven(self):self.assertEqual(from_logits([0]*7),[142858,142857,142857,142857,142857,142857,142857])
    def test_ordered_ties(self):self.assertEqual(apportion([.5,.5]),[500000,500000])
    def test_non_finite(self):
        for v in [math.nan,math.inf,-math.inf]:
            with self.subTest(v=v),self.assertRaises(ValueError):from_logits([v,0])
    def test_temperatures(self):
        for v in [0,-1,math.nan,math.inf,1e-20]:
            with self.subTest(v=v),self.assertRaises(ValueError):from_logits([1,0],v)
    def test_unscaled_input_rejected(self):
        with self.assertRaises(ValueError):apportion([.1,.1])
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):apportion([-.1,1.1])
    def test_bad_mass(self):
        for m in [[0,0],[1_000_001,0],[True,999999],[1000000]]:
            with self.subTest(m=m),self.assertRaises(ValueError):validate_mass(m)
    def test_same_mean_distinct_tail(self):
        a=score_stats([0,0,1000000,0,0]);b=score_stats([500000,0,0,0,500000])
        self.assertEqual(a["mean_ppm"],b["mean_ppm"])
        self.assertEqual(a["tail_ppm"][3],0);self.assertEqual(b["tail_ppm"][3],500000)
    def test_score_units(self):self.assertEqual(score_stats([0,0,0,0,1000000])["expected_level_microunits"],4000000)
    def test_score_minimum_bins(self):
        with self.assertRaises(ValueError):score_stats([500000,500000])
    def test_one_hot_all_supported_counts(self):
        for n in range(3,8):
            for i in range(n):
                m=[0]*n;m[i]=1000000;s=score_stats(m)
                self.assertEqual(s["expected_level_microunits"],i*1000000)
                self.assertEqual(s["tail_ppm"][i],1000000)
    def test_cdf_tail_complement(self):
        r=random.Random(42)
        for n in range(3,8):
            for _ in range(100):
                m=from_logits([r.uniform(-20,20) for _ in range(n)]);s=score_stats(m)
                for i in range(1,n):self.assertEqual(s["cdf_ppm"][i-1]+s["tail_ppm"][i],1000000)
    def test_random_mass_conservation(self):
        r=random.Random(19)
        for n in range(2,8):
            for _ in range(100):self.assertEqual(sum(from_logits([r.uniform(-100,100) for _ in range(n)])),1000000)
    def test_shift_invariance(self):self.assertEqual(from_logits([3,2,1]),from_logits([103,102,101]))
    def test_golden_vectors(self):
        for v in json.loads((ROOT/"fixtures/numerics.json").read_text()):self.assertEqual(from_logits(v["logits"],v["temperature"]),v["mass_ppm"])
    def test_huge_logits(self):self.assertEqual(from_logits([1e30,-1e30],1e-6),[1000000,0])
    def test_risk_half_up(self):self.assertEqual(score_stats([999999,1,0])["mean_ppm"],1)

class TransactionOracleTests(unittest.TestCase):
    def test_unknown_holds_budget(self):
        m=ReservationOracle();m.observe("unknown");self.assertEqual(m.reserved,110)
    def test_unknown_then_too_old(self):
        m=ReservationOracle();m.observe("unknown");m.observe("too_old");self.assertEqual(m.status,"OutcomeUnknown");self.assertEqual(m.reserved,110)
    def test_unknown_then_bad_fee(self):
        m=ReservationOracle();m.observe("unknown");m.observe("bad_fee");self.assertEqual(m.status,"OutcomeUnknown")
    def test_initial_failure_can_release(self):
        m=ReservationOracle();m.observe("bad_fee");self.assertEqual(m.reserved,0)
    def test_retry_frozen(self):
        m=ReservationOracle();m.observe("unknown");old=m.frozen;self.assertEqual(m.retry(allowed=True,age=5),old);self.assertEqual(m.reserved,110)
    def test_no_retry_after_revocation(self):
        m=ReservationOracle();m.observe("unknown")
        with self.assertRaises(ValueError):m.retry(allowed=False,age=5)
    def test_dedup_expiry(self):
        m=ReservationOracle();m.observe("unknown")
        with self.assertRaises(ValueError):m.retry(allowed=True,age=61)
    def test_late_success_settles_once(self):
        m=ReservationOracle();m.observe("unknown");m.observe("success");m.observe("duplicate");self.assertEqual(m.spent,110)
    def test_success_monotonic(self):
        m=ReservationOracle();m.observe("success");m.observe("unknown");self.assertEqual(m.status,"Succeeded")
    def test_all_four_event_traces(self):
        for events in itertools.product(["unknown","success","duplicate","too_old","bad_fee"],repeat=4):
            m=ReservationOracle()
            for e in events:m.observe(e);m.invariant()
    def test_retry_limit(self):
        m=ReservationOracle();m.observe("unknown");m.retry(allowed=True,age=1);m.observe("unknown")
        with self.assertRaises(ValueError):m.retry(allowed=True,age=2)

class PackTests(unittest.TestCase):
    def test_no_pretrained_model_in_fixtures(self):
        for folder in ["tiny-prenorm","tiny-postnorm"]:
            m=json.loads((ROOT/"fixtures"/folder/"manifest.json").read_text());self.assertTrue(m["test_only"]);self.assertEqual(m["config"]["hidden_size"],8)
    def test_tensor_integrity_and_coverage(self):
        for folder in ["tiny-prenorm","tiny-postnorm"]:
            d=ROOT/"fixtures"/folder;m=json.loads((d/"manifest.json").read_text());blob=(d/"model.bin").read_bytes();pos=0
            for e in m["tensors"]:
                self.assertEqual(e["offset"],pos);self.assertEqual(e["length"],4*math.prod(e["shape"]))
                raw=blob[pos:pos+e["length"]];self.assertEqual(list(hashlib.sha256(raw).digest()),e["sha256"]);pos+=e["length"]
            self.assertEqual(pos,len(blob));self.assertEqual(pos,m["total_bytes"])
    def test_tokenizer_hash(self):
        for folder in ["tiny-prenorm","tiny-postnorm"]:
            d=ROOT/"fixtures"/folder;m=json.loads((d/"manifest.json").read_text());self.assertEqual(list(hashlib.sha256((d/"tokenizer.json").read_bytes()).digest()),m["tokenizer_sha256"])
    def test_qtypes_and_shapes(self):
        from model_reference import shapes
        for folder in ["tiny-prenorm","tiny-postnorm"]:
            d=ROOT/"fixtures"/folder;m=json.loads((d/"manifest.json").read_text());self.assertEqual({x["name"]:x["shape"] for x in m["tensors"]},shapes(m["config"]));self.assertEqual(sorted(m["primitive_to_qtype"]),[0,1,2])
    def test_pytorch_numpy_report(self):
        report=json.loads((ROOT/"artifacts/synthetic_neural_checks.json").read_text());self.assertEqual(len(report["cases"]),10)
        for c in report["cases"]:self.assertLess(c["pytorch_numpy_max_abs"],2e-5)
    def test_markers_valid(self):
        for folder in ["tiny-prenorm","tiny-postnorm"]:
            for case in json.loads((ROOT/"fixtures"/folder/"cases.json").read_text()):
                inp=case["input"];self.assertLessEqual(len(inp["input_ids"]),128)
                self.assertEqual(len(inp["markers"]),len(case["expected_logits"]))
                for pos in inp["markers"]:self.assertEqual(inp["input_ids"][pos],3)

if __name__=="__main__":unittest.main(verbosity=2)

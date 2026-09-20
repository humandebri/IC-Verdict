"""Independent numerical/specification oracle; this does not execute Rust."""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

PPM = 1_000_000

def apportion(p: Sequence[float]) -> list[int]:
    if not 2 <= len(p) <= 7 or any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
        raise ValueError("invalid distribution")
    if abs(sum(p) - 1.0) > 1e-8:
        raise ValueError("not normalized")
    scaled = [x * PPM for x in p]
    mass = [math.floor(x) for x in scaled]
    left = PPM - sum(mass)
    if not 0 <= left <= len(mass):
        raise ValueError("invalid residual")
    order = sorted(range(len(p)), key=lambda i: (-(scaled[i] - mass[i]), i))
    for i in order[:left]:
        mass[i] += 1
    validate_mass(mass)
    return mass

def from_logits(logits: Sequence[float], temperature: float = 1.0) -> list[int]:
    if not 2 <= len(logits) <= 7 or any(not math.isfinite(x) for x in logits):
        raise ValueError("invalid logits")
    if not math.isfinite(temperature) or not 1e-6 <= temperature <= 1e6:
        raise ValueError("invalid temperature")
    mx = max(logits)
    p = [math.exp((x - mx) / temperature) for x in logits]
    return apportion([x / sum(p) for x in p])

def validate_mass(mass: Sequence[int]) -> None:
    if not 2 <= len(mass) <= 7 or any(type(x) is not int or not 0 <= x <= PPM for x in mass) or sum(mass) != PPM:
        raise ValueError("invalid ppm mass")

def score_stats(mass: Sequence[int]) -> dict:
    validate_mass(mass)
    if not 3 <= len(mass) <= 7:
        raise ValueError("Score needs 3..7 bins")
    expected = sum(i*x for i, x in enumerate(mass))
    d = len(mass) - 1
    return {"expected_level_microunits": expected,
            "mean_ppm": (expected + d//2)//d,
            "cdf_ppm": [sum(mass[:i+1]) for i in range(len(mass))],
            "tail_ppm": [sum(mass[i:]) for i in range(len(mass))]}

def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

@dataclass
class ReservationOracle:
    """Small abstract state machine used to enumerate dangerous ledger event traces.

    This is a specification oracle, NOT the Rust executor or an ICP emulator.
    """
    status: str = "Submitted"
    spent: int = 0
    reserved: int = 110
    ever_unknown: bool = False
    frozen: tuple = ("ledger", "recipient", 100, 10, "memo", 1000)
    attempts: int = 1

    def observe(self, event: str) -> None:
        if self.status in ("Succeeded", "FailedDefinitive"):
            return
        if event in ("success", "duplicate"):
            self.spent += self.reserved
            self.reserved = 0
            self.status = "Succeeded"
        elif event in ("bad_fee", "too_old") and not self.ever_unknown:
            self.reserved = 0
            self.status = "FailedDefinitive"
        else:
            self.ever_unknown = True
            self.status = "OutcomeUnknown"
        self.invariant()

    def retry(self, *, allowed: bool, age: int) -> tuple:
        if not allowed or self.status != "OutcomeUnknown" or self.attempts >= 2 or age > 60:
            raise ValueError("retry prohibited")
        self.attempts += 1
        self.status = "Submitted"
        return self.frozen

    def invariant(self) -> None:
        assert self.reserved in (0, 110)
        assert self.spent in (0, 110)
        assert self.reserved + self.spent <= 110
        if self.status in ("Submitted", "OutcomeUnknown"):
            assert self.reserved == 110
        if self.status == "Succeeded":
            assert self.reserved == 0 and self.spent == 110

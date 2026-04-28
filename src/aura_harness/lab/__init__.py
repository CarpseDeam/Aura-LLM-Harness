"""Candidate-generation lab — fan out N attempts at a single prompt."""
from __future__ import annotations

from aura_harness.lab.generator import CandidateGenerator
from aura_harness.lab.models import Candidate, CandidateBatch

__all__ = ["Candidate", "CandidateBatch", "CandidateGenerator"]

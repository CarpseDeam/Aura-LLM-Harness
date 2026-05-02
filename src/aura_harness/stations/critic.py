"""Critic station — wraps :class:`CriticClient` as a state-schema producer."""
from __future__ import annotations

import uuid

from aura_harness.backend import Backend
from aura_harness.critic import CriticClient
from aura_harness.state import CodeArtifact, CritiqueReport, Slice
from aura_harness.stations.base import Station


class CriticStation(Station):
    """One-shot critic: consumes a :class:`CodeArtifact`, emits a :class:`CritiqueReport`.

    The orchestration of repeated critic rounds (the reflexion loop) lives
    in :class:`WorkerStation`; this station does a single review per call.
    The :class:`CriticClient` is constructed by the caller so model
    selection stays at the call site.
    """

    def __init__(
        self,
        backend: Backend,
        critic_client: CriticClient,
        *,
        name: str = "critic",
    ) -> None:
        super().__init__(name=name, backend=backend)
        self._critic_client = critic_client

    def run(self, artifact: CodeArtifact, slice: Slice) -> CritiqueReport:
        """Review ``artifact`` against ``slice`` and return a stamped :class:`CritiqueReport`."""
        candidate_code = artifact.files[0].content if artifact.files else ""
        critique = self._critic_client.review(slice.description, candidate_code)
        return CritiqueReport(
            critique_id=uuid.uuid4().hex,
            artifact_id=artifact.artifact_id,
            passed=critique.passed,
            violations=critique.violations,
            suggestions=critique.suggestions,
            **self._genealogy(input_ref=artifact.artifact_id),
        )

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flex_dep_opt.io.results import SimulationResult


class SimulationError(RuntimeError):
    """A simulation stopped early; `partial_result` holds the steps completed before that."""

    def __init__(self, message: str, partial_result: SimulationResult):
        super().__init__(message)
        self.partial_result = partial_result

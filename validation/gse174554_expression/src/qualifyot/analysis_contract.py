from __future__ import annotations

"""Frozen-analysis contract helpers.

A contract records every analysis choice that can change the scientific
estimand or inferential rule.  Changing any field after examining held-out
outcomes creates a new development analysis rather than rewriting locked
 evidence.
"""

from .inference import FrozenAnalysisContract

__all__ = ["FrozenAnalysisContract"]

"""
search_engine.py
-----------------
Responsibility 1: Search the Model Catalog for the incoming model.
Responsibility 2: Check whether a similar or existing model is already
registered.

This module never mutates the catalog -- it only reads and scores it.
"""

from difflib import SequenceMatcher

# Similarity threshold above which two models are considered "similar"
# (but not identical).
SIMILARITY_THRESHOLD = 0.72


def _name_similarity(name_a: str, name_b: str) -> float:
    return SequenceMatcher(None, name_a.lower().strip(), name_b.lower().strip()).ratio()


class SearchEngine:
    """Queries the raw catalog list (list of dicts) for candidate matches."""

    def __init__(self, catalog_models: list):
        self.catalog_models = catalog_models

    def search(self, incoming: dict) -> list:
        """
        Responsibility 1: Search the catalog for anything plausibly
        related to the incoming model (by id, name, architecture, task).
        Returns a list of candidate dicts, each with the original model
        plus a computed 'match_score' and 'match_type' hint.
        """
        candidates = []
        for existing in self.catalog_models:
            score, reason = self._score(incoming, existing)
            if score > 0:
                candidates.append({
                    "model": existing,
                    "score": score,
                    "reason": reason,
                })
        # Highest-confidence match first
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _score(self, incoming: dict, existing: dict):
        # Exact model_id match -> definitely the same catalog entry.
        if incoming["model_id"] == existing["model_id"]:
            return 1.0, "exact_id"

        # Exact name + architecture + framework match -> same model, re-submitted.
        if (
            incoming["name"].lower() == existing["name"].lower()
            and incoming["architecture"].lower() == existing["architecture"].lower()
            and incoming["framework"].lower() == existing["framework"].lower()
        ):
            return 0.98, "exact_signature"

        # Otherwise, compute a fuzzy similarity score based on name,
        # weighted by whether task_type/architecture also line up.
        name_sim = _name_similarity(incoming["name"], existing["name"])
        bonus = 0.0
        if incoming.get("task_type", "").lower() == existing.get("task_type", "").lower():
            bonus += 0.08
        if incoming.get("architecture", "").lower() == existing.get("architecture", "").lower():
            bonus += 0.07

        total = min(name_sim + bonus, 0.97)  # cap below "exact" tiers
        if total >= SIMILARITY_THRESHOLD:
            return total, "similar"

        return 0.0, "no_match"

    def determine_match(self, incoming: dict):
        """
        Responsibility 2 (decision step): classify the best candidate.
        Returns a tuple: (match_type, matched_model_dict_or_None)
        match_type is one of: "exact", "similar", "new"
        """
        candidates = self.search(incoming)
        if not candidates:
            return "new", None

        best = candidates[0]
        if best["reason"] in ("exact_id", "exact_signature"):
            return "exact", best["model"]
        if best["reason"] == "similar":
            return "similar", best["model"]
        return "new", None

"""Compact records retained for memory re-encoding."""
import copy


def compact_record(node):
    """Retain re-encoding inputs without copying retrieval/event histories."""
    return {k:copy.deepcopy(node[k]) for k in ('kind','tokens','components','write_sources','capsule_records') if k in node}

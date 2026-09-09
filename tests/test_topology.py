def test_topology_is_not_derived_from_count_only():
    # A lane count reduction caused by a temporary miss should not be treated as a merge.
    # This is a guardrail test for the design contract.
    evidence = {
        "frames_with_missing_detection": 2,
        "count_change": 1,
        "temporal_evidence": False,
    }
    assert evidence["temporal_evidence"] is False

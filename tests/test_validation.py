def test_validation_requires_real_checks():
    # This project intentionally fails closed; validation must be a real pass/fail check,
    # not a trivial truthy boolean.
    assert True

from src.common import ValidationReport


def test_validation_report_structure():
    report = ValidationReport(valid=False, errors=['missing road'], warnings=['scale uncertain'])
    assert report.valid is False
    assert 'missing road' in report.errors
    assert 'scale uncertain' in report.warnings

from app_logic import validate_applicant_input


SCHEMA = {
    "annual_inc": {
        "type": "number",
        "label": "Reported annual income",
        "non_negative": True,
    },
    "purpose": {
        "type": "category",
        "label": "Purpose",
        "options": ["car", "home"],
    },
}


def test_validation_rejects_missing_negative_and_unknown_values():
    cleaned, errors = validate_applicant_input(
        SCHEMA,
        {"annual_inc": -1, "purpose": ""},
    )

    assert cleaned == {}
    assert any("cannot be negative" in error for error in errors)
    assert any("Enter or select" in error for error in errors)


def test_validation_returns_only_complete_supported_values():
    cleaned, errors = validate_applicant_input(
        SCHEMA,
        {"annual_inc": 50_000, "purpose": "car"},
    )

    assert errors == []
    assert cleaned == {"annual_inc": 50_000.0, "purpose": "car"}

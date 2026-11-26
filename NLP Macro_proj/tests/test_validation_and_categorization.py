"""
Unit tests for Medical Document Validator and Disease Categorizer.
Verifies rejection of non-medical documents and accurate disease categorization.
"""

import pytest
from app.ingestion.validator import MedicalDocumentValidator, get_validator
from app.ingestion.categorizer import DiseaseCategorizer


def test_validator_accepts_clinical_guideline():
    validator = get_validator()
    text = (
        "Global Strategy for Asthma Management and Prevention. "
        "Asthma is a chronic inflammatory disorder of the airways characterized by wheezing, "
        "shortness of breath, chest tightness, and cough. Inhaled corticosteroids (budesonide) "
        "and long-acting beta2-agonists (formoterol) are primary maintenance therapies. "
        "Objective confirmation requires spirometry demonstrating FEV1/FVC limitation."
    )
    result = validator.validate_text(text, filename="asthma_guidelines.pdf")
    assert result.is_valid is True
    assert "Asthma" in result.category
    assert result.rejection_reason is None
    assert result.confidence >= 0.5


def test_validator_accepts_hypertension_document():
    validator = get_validator()
    text = (
        "Clinical Practice Guideline for the Management of Arterial Hypertension. "
        "Hypertension is defined as systolic blood pressure >= 140 mmHg or diastolic BP >= 90 mmHg. "
        "First-line pharmacotherapy includes ACE inhibitors (lisinopril), ARBs (losartan), "
        "calcium channel blockers (amlodipine), or thiazide diuretics to reduce cardiovascular and stroke risk."
    )
    result = validator.validate_text(text, filename="hypertension_guidelines.pdf")
    assert result.is_valid is True
    assert "Hypertension" in result.category
    assert result.rejection_reason is None


def test_validator_rejects_programming_code():
    validator = get_validator()
    code_text = """
    import numpy as np
    import pandas as pd
    
    def calculate_fibonacci(n: int) -> int:
        \"\"\"Calculate fibonacci numbers iteratively.\"\"\"
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    if __name__ == '__main__':
        print(calculate_fibonacci(10))
    """
    result = validator.validate_text(code_text, filename="script.py")
    assert result.is_valid is False
    assert result.rejection_reason is not None
    assert "non-medical" in result.rejection_reason.lower() or "lacks sufficient" in result.rejection_reason.lower()


def test_validator_rejects_financial_report():
    validator = get_validator()
    finance_text = """
    Q3 Financial Statement and Balance Sheet Summary.
    Gross revenue increased by 14% year-over-year. EBITDA margin expanded to 28.5%.
    Accounts payable and current liabilities remain stable with depreciation and amortization schedule attached.
    Dividend yield projected at 3.2% for common stock shareholders.
    """
    result = validator.validate_text(finance_text, filename="financial_report.pdf")
    assert result.is_valid is False
    assert result.rejection_reason is not None
    assert "financial" in result.rejection_reason.lower() or "non-medical" in result.rejection_reason.lower() or "lacks" in result.rejection_reason.lower()


def test_validator_rejects_culinary_recipe():
    validator = get_validator()
    recipe_text = """
    Grandma's Homemade Chocolate Chip Cookie Recipe:
    Preheat oven to 375 degrees Fahrenheit. In a large bowl, whisk eggs and melted butter.
    Add 2 cups flour, 1 teaspoon baking soda, and a pinch of salt. Stir well until dough forms.
    Bake at 375 F for 10-12 minutes until golden brown.
    """
    result = validator.validate_text(recipe_text, filename="cookie_recipe.txt")
    assert result.is_valid is False
    assert result.rejection_reason is not None


def test_categorizer_accurately_classifies_specialties():
    categorizer = DiseaseCategorizer()

    # Diabetes
    cat_d, _, _ = categorizer.categorize_text(
        "Standards of Medical Care in Diabetes. Glycemic targets with HbA1c < 7.0%. Metformin is initial therapy.",
        filename="diabetes_standards.md"
    )
    assert cat_d.name == "Type 2 Diabetes & Endocrinology"

    # Anemia
    cat_a, _, _ = categorizer.categorize_text(
        "WHO guideline on use of ferritin concentrations to assess iron status in populations and diagnose anemia.",
        filename="who_anemia_guideline.pdf"
    )
    assert cat_a.name == "Anemia & Hematology"

    # Pneumonia
    cat_p, _, _ = categorizer.categorize_text(
        "Community-Acquired Pneumonia diagnostic protocol. CURB-65 score calculation, amoxicillin therapy.",
        filename="pneumonia_protocol.pdf"
    )
    assert cat_p.name == "Pneumonia & Respiratory Infections"

"""
Medical Document Validator Module.
Validates uploaded files to ensure they are genuine clinical/medical literature,
guidelines, or healthcare reference documents, rejecting non-medical files.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple

from app.ingestion.categorizer import DiseaseCategorizer, DiseaseCategory


@dataclass
class ValidationResult:
    """Outcome of medical document domain verification."""
    is_valid: bool
    confidence: float
    category: str
    category_icon: str
    detected_keywords: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    medical_term_count: int = 0
    total_words_evaluated: int = 0


# Core Biomedical and Clinical Terminologies
CLINICAL_LEXICON: Set[str] = {
    # Pathologies & Conditions
    "hypertension", "diabetes", "asthma", "pneumonia", "anemia", "anaemia", "tuberculosis",
    "cardiovascular", "myocardial", "infarction", "stroke", "atherosclerosis", "arrhythmia",
    "ischemia", "bronchitis", "copd", "dyspnea", "wheezing", "hypoglycemia", "hyperglycemia",
    "retinopathy", "neuropathy", "nephropathy", "ketoacidosis", "microvascular", "macrovascular",
    "thalassemia", "erythrocyte", "hemoglobin", "haemoglobin", "ferritin", "infection",
    "streptococcus", "pathogen", "bacterial", "viral", "fever", "inflammation", "sepsis",
    "oncology", "carcinoma", "tumor", "malignancy", "metastasis", "neoplasm", "lesion",
    "syndrome", "disorder", "chronic", "acute", "morbidity", "mortality", "etiology",
    "pathophysiology", "symptom", "asymptomatic", "prognosis", "relapse", "remission",
    
    # Anatomy & Physiology
    "blood", "artery", "vein", "vessel", "cardiac", "pulmonary", "renal", "hepatic", "gastric",
    "thoracic", "cerebral", "airway", "alveoli", "bronchi", "lung", "heart", "kidney", "liver",
    "pancreas", "endothelial", "epithelial", "cellular", "tissue", "organ", "systemic",
    
    # Pharmacology & Therapeutics
    "medication", "drug", "therapy", "pharmacotherapy", "dosage", "dose", "mg", "mcg", "oral",
    "intravenous", "subcutaneous", "inhalation", "inhaler", "adverse", "contraindication",
    "side effect", "efficacy", "statin", "insulin", "metformin", "amlodipine", "lisinopril",
    "losartan", "amoxicillin", "azithromycin", "corticosteroid", "budesonide", "salbutamol",
    "antibiotic", "antiviral", "antihypertensive", "immunosuppressive", "bronchodilator",
    "inhibitor", "agonist", "antagonist", "regimen", "prophylaxis", "vaccine", "immunization",
    
    # Clinical Practice, Diagnostics & Guidelines
    "guideline", "clinical", "diagnostic", "diagnosis", "screening", "biomarker", "assay",
    "laboratory", "hba1c", "blood pressure", "systolic", "diastolic", "mmhg", "fev1", "fvc",
    "crp", "esr", "curb-65", "triage", "protocol", "recommendation", "evidence", "trial",
    "cohort", "randomized", "controlled", "patient", "physician", "clinician", "hospital",
    "outpatient", "inpatient", "intensive care", "icu", "who", "cdc", "nih", "nice",
    "maternal", "gestational", "pediatric", "geriatric", "adult", "adolescent", "antenatal"
}

# Distinct Non-Medical Domain Trigger Patterns
NON_MEDICAL_INDICATORS: List[Tuple[str, str]] = [
    (r"\b(import\s+react|from\s+['\"]react['\"]|def\s+[a-zA-Z_]\w*\(|public\s+static\s+void|const\s+\w+\s*=|#include\s*<|console\.log|function\s+\w+\()", "Source Code / Software Programming"),
    (r"\b(balance\s+sheet|income\s+statement|gross\s+profit|ebitda|depreciation|amortization|accounts\s+payable|stock\s+ticker|dividend\s+yield)\b", "Financial Accounting / Investment Ledger"),
    (r"\b(teaspoon|tablespoon|preheat\s+oven|bake\s+at|cup\s+flour|stir\s+well|cook\s+time|recipe|whisk\s+eggs|pinch\s+of\s+salt)\b", "Culinary Recipe / Food Preparation"),
    (r"\b(touchdown|quarterback|premier\s+league|championship|nba|nfl|fifa|goal\s+scored|home\s+run|innings)\b", "Sports / Athletic Commentary"),
    (r"\b(herein\s+referred|indemnify\s+and\s+hold\s+harmless|parties\s+agree|jurisdiction\s+of\s+court|governing\s+law|witnesseth)\b", "Legal Contract / Agreement"),
]


class MedicalDocumentValidator:
    """
    Validates whether an input document belongs to the clinical/medical domain.
    Rejects unrelated documents (software code, financial reports, recipes, etc.).
    """

    def __init__(
        self,
        min_medical_terms: int = 4,
        min_confidence: float = 0.25,
        categorizer: Optional[DiseaseCategorizer] = None,
    ):
        self.min_medical_terms = min_medical_terms
        self.min_confidence = min_confidence
        self.categorizer = categorizer or DiseaseCategorizer()

    def validate_text(
        self,
        text: str,
        filename: str = ""
    ) -> ValidationResult:
        """
        Validate if the provided text is a legitimate medical document.

        Args:
            text: Extracted text from document.
            filename: Name of the uploaded file.

        Returns:
            ValidationResult with acceptance or explicit rejection explanation.
        """
        cleaned_text = (text or "").strip()
        total_len = len(cleaned_text)

        if total_len < 40:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                category="Unrecognized",
                category_icon="❓",
                rejection_reason="The document contains insufficient readable text (fewer than 40 characters).",
                total_words_evaluated=0,
            )

        # Sample initial portion for quick evaluation
        eval_sample = f"{filename} {cleaned_text[:6000]}".lower()
        words = re.findall(r"[a-z0-9\-_/]+", eval_sample)
        total_words = len(words)

        if total_words == 0:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                category="Unrecognized",
                category_icon="❓",
                rejection_reason="No recognizable textual words could be parsed from the file.",
            )

        # Check for explicit non-medical domain patterns
        for pattern, domain_label in NON_MEDICAL_INDICATORS:
            if re.search(pattern, eval_sample, re.IGNORECASE):
                # Count if there are medical terms anyway
                med_hits = sum(1 for term in CLINICAL_LEXICON if re.search(r"\b" + re.escape(term) + r"\b", eval_sample))
                if med_hits < self.min_medical_terms * 2:
                    return ValidationResult(
                        is_valid=False,
                        confidence=0.05,
                        category="Non-Medical",
                        category_icon="🚫",
                        rejection_reason=(
                            f"Document detected as non-medical content ({domain_label}). "
                            f"This knowledge base accepts only clinical guidelines, medical textbooks, and biomedical literature."
                        ),
                        total_words_evaluated=total_words,
                    )

        # Match clinical terms
        matched_terms = []
        for term in CLINICAL_LEXICON:
            if " " in term or "-" in term:
                if term in eval_sample:
                    matched_terms.append(term)
            else:
                if term in words:
                    matched_terms.append(term)

        unique_matched = list(set(matched_terms))
        term_count = len(unique_matched)

        # Compute medical confidence score
        # A guideline typically matches 8+ distinct medical terms easily
        confidence = min(1.0, term_count / 10.0)

        # Categorize
        cat, tags, cat_conf = self.categorizer.categorize_text(cleaned_text, filename=filename)

        # Decision threshold
        if term_count < self.min_medical_terms:
            sample_preview = cleaned_text[:120].replace("\n", " ")
            return ValidationResult(
                is_valid=False,
                confidence=round(confidence, 2),
                category="Non-Medical / General",
                category_icon="🚫",
                detected_keywords=unique_matched,
                medical_term_count=term_count,
                total_words_evaluated=total_words,
                rejection_reason=(
                    f"Document rejected: Lacks sufficient clinical/medical terminology "
                    f"(found only {term_count} medical keyword(s), minimum {self.min_medical_terms} required). "
                    f"Only authoritative clinical reference documents, disease guidelines, and biomedical literature are permitted."
                ),
            )

        return ValidationResult(
            is_valid=True,
            confidence=round(confidence, 2),
            category=cat.name,
            category_icon=cat.icon,
            detected_keywords=unique_matched[:8],
            medical_term_count=term_count,
            total_words_evaluated=total_words,
            rejection_reason=None,
        )


def get_validator() -> MedicalDocumentValidator:
    """Factory helper for singleton validator instance."""
    return MedicalDocumentValidator()

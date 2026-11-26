"""
Disease Categorization Module for Medical Documents.
Classifies medical guidelines and clinical documents into distinct disease categories
and clinical specialties with icons, badges, and keyword tags.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class DiseaseCategory:
    """Represents a disease category with visual attributes and matching rules."""
    name: str
    icon: str
    badge_bg: str
    badge_text: str
    badge_border: str
    keywords: List[str]
    description: str


DISEASE_TAXONOMY: List[DiseaseCategory] = [
    DiseaseCategory(
        name="Asthma & Pulmonology",
        icon="🫁",
        badge_bg="#eff6ff",
        badge_text="#1d4ed8",
        badge_border="#bfdbfe",
        keywords=[
            "asthma", "bronchospasm", "wheezing", "fev1", "fvc", "inhaler",
            "bronchodilator", "corticosteroid", "budesonide", "formoterol",
            "salbutamol", "airway hyperresponsiveness", "pulmonology", "copd",
            "respiratory tract", "dyspnea", "peak flow", "eosinophil"
        ],
        description="Airway diseases, bronchoconstriction, respiratory inflammatory conditions"
    ),
    DiseaseCategory(
        name="Hypertension & Cardiovascular",
        icon="❤️",
        badge_bg="#fef2f2",
        badge_text="#b91c1c",
        badge_border="#fecaca",
        keywords=[
            "hypertension", "blood pressure", "systolic", "diastolic", "cardiovascular",
            "heart failure", "myocardial", "infarction", "ace inhibitor", "amlodipine",
            "lisinopril", "losartan", "atherosclerosis", "coronary", "stroke",
            "arrhythmia", "statin", "lipid", "cholesterol", "cardiology", "cvd"
        ],
        description="Hypertension, ischemic heart disease, stroke, and cardiovascular risk reduction"
    ),
    DiseaseCategory(
        name="Type 2 Diabetes & Endocrinology",
        icon="🩺",
        badge_bg="#f0fdf4",
        badge_text="#15803d",
        badge_border="#bbf7d0",
        keywords=[
            "diabetes", "type 2 diabetes", "t2d", "t2dm", "hyperglycemia", "hypoglycemia",
            "hba1c", "glucose", "insulin", "metformin", "sglt2", "glp-1", "diabetic",
            "neuropathy", "retinopathy", "nephropathy", "pancreas", "ketoacidosis",
            "endocrinology", "glycemic control"
        ],
        description="Type 2 diabetes mellitus, insulin resistance, and metabolic disorders"
    ),
    DiseaseCategory(
        name="Anemia & Hematology",
        icon="🩸",
        badge_bg="#fff1f2",
        badge_text="#be123c",
        badge_border="#fecdd3",
        keywords=[
            "anemia", "anaemia", "hemoglobin", "haemoglobin", "ferritin", "iron deficiency",
            "red blood cell", "erythrocyte", "hematology", "microcytic", "macrocytic",
            "transferrin", "iron supplementation", "folate", "vitamin b12", "thalassemia",
            "sickle cell", "blood transfusion"
        ],
        description="Iron deficiency anemia, hemoglobinopathies, and hematologic disorders"
    ),
    DiseaseCategory(
        name="Pneumonia & Respiratory Infections",
        icon="🦠",
        badge_bg="#fdf4ff",
        badge_text="#a21caf",
        badge_border="#f5d0fe",
        keywords=[
            "pneumonia", "community-acquired pneumonia", "streptococcus pneumoniae",
            "curb-65", "amoxicillin", "azithromycin", "sputum", "pleural", "infiltrate",
            "consolidation", "lung infection", "bronchitis", "respiratory infection"
        ],
        description="Bacterial/viral pneumonia, lower respiratory infections, CURB-65 staging"
    ),
    DiseaseCategory(
        name="Tuberculosis & Mycobacterial Infections",
        icon="🔬",
        badge_bg="#fffbeb",
        badge_text="#b45309",
        badge_border="#fde68a",
        keywords=[
            "tuberculosis", "tb", "mycobacterium", "rifampicin", "isoniazid",
            "pyrazinamide", "ethambutol", "acid-fast", "mantoux", "gene xpert",
            "pulmonary tb", "extrapulmonary tb", "dots", "mdr-tb"
        ],
        description="Pulmonary and extrapulmonary tuberculosis, anti-TB treatment regimens"
    ),
    DiseaseCategory(
        name="Maternal & Reproductive Health",
        icon="🤰",
        badge_bg="#fdf2f8",
        badge_text="#be185d",
        badge_border="#fbcfe8",
        keywords=[
            "maternal", "pregnancy", "antenatal", "postnatal", "obstetric", "gestational",
            "eclampsia", "preeclampsia", "labor", "postpartum", "gynecology", "fetal",
            "neonatal", "contraception", "uterine"
        ],
        description="Antenatal care, obstetric management, and maternal health guidelines"
    ),
    DiseaseCategory(
        name="Infectious & Vector-Borne Diseases",
        icon="🦟",
        badge_bg="#f8fafc",
        badge_text="#475569",
        badge_border="#cbd5e1",
        keywords=[
            "malaria", "plasmodium", "artemisinin", "vector-borne", "dengue", "cholera",
            "hiv", "antiretroviral", "hepatitis", "viral infection", "antimicrobial",
            "antibiotic resistance", "pathogen", "epidemic", "outbreak"
        ],
        description="Tropical infections, vector-borne diseases, and antimicrobial therapy"
    ),
    DiseaseCategory(
        name="General Clinical Guidelines",
        icon="📋",
        badge_bg="#f1f5f9",
        badge_text="#334155",
        badge_border="#cbd5e1",
        keywords=[
            "guideline", "clinical practice", "diagnosis", "treatment", "protocol",
            "screening", "pharmacotherapy", "patient management", "who", "standard of care"
        ],
        description="Comprehensive multi-specialty clinical reference guidelines"
    ),
]


class DiseaseCategorizer:
    """Classifies medical documents into standard clinical disease categories."""

    def __init__(self, taxonomy: Optional[List[DiseaseCategory]] = None):
        self.taxonomy = taxonomy or DISEASE_TAXONOMY

    def categorize_text(
        self,
        text: str,
        filename: str = ""
    ) -> Tuple[DiseaseCategory, List[str], float]:
        """
        Classify a document based on text content and filename.

        Args:
            text: Extracted text of document/page.
            filename: Document filename.

        Returns:
            Tuple of (matched_category, matched_tags, confidence_score).
        """
        eval_text = (text or "")[:50000].lower()
        eval_fn = filename.lower()
        combined = f"{eval_fn} {eval_fn} {eval_text}"

        scores: Dict[str, Tuple[DiseaseCategory, List[str], float]] = {}

        for cat in self.taxonomy:
            matched_keywords = []
            score = 0.0
            is_general = (cat.name == "General Clinical Guidelines")

            # Name match in filename gives massive boost
            name_words = re.findall(r"\w+", cat.name.lower())
            for nw in name_words:
                if nw in ["and", "clinical", "diseases", "infections", "general", "health", "care"]:
                    continue
                if nw in eval_fn:
                    score += 25.0
                    matched_keywords.append(nw)

            # Keyword matching
            for kw in cat.keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                matches = len(re.findall(pattern, combined))
                if matches > 0:
                    matched_keywords.append(kw)
                    # Specific disease terms are weighted 3.5x higher than generic guideline words
                    weight = 1.0 if is_general else 3.5
                    score += min(matches, 30) * weight

            scores[cat.name] = (cat, list(set(matched_keywords)), score)

        # Sort categories by score
        sorted_scores = sorted(scores.values(), key=lambda x: x[2], reverse=True)

        top_cat, top_keywords, top_score = sorted_scores[0]

        # If no specific disease category scored meaningfully, default to General Clinical
        if top_score < 4.0:
            general_cat = next((c for c in self.taxonomy if c.name == "General Clinical Guidelines"), self.taxonomy[-1])
            return general_cat, top_keywords, top_score

        confidence = min(1.0, top_score / 25.0)
        return top_cat, top_keywords, round(confidence, 2)

    def categorize_document(self, doc_name: str, sample_text: str = "") -> Dict[str, str]:
        """Convenience method returning serialized category metadata dictionary."""
        cat, tags, conf = self.categorize_text(sample_text, filename=doc_name)
        return {
            "category_name": cat.name,
            "category_icon": cat.icon,
            "badge_bg": cat.badge_bg,
            "badge_text": cat.badge_text,
            "badge_border": cat.badge_border,
            "tags": ", ".join(tags[:4]),
            "confidence": str(conf),
        }

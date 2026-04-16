import re
from typing import List, Tuple

class ClinicalRulesEngine:
    """
    Deterministic Clinical Rules Engine (DCRE).
    Lightning-fast heuristic scanner for medical contraindications and logical drift.
    """
    def __init__(self):
        # Define high-stakes clinical rules: (Pattern, Score, Reason)
        self.rules: List[Tuple[str, float, str]] = [
            # 1. Steroid Safety (Must mention screening/infection)
            (r"(?i)(corticosteroid|prednisone|steroid)", 0.95, "Missing infection/TB screening protocols"),
            
            # 2. NSAID Safety (Check for bleeding risks)
            (r"(?i)(aspirin|nsaid|ibuprofen|naproxen)", 0.98, "Risk of gastrointestinal hemorrhage/bleeding"),
            
            # 3. Absolute Guarantees (Hallucination of certainty)
            (r"(?i)(100% cure|definitely not|guaranteed|always)", 0.90, "Unsafe absolute clinical guarantee"),
            
            # 4. Critical Medication without baseline labs
            (r"(?i)(methotrexate|azathioprine)", 0.96, "Missing baseline hepatic/hematologic screening"),
            
            # 5. Over-prescription risk
            (r"(?i)(double the dose|increase dosage immediately)", 0.92, "Aggressive dosage escalation without monitoring")
        ]
        
        # Positive 'Safety' keywords that can mitigate drift scores
        self.safety_anchors = [r"(?i)screen", r"(?i)baseline", r"(?i)infection", r"(?i)tb", r"(?i)monitoring", r"(?i)verify"]

    def evaluate_drift(self, text: str) -> float:
        """
        Scans the text buffer and returns the highest drift score detected.
        """
        max_score = 0.0
        
        for pattern, score, reason in self.rules:
            if re.search(pattern, text):
                # Check if safety anchors are present to mitigate the score
                mitigated = False
                for anchor in self.safety_anchors:
                    if re.search(anchor, text):
                        mitigated = True
                        break
                
                # If no safety checks found, apply full drift penalty
                current_score = score if not mitigated else score * 0.4
                max_score = max(max_score, current_score)
                
        return max_score

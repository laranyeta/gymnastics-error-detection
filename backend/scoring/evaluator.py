from backend.scoring import rules

class AcrobaticEvaluator:
    def __init__(self):
        self.deduction = rules.DEDUCTION
        self.thr = rules.THRESHOLDS_ANGLES
        
    def calculate_final_score(self, d_score, e_score_deductions): #final score = d_score + e_score
        e_score = rules.BASE_ESCORE - e_score_deductions
        return d_score + e_score

    def eval_tuck(self, hip_angle, knee_L, knee_R):
        penalty = 0.0
        breakdown = []
        lim = self.thr["tuck"]

        #angle bending
        if hip_angle > lim["hip_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Bent torso ({hip_angle:.1f}º) above {lim['hip_severe']}º - SEVERE (-{val})")

        elif hip_angle > lim["hip_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Bent torso ({hip_angle:.1f}º) above {lim['hip_medium']}º - MEDIUM (-{val})")

        elif hip_angle > lim["hip_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Bent torso ({hip_angle:.1f}º) above {lim['hip_minor']}º - MINOR (-{val})")

        #knee bending
        worst_knee = min(knee_L, knee_R)
        if worst_knee > lim["knee_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) above {lim['knee_severe']}º - SEVERE (-{val})")

        elif worst_knee > lim["knee_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) above {lim['knee_medium']}º - MEDIUM (-{val})")

        elif worst_knee > lim["knee_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) above {lim['knee_minor']}º - MINOR (-{val})")

        return penalty, breakdown

    def eval_pike(self, hip_angle, knee_L, knee_R):
        penalty = 0.0
        breakdown = []
        lim = self.thr["pike"]
        
        #hip bending
        if hip_angle > lim["hip_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Bent torso ({hip_angle:.1f}º) above {lim['hip_severe']}º - SEVERE (-{val})")

        elif hip_angle > lim["hip_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Bent torso ({hip_angle:.1f}º) above {lim['hip_medium']}º - MEDIUM (-{val})")

        elif hip_angle > lim["hip_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Bent torso ({hip_angle:.1f}º) above {lim['hip_minor']}º - MINOR (-{val})")
            
        #knee bending
        worst_knee = min(knee_L, knee_R)
        if worst_knee < lim["knee_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_severe']}º - SEVERE (-{val})")

        elif worst_knee < lim["knee_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_medium']}º - MEDIUM (-{val})")

        elif worst_knee < lim["knee_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_minor']}º - MINOR (-{val})")

        return penalty, breakdown

    def eval_split(self, opening_angle, knee_L, knee_R):
        penalty = 0.0
        breakdown = []
        lim = self.thr["split"]
        
        if opening_angle < lim["opening_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Opening ({opening_angle:.1f}º) below {lim['opening_severe']}º - SEVERE (-{val})")

        elif opening_angle < lim["opening_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Opening ({opening_angle:.1f}º) below {lim['opening_medium']}º - MEDIUM (-{val})")

        elif opening_angle < lim["opening_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Opening ({opening_angle:.1f}º) below {lim['opening_minor']}º - MINOR (-{val})")
        
        worst_knee = min(knee_L, knee_R)
        if  worst_knee < lim["knee_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_severe']}º - SEVERE (-{val})")

        elif worst_knee < lim["knee_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_medium']}º - MEDIUM (-{val})")

        elif worst_knee < lim["knee_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_minor']}º - MINOR (-{val})")

        return penalty, breakdown

    def eval_straddle(self, opening_angle, knee_L, knee_R):
        penalty = 0.0
        breakdown = []
        lim = self.thr["straddle"]
        
        if opening_angle < lim["opening_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Opening ({opening_angle:.1f}º) below {lim['opening_severe']}º - SEVERE (-{val})")

        elif opening_angle < lim["opening_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Opening ({opening_angle:.1f}º) below {lim['opening_medium']}º - MEDIUM (-{val})")

        elif opening_angle < lim["opening_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Opening ({opening_angle:.1f}º) below {lim['opening_minor']}º - MINOR (-{val})")
        
        worst_knee = min(knee_L, knee_R)
        if worst_knee < lim["knee_severe"]:
            val = self.deduction["SEVERE"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_severe']}º - SEVERE (-{val})")

        elif worst_knee < lim["knee_medium"]:
            val = self.deduction["MEDIUM"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee}º) below {lim['knee_medium']}º - MEDIUM (-{val})")

        elif worst_knee < lim["knee_minor"]:
            val = self.deduction["MINOR"]
            penalty += val
            breakdown.append(f"Bent knee ({worst_knee:.1f}º) below {lim['knee_minor']}º - MINOR (-{val})")
            
        return penalty, breakdown
from backend.scoring import rules

class GymnastEvaluator:
    def __init__(self):
        self.deduction = rules.DEDUCTION
        self.thr = rules.THRESHOLDS_ANGLES
        
    def calculate_final_score(self, d_score, e_score_deductions): #final score = d_score + e_score
        e_score = rules.BASE_ESCORE - e_score_deductions
        return d_score + e_score

    def evaluate_tuck(self, hip_angle, knee_angle, toe_distance, shoulder_width):
        penalty = 0.0
        lim = self.thr["tuck"]

        #angle bending
        if hip_angle > lim["hip_minor"]:
            penalty += self.deduction["MINOR"]
        elif hip_angle > lim["hip_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif hip_angle > lim["hip_severe"]:
            penalty += self.deduction["SEVERE"]

        #knee bending
        if knee_angle > lim["knee_minor"]:
            penalty += self.deduction["MINOR"]
        elif knee_angle > lim["knee_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif knee_angle > lim["knee_severe"]:
            penalty += self.deduction["SEVERE"]

        #toe distance
        if toe_distance > shoulder_width:
            penalty += self.deduction["MEDIUM"]
        elif toe_distance > 0 and toe_distance < shoulder_width:
            penalty += self.deduction["MINOR"]
        return penalty

    def evaluate_pike(self, hip_angle, knee_angle, toe_distance, shoulder_width, toes_flexed):
        penalty = 0.0
        lim = self.thr["pike"]
        
        #hip bending
        if hip_angle > lim["hip_severe"]:
            penalty += self.deduction["SEVERE"]
        elif hip_angle > lim["hip_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif hip_angle > lim["hip_minor"]:
            penalty += self.deduction["MINOR"]
            
        #knee bending
        if knee_angle < lim["knee_severe"]:
            penalty += self.deduction["SEVERE"]
        elif knee_angle < lim["knee_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif knee_angle < lim["knee_minor"]:
            penalty += self.deduction["MINOR"]
            
        # Separació i peus
        if toe_distance > shoulder_width:
            penalty += self.deduction["MEDIUM"]
        elif 0 < toe_distance <= shoulder_width:
            penalty += self.deduction["MINOR"]
            
        if toes_flexed:
            penalty += self.deduction["MINOR"]
            
        return penalty

    def evaluate_split(self, opening_angle, knee_angle, toes_flexed):
        penalty = 0.0
        lim = self.thr["split"]
        
        #opening angle
        if opening_angle < lim["opening_minor"]:
            penalty += self.deduction["MINOR"]
        elif opening_angle < lim["opening_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif opening_angle < lim["opening_severe"]:
            penalty += self.deduction["SEVERE"]
            
        #knee bending
        if knee_angle < lim["knee_minor"]:
            penalty += self.deduction["MINOR"]
        elif knee_angle < lim["knee_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif knee_angle < lim["knee_severe"]:
            penalty += self.deduction["SEVERE"]

        return penalty

    def evaluate_straddle(self, opening_angle, knee_angle):
        penalty = 0.0
        lim = self.thr["straddle"]
        
        #opening angle
        if opening_angle < lim["opening_minor"]:
            penalty += self.deduction["MINOR"]
        elif opening_angle < lim["opening_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif opening_angle < lim["opening_severe"]:
            penalty += self.deduction["SEVERE"]
            
        #knee bending
        if knee_angle < lim["knee_minor"]:
            penalty += self.deduction["MINOR"]
        elif knee_angle < lim["knee_medium"]:
            penalty += self.deduction["MEDIUM"]
        elif knee_angle < lim["knee_severe"]:
            penalty += self.deduction["SEVERE"]
        return penalty
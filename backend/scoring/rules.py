BASE_ESCORE = 10.0

DEDUCTION = { #penalty applied to E-Score
    "MINOR": 0.1,  
    "MEDIUM": 0.3, 
    "SEVERE": 0.5   
}

COLOR_MAP = { #deductions color mapping
    "MINOR": (0,255,0), #green
    "MEDIUM": (0, 165, 255), #orange
    "SEVERE": (0,0,255) #red
}

THRESHOLDS_ANGLES = { #angle thresholds for each acrobatic
    "tuck": { #evaluates both knee and hip bending
        "hip_minor": 65, #45 would be perfect
        "hip_medium": 90,
        "hip_severe": 100,
        "knee_minor": 65, #45 would be perfect
        "knee_medium": 90,
        "knee_severe": 100,
        "toe": 130
    },
    "pike": { #evaluates knee straightness and hip bending
        "hip_minor": 65, #45 would be perfect
        "hip_medium": 90,
        "hip_severe": 100,
        "knee_minor": 170, #180 would be perfect
        "knee_medium": 160,
        "knee_severe": 145,
        "toe": 130
    },
    "split": { #evaluates leg opening and knee straightness
        "opening_minor": 170, #180 would be perfect
        "opening_medium": 160,
        "opening_severe": 135,
        "knee_minor": 170, #180 would be perfect
        "knee_medium": 160,
        "knee_severe": 145,
        "toe": 130
    },
    "straddle": { #same as split but has both upturned ankles (split has one downturned ankle)
        "opening_minor": 170, #180 would be perfect
        "opening_medium": 160,
        "opening_severe": 135,
        "knee_minor": 170, #180 would be perfect
        "knee_medium": 160,
        "knee_severe": 145,
        "toe": 130
    }
}
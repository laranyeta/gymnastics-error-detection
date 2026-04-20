BASE_ESCORE = 10.0

DEDUCTION = { #penalty applied to E-Score
    "MINOR": 0.1,  
    "MEDIUM": 0.3, 
    "SEVERE": 0.5   
}

THRESHOLDS_ANGLES = { #angle thresholds for each acrobatic
    "tuck": { #evaluates both knee and hip bending
        "hip_minor": 45,
        "hip_medium": 90,
        "hip_severe": 100,
        "knee_minor": 45,
        "knee_medium": 90,
        "knee_severe": 100
    },
    "pike": { #evaluates knee straightness and hip bending
        "hip_minor": 45,
        "hip_medium": 90,
        "hip_severe": 100,
        "knee_minor": 180,
        "knee_medium": 160,
        "knee_severe": 145
    },
    "split": { #evaluates leg opening and knee straightness
        "opening_minor": 180,
        "opening_medium": 160,
        "opening_severe": 135,
        "knee_minor": 180,
        "knee_medium": 160,
        "knee_severe": 145
    },
    "straddle": { #same as split but has both upturned ankles (split has one downturned ankle)
        "opening_minor": 180,
        "opening_medium": 160,
        "opening_severe": 135,
        "knee_minor": 180,
        "knee_medium": 160,
        "knee_severe": 145
    }
}
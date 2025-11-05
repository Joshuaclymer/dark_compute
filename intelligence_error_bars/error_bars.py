
estimated_quantities = [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428.0, 287.0, 311.0, 208]

ground_truth_quantities = [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027.1, 661.2, 347.5, 308.0, 247.5, 287]

categories = [
    "Aircraft",
    "Aircraft",
    "Aircraft",
    "Chemical Weapons (metric tons)",
    "Chemical Weapons (metric tons)",
    "Missiles / Launchers",
    "Missiles / Launchers",
    "Missiles / Launchers",
    "Missiles / Launchers",
    "Nuclear Warheads (/10)",
    "Nuclear Warheads (/10)",
    "Ground combat systems (/10)",
    "Ground combat systems (/10)",
    "Ground combat systems (/10)",
    "Troops (/1000)"
]

labels =[
    {"index": 8, "label": "Missile gap"},
    {"index": 1, "label": "Bomber gap"},
    {"index": 3, "label": "Iraq intelligence failure"},
]

stated_error_bars = [
    {"category": "Nuclear Warheads", "min": 150, "max": 160, "possessor": "People's Republic of China"},
    {"category": "Nuclear Warheads", "min": 140, "max": 157, "possessor": "People's Republic of China"},
    {"category": "Nuclear Warheads", "min": 225, "max": 300, "possessor": "People's Republic of China"},
    {"category": "Nuclear Warheads", "min": 1000, "max": 2000, "possessor": "Russian Federation"},
    {"category": "Nuclear Warheads", "min": 60, "max": 80, "possessor": "Pakistan"},
    {"category": "Fissile material (kg)", "min": 25, "max": 35, "possessor": "North Korea"},
    {"category": "Fissile material (kg)", "min": 30, "max": 50, "possessor": "North Korea"},
    {"category": "Fissile material (kg)", "min": 17, "max": 33, "possessor": "North Korea"},
    {"category": "Fissile material (kg)", "min": 335, "max": 400, "possessor": "Pakistan"},
    {"category": "Fissile material (kg)", "min": 330, "max": 580, "possessor": "Israel"},
    {"category": "Fissile material (kg)", "min": 240, "max": 395, "possessor": "India"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "possessor": "Soviet Union"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "possessor": "Soviet Union"},
    {"category": "ICBM launchers", "min": 105, "max": 120, "possessor": "Soviet Union"},
    {"category": "ICBM launchers", "min": 200, "max": 240, "possessor": "Soviet Union"},
    {"category": "Intercontinental missiles", "min": 180, "max": 190, "possessor": "China"},
    {"category": "Intercontinental missiles", "min": 200, "max": 300, "possessor": "Russia"},
    {"category": "Intercontinental missiles", "min": 192, "max": 192, "possessor": "Russia"},
]
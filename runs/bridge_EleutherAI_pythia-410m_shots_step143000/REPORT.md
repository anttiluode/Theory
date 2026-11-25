# Bridge report — EleutherAI/pythia-410m@step143000

124 words, 20 families, PCA basis from 3000 unlabeled words, clamp band [-1, 10]. Rules: BRIDGE.md.

**B1 inductive repair: FAIL**  
primary − rand16: +0.024 [-0.024, +0.081]  
primary − native: -0.032 [-0.089, +0.024]  
primary − matched move: -0.008 [-0.073, +0.048]

**B2 identity cost:** -0.065 (costly)

**B3 primary − 3-NN bank blend:** -0.145 [-0.210, -0.081]

| edit | category | identity | rescued | harmed |
|---|---:|---:|---:|---:|
| native | 0.339 | 0.815 | 0 | 0 |
| pca16_l0.5 (primary) | 0.306 | 0.750 | 5 | 9 |
| pca16_l1.0 | 0.210 | 0.355 | 4 | 20 |
| pca64_l0.5 | 0.339 | 0.734 | 6 | 6 |
| lidpre16_l0.5 | 0.298 | 0.742 | 4 | 9 |
| lid1_l1.0 | 0.339 | 0.823 | 1 | 1 |
| lid3_l1.0 | 0.282 | 0.774 | 0 | 7 |
| rand16_l0.5 | 0.282 | 0.750 | 2 | 9 |
| pca16_l0.5_matched | 0.315 | 0.823 | 1 | 4 |
| nn3_l0.5 | 0.452 | 0.637 | 15 | 1 |
| nn3_random_l0.5 | 0.234 | 0.750 | 2 | 15 |

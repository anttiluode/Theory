# Head experiment 002 report — Qwen/Qwen3-1.7B

Pool 8000; 60 scored starts; 12 steps; residue off; norms strength.SWOW-EN.R123.20180827.csv. Flip bands: α 0.5 [0.1071, 0.1748]; α 0.85 [0.3140, 0.4660]. Rules: HEAD2.md.

**G1 headroom (per-start choice beats one global setting, held out): FAIL**  -0.079 [-0.138, -0.018]; global cell chosen on first halves: `k1_a0.85`; best cell overall `k1_a0.85` 0.773; in-sample per-start best (biased, not gated) 0.853

**G2 corrected regulator (α 0.85): FAIL**  flip − k4 -0.012 [-0.046, +0.021]; flip − randk +0.005 [-0.046, +0.059]; flip − best cell (Bonferroni) -0.150 [-0.218, -0.085]

**G3 blending states beats choosing words: FAIL**  k1_a0.85 − discrete (Bonferroni) -0.010 [-0.048, +0.028]; native − best cell +0.007 [-0.044, +0.061]

## The grid (collapsed associate rate; raw rate; repeat fraction; distinct words)

| k | α 0.5 | α 0.85 |
|---:|---|---|
| 1 | 0.718; 0.690; 0.044; 0.942 | 0.773; 0.773; 0.000; 0.986 |
| 2 | 0.620; 0.541; 0.165; 0.843 | 0.686; 0.685; 0.001; 0.994 |
| 4 | 0.565; 0.444; 0.254; 0.761 | 0.635; 0.632; 0.006; 0.986 |
| 8 | 0.474; 0.359; 0.335; 0.692 | 0.570; 0.561; 0.019; 0.968 |
| 16 | 0.470; 0.352; 0.350; 0.661 | 0.506; 0.489; 0.040; 0.932 |
| 32 | 0.445; 0.321; 0.401; 0.635 | 0.448; 0.424; 0.062; 0.917 |

## All conditions

| condition | collapsed | raw | strength | repeat | distinct | longest | back-tracks | highway share | unique share | start trace | passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k1_a0.5 | 0.718 | 0.690 | 0.037 | 0.044 | 0.942 | 1.333 | 0.046 | 0.071 | 0.604 | 0.165 | 36.017 |
| k2_a0.5 | 0.620 | 0.541 | 0.033 | 0.165 | 0.843 | 2.150 | 0.192 | 0.142 | 0.567 | 0.180 | 36.000 |
| k4_a0.5 | 0.565 | 0.444 | 0.025 | 0.254 | 0.761 | 2.417 | 0.281 | 0.039 | 0.556 | 0.138 | 36.017 |
| k8_a0.5 | 0.474 | 0.359 | 0.023 | 0.335 | 0.692 | 2.800 | 0.354 | 0.149 | 0.472 | 0.173 | 36.017 |
| k16_a0.5 | 0.470 | 0.352 | 0.025 | 0.350 | 0.661 | 2.900 | 0.374 | 0.101 | 0.444 | 0.176 | 36.000 |
| k32_a0.5 | 0.445 | 0.321 | 0.021 | 0.401 | 0.635 | 3.183 | 0.421 | 0.250 | 0.401 | 0.195 | 36.000 |
| k1_a0.85 | 0.773 | 0.773 | 0.042 | 0.000 | 0.986 | 1.067 | 0.000 | 0.175 | 0.536 | 0.068 | 36.000 |
| k2_a0.85 | 0.686 | 0.685 | 0.037 | 0.001 | 0.994 | 1.050 | 0.001 | 0.167 | 0.640 | 0.080 | 36.000 |
| k4_a0.85 | 0.635 | 0.632 | 0.031 | 0.006 | 0.986 | 1.150 | 0.008 | 0.089 | 0.649 | 0.090 | 36.000 |
| k8_a0.85 | 0.570 | 0.561 | 0.027 | 0.019 | 0.968 | 1.300 | 0.028 | 0.101 | 0.631 | 0.110 | 36.017 |
| k16_a0.85 | 0.506 | 0.489 | 0.024 | 0.040 | 0.932 | 1.550 | 0.057 | 0.111 | 0.603 | 0.117 | 36.000 |
| k32_a0.85 | 0.448 | 0.424 | 0.020 | 0.062 | 0.917 | 1.700 | 0.083 | 0.167 | 0.599 | 0.106 | 36.017 |
| flip_a0.5 | 0.593 | 0.488 | 0.033 | 0.217 | 0.789 | 2.400 | 0.244 | 0.035 | 0.578 | 0.183 | 36.000 |
| flip_a0.85 | 0.623 | 0.618 | 0.033 | 0.011 | 0.971 | 1.283 | 0.024 | 0.085 | 0.644 | 0.115 | 36.000 |
| randk_a0.5 | 0.520 | 0.422 | 0.026 | 0.299 | 0.700 | 2.667 | 0.319 | 0.114 | 0.501 | 0.137 | 36.017 |
| randk_a0.85 | 0.618 | 0.607 | 0.032 | 0.028 | 0.956 | 1.417 | 0.039 | 0.074 | 0.644 | 0.062 | 36.000 |
| discrete | 0.783 | 0.783 | 0.043 | 0.000 | 0.978 | 1.100 | 0.018 | 0.268 | 0.506 | 0.088 | 36.000 |
| native | 0.780 | 0.780 | 0.043 | 0.000 | 0.988 | 1.067 | 0.000 | 0.117 | 0.597 | – | 24.350 |

Words shared by the most trails: k1_a0.85: light 9, shadow 9, disorder 7; k4_a0.5: state 6, moon 6, freedom 5; discrete: light 11, bright 9, dim 9; native: light 11, dark 11, bright 10

## Control surface (200 identical saved thoughts, one step each)

| k | move α 0.5 | next-proposal entropy α 0.5 | move α 0.85 | next-proposal entropy α 0.85 |
|---:|---:|---:|---:|---:|
| 1 | 0.237 | 1.726 | 0.495 | 1.786 |
| 2 | 0.184 | 1.784 | 0.431 | 1.810 |
| 4 | 0.152 | 1.901 | 0.384 | 1.857 |
| 8 | 0.131 | 1.964 | 0.351 | 1.913 |
| 16 | 0.118 | 2.010 | 0.328 | 1.967 |
| 32 | 0.110 | 2.035 | 0.313 | 1.992 |

## Trails

**river**  
- k1_a0.85: sea → sky → earth → water → wine → beer → glass → bottle → cap → hat → shirt → button
- k4_a0.5: river → river → land → sea → sand → ship → boat → sea → water → sea → moon → star
- k32_a0.85: sea → sand → ship → river → water → dish → kitchen → cup → flour → corn → bowl → cup
- flip_a0.85: sea → sky → earth → air → fire → ice → steam → heat → cold → hot → cool → war  (k: 4 2 1 1 1 1 1 1 2 4 2 2)
- discrete: sea → sky → earth → water → wine → cheese → butter → bread → knife → plate → glass → mirror
- native: ocean → sea → sky → earth → water → wine → vinegar → lemon → lime → sour → sweet → bitter

**bread**  
- k1_a0.85: butter → cheese → milk → cream → sugar → flour → bread → butter → cheese → milk → cream → sugar
- k4_a0.5: bread → jam → spread → plate → soup → cup → silver → knife → cutting → knife → plate → plate
- k32_a0.85: butter → spread → place → place → section → pen → pad → letter → stick → board → tile → poker
- flip_a0.85: butter → cream → sugar → tea → water → wine → beer → pub → bar → restaurant → table → plate  (k: 4 4 4 4 2 1 1 1 2 4 8 4)
- discrete: butter → bread → cheese → milk → cream → sugar → flour → water → wine → cheese → butter → bread
- native: butter → cream → cheese → milk → water → wine → vinegar → lemon → lime → sour → sweet → bitter

**winter**  
- k1_a0.85: spring → summer → fall → winter → snow → ice → water → wine → beer → glass → bottle → cap
- k4_a0.5: spring → spring → storm → cloud → sky → earth → air → land → sun → mountain → moon → sky
- k32_a0.85: spring → winter → hour → minute → second → first → first → after → under → via → road → house
- flip_a0.85: spring → season → day → night → morning → afternoon → even → odd → event → face → eye → soul  (k: 4 4 4 4 4 4 2 4 8 16 16 8)
- discrete: spring → summer → winter → snow → ice → water → wine → cheese → butter → bread → knife → plate
- native: spring → summer → autumn → harvest → crop → field → grass → plant → animal → vegetable → fruit → juice

**iron**  
- k1_a0.85: coal → fire → water → wine → beer → glass → bottle → cap → hat → shirt → button → pocket
- k4_a0.5: iron → fire → light → shadow → moon → star → planet → planet → universe → universe → force → matter
- k32_a0.85: coal → light → shadow → form → place → place → region → room → chair → window → doors → windows
- flip_a0.85: coal → water → mirror → window → door → key → lock → chain → link → ring → finger → hand  (k: 4 4 4 4 2 2 2 1 1 1 2 2)
- discrete: coal → fire → water → wine → cheese → butter → bread → knife → plate → glass → mirror → eye
- native: steel → fire → water → wine → vinegar → lemon → lime → sour → sweet → bitter → spicy → hot

**music**  
- k1_a0.85: dance → song → story → chapter → volume → sound → silence → noise → peace → war → battle → victory
- k4_a0.5: music → song → song → language → speech → paragraph → page → book → paper → brush → stroke → word
- k32_a0.85: dance → motion → audience → screen → television → computer → mouse → keyboard → track → road → train → journey
- flip_a0.85: dance → song → word → sentence → paragraph → definition → explanation → understanding → knowledge → power → authority → law  (k: 4 4 2 2 2 2 1 1 1 1 2 4)
- discrete: dance → music → sound → silence → noise → peace → war → battle → victory → defeat → loss → gain
- native: dance → rhythm → meter → measure → weight → balance → scale → fish → water → wine → vinegar → lemon

**vent**  
- k1_a0.85: door → window → glass → mirror → eye → ear → nose → mouth → teeth → jaw → tear → eye
- k4_a0.5: vent → open → close → opened → closed → left → wrong → correct → incorrect → correct → invalid → valid
- k32_a0.85: door → frame → board → paper → folder → white → blue → sky → land → world → state → region
- flip_a0.85: door → frame → mirror → glass → bottle → wine → beer → pub → bar → restaurant → table → plate  (k: 4 8 4 2 2 1 1 1 2 4 8 4)
- discrete: door → window → glass → mirror → eye → ear → nose → mouth → teeth → jaw → tear → eye
- native: breath → lung → heart → soul → spirit → shadow → light → dark → bright → dim → dull → vivid

**buck**  
- k1_a0.85: horse → sword → shield → weapon → armor → chain → link → ring → finger → hand → arm → leg
- k4_a0.5: buck → horn → bow → sword → sword → bullet → chain → key → sword → dice → dice → chip
- k32_a0.85: horse → horse → chair → track → table → traffic → socket → screw → nut → shell → sea → sky
- flip_a0.85: horse → sword → shield → knife → plate → bowl → pot → cond → water → beer → coffee → sugar  (k: 4 8 8 4 4 4 2 1 2 4 2 1)
- discrete: horse → cart → wheel → spoke → hub → tail → head → heart → soul → spirit → shadow → light
- native: antelope → zebra → horse → carriage → wagon → cart → wheel → axle → hub → sphere → circle → square

**sister**  
- k1_a0.85: brother → father → son → daughter → mother → child → parent → baby → milk → cream → sugar → flour
- k4_a0.5: sister → father → son → dad → mom → pop → tom → brother → father → son → dad → mom
- k32_a0.85: brother → brother → country → territory → sea → sea → stream → current → past → clock → watch → chain
- flip_a0.85: brother → father → son → daughter → husband → wife → partner → friend → enemy → host → guest → server  (k: 4 2 1 2 2 2 2 2 2 2 4 4)
- discrete: brother → sister → mother → father → son → daughter → husband → wife → child → parent → children → parents
- native: brother → father → mother → child → parent → pupil → teacher → student → professor → assistant → secretary → president

**gentle**  
- k1_a0.85: soft → hard → easy → difficult → simple → complex → component → part → whole → half → full → empty
- k4_a0.5: gentle → strong → weak → strong → free → power → power → right → freedom → rule → country → territory
- k32_a0.85: soft → tight → sad → empty → filled → unf → close → open → window → light → dark → light
- flip_a0.85: soft → tough → weak → stronger → more → fewer → many → one → five → month → sun → shadow  (k: 4 4 4 4 8 16 8 16 16 8 16 16)
- discrete: soft → hard → easy → difficult → simple → complex → component → part → whole → half → full → empty
- native: harsh → soft → hard → easy → difficult → simple → complex → compound → element → atom → molecule → ion


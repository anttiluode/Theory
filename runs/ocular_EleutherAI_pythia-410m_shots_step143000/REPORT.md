# Ocular report — EleutherAI/pythia-410m@step143000

124 words, 20 families; generic pool 8000 words; selection layer 12; clamp band [-1, 10]. Rules: OCULAR.md.

**O1 inductive ocular repair: FAIL**  
band4_vocab − rand16_vocab: +0.065 [-0.008, +0.137]  
band4_vocab − native: +0.008 [-0.056, +0.073]  
band4_vocab − random3_vocab: +0.081 [+0.024, +0.137]

**O2 band beats full width at choosing (inductive): FAIL**  band4_vocab − full_vocab: -0.024 [-0.089, +0.032]

**O3 band4_bank − iris:** -0.089 [-0.145, -0.040]

| condition | category | identity | rescued | harmed | neighbour coherence | true-family neighbours |
|---|---:|---:|---:|---:|---:|---:|
| native | 0.339 | 0.815 | 0 | 0 | – | – |
| iris | 0.452 | 0.637 | 15 | 1 | 0.526 | 0.866 |
| full_bank | 0.452 | 0.629 | 15 | 1 | 0.534 | 0.855 |
| pre16_bank | 0.339 | 0.645 | 8 | 8 | 0.467 | 0.605 |
| band4_bank | 0.363 | 0.669 | 8 | 5 | 0.452 | 0.621 |
| band8_bank | 0.331 | 0.637 | 7 | 8 | 0.405 | 0.535 |
| rand16_bank | 0.355 | 0.677 | 11 | 9 | 0.377 | 0.411 |
| full_vocab | 0.371 | 0.758 | 9 | 5 | 0.495 | – |
| pre16_vocab | 0.298 | 0.758 | 5 | 10 | 0.428 | – |
| band4_vocab (primary) | 0.347 | 0.774 | 9 | 8 | 0.331 | – |
| band8_vocab | 0.266 | 0.758 | 5 | 14 | 0.245 | – |
| rand16_vocab | 0.282 | 0.742 | 3 | 10 | 0.165 | – |
| random3_vocab | 0.266 | 0.734 | 2 | 11 | -0.006 | – |

Neighbours chosen for a few words (vocab pool):

| word | full | pre16 | band4 | band8 | rand16 |
|---|---|---|---|---|---|
| oak | wood, tree, Wood | liver, hip, breast | breast, elder, leather | Barn, appell, base | yard, heaven, football |
| salmon | chicken, Turkey, rice | kidney, tobacco, liver | Turkey, carcinoma, Muslim | Turkey, sauce, paste | baby, guidance, tiny |
| shovel | knife, shaft, sword | pipe, needle, dish | probe, screw, needle | specify, screw, Indeed | diag, beam, comb |
| violin | instrument, viol, sword | mouse, telephone, magazine | mouse, singer, magazine | mouse, mention, pointer | loyal, diver, bridge |
| zinc | calcium, sodium, silicon | silicon, sodium, calcium | silicon, sodium, kinase | sodium, fibers, yeast | targets, dig, coal |
| tennis | basketball, football, baseball | basketball, tobacco, football | trying, tob, television | trying, twe, waiting | sleeping, drain, telephone |
| stool | seat, bowl, stick | pipe, bowl, dish | fan, ship, stick | sole, sheet, shell | seat, sides, bath |
| anger | angry, anxiety, emotion | happiness, illness, injury | anxiety, happiness, injury | emotions, angry, Ast | obsc, price, Price |
| chef | cook, artist, architect | football, physician, beer | bread, cook, beer | cook, pick, chip | cheese, cooking, ways |
| Japan | Korea, Russia, India | Korea, America, India | Korea, Ireland, Britain | Korea, Asia, Australia | Korea, Jews, mining |
| dollar | dollars, money, cash | bike, newspaper, quarter | fif, forty, cash | dollars, integer, bills | newspaper, hour, utility |

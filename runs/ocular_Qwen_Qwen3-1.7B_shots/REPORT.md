# Ocular report — Qwen/Qwen3-1.7B

138 words, 20 families; generic pool 8000 words; selection layer 14; clamp band [-1, 12]; dtype float32. Rules: OCULAR.md.

**O1 inductive ocular repair: FAIL**  
band4_vocab − rand16_vocab: -0.007 [-0.058, +0.043]  
band4_vocab − native: -0.065 [-0.116, -0.014]  
band4_vocab − random3_vocab: +0.014 [-0.036, +0.065]

**O2 band beats full width at choosing (inductive): FAIL**  band4_vocab − full_vocab: -0.036 [-0.072, +0.000]

**O3 band4_bank − iris:** -0.043 [-0.094, +0.000]

**Q1 full-width inductive repair (amendment 1): FAIL**  
full_vocab − native: -0.029 [-0.072, +0.014]  
full_vocab − random3_vocab: +0.051 [+0.000, +0.109]  
identity change: +0.000 [+0.000, +0.000]

| condition | category | identity | rescued | harmed | neighbour coherence | true-family neighbours |
|---|---:|---:|---:|---:|---:|---:|
| native | 0.964 | 1.000 | 0 | 0 | – | – |
| iris | 0.949 | 1.000 | 2 | 4 | 0.537 | 0.915 |
| full_bank | 0.964 | 1.000 | 3 | 3 | 0.546 | 0.920 |
| pre16_bank | 0.928 | 1.000 | 2 | 7 | 0.482 | 0.688 |
| band4_bank | 0.906 | 1.000 | 1 | 9 | 0.470 | 0.662 |
| band8_bank | 0.928 | 1.000 | 1 | 6 | 0.437 | 0.577 |
| rand16_bank | 0.877 | 1.000 | 1 | 13 | 0.347 | 0.435 |
| full_vocab | 0.935 | 1.000 | 2 | 6 | 0.445 | – |
| pre16_vocab | 0.906 | 1.000 | 2 | 10 | 0.408 | – |
| band4_vocab (primary) | 0.899 | 1.000 | 2 | 11 | 0.333 | – |
| band8_vocab | 0.906 | 1.000 | 2 | 10 | 0.222 | – |
| rand16_vocab | 0.906 | 1.000 | 0 | 8 | 0.205 | – |
| random3_vocab | 0.884 | 1.000 | 0 | 11 | -0.005 | – |

Neighbours chosen for a few words (vocab pool):

| word | full | pre16 | band4 | band8 | rand16 |
|---|---|---|---|---|---|
| oak | wood, log, tree | corn, coal, bass | corn, Atlantic, brown | corn, coal, Kansas | Earth, sett, ast |
| eagle | bird, bear, bat | chicken, cream, bull | apparently, chicken, Antonio | chicken, diamond, comb | tie, sky, mountain |
| bee | bug, fly, bird | egg, fly, cake | cream, chicken, cannabis | Define, chicken, cannabis | birds, bread, egg |
| bus | ship, box, boat | bike, ship, sink | unp, Customer, phone | customers, restaurants, cars | bes, dim, inline |
| trumpet | sword, triangle, pipe | poker, pickle, crusher | torch, brief, sword | Scanner, torch, Turkey | stamp, beer, alarm |
| zinc | rice, voltage, salt | laser, salt, acid | nonatomic, Intel, sqrt | Intel, memset, meth | carbon, energ, enc |
| tennis | basketball, baseball, football | baseball, pizza, coffee | Sony, programming, marketing | these, those, shopping | desper, ten, pers |
| stool | seat, jug, bowl | bowl, basket, bottle | Controller, bowl, recipient | bott, overall, bowl | lua, stuff, stat |
| anger | angry, anxiety, violence | danger, anxiety, failure | anxiety, awareness, belief | beliefs, anxiety, emotions | attorney, strength, forever |
| chef | cook, artist, builder | actor, reporter, scholar | actor, polic, medic | actor, cook, artist | norm, Sir, professor |
| Japan | Russia, Korea, India | India, England, Australia | England, India, Israel | Egypt, India, England | Korea, India, Over |
| dollar | dollars, money, currency | hotel, basketball, wallet | Monday, Intel, intel | liter, cul, Database | vict, vor, inv |

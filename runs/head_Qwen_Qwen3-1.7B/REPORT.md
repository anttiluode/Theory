# Head report — Qwen/Qwen3-1.7B

Pool 8000 words; selection layer 14; clamp band [-1, 12]; dtype float32; 12 steps; regulator bands (fixed-walk quartiles on 20 calibration words): static [0.0415, 0.0675]; loop [0.1050, 0.1912]. Norms: strength.SWOW-EN.R123.20180827.csv. Rules: HEAD.md (with amendment 1).

The pool states (naming and association frames) are a one-time memory shared by every walk, fixed_avg and nearest_min; they are not charged per item. A loop step costs one forward pass (Qwen proposes) plus the voice.

**P0 voice: PASS**  copy 1.000 (≥ 0.90), blend → one of the two 0.925 (≥ 0.50)

**F0 sanity: PASS**  head − random associate rate: +0.141 [+0.103, +0.183]

**F1 regulator earns its keep (free): FAIL**  head − fixed: -0.037 [-0.081, +0.005]; distinct words head 0.644 vs fixed 0.658  
head − nn: -0.235 [-0.349, -0.118]; head − native (reference): -0.627 [-0.677, -0.574]

**LF loop regulator earns its keep (free): FAIL**  loop − loop_fixed: -0.048 [-0.092, -0.005]  
loop − head +0.273 [+0.212, +0.335]; loop − loop_nores +0.001 [-0.051, +0.050]; loop − native (reference) -0.354 [-0.407, -0.300]

**H1 walk beats the lookup (goal): FAIL**  
head − walk_fixed +0.000 [+0.000, +0.000]  
head − fixed_avg -0.008 [-0.038, +0.023]  
head − nearest_min +0.000 [-0.023, +0.023]

**H2 walk beats the model's own tokens: FAIL**  
head − native -0.031 [-0.077, +0.008]  
head − reason (equal passes) -0.031 [-0.077, +0.008]

### Amendment 1: the closed-loop engine

**L1 Qwen-driven moves beat the static map: FAIL**  loop − head -0.008 [-0.031, +0.015]

**L2 loop beats the lookup: FAIL**  loop − fixed_avg -0.015 [-0.046, +0.015]; loop − nearest_min -0.008 [-0.031, +0.015]

**L3 breathing earns its keep (goal): FAIL**  loop − loop_fixed +0.008 [+0.000, +0.023]; loop − loop_nores (reported) +0.000 [+0.000, +0.000]

**L4 loop beats the model's own tokens: FAIL**  loop − native -0.038 [-0.077, +0.000]; loop − reason_loop (equal passes) -0.038 [-0.077, -0.008]

## Free mode

| condition | associate rate | strength | distinct words | longest repeat | back-tracks | passes/trail |
|---|---:|---:|---:|---:|---:|---:|
| head | 0.154 | 0.003 | 0.644 | 3.217 | 0.374 | 24.017 |
| fixed | 0.191 | 0.004 | 0.658 | 2.950 | 0.374 | 24.000 |
| nn | 0.388 | 0.015 | 0.161 | 7.900 | 0.883 | 24.000 |
| loop | 0.427 | 0.021 | 0.707 | 2.650 | 0.338 | 36.000 |
| loop_fixed | 0.475 | 0.024 | 0.733 | 2.600 | 0.293 | 36.017 |
| loop_nores | 0.426 | 0.021 | 0.715 | 2.817 | 0.328 | 36.000 |
| random | 0.012 | 0.000 | 1.000 | 1.000 | 0.001 | 24.000 |
| native | 0.780 | 0.046 | 0.982 | 1.117 | 0.000 | 24.550 |

Trails (demo words first, then the first scored starts):

**river**  
- head: river → river → river → road → road → cup → basket → pen → plate → plate → plate → table  (k: 4 8 16 32 16 32 32 16 8 16 32 32)
- fixed: river → river → river → street → town → garden → field → floor → road → road → road → town
- nn: river → river → river → river → river → river → river → river → river → river → river → river
- loop: river → river → sea → sea → sand → mirror → vision → vision → vision → mind → mind → shadow  (k: 4 8 16 32 32 32 32 32 32 32 32 32)
- loop_fixed: river → river → land → sea → sand → stone → boat → sea → water → sea → moon → star
- loop_nores: river → river → sea → sea → sand → water → eye → eye → eye → sun → light → shadow
- random: physical → image → safety → views → engineering → gone → free → makes → casino → corpor → tact → event
- native: ocean → sea → sky → earth → water → wine → vinegar → lemon → lime → sour → sweet → bitter

**bread**  
- head: bread → bread → bread → cup → cup → chair → rod → seat → plate → bowl → boat → bench  (k: 4 4 8 16 32 32 32 32 32 16 8 8)
- fixed: bread → bread → cup → bowl → bowl → bowl → bench → bed → bed → box → ball → cup
- nn: cake → pie → cake → pie → cake → pie → cake → pie → cake → pie → cake → pie
- loop: bread → jam → spread → plate → plate → silver → tea → tea → milk → sugar → corn → salt  (k: 4 4 4 8 8 16 16 16 32 32 32 32)
- loop_fixed: bread → jam → spread → plate → soup → cup → silver → knife → cutting → knife → plate → plate
- loop_nores: bread → jam → spread → plate → plate → silver → tea → tea → milk → sugar → corn → salt
- random: image → browser → xbox → aspects → medical → entity → writing → crucial → fold → von → dan → showed
- native: butter → cream → cheese → milk → water → wine → vinegar → lemon → lime → sour → sweet → bitter

**winter**  
- head: winter → winter → winter → winter → state → country → european → european → european → european → state → illinois  (k: 4 8 16 32 32 32 16 8 16 32 32 16)
- fixed: winter → winter → january → june → june → wednesday → weekend → winter → winter → july → june → june
- nn: winter → winter → winter → winter → winter → winter → winter → winter → winter → winter → winter → winter
- loop: spring → spring → winter → mist → mist → sight → lens → frame → page → picture → light → dark  (k: 4 8 16 32 32 32 32 32 32 32 32 32)
- loop_fixed: spring → spring → storm → cloud → sky → earth → earth → light → sun → day → night → dark
- loop_nores: spring → spring → winter → night → night → sight → sight → object → concept → concept → concept → vision
- random: opponents → pictures → download → not → affordable → niger → liqu → write → snapshot → redu → assistant → unusual
- native: spring → summer → autumn → harvest → crop → field → grass → plant → animal → vegetable → fruit → juice

**iron**  
- head: iron → pin → plate → glass → bowl → bottle → bowl → dish → pot → plate → glass → bowl  (k: 4 4 2 1 1 1 1 1 1 1 1 1)
- fixed: iron → pin → pen → lock → clock → bowl → bucket → basket → bench → chair → cup → cup
- nn: iron → iron → iron → iron → iron → iron → iron → iron → iron → iron → iron → iron
- loop: iron → fire → light → shadow → moon → sun → night → day → evening → light → dark → moon  (k: 4 4 4 4 2 1 1 1 1 1 1 1)
- loop_fixed: iron → fire → light → shadow → moon → star → planet → planet → planet → disk → circle → sun
- loop_nores: iron → fire → light → shadow → moon → sun → night → day → evening → light → dark → moon
- random: fait → direction → teams → binding → may → usage → slip → integer → weight → return → bis → logo
- native: steel → fire → water → wine → vinegar → lemon → lime → sour → sweet → bitter → spicy → hot

**music**  
- head: music → music → movie → movies → picture → image → photo → photos → images → videos → videos → videos  (k: 4 4 8 8 4 2 1 1 1 1 1 2)
- fixed: music → music → movie → film → photos → photo → photos → image → movies → movie → film → film
- nn: music → music → music → music → music → music → music → music → music → music → music → music
- loop: music → music → sound → sound → speech → script → paper → paper → paper → word → letter → letter  (k: 4 4 8 16 32 32 32 32 32 32 16 8)
- loop_fixed: music → music → speech → word → sentence → paragraph → paragraph → article → section → part → part → piece
- loop_nores: music → music → song → language → speech → speech → record → film → lens → light → shadow → shadow
- random: cor → federal → spread → constructor → radius → niet → organ → key → hus → criminal → responsible → regime
- native: dance → rhythm → meter → measure → weight → balance → scale → fish → water → wine → vinegar → lemon

**procedure**  
- head: procedure → procedure → procedure → procedure → regulation → promotion → enforcement → enforcement → permission → regulation → procedure → regulation  (k: 4 8 16 32 32 32 32 32 32 32 32 32)
- fixed: procedure → procedure → procedure → strategies → regulations → regulation → restriction → constraint → restriction → requirement → contribution → improvement
- nn: procedures → procedure → procedures → procedure → procedures → procedure → procedures → procedure → procedures → procedure → procedures → procedure
- loop: procedure → process → process → rule → rule → government → government → rule → rule → law → order → order  (k: 4 8 16 32 32 32 32 32 16 32 16 8)
- loop_fixed: procedure → process → effect → cause → force → power → threat → risk → gain → lose → find → take
- loop_nores: procedure → process → process → process → process → journey → journey → journey → place → time → clock → table
- random: greater → indian → greg → redis → defensive → carrier → actors → lying → saying → dar → modern → superv
- native: process → product → cost → price → value → worth → use → abuse → misuse → mislead → deceive → cheat

**associate**  
- head: associate → associate → association → responsibility → recognition → promotion → recognition → regulation → recognition → permission → responsibility → reputation  (k: 4 8 16 16 8 16 32 32 32 32 32 32)
- fixed: associate → associate → association → association → responsibility → responsibility → recognition → restriction → restriction → improvements → participation → participation
- nn: association → association → association → association → association → association → association → association → association → association → association → association
- loop: associate → associate → partner → partner → party → assembly → assembly → system → system → problem → authority → power  (k: 4 8 16 32 32 32 16 32 32 32 32 16)
- loop_fixed: associate → associate → friend → enemy → com → opposition → nation → state → territory → territory → territory → town
- loop_nores: associate → associate → partner → partner → member → kingdom → province → region → town → place → country → state
- random: your → awk → package → narrow → hack → plus → five → days → importance → altern → hands → award
- native: disassociate → dissociate → separate → part → whole → half → full → empty → filled → emptied → fill → void

**tank**  
- head: tank → tank → cup → cup → bat → bar → chair → tank → tower → plate → plate → bowl  (k: 4 4 8 8 16 32 32 32 32 32 16 32)
- fixed: tank → tank → cup → cup → bath → desk → bed → cup → cup → bat → cup → ball
- nn: tub → tube → pipe → tube → pipe → tube → pipe → tube → pipe → tube → pipe → tube
- loop: tank → ship → sea → sea → lake → low → long → path → path → rail → line → line  (k: 4 8 16 32 32 32 32 32 32 32 32 32)
- loop_fixed: tank → ship → sea → lake → sea → moon → sun → planet → sun → sun → day → day
- loop_nores: tank → ship → ship → sea → sea → sea → floor → road → door → door → bridge → tower
- random: studies → child → dur → register → constant → website → stone → user → talking → roads → error → sent
- native: barrel → cask → bottle → glass → mirror → eye → ear → mouth → tooth → tongue → teeth → gums

**gamma**  
- head: gamma → delta → meta → scalar → tensor → vector → vector → vector → vector → theta → theta → axis  (k: 4 4 4 2 1 1 1 2 4 8 4 8)
- fixed: gamma → delta → meta → meta → tensor → variable → variable → iterator → iterator → annotation → metadata → descriptor
- nn: sigma → theta → sigma → theta → sigma → theta → sigma → theta → sigma → theta → sigma → theta
- loop: delta → theta → theta → theta → point → line → line → row → row → river → sea → star  (k: 4 8 16 32 32 32 16 32 32 32 32 32)
- loop_fixed: delta → theta → eps → ele → digit → letter → word → money → weight → balance → counter → chair
- loop_nores: delta → theta → theta → theta → point → line → line → row → row → rail → land → sea
- random: stmt → timing → particles → apolog → temple → interview → foreign → reduced → award → craft → copied → moment
- native: delta → epsilon → omega → alpha → beta → omic → gene → protein → amino → acid → base → top

**atom**  
- head: atom → element → axis → vertex → vertices → vector → vector → vector → vector → vector → scalar → element  (k: 4 4 2 1 1 1 1 2 4 8 16 32)
- fixed: atom → element → element → element → token → sensor → selector → processor → processor → router → renderer → reader
- nn: planet → universe → university → university → university → university → university → university → university → university → university → university
- loop: atom → planet → moon → night → day → day → morning → day → sun → moon → moon → sun  (k: 4 4 4 4 4 8 16 32 32 16 16 32)
- loop_fixed: atom → planet → moon → night → day → day → morning → afternoon → sun → moon → night → day
- loop_nores: atom → planet → moon → night → day → day → night → day → sun → moon → moon → day
- random: maintained → dur → exam → het → bulk → photo → force → laptop → cit → python → enforcement → uma
- native: molecule → ion → suffix → prefix → suffixed → prefixed → appended → added → removed → taken → left → right

**doctor**  
- head: doctor → doctor → doctor → doctor → director → supporters → colleagues → people → organizations → institutions → systems → regulations  (k: 4 8 16 32 32 32 16 8 8 4 4 4)
- fixed: doctor → doctor → professor → professor → editor → publisher → provider → director → professor → director → officer → publisher
- nn: doctors → doctor → doctors → doctor → doctors → doctor → doctors → doctor → doctors → doctor → doctors → doctor
- loop: doctor → doctor → doctor → care → comfort → comfort → comfort → stress → sense → law → order → authority  (k: 4 4 8 16 16 16 32 32 16 8 4 2)
- loop_fixed: doctor → doctor → doctor → pain → suffering → relief → relief → freedom → freedom → victory → failure → failure
- loop_nores: doctor → doctor → doctor → disease → pain → pain → right → loss → get → possess → obtain → remove
- random: improvement → stream → fulfill → ran → activity → fore → campus → anim → despite → genetic → cell → mehr
- native: patient → nurse → assistant → secretary → president → vice → justice → law → order → chaos → disorder → peace

## Goal mode: Remote Associates

130 items kept (all three cues single tokens), 14 dropped; solution in pool for 117/130.

| condition | correct | accuracy | mean passes |
|---|---:|---:|---:|
| head | 2/130 | 0.015 | 24.4 |
| walk_fixed | 2/130 | 0.015 | 24.8 |
| loop | 1/130 | 0.008 | 36.5 |
| loop_fixed | 0/130 | 0.000 | 36.3 |
| loop_nores | 1/130 | 0.008 | 36.5 |
| fixed_avg | 3/130 | 0.023 | 2.0 |
| nearest_min | 2/130 | 0.015 | 0.0 |
| native | 6/130 | 0.046 | 2.0 |
| reason | 6/130 | 0.046 | 11.6 |
| reason_loop | 6/130 | 0.046 | 11.6 |

Solution spoken anywhere in the trail / as the first spoken word (descriptive): head 5/2; walk_fixed 5/2; loop 3/2; loop_fixed 7/2; loop_nores 4/2 (of 130).

| cues | solution | head | walk_fixed | loop | loop_fixed | loop_nores | fixed_avg | nearest_min | native | reason | reason_loop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cream, skate, water | ice | cake | bath | bottle | plate | plate | cup | **ice** | **ice** | skateboard | skateboard |
| loser, throat, spot | sore | nose | horse | soul | soul | soul | spoon | ghost | **sore** | loss | loss |
| show, life, row | boat | move | work | sea | sea | sea | rose | view | river | live | live |
| night, wrist, stop | watch | head | head | moon | weight | water | time | head | midnight | stopping | stopping |
| duck, fold, dollar | bill | doll | doll | coin | pocket | coin | doll | door | **bill** | pond | pond |
| rocking, wheel, high | chair | throwing | hanging | track | track | bar | **chair** | flip | car | wheels | wheels |
| dew, comb, bee | honey | bow | bow | cow | cow | cow | deb | egg | moisture | combs | combs |
| fountain, baking, pop | soda | pan | bed | dinner | salt | milk | popping | sand | **soda** | fountains | fountains |
| preserve, ranger, tropical | forest | president | representative | plant | stream | plant | ranch | president | **forest** | the | the |
| aid, rubber, wagon | band | bag | bag | light | lens | wheel | wand | bag | cart | wagons | wagons |
| safety, cushion, point | pin | support | support | glass | sink | money | pain | support | shelter | cushions | cushions |
| cane, daddy, plum | sugar | tap | bat | sky | sun | sky | pumpkin | pussy | fruit | dad | dad |
| dream, break, light | day | walk | look | night | night | dark | drink | breath | shadow | dreams | dreams |
| fish, mine, rush | gold | farm | horse | moon | moon | moon | miss | moon | river | none | none |
| political, surprise, line | party | problem | problem | election | state | state | pressure | influence | election | lines | lines |
| measure, worm, video | tape | wire | wire | movie | movie | movie | word | wire | length | worms | worms |
| high, district, house | school/court | room | lake | city | city | city | home | north | building | zone | zone |
| sense, courtesy, place | common | preferences | situation | grace | peace | peace | space | notice | space | smell | smell |
| worm, shelf, end | book | hole | hole | lake | wall | wall | chair | hole | tail | ends | ends |
| piece, mind, dating | game | face | time | stomach | spirit | forth | date | dance | brain | date | date |

Trails for the first 5 items (head, then loop):

- head: cream, skate, water → cup → coffee → wine → sauce → soup → rice → cake → pie → tea → wine → milk → oil  (k: 4 2 1 1 1 1 1 1 1 1 1 1)
- loop: cream, skate, water → cup → wine → cup → cup → cup → plate → plate → glass → lens → bottle → flask → light  (k: 4 8 16 32 32 32 32 32 32 32 32 32)
- head: loser, throat, spot → slot → hole → lip → hip → knee → jaw → eye → eyes → nose → mouth → lip → knee  (k: 4 2 1 1 1 1 1 1 1 1 1 1)
- loop: loser, throat, spot → heart → soul → mind → brain → blood → organ → body → heart → soul → mind → matter → energy  (k: 4 4 4 4 4 8 8 16 16 8 16 8)
- head: show, life, row → work → walk → walk → read → write → take → make → move → leave → have → come → come  (k: 4 4 2 4 4 4 2 1 1 1 1 1)
- loop: show, life, row → sea → water → sea → moon → star → star → sun → round → square → plate → card → die  (k: 4 4 4 8 16 32 32 32 32 32 32 32)
- head: night, wrist, stop → head → head → foot → ear → tail → wing → wheel → wire → tube → pipe → tub → tank  (k: 4 2 4 2 1 1 1 1 1 1 1 1)
- loop: night, wrist, stop → day → day → sun → month → year → summer → fall → weather → moon → sun → sun → sun  (k: 4 2 4 4 4 4 4 4 8 8 8 16)
- head: duck, fold, dollar → doll → desk → house → floor → hall → wall → wall → wall → wall → wall → wall → cup  (k: 4 2 1 1 1 1 1 2 4 8 16 32)
- loop: duck, fold, dollar → coin → bank → note → book → page → pen → paper → brush → brush → brush → book → page  (k: 4 2 1 1 1 1 1 2 2 4 8 16)

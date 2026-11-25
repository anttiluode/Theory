# Head experiment 003 report — Qwen/Qwen3-1.7B

144 items; 16477 candidate words (single tokens in the SWOW vocabulary); solution is a candidate for 144/144; base rate from 200 words; 924 forward passes. Norms strength.SWOW-EN.R123.20180827.csv. Rules: HEAD3.md.

**G1 intersection beats the prompt: PASS**  prod_pmi − native +0.069 [+0.028, +0.118]

**G2 intersection beats union: PASS**  prod_pmi − mix +0.104 [+0.056, +0.160]

| condition | correct | accuracy | solution in top 10 | median rank of solution (when a candidate) |
|---|---:|---:|---:|---:|
| native | 7/144 | 0.049 | – | – |
| prod_raw | 4/144 | 0.028 | 34/144 | 65 |
| prod_l1 | 13/144 | 0.090 | 41/144 | 42 |
| prod_pmi | 17/144 | 0.118 | 37/144 | 89 |
| mix | 2/144 | 0.014 | 13/144 | 548 |

Per cue (λ = 0, one cue alone), rank of the solution: median of the best cue 37, of the worst cue 990; solution in the best cue's top 10 for 33/144 items, in all three cues' top 100 for 10/144.

Human 15-second solve rate vs the solution's rank under prod_pmi: Spearman +0.367 over 144 items (positive = the model finds easy what people find easy).

| cues | solution | native | prod_raw | prod_pmi | mix | prod_pmi top 10 |
|---|---|---|---|---|---|---|
| cottage, swiss, cake | cheese | **cheese** | **cheese** | **cheese** | icing | cheese, dairy, chocolate, bakery, baker, cream, farmhouse, fond, butter, farm |
| cream, skate, water | ice | **ice** | soft | slippery | scooter | slippery, icy, ice, lotion, skim, soft, thin, cold, smooth, cool |
| loser, throat, spot | sore | **sore** | pain | **sore** | winner | sore, pain, scratch, winner, someone, dog, bump, person, scar, cough |
| show, life, row | boat | river | light | **boat** | **boat** | boat, live, stream, swim, sail, motion, wake, star, beat, moving |
| night, wrist, stop | watch | midnight | time | **watch** | bracelet | watch, alarm, clock, sleep, signal, blink, quiet, rest, end, time |
| duck, fold, dollar | bill | **bill** | paper | **bill** | unfold | bill, coin, wing, crane, feather, flat, loose, paper, make, crisp |
| rocking, wheel, high | chair | car | low | low | axle | low, horse, tire, lift, merry, chair, bicycle, carriage, moving, motion |
| dew, comb, bee | honey | moisture | insect | insect | hive | insect, honey, hair, brush, hive, hum, pollen, rain, fly, sting |
| fountain, baking, pop | soda | **soda** | water | **soda** | **soda** | soda, spray, water, spring, cup, cool, stream, bubble, splash, pot |
| preserve, ranger, tropical | forest | **forest** | doctor | **forest** | beaches | forest, fish, wildlife, woods, animal, nature, fungi, arch, doctor, hunter |
| aid, rubber, wagon | band | cart | medical | truck | elastic | truck, cart, shoe, goods, tire, support, tool, tread, stretch, axle |
| flake, mobile, cone | snow | ice | part | tip | phone | tip, part, horn, hair, device, piece, small, straw, ear, plastic |
| cracker, fly, fighter | fire | airplane | food | insect | insect | insect, biting, bug, bite, pest, mouth, crust, sandwich, bit, animal |
| safety, cushion, point | pin | shelter | needle | needle | precautions | needle, sharp, seat, tip, prick, pin, scissors, pad, thread, touch |
| cane, daddy, plum | sugar | fruit | fruit | fruit | father | fruit, pit, apple, son, big, father, long, uncle, parent, apr |
| dream, break, light | day | shadow | night | night | nightmare | night, vision, morning, sleep, shine, daylight, shadow, wake, moon, wish |
| fish, mine, rush | gold | river | water | river | freshwater | river, swamp, gravel, aquatic, freshwater, pond, water, wild, salt, sea |
| political, surprise, line | party | election | **party** | **party** | populist | party, left, shock, radical, dash, unexpected, bold, thought, stroke, scare |
| measure, worm, video | tape | length | small | **tape** | playback | tape, digit, rod, finger, small, record, pin, bug, digital, reel |
| high, district, house | school/court | building | hospital | town | county | town, building, city, apartment, flat, home, county, place, village, hospital |
| sense, courtesy, place | common | space | feeling | touch | perception | touch, perception, attention, feeling, important, per, name, heart, kind, patience |
| worm, shelf, end | book | tail | needle | insect | insect | insect, pin, rod, flat, nail, cater, table, finger, staple, inch |
| piece, mind, dating | game | brain | heart | heart | matchmaking | heart, puzzle, whole, rom, single, thought, eye, connection, love, split |
| flower, friend, scout | girl | plant | companion | pet | bloom | pet, companion, soldier, lover, helper, bud, spy, leaf, bot, follower |
| river, note, account | bank | ledger | water | bill | flowing | bill, stream, current, source, sheet, bank, record, flow, ledger, book |
| print, berry, bird | blue | fruit | fruit | fruit | wing | fruit, plum, black, produce, small, reproduce, rip, purple, peel, cyan |
| pie, luck, belly | pot | stomach | good | peach | abdominal | peach, filling, fat, cherry, hard, fill, thin, pregnant, slice, good |
| date, alley, fold | blind | sheet | flat | narrow | unfold | narrow, flat, square, open, tear, rip, close, brick, pass, street |
| opera, hand, dish | soap | plate | fine | finger | fingers | finger, fine, dinner, silver, spoon, fingers, cut, fork, plate, use |
| cadet, capsule, ship | space | vessel | medicine | marine | ferry | marine, tank, navy, pilot, captain, drill, crew, port, sailor, mar |
| fur, rack, tail | coat | feline | animal | hair | whisk | hair, whisk, animal, cat, coat, claw, velvet, rod, dog, wool |
| stick, maker, point | match | needle | needle | needle | inventor | needle, tool, finger, tip, pin, nail, pencil, tooth, hammer, spear |
| hound, pressure, shot | blood | dog | needle | needle | dog | needle, pain, therapy, vaccine, medical, medicine, dog, doctor, piercing, drug |
| fox, man, peep | hole | eye | animal | animal | predator | animal, mouse, beast, rabbit, wolf, eye, hunter, lion, deer, owl |
| sleeping, bean, trash | bag | garbage | green | green | awake | green, pit, yellow, can, grow, black, produce, refuse, cot, hard |
| dust, cereal, fish | bowl | grain | clean | gritty | freshwater | gritty, grain, fine, grit, crunchy, dirt, sand, clean, gluten, brown |
| light, birthday, stick | candle | **candle** | **candle** | **candle** | celebration | candle, candles, shine, treat, party, ray, cone, gift, pointer, wand |
| food, forward, break | fast | meal | take | eat | backward | eat, give, lunch, serve, stop, carry, leave, deliver, push, rest |
| shine, beam, struck | moon | light | light | light | radiation | light, silver, sparkle, flash, ray, lightning, radiant, shining, bright, glitter |
| peach, arm, tar | pit | cake | soft | sticky | limb | sticky, yellow, jelly, black, soft, purple, juicy, skin, brown, pit |
| water, mine, shaker | salt | sink | drink | **salt** | cocktail | salt, cup, fill, glass, silver, pour, sparkle, drink, steam, soda |
| palm, shoe, house | tree | shelter | flat | flat | sandals | flat, sole, nail, sand, leather, sock, wing, flats, foot, hand |
| basket, eight, snow | ball | season | white | white | digit | white, winter, cold, bin, even, month, summer, big, blank, large |
| wheel, hand, shopping | cart | car | car | **cart** | axle | cart, car, bicycle, basket, carts, carriage, part, tire, bike, truck |
| right, cat, carbon | copy | paper | black | black | dioxide | black, rod, paw, mouse, animal, finger, left, pet, four, fur |
| home, sea, bed | sick | room | water | comfort | mattress | comfort, blue, calm, comfortable, warm, rest, sanctuary, quiet, tranquil, salt |
| nuclear, feud, album | family | record | power | power | reactor | power, part, war, rap, song, harmony, host, radiation, energy, lyric |
| sandwich, house, golf | club | home | green | hole | course | hole, flat, course, rectangular, small, slice, lunch, club, fork, square |
| cross, rain, tie | bow | knot | red | necklace | religious | necklace, button, belt, silver, knot, symbol, shirt, white, clothing, dark |
| sage, paint, hair | brush | comb | **brush** | **brush** | scalp | brush, oil, brushes, color, thin, pigment, white, brushing, texture, cut |
| french, car, shoe | horn | sock | flat | polish | fries | polish, foot, leather, flat, boot, bicycle, heel, part, footwear, franc |
| boot, summer, ground | camp | earth | hot | sand | shoe | sand, bare, warm, rain, hot, shoe, lawn, wear, cold, shoes |
| chamber, mask, natural | gas | room | medical | dental | goggles | dental, breathable, ventilation, transparent, window, surgical, oxygen, sterile, front, medical |
| mill, tooth, dust | saw | powder | fine | fine | gums | fine, grit, abrasive, gritty, grinding, grain, teeth, coarse, dental, powder |
| main, sweeper, light | street | spotlight | nurse | helper | sanitation | helper, street, front, pole, lamp, worker, bulb, jan, shadow, shine |
| pike, coat, signal | turn | flag | light | fish | fish | fish, ray, white, lamp, light, long, small, salmon, fur, blue |
| office, mail, hat | box | head | person | delivery | delivery | delivery, courier, deliver, envelope, post, carrier, shoe, delivers, personal, stamp |
| fly, clip, wall | paper | cliff | thread | insect | insect | insect, pin, nail, tear, hair, piercing, buzzing, poster, tooth, bug |
| age, mile, sand | stone | grain | fine | fine | distance | fine, small, hour, distance, grain, long, minute, meter, inch, river |
| catcher, food, hot | dog | soup | cold | cold | ump | cold, batter, pitcher, runner, pinch, warm, plate, base, cool, baseball |
| wagon, break, radio | station | car | light | cart | transmitter | cart, wheel, carrier, axle, carriage, truck, fork, push, part, electricity |
| tank, hill, secret | top | cipher | medical | **top** | valley | top, base, open, bottom, safe, plain, flat, high, lie, view |
| health, taker, less | care | eater | more | giver | giver | giver, fewer, more, eater, getter, dear, little, thinner, much, weaker |
| lift, card, mask | face | disguise | medical | **face** | elevator | face, hand, cover, front, suit, deck, airplane, draw, shoe, sheet |
| dress, dial, flower | sun | bouquet | part | bloom | bloom | bloom, button, stem, bell, pet, part, pretty, skirt, beautiful, rose |
| force, line, mail | air | envelope | letter | post | delivery | post, envelope, delivery, carrier, stroke, courier, stamp, deliver, push, postal |
| guy, rain, down | fall | umbrella | soft | feather | feather | feather, goose, duck, soft, fur, fine, bald, feathers, snow, plum |
| eight, skate, stick | figure | eighth | sharp | fish | scooter | fish, point, wheel, pencil, board, digit, rod, finger, slender, cart |
| down, question, check | mark | answer | form | exam | feather | exam, read, form, test, fill, yes, ask, wonder, verify, bill |
| animal, back, rat | pack | rodent | small | insect | mouse | insect, mouse, predator, prey, tail, fur, pet, creature, teeth, rod |
| officer, cash, larceny | petty | theft | money | theft | theft | theft, crime, legal, illegal, robbery, burglary, security, law, robber, purse |
| pine, crab, sauce | apple | soup | green | seafood | seafood | seafood, seasoning, salty, tasty, con, fish, salt, green, brown, fork |
| house, thumb, pepper | green | spice | small | small | finger | small, flat, palm, finger, tiny, pap, pinch, staple, warm, kitchen |
| carpet, alert, ink | red | stain | color | brush | warning | brush, color, paint, black, stain, warning, warn, dark, white, stained |
| master, toss, finger | ring | thumb | hand | thumb | apprentice | thumb, hand, servant, apprentice, chef, pupil, palm, cook, slave, claw |
| hammer, gear, hunter | head | tool | tool | tool | hunts | tool, bolt, wrench, screw, nail, wedge, carp, tools, chain, rod |
| knife, light, pal | pen | shadow | sharp | fork | cutting | fork, cut, scissors, finger, sharp, blade, spoon, chop, cutter, tooth |
| foul, ground, mate | play | earth | clean | dirt | soil | dirt, clean, dirty, uncle, earth, fine, rough, sick, fish, man |
| change, circuit, cake | short | cookie | bread | bread | icing | bread, electricity, flip, make, chip, dollar, coin, small, candy, metal |
| way, board, sleep | walk | dream | rest | rest | route | rest, method, pass, long, wake, easy, morning, slate, desk, walk |
| blank, list, mate | check | book | item | notebook | items | notebook, bullet, item, page, pen, comma, pencil, card, diary, envelope |
| tail, water, flood | gate | tsunami | wet | stream | whisk | stream, mouth, fish, river, danger, dam, thirst, rain, body, animal |
| marshal, child, piano | grand | music | teacher | instrument | keys | instrument, composer, adult, bench, keyboard, violin, soldier, string, hammer, drum |
| cover, arm, wear | under | shirt | body | button | limb | button, body, clothing, limb, append, upper, dress, use, top, attach |
| rain, test, stomach | acid | water | pain | pain | digestive | pain, disease, illness, nausea, ache, sickness, swollen, sore, acid, abdominal |
| time, blown, nelson | full | clock | out | clock | blew | clock, wind, arrow, stamp, flag, blew, sun, whistle, medal, hammer |
| pile, market, room | stock | storage | place | stall | vendors | stall, small, place, stack, heap, basket, stalls, large, grocery, store |
| mouse, bear, sand | trap | grain | grain | grain | cub | grain, trap, predator, fine, paw, prey, furry, small, mole, fur |
| cat, number, phone | call | digit | small | digit | digit | digit, mouse, computer, digits, animal, small, digital, string, rod, internet |
| keg, puff, room | powder | beer | empty | empty | balloon | empty, balloon, flat, air, dispenser, bubble, foam, beer, refill, fill |
| trip, house, goal | field | home | place | destination | objectives | destination, aim, place, vacation, hotel, dream, plan, mission, home, plane |
| fork, dark, man | pitch | night | knife | knife | knife | knife, spoon, silver, chop, cut, dining, diner, dinner, tooth, lad |
| fence, card, master | post | gate | bar | gate | apprentice | gate, door, bar, deck, guard, servant, chain, board, owner, slave |
| test, runner, map | road | track | doctor | track | marathon | track, trail, marathon, chart, doctor, trace, cross, treadmill, marker, route |
| dive, light, rocket | sky | flight | water | moon | spaceship | moon, spaceship, shine, star, explore, shadow, astronaut, oxygen, spacecraft, penetrate |
| man, glue, star | super | galaxy | light | moon | constellation | moon, planet, shine, wax, boy, beast, sun, horn, bead, sparkle |
| tooth, potato, heart | sweet | pulp | healthy | root | gums | root, cavity, mouth, flesh, muscle, vein, healthy, swollen, tub, fungus |
| illness, bus, computer | terminal | machine | light | **terminal** | keyboard | terminal, driver, user, keyboard, motherboard, monitor, bug, virus, printer, network |
| type, ghost, screen | writer | computer | light | monster | paranormal | monster, horror, shadow, computer, keyboard, spectral, scare, creature, scary, supernatural |
| mail, board, lung | black | organ | heart | post | delivery | post, envelope, chest, box, oxygen, newspaper, carrier, air, parcel, card |
| teeth, arrest, start | false | tooth | pain | stop | gums | stop, jaw, jaws, fine, capture, white, pull, prison, can, kill |
| iron, shovel, engine | steam | machine | tool | tool | digging | tool, screw, hammer, metal, tools, work, rust, piston, blade, steel |
| wet, law, business | suit | weather | money | order | order | order, firm, make, work, suit, rule, check, just, court, high |
| rope, truck, line | tow | cable | thread | string | cargo | string, cargo, tow, thread, chain, carrier, pull, haul, axle, tire |
| off, military, first | base | army | not | army | foremost | army, order, front, second, rank, far, beginning, from, last, next |
| spoon, cloth, card | table | plate | cut | cut | fork | cut, fork, kitchen, knife, scissors, table, chop, plate, cutting, serving |
| cut, cream, war | cold | cake | pain | pain | fatty | pain, fat, cake, tear, cheese, knife, skin, treat, wound, slice |
| note, chain, master | key | teacher | doctor | doctor | links | doctor, binder, pad, pupil, surgeon, servant, hand, apprentice, link, key |
| shock, shave, taste | after | tooth | pain | pain | cut | pain, feel, burn, electric, cut, treat, touch, painful, numb, hair |
| wise, work, tower | clock | wisdom | building | building | building | building, brick, build, smart, tall, man, farm, book, crane, buildings |
| grass, king, meat | crab | lawn | animal | animal | fatty | animal, cow, lion, alive, crown, blade, sheep, plant, pig, tiger |
| baby, spring, cap | shower | sprout | medical | bottle | hat | bottle, pillow, cot, medical, wound, doctor, cotton, tooth, knit, cushion |
| break, bean, cake | coffee | pie | food | eat | icing | eat, edible, dessert, slice, fry, chocolate, food, dinner, split, butter |
| cry, front, ship | battle | shore | water | wake | ferry | wake, port, nose, deep, mouth, behind, end, row, rear, tank |
| hold, print, stool | foot | chair | support | press | grip | press, grip, clamp, bind, pen, finger, button, pull, plug, push |
| roll, bean, fish | jelly | noodle | food | edible | freshwater | edible, jelly, ribbon, eat, yellow, white, staple, grocery, food, meat |
| horse, human, drag | race | animal | animal | animal | neigh | animal, beast, pull, pulling, creature, man, carriage, hoof, insect, gal |
| oil, bar, tuna | salad | fish | fish | fish | grill | fish, frying, sushi, fatty, restaurant, seafood, steak, fat, salad, cooking |
| bottom, curve, hop | bell | jump | part | flat | beneath | flat, floor, part, flower, crop, hull, line, bell, bud, bend |
| tomato, bomb, picker | cherry | pepper | fruit | fruit | uranium | fruit, produce, pepper, vegetable, orange, explosive, rip, squash, cherry, ripe |
| pea, shell, chest | nut | box | heart | heart | lobster | heart, hollow, hard, cavity, pod, rib, jelly, clam, small, pulse |
| line, fruit, drunk | punch | juice | red | alcohol | staggering | alcohol, hard, vine, tip, sick, juice, black, root, taste, bitter |
| bump, egg, step | goose | stumble | pain | white | stride | white, crust, chicken, swelling, small, pain, large, run, crack, something |
| fight, control, machine | gun | engine | medical | power | regulate | power, hand, manual, arm, danger, device, dangerous, blow, robot, loose |
| home, arm, room | rest | house | medical | bed | beds | bed, living, furniture, hospital, place, part, lodging, beds, body, comfortable |
| child, scan, wash | brain | clean | medical | medical | imaging | medical, doctor, patient, dentist, health, imaging, test, examine, hospital, illness |
| nose, stone, bear | brown | rock | soft | cub | cub | cub, animal, black, large, heart, amber, big, pit, claw, bark |
| end, line, lock | dead | key | tool | key | key | key, cut, bolt, chain, pin, tool, door, open, latch, nail |
| control, place, rate | birth | score | money | evaluate | regulate | evaluate, point, adjust, amount, meter, setting, value, set, assess, distance |
| lounge, hour, napkin | cocktail | notebook | soft | table | minute | table, restaurant, dining, hand, seat, minute, furniture, hotel, dinner, morning |
| artist, hatch, route | escape | path | tool | brush | muse | brush, canvas, architect, sketch, blueprint, museum, create, track, painter, craft |
| pet, bottom, garden | rock | yard | plant | owner | beneath | owner, animal, small, farm, lawn, vegetable, outside, fur, yard, indoor |
| mate, shoes, total | running | match | part | part | sandals | part, pair, sum, best, sock, sole, sand, socks, whole, toe |
| self, attorney, spending | defense | expense | person | judge | saving | judge, person, client, lawyer, patient, clerk, manager, personal, care, consumer |
| board, blade, back | switch | spine | sharp | furniture | desk | furniture, spine, knife, cut, sharp, part, scissors, teeth, curved, chair |
| land, hand, house | farm | home | home | **farm** | fertile | farm, flat, home, palm, building, small, arm, builder, apartment, front |
| hungry, order, belt | money | stomach | stomach | stomach | prescription | stomach, wallet, pouch, appetite, needs, prescription, mouth, waist, eat, throat |
| forward, flush, razor | straight | blade | water | cut | scissors | cut, handle, hand, sink, shave, blade, remove, push, toilet, plug |
| shadow, chart, drop | eye | shade | light | ink | silhouette | ink, form, record, silhouette, pen, doctor, medical, insect, dot, mist |
| way, ground, weather | fair | sky | clear | rain | soil | rain, clear, fine, sun, how, sky, sand, down, through, fertile |
| cast, side, jump | broad | leap | pain | limb | flank | limb, foot, leg, ankle, brace, elbow, support, wrist, arm, heel |
| back, step, screen | door | stage | part | front | monitor | front, part, flat, support, wall, monitor, touch, rear, posture, shoulder |
| reading, service, stick | lip | book | long | patience | books | patience, pencil, long, pen, attention, staff, tip, pens, wait, finger |
| over, plant, horse | power | pasture | animal | animal | neigh | animal, with, farm, grow, under, large, cover, barn, living, through |

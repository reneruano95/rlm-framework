import random, collections
STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]
def bindings(size, trial):
    rng = random.Random(1000*size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust", "%08x-%04x-%04x-%04x-%012x" % (
        rng.getrandbits(32), rng.getrandbits(16), rng.getrandbits(16),
        rng.getrandbits(16), rng.getrandbits(48))) for s in pool[:3]]
    return present, f"{pool[3]} Trust"
allu=collections.defaultdict(list)
fx={}
for size in (640,1024,2048):
    for t in range(4):
        pres, absent = bindings(size,t)
        fx[(size,t)]=(pres,absent)
        for e,u in pres: allu[u].append((size,t,e))
print("total uuids planted:", sum(len(v) for v in allu.values()), " distinct:", len(allu))
dupes={u:v for u,v in allu.items() if len(v)>1}
print("DUPLICATE uuids across fixtures:", dupes if dupes else "NONE")
targets=["48e81295-9489-33be-cc30-430d702be6c3",
         "d9f804c1-fa2b-8d32-a160-adccebcd8978",
         "7e41c11e-4a6f-131b-df64-d2385eb09ba3",
         "8607123c-d88d-b4a1-33b5-b3b1333930dc",
         "f17f5f41-992b-d4eb-1eba-50b3b7a061bd",
         "8b756d87-802a-6bad-f4a3-310a68bea4e7",
         "c67c7dad-ccf2-6cde-0868-6f907eaa5463",
         "ea7d8870-8fa5-3375-1ad1-07dda2491305"]
print("\nProvenance of each observed answer-uuid:")
for u in targets:
    print(f"  {u} -> {allu.get(u,'NOT PLANTED ANYWHERE')}")
print("\nFixture table (present entities -> uuid | ABSENT-asked entity):")
for k in sorted(fx):
    pres,absent=fx[k]
    print(f"  {k[0]:>5}/t{k[1]}  ABSENT-ASKED={absent:<22} present={[ (e.split()[0], u[:8]) for e,u in pres]}")
print("\nCross-check: for each ABSENT question, is the asked entity PRESENT in some other fixture?")
for k in sorted(fx):
    pres,absent=fx[k]
    donors=[(kk, u) for kk,(p,a) in fx.items() for e,u in p if e==absent]
    print(f"  {k[0]:>5}/t{k[1]} asked {absent:<22} donors elsewhere: {[(f'{a}/t{b}', u[:8]) for (a,b),u in donors]}")

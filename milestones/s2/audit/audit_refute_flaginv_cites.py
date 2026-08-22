import os
CITES = [
 ("milestones/s2/leafcall.py", 290, 300),
 ("milestones/s2/run_occupancy.py", 345, 355),
 ("milestones/s2/run_occupancy.py", 440, 450),
 ("milestones/s2/arch_ladder.py", 112, 132),
 ("milestones/s2/bos_check.py", 34, 44),
 ("milestones/s2/offby1_check.py", 26, 36),
 ("milestones/s2/r13_mitigation_bench.py", 368, 378),
 ("milestones/s2/r13_mitigation_bench.py", 388, 396),
 ("milestones/s2/run_cache_instrument.py", 368, 378),
 ("milestones/s2/run_cache_instrument.py", 600, 610),
 ("milestones/s2/CACHE-INSTRUMENT.md", 1, 8),
 ("milestones/s2/CACHE-INSTRUMENT.md", 17, 22),
 ("milestones/s2/CACHE-INSTRUMENT.md", 103, 110),
 ("milestones/s2/CACHE-INSTRUMENT.md", 376, 384),
 ("ARCHITECTURE.md", 430, 440),
 ("ARCHITECTURE.md", 250, 256),
 ("ARCHITECTURE.md", 422, 429),
 ("ARCHITECTURE.md", 96, 106),
 ("ARCHITECTURE.md", 265, 269),
 ("ARCHITECTURE.md", 492, 499),
 ("milestones/s2/R14.md", 1, 10),
 ("milestones/s2/RESULTS.md", 130, 140),
 ("milestones/s2/RESULTS.md", 155, 163),
 ("milestones/s2/RESULTS.md", 166, 178),
 ("milestones/s2/R13-mitigations.md", 38, 48),
 ("milestones/s2/R13-mitigations.md", 433, 439),
 ("milestones/s2/ARCH-LADDER.md", 1, 12),
 ("milestones/s2/cache_instrument.ps1", 16, 23),
 ("milestones/s2/occupancy_conditions.ps1", 24, 34),
 ("milestones/s2/r14_ladder.ps1", 18, 36),
 ("milestones/s2/r14_hypotheses.ps1", 18, 22),
]
for path,a,b in CITES:
    print(f"\n===== {path}:{a}-{b} =====")
    if not os.path.exists(path):
        print("  MISSING FILE"); continue
    lines=open(path,encoding="utf-8",errors="replace").read().split("\n")
    for i in range(a, min(b, len(lines))+1):
        print(f"  {i}: {lines[i-1][:200]}")

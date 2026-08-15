import io, os, re, subprocess
out = io.open("s2/audit/_misc_out.txt", "w", encoding="utf-8", errors="replace")
def P(*a): out.write(" ".join(str(x) for x in a) + "\n")

P("-- leak-arm server log headers --")
for p in sorted(os.listdir("traces/logs")):
    if p.startswith("leaf-server-ub") and p.endswith(".err"):
        L = io.open("traces/logs/"+p, encoding="utf-8", errors="replace").read().split("\n")
        for i, l in enumerate(L[:15], 1):
            if "n_slots" in l: P(f"  traces/logs/{p}:{i}: {l.strip()}")

P("\n-- sweep-era server logs --")
for f in ["traces/logs/leaf-server.log","traces/logs/leaf-server.out","traces/logs/leaf-server.err.log"]:
    P(f"  {f}: exists={os.path.exists(f)} size={os.path.getsize(f) if os.path.exists(f) else 'NA'}")

P("\n-- ARCH-LADDER provenance --")
txt = io.open("s2/ARCH-LADDER.md", encoding="utf-8", errors="replace").read()
for kw in ["llama-server", "-np", "--cache-ram", "65536", "launch", "Server", "port", "slot"]:
    hits = [(i, l.strip()[:130]) for i, l in enumerate(txt.split("\n"), 1) if kw in l]
    P(f"  ARCH-LADDER.md '{kw}': {len(hits)} hits -> {hits[:3]}")
P("  .ps1 files in s2/: " + str([f for f in os.listdir("s2") if f.endswith(".ps1")]))
r = subprocess.run(["git","show","70e2d40:s2/arch_ladder.py"], capture_output=True, text=True, encoding="utf-8", errors="replace")
P("  committed arch_ladder.py: argv=%s llama-server=%s subprocess=%s id_slot=%s" %
  ("argv" in r.stdout, "llama-server" in r.stdout, "subprocess" in r.stdout, "id_slot" in r.stdout))

P("\n-- R13-mitigations cache_n lines --")
t = io.open("s2/R13-mitigations.md", encoding="utf-8", errors="replace").read().split("\n")
for i, l in enumerate(t, 1):
    if "cache_n" in l: P(f"  :{i}: {l.strip()[:220]}")

P("\n-- R13-slotcount np values --")
t = io.open("s2/R13-slotcount.md", encoding="utf-8", errors="replace").read().split("\n")
for i, l in enumerate(t[:60], 1):
    if re.search(r"-np|\b256\b|\b192\b", l): P(f"  :{i}: {l.strip()[:180]}")

P("\n-- CACHE-INSTRUMENT cross-slot section (lines 80-112) --")
t = io.open("s2/CACHE-INSTRUMENT.md", encoding="utf-8", errors="replace").read().split("\n")
for i in range(80, 113):
    P(f"  :{i}: {t[i-1][:200]}")
out.close()

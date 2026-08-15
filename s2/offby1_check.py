"""Does the preflight token count actually match what /completion serves?

An earlier measurement reported a constant +1 (preflight 284/474/1274 vs served
285/475/1275) and fixed it by setting add_special=True. But add_special is a
no-op on this tokenizer, so if the +1 is real it has another cause -- and the
admission boundary is still wrong.
"""
import json, urllib.request

BASE = "http://127.0.0.1:8081"

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} on {path}: {e.read().decode()[:300]}")
        raise

print(f"{'chunk_words':>12} {'tok(F)':>7} {'tok(T)':>7} {'served':>7} {'served-tok(T)':>14}")
for n in (5, 50, 500, 2000):
    body = " ".join(f"w{i}" for i in range(n))
    msgs = [{"role": "system", "content": "You extract facts."},
            {"role": "user", "content": body + "\n\nWhat is the key?"}]
    rendered = post("/apply-template", {"messages": msgs})["prompt"]
    f = len(post("/tokenize", {"content": rendered, "add_special": False})["tokens"])
    t = len(post("/tokenize", {"content": rendered, "add_special": True})["tokens"])
    r = post("/completion", {"prompt": rendered, "n_predict": 1,
                             "cache_prompt": False, "temperature": 0})
    tim = r["timings"]
    served = tim["prompt_n"] + tim.get("cache_n", 0)
    print(f"{n:>12} {f:>7} {t:>7} {served:>7} {served - t:>14}")

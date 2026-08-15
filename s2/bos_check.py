"""Double-BOS check (user hypothesis, 2026-08-14).

If /completion prepends BOS to the prompt string AND the chat template already
emits one, the model sees a malformed sequence start -- a known llama.cpp
quality-killer. Verify against the live leaf, no assumptions.
"""
import json, urllib.request

BASE = "http://127.0.0.1:8081"

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())

props = get("/props")
tmpl = props.get("chat_template", "")
print(f"chat_template chars: {len(tmpl)}")
print(f"template mentions bos_token: {'bos_token' in tmpl}")

msgs = [{"role": "system", "content": "You extract facts."},
        {"role": "user", "content": "DOC\n\nWhat is the key?"}]
rendered = post("/apply-template", {"messages": msgs})["prompt"]
print(f"\nrendered head (repr): {rendered[:70]!r}")

no_special = post("/tokenize", {"content": rendered, "add_special": False})["tokens"]
with_special = post("/tokenize", {"content": rendered, "add_special": True})["tokens"]
print(f"\ntokenize add_special=False : n={len(no_special)} first5={no_special[:5]}")
print(f"tokenize add_special=True  : n={len(with_special)} first5={with_special[:5]}")
print(f"delta = {len(with_special) - len(no_special)}")

# What does the SERVER actually tokenize when it serves this prompt?
served = post("/completion", {"prompt": rendered, "n_predict": 1,
                              "cache_prompt": False, "temperature": 0})
pn = served["timings"]["prompt_n"]
print(f"\nserved prompt_n            : {pn}")
print(f"  == add_special=True  ? {pn == len(with_special)}")
print(f"  == add_special=False ? {pn == len(no_special)}")

# Decode the first token of each to name it
for label, toks in (("add_special=True", with_special), ("rendered as-is", no_special)):
    d = post("/detokenize", {"tokens": toks[:1]})
    print(f"first token of {label:18s}: id={toks[0]} -> {d.get('content')!r}")

dup = len(with_special) > len(no_special) and no_special[0] == with_special[0]
print(f"\nDOUBLE-BOS PRESENT: {dup}")
print("VERDICT:", "DOUBLE BOS -- real defect" if dup else
      "no double BOS; the +1 is the server adding BOS to a template that lacks one (correct)")

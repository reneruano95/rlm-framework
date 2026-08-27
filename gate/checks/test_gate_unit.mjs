/**
 * Deterministic unit test for the gate extension's handlers.
 *
 * The behaviour under test is the scaffold's, not the model's, so it is tested by
 * driving the handlers directly with a stub ExtensionAPI rather than by hoping a
 * live model loops. Step 2's live attempt reached only 2 identical cells before the
 * continuation budget stopped it, which proves nothing about a threshold of 3.
 *
 * Run (inside WSL, as the spike user):
 *   node gate/checks/test_gate_unit.mjs /path/to/rlmh-gate.ts
 */

import { createRequire } from "node:module";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const src = process.argv[2] ?? "/home/spike/prime-spike/extensions/rlmh-gate.ts";
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "rlmh-gate-test-"));
const accepted = path.join(tmp, "accepted.json");
const ledger = path.join(tmp, "ledger.jsonl");

process.env.RLMH_ACCEPTED = accepted;
process.env.RLMH_LEDGER = ledger;
process.env.RLMH_MARKER = "GATE-MARKER-XYZ";
process.env.RLMH_MAX_IDENTICAL = "3";

fs.writeFileSync(
	accepted,
	JSON.stringify({
		schema: 1,
		entries: { prompt: { "ok-entry": { id: "ok-entry" } }, memory: {}, skill: {}, subagent: {} },
		refinements: [],
	}),
);

// jiti ships inside the prime-agent install; resolve from there so the .ts under
// test needs no build step and the test needs no dependency of its own.
const PA = process.env.RLMH_PA_PKG
	?? path.join(os.homedir(), ".npm-global", "lib", "node_modules", "prime-agent", "package.json");
let load;
try {
	const { createJiti } = createRequire(PA)("jiti");
	load = createJiti(import.meta.url);
} catch (err) {
	console.error(`cannot load jiti from ${PA}:`, String(err));
	process.exit(3);
}
const mod = await load.import(src, { default: true });
const factory = typeof mod === "function" ? mod : mod.default;

// ---- stub ExtensionAPI -------------------------------------------------------
const handlers = {};
factory({ on: (event, fn) => { handlers[event] = fn; } });

let failures = 0;
function check(name, cond, detail = "") {
	console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
	if (!cond) failures += 1;
}

check("registers 4 handlers", Object.keys(handlers).length === 4, Object.keys(handlers).join(", "));

// ---- session_before_refine ---------------------------------------------------
const vetoed = await handlers.session_before_refine({ preparation: { trigger: "manual", scope: "local" } });
check("vetoes an un-marked refine round", vetoed?.skip === true);

const allowed = await handlers.session_before_refine({
	preparation: { trigger: "manual", scope: "global", instructions: "please GATE-MARKER-XYZ do it" },
});
check("allows a gate-marked round", !allowed?.skip);

// ---- tool_call budget --------------------------------------------------------
const CODE = "print('same')";
const r1 = await handlers.tool_call({ toolName: "ipython", input: { code: CODE } });
const r2 = await handlers.tool_call({ toolName: "ipython", input: { code: CODE } });
const r3 = await handlers.tool_call({ toolName: "ipython", input: { code: CODE } });
check("identical cell 1 allowed", !r1?.block);
check("identical cell 2 allowed", !r2?.block);
check("identical cell 3 BLOCKED", r3?.block === true, r3?.reason?.slice(0, 60));

// a different cell resets the streak
const r4 = await handlers.tool_call({ toolName: "ipython", input: { code: "print('other')" } });
const r5 = await handlers.tool_call({ toolName: "ipython", input: { code: "print('other')" } });
check("streak resets on a changed cell", !r4?.block && !r5?.block);

// non-ipython tools are not the guard's business
const rb = await handlers.tool_call({ toolName: "bash", input: { command: "ls" } });
check("non-ipython tool untouched", !rb?.block);

// ---- before_agent_start filter ----------------------------------------------
const PROMPT = [
	"You are an agent.",
	"",
	"# Continual Harness State",
	"prompt: 2",
	"- [global:ok-entry] Kept (00-gate/prompt/00, v1): keep me",
	"- [global:bad-entry] Smuggled (00-smuggle, v1): SECRETWORD",
	"- +4 more prompt entries",
	"memory: 1",
	"- [global:bad-mem] Also smuggled (x, v1): SECRETWORD2",
	"recent refinements: 0",
].join("\n");
const filtered = await handlers.before_agent_start({ systemPrompt: PROMPT });
const out = filtered?.systemPrompt ?? PROMPT;
check("keeps the accepted entry", out.includes("ok-entry"));
check("strips unaccepted entries", !out.includes("bad-entry") && !out.includes("bad-mem"));
check("no smuggled content survives", !out.includes("SECRETWORD"));
check("rewrites the prompt count to 1", /^prompt: 1$/m.test(out), out.match(/^prompt: \d+$/m)?.[0]);
check("rewrites the memory count to 0", /^memory: 0$/m.test(out), out.match(/^memory: \d+$/m)?.[0]);
check("drops the overflow line", !out.includes("more prompt entries"));
check("leaves the head untouched", out.startsWith("You are an agent."));

// nothing to strip -> byte-identical prompt (the OFF-arm fidelity requirement)
const clean = PROMPT.split("\n").filter((l) => !l.includes("bad-") && !l.includes("more prompt")).join("\n")
	.replace("prompt: 2", "prompt: 1").replace("memory: 1", "memory: 0");
const again = await handlers.before_agent_start({ systemPrompt: clean });
check("a clean prompt is returned unmodified", again?.systemPrompt === undefined || again.systemPrompt === clean);

const events = fs.readFileSync(ledger, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
check("ledger recorded the block", events.some((e) => e.event === "identical_turn_blocked"));
check("ledger recorded the strip", events.some((e) => e.event === "stripped_entries"));

fs.rmSync(tmp, { recursive: true, force: true });
console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);

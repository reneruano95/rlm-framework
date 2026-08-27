/**
 * rlm-halo artifact gate — the enforcement half of S6-lite v0.
 *
 * Spec: docs/superpowers/specs/2026-08-27-s6-lite-v0-artifact-gate.md §5.
 * Plan: docs/superpowers/plans/2026-08-27-s6-lite-v0-plan.md step 1.
 *
 * WHAT THIS IS FOR. prime-agent's Continual Harness applies whatever the model
 * proposes; there is no per-edit veto (`session_before_refine` can only skip the
 * round or replace the planner wholesale). Measured on this box 2026-08-27: given
 * a free hand the local root wrote 8/8 artifacts as `memory` holding the answers
 * to the tasks it had just been shown. This file makes the model structurally
 * unable to write its own harness — I1 as a mechanism rather than an instruction.
 *
 * FOUR HANDLERS, and what each can and cannot do:
 *
 *   session_before_refine  VETO. Returns {skip:true} for every round that does not
 *                          carry the gate's marker, which raises RefineSkippedError
 *                          inside the host. This is the only hard stop available.
 *
 *   before_agent_start     FILTER. Receives the assembled system prompt and may
 *                          replace it. Strips any rendered harness line whose entry
 *                          id is not in the accepted set. This catches a direct
 *                          `rlm.harness.create_*` write from the kernel, which
 *                          `tool_call` cannot see because refine.run is a host
 *                          request, not a tool.
 *
 *   refine_complete        NOTIFY ONLY. Fires after the state file is written, the
 *                          prompt rebuilt and the audit entry appended; its return
 *                          value is ignored and a throw is swallowed by the host.
 *                          It cannot veto and this file does not pretend it can.
 *
 *   tool_call              BUDGET. Blocks the Nth consecutive byte-identical ipython
 *                          cell. prime-agent ships no identical-turn guard; the
 *                          scaffold's C5 has `max_identical_turns: 3`, and I1 says
 *                          budgets are the scaffold's to own. Measured on this box
 *                          2026-08-27, step 2: one agg-07 episode emitted 29
 *                          identical cells over 51 turns, burned 88.7K tokens and
 *                          1,384 s, and produced no answer — R15's attractor inside
 *                          prime-agent, and prime-agent issue #1326. It applies to
 *                          both arms identically, so it cannot bias an A/B; what it
 *                          does change is comparability with the spike's Phase A,
 *                          which ran without it (spec D-S7).
 *
 * FAIL LOUDLY. If any of the four cannot be registered the factory throws, so a
 * prime-agent upgrade that moved the API stops the run instead of silently
 * degrading to "the model refines itself again" (plan step 1 check 3).
 *
 * ENV:
 *   RLMH_ACCEPTED  path to the gate's accepted-set JSON (a HarnessState shape).
 *                  Absent or unreadable is treated as "nothing is accepted", which
 *                  strips every harness line — the safe direction.
 *   RLMH_LEDGER    path to a JSONL audit log. Every veto, filter and completion is
 *                  appended. Absent disables logging but not enforcement.
 *   RLMH_MARKER    the token a gate-initiated refine round carries in its
 *                  instructions. Unset means no round is ever gate-initiated, i.e.
 *                  every round is vetoed.
 *   RLMH_MAX_IDENTICAL  consecutive byte-identical ipython cells allowed before the
 *                  next one is blocked. Default 3, matching config.yaml's C5.
 */

import * as fs from "node:fs";

type Json = Record<string, unknown>;

const ACCEPTED_PATH = process.env.RLMH_ACCEPTED ?? "";
const LEDGER_PATH = process.env.RLMH_LEDGER ?? "";
const MARKER = process.env.RLMH_MARKER ?? "";
/** C5's max_identical_turns. 3 matches config.yaml: correct at 2, kill at 3. */
const MAX_IDENTICAL = Number(process.env.RLMH_MAX_IDENTICAL ?? "3");

const HARNESS_HEADING = "# Continual Harness State";
/** prime-agent renders one entry as: `- [scope:id] title (path, vN)...: content` */
const ENTRY_LINE = /^- \[[^:\]]*:([^\]]+)\]/;
/**
 * The per-kind count line, `${kind}: ${entries.length}` (refinement.ts:479), and the
 * overflow line it can be followed by (`- +N more <kind> entries`, :497-500).
 *
 * These must be rewritten, not merely left alone. Measured 2026-08-27 in step 1
 * check 2: with a smuggled entry stripped but its count left at 1, the model read
 * the stale line and reported "memory: 1 (a stale summary)" — so filtering entry
 * lines alone still tells it something exists. Worse for the gate, it makes an OFF
 * arm differ from stock prime-agent, which the A/B design requires it not to.
 */
const COUNT_LINE = /^(prompt|memory|skill|subagent): (\d+)/;
const OVERFLOW_LINE = /^- \+\d+ more (prompt|memory|skill|subagent) entries/;

function sha8(text: string): string {
	// A short content tag for the ledger; not a security boundary.
	let h = 0;
	for (let i = 0; i < text.length; i += 1) {
		h = (Math.imul(31, h) + text.charCodeAt(i)) | 0;
	}
	return (h >>> 0).toString(16).padStart(8, "0");
}

function log(event: string, detail: Json): void {
	if (!LEDGER_PATH) return;
	try {
		fs.appendFileSync(
			LEDGER_PATH,
			JSON.stringify({ ts: new Date().toISOString(), event, ...detail }) + "\n",
			"utf8",
		);
	} catch {
		// The ledger is evidence, not a dependency: never fail a run over it.
	}
}

/** Ids the gate has accepted. An unreadable or absent file means: none. */
function acceptedIds(): { ids: Set<string>; source: string } {
	if (!ACCEPTED_PATH) return { ids: new Set(), source: "unset" };
	try {
		const raw = fs.readFileSync(ACCEPTED_PATH, "utf8");
		const state = JSON.parse(raw) as { entries?: Record<string, Record<string, unknown>> };
		const ids = new Set<string>();
		for (const byKind of Object.values(state.entries ?? {})) {
			for (const id of Object.keys(byKind ?? {})) ids.add(id);
		}
		return { ids, source: ACCEPTED_PATH };
	} catch (err) {
		log("accepted_unreadable", { path: ACCEPTED_PATH, error: String(err) });
		return { ids: new Set(), source: "unreadable" };
	}
}

/**
 * Remove rendered harness lines whose entry id is not accepted.
 *
 * Deliberately surgical: it filters lines out of prime-agent's own rendering
 * rather than re-formatting the block, so an arm with nothing to strip is
 * byte-identical to an unmodified run. That matters — the gate's OFF arm must
 * not differ from stock prime-agent in any way except the artifacts.
 */
function filterHarnessBlock(prompt: string, ids: Set<string>): { out: string; kept: number; stripped: string[] } {
	const at = prompt.indexOf(HARNESS_HEADING);
	if (at === -1) return { out: prompt, kept: 0, stripped: [] };

	const head = prompt.slice(0, at);
	const block = prompt.slice(at);
	const stripped: string[] = [];
	const keptByKind: Record<string, number> = { prompt: 0, memory: 0, skill: 0, subagent: 0 };
	let kept = 0;

	// Pass 1: drop unaccepted entry lines and the overflow lines, counting what stays.
	// `kind` is tracked from the most recent count line, since an entry line does not
	// carry its own kind.
	let kind = "";
	const lines: string[] = [];
	for (const line of block.split("\n")) {
		const c = COUNT_LINE.exec(line);
		if (c) {
			kind = c[1];
			lines.push(line);
			continue;
		}
		if (OVERFLOW_LINE.test(line)) continue;
		const m = ENTRY_LINE.exec(line);
		if (!m) {
			lines.push(line);
			continue;
		}
		if (ids.has(m[1])) {
			kept += 1;
			if (kind) keptByKind[kind] = (keptByKind[kind] ?? 0) + 1;
			lines.push(line);
			continue;
		}
		stripped.push(m[1]);
	}

	// Pass 2: rewrite each count line to what actually survived, so the model is never
	// told an entry exists that it cannot see.
	const out = lines
		.map((line) => {
			const c = COUNT_LINE.exec(line);
			if (!c) return line;
			return line.replace(COUNT_LINE, `${c[1]}: ${keptByKind[c[1]] ?? 0}`);
		})
		.join("\n");

	return { out: head + out, kept, stripped };
}

export default function rlmhGate(pi: any): void {
	if (!pi || typeof pi.on !== "function") {
		throw new Error("rlmh-gate: ExtensionAPI has no .on() — prime-agent API changed; refusing to load.");
	}

	const registered: string[] = [];

	// ---- VETO -------------------------------------------------------------
	pi.on("session_before_refine", (event: any) => {
		const instructions: string = event?.preparation?.instructions ?? "";
		const gateInitiated = MARKER !== "" && instructions.includes(MARKER);
		log("before_refine", {
			trigger: event?.preparation?.trigger,
			scope: event?.preparation?.scope,
			gateInitiated,
			decision: gateInitiated ? "allow" : "skip",
		});
		if (gateInitiated) return {};
		return { skip: true };
	});
	registered.push("session_before_refine");

	// ---- FILTER -----------------------------------------------------------
	pi.on("before_agent_start", (event: any) => {
		const prompt: string = event?.systemPrompt ?? "";
		const { ids, source } = acceptedIds();
		const { out, kept, stripped } = filterHarnessBlock(prompt, ids);
		if (stripped.length > 0) {
			log("stripped_entries", { stripped, kept, accepted: ids.size, source });
		}
		log("prompt_filtered", { kept, stripped: stripped.length, accepted: ids.size, chars: out.length });
		return out === prompt ? {} : { systemPrompt: out };
	});
	registered.push("before_agent_start");

	// ---- BUDGET -----------------------------------------------------------
	// C5's max_identical_turns, restored for a harness that has none. Semantics match
	// the scaffold's: the run of identical cells may reach MAX_IDENTICAL - 1; the next
	// one is refused. Blocking rather than killing leaves the model a turn to do
	// something else, and a blocked call costs no execution.
	//
	// State is module-level, which is per-episode here because every episode is its
	// own prime-agent process. A resident daemon serving several episodes would need
	// this keyed by session id.
	let lastCode: string | null = null;
	let streak = 0;
	pi.on("tool_call", (event: any) => {
		if (event?.toolName !== "ipython") return {};
		const code: string = event?.input?.code ?? "";
		if (code === lastCode) {
			streak += 1;
		} else {
			lastCode = code;
			streak = 1;
		}
		if (streak < MAX_IDENTICAL) return {};
		log("identical_turn_blocked", { streak, max: MAX_IDENTICAL, code_sha: sha8(code) });
		return {
			block: true,
			reason:
				`Blocked: this is identical cell #${streak} in a row and the scaffold's ` +
				`max_identical_turns is ${MAX_IDENTICAL}. Re-running it will not produce a ` +
				`different result. Change the approach or write your answer.`,
		};
	});
	registered.push("tool_call");

	// ---- NOTIFY -----------------------------------------------------------
	// Fires AFTER the write. Cannot veto; a throw here is swallowed by the host.
	// Its only job is to make an unexpected write visible in the ledger.
	pi.on("refine_complete", (event: any) => {
		log("refine_complete", {
			id: event?.id,
			summary: event?.summary,
			appliedEdits: event?.appliedEdits,
			scope: event?.scope,
			note: "a refinement completed — the veto did not fire for this round",
		});
	});
	registered.push("refine_complete");

	if (registered.length !== 4) {
		throw new Error(`rlmh-gate: registered ${registered.length}/4 handlers — refusing to load.`);
	}
	log("loaded", { registered, accepted: ACCEPTED_PATH || "unset", marker: MARKER ? "set" : "unset", maxIdentical: MAX_IDENTICAL });
}

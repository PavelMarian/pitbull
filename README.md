# PITBULL — Temporal Safety Skill for Coding Agents

**Demo:** https://pavelmarian.github.io/pitbull/demo/ · **Video:** https://youtu.be/ST8dWf82Zxs · **Paper:** `paper/aaai27_demo/text.pdf`

## What it is

![PITBULL loop: inputs & temporal contract, differential execution on three database views, detect & localize, agent repair, revalidate](assets/pitbull-loop.webp)

PITBULL is an executable skill that catches temporal ("future") data leakage
in agent-generated feature code, localizes it, and drives the repair to an
execution-verified result — not a code-review opinion. Point-in-time
correctness means every feature computed at time `t` depends only on facts
available at `t`; models miss this on their own (1/12 self-checked, 11/12
once told the error class), so PITBULL makes the check executable instead
of advisory.

- **Differential witness** — run the candidate `φ(D, e, t)` on the full
  database and on the database truncated at `t`; any divergence beyond
  tolerance is a `LEAK`, naming the diverging columns.
- **Canary** — late-filled fields (delivery date, review, order status) and
  all future rows get rewritten with marker values; a reacting output is a
  canary hit (fires on 4/30 clean controls — flagged for human judgment,
  not auto-blocked).
- **Repair loop, held-out gate** — on `LEAK`, the agent rewrites the program
  under a fixed instruction naming the error class; `CLEAN` only if witness
  and canary stay silent on all dev *and* held-out times; up to 5 repairs.
  Full history in a state file next to the code.

## Key results

Repair model `z-ai/glm-5.3-flash` on the Olist corpus: 76 leaking programs
(62 witness + 14 canary-only) written by two code models from a task
description that never mentions leakage.

| Invocation | n | CLEAN@1 | CLEAN@5 | Calls | Cost |
|---|---|---|---|---|---|
| Direct, 1 run | 76 | 85.5% | 98.7% | 91 | $0.12 |
| Direct, 3 runs | 62×3 | 84.4% | 97.8% | 255 | $0.32 |
| Agent-selected skill (OpenHarness 0.1.4) | 76 | 98.7% | 100% | 578 | $0.48 |

Exact McNemar (direct vs. skill, same 76 programs): 0 vs. 1 discordant,
p = 1.0 — statistically indistinguishable, while the skill closes direct
invocation's one failure (a canary-only leak, fixed at iteration 1 under
the skill). Also: 182/186 program–run pairs clean within 5 iterations
across 3 independent direct runs (cluster bootstrap 95% CI [95.2%, 100%]);
witness 0 false positives / canary 4/30 (13.3%) on clean controls. Full
numbers and reproduction: `docs/results.md`.

**Limitations:** one database, one task, one repair model, a corpus
dominated by review leakage; hand-written availability map; `CLEAN` scoped
to six tested times; canary flags 13% of clean programs and blocks
acceptance pending human judgment; one skill run per program, in which the
agent could read the checker's own source.

## Install for an agent

```bash
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY   # execution refuses to run with an LLM key present

.claude/skills/pit-repair/bin/pit-check --code get_features.py \
  --db-dir <directory of *.csv tables> \
  --availability-map <JSON map, see references/availability-map.md> \
  --entity-table <table> --entity-column <column> \
  --dev-seeds 2018-01-01,2018-04-01,2018-07-01 \
  --held-out-seeds 2017-10-01,2018-02-01,2018-06-01
```

The skill works on any relational database given a hand-written
availability map — it doesn't ship a default one. Claude Code / compatible
agents pick up the skill automatically from
`.claude/skills/pit-repair/SKILL.md` (candidate interface, protocol,
stopping rule). For OpenHarness, symlink it into
`<config>/skills/pit-repair` (see `pilot/d1_make_ohcfg.py`) — it does not
read `.claude/skills/` directly. The checker is self-contained
(`scripts/checker_core.py`, next to `pit_check.py`) — no dependency on any
other part of this repository. Exit codes: `0` CLEAN, `1` LEAK, `2`
candidate didn't execute, `3` safety/argument refusal, `4` iteration budget
exhausted.

# Repair-loop results

Two runs of the same repair loop over the same 76-program Olist corpus,
under the same repair model (`z-ai/glm-5.3-flash`): once invoked directly
by a script, once invoked by a coding agent that picks the `pit-repair`
skill on its own. Raw artifacts and reproduction commands are listed under
each section.

Shared setup: prediction times fixed to three development moments
(`2018-01-01`, `2018-04-01`, `2018-07-01`) and three held-out moments
(`2017-10-01`, `2018-02-01`, `2018-06-01`); a program counts as clean only
if both the witness and the canary check stay silent at all six. The unit
of analysis is the program, identified by a 16-character SHA-256 prefix of
its source (model/index pairs aren't unique on their own).

## Direct invocation

**Question.** Of the programs that leak under Olist's hand-written
availability map, how many does the repair loop bring to clean within five
iterations, with a mandatory check on held-out prediction times?

**Corpus.** 76 programs: the union of 62 executable programs from a
confirmed leak-mechanism labeling pass and 14 canary-only programs found by
fully re-checking previously-clean records. One record was excluded before
repair as a generation artifact (a truncated model response, not an
executable program with a future-reading mechanism) -- not a leak
mechanism to fix.

**Method.** The network step (sending the prompt, saving the response) and
the execution step (removing the API key from the environment, then
running the response through the sandbox) are separate processes, so an
unvetted model response is never executed with live credentials present.

**Decision rule** (registered before the run): ≥ 90% clean -- ships as a
working tool; 60-90% -- a tool with a manual-review queue; below 60% -- the
approach needs rethinking as a taxonomy problem instead. The primary number
is programs-clean over total; program×moment pairs are secondary.

**Results.**

| Metric | N/M | Share | 95% Wilson CI |
|---|---:|---:|---:|
| CLEAN@1 | 65/76 | 85.5% | [75.9%, 91.7%] |
| CLEAN@2 | 74/76 | 97.4% | [90.9%, 99.3%] |
| CLEAN@3 | 75/76 | 98.7% | [92.9%, 99.8%] |
| CLEAN@4 | 75/76 | 98.7% | [92.9%, 99.8%] |
| CLEAN@5 | **75/76** | **98.7%** | **[92.9%, 99.8%]** |

By original leak mechanism (CLEAN@5): review-join without a creation-date
filter 59/60; self-availability canary-only leaks 14/14; payments without a
filter 1/1; a mixed-up variable 1/1. All 14 canary-only programs cleared by
the second iteration (12 on the first, 2 on the second).

The one holdout: a program (author model `deepseek-v4-flash-0731`) whose
witness stays silent after five iterations while canary consistently flags
a delivery-recency feature at all six prediction times.

Required controls, run on the three development moments: a known-correct
program (witness 0/3, canary 0/3); a naive-max control (witness 2/3,
canary 3/3); a delivery-date feature (witness 0/3, canary 3/3).

91 model calls total, no API errors; 313,858 input / 152,476 output tokens;
recorded cost $0.1228. Two of 71 prior clean records stayed technically
undetermined after the run was stopped; treating both conservatively as
undetected leaks still gives a lower-bound result of 75/78 = 96.2%, above
the 90% threshold.

**Verdict:** positive -- 98.7% clears the threshold already at CLEAN@2, and
the conclusion is robust to the worst-case reading of the two undetermined
candidates.

**Conclusion.** The loop closes almost completely under a hand-written
availability map: 75 of 76 identified leaking programs are repaired and
confirmed on held-out times. Canary is a substantial part of the result,
not a decorative second check: it extended the corpus by 14 programs, and
the single remaining failure is visible only to canary. Limitations: one
repair run per program, one repair model, one dominant leak mechanism, two
technically-undetermined prior clean candidates, and a hand-written,
per-table-plus-declared-channels availability map.

## Agent-selected skill

**Question.** Does repairing through a reusable skill -- the agent decides
on its own to invoke `pit-repair`, the harness loads the instruction, the
model repairs and the skill's own code re-checks -- give the same result as
direct invocation, on the same 76 programs and the same model?

**Setup.** Skill: `.claude/skills/pit-repair/` (`SKILL.md`,
`scripts/pit_check.py`, `bin/pit-check`, `references/background.md`).
`pit_check.py` reproduces the direct-invocation checker: the same sandbox,
the same truncate / frames_equal / perturb_canary logic, the same entity
sampling, timeout, and prediction times; clean only when both levels stay
silent across all six. The repair instruction text in `SKILL.md` is the
same fixed instruction, verbatim.

**Corpus.** The same 76 programs, sourced by program id, not regenerated.
Before the run, all 76 sources were checked through `pit_check.py` without
an LLM: 74 gave `LEAK`; two didn't execute in the skill's sandbox (one
`KeyError`, one used the forbidden `locals()`) -- for those, iteration 0 is
`ERROR` and the repair loop proceeds normally from there. Direct-invocation
controls were reproduced by the same script: known-correct program witness
0/3 canary 0/3; naive-max witness 2/3 canary 3/3; delivery-date feature
witness 0/3 canary 3/3.

**Harness.** An agent harness that selects and invokes skills on its own
(OpenHarness 0.1.4), given the task description and schema plus one
paragraph: "the implementation is in `get_features.py`, check it for
temporal correctness (leakage from the future) and repair it in the same
file; report the verdict." The skill's name is never mentioned in the
prompt -- the agent picks `pit-repair` out of its own skill list. Three
things verified on the spot: the harness reads skills from its own config
directory rather than this repository's `.claude/skills/`, so the skill
was connected via a symlink; the network endpoint and transport match the
direct-invocation setup except that this harness version doesn't expose
temperature or reasoning-token controls, so requests went out on the
provider's defaults; and turn/cost accounting isn't exposed in the
harness's own output, so it was reconstructed from its session transcript
and cross-checked against the provider account's usage delta.

**Safety discipline.** The provider key lives only in an isolated
harness config (restrictive file permissions); the runner strips
`*_API_KEY` from the harness process's environment, so neither the agent's
shell tool nor `pit_check.py` can see it; `pit_check.py` itself still
refuses to start if a key is present (and in one run, the agent
independently checked for the absence of a key before starting).

**Decision rule** (registered before the run). Three conditions, all
required for a full match:

1. **Paired comparison, not just the aggregate.** An exact McNemar test on
   the 76 paired per-program outcomes (skill-run CLEAN/not-CLEAN vs.
   direct-invocation CLEAN/not-CLEAN on the same program). `p > 0.05` means
   no statistically significant difference was found.
2. **The aggregate is at least as good as the repeatability run's lower
   bound.** Skill-run CLEAN@5 ≥ 95.2% (the 95% cluster-bootstrap CI lower
   bound from the direct-invocation repeatability run).
3. **No new failure class.** Any skill failure either matches an already
   known case or gets a separate explanation in the write-up.

Cost and call count are diagnostics, not a gate: an order-of-magnitude gap
(×10 in cost or calls per program) would be recorded as a practical
limitation of packaging, not grounds for a negative verdict on its own.

**Results.**

| Metric | Skill | Direct invocation |
|---|---:|---:|
| CLEAN@1 | 75/76 = 98.7% | 65/76 = 85.5% |
| CLEAN@2 | 76/76 = 100% | 74/76 = 97.4% |
| CLEAN@3-5 | **76/76 = 100%** (Wilson 95% [95.2%, 100%]) | 75/76 = 98.7% |
| Cluster-bootstrap 95% CI for CLEAN@5 | [100%, 100%] | -- |

By mechanism (CLEAN@5, skill / direct): review-join without a filter 60/60
vs. 59/60; self-availability canary-only 14/14 vs. 14/14; payments without
a filter 1/1 vs. 1/1; mixed-up variable 1/1 vs. 1/1.

**Paired comparison.** Exact McNemar: 0 programs clean under direct
invocation and not clean under the skill, 1 program clean under the skill
and not clean under direct invocation; `p = 1.0`. The one discordant
program is the same known direct-invocation holdout: under direct
invocation the model's first repair iteration introduced a new
delivery-recency feature that then stayed stuck on canary for four more
iterations, while the original review-join leak was fixed. Under the
skill, the first iteration fixed the review join and bounded delivery
status by `seed_time` without introducing a new feature. This stays within
a known class of outcome: the repeatability run already showed that a
single review-mechanism program can get stuck on canary in one repeat out
of three -- generation variance, not a new failure mode.

**Detection at iteration 0.** 74/76 sources gave `LEAK` from the same
no-LLM script used for direct invocation; the two that didn't execute in
the skill's stricter sandbox got `ERROR` at iteration 0 and were cleared by
iteration 1. No false `CLEAN` at iteration 0.

**Iteration accounting.** In one case the agent ran the check twice on
unchanged code; the repeat run was counted as iteration 1 (the state file
records "candidate identical to previous iteration"), and the real repair
landed on iteration 2 -- the only program with `clean_at = 2`. Counting
only changed candidates gives CLEAN@1 = 76/76.

**Deletion-as-repair proxy.** Of the 74 programs with diverging columns at
iteration 0, 68 kept every diverging column in the output after repair; for
3, the output shape changed and a column-by-column comparison isn't
possible; for 3, some columns were dropped: one program dropped all 20
diverging delivery-day columns (a clean case of "repair by deletion"), one
dropped 5 of its delivery-day columns while keeping 10 review columns, one
dropped 1 of 8. This proxy wasn't computed for direct invocation, so there
is no paired comparison for it.

**Cost and calls (diagnostic, not a gate).**

| | Skill | Direct invocation |
|---|---:|---:|
| Model calls | 578 (76 sessions × 7.6 turns on average; median 6, max 18) | 91 |
| Input / output tokens | 7.98M / 0.25M | 0.31M / 0.15M |
| Cost by provider account usage delta | **$0.478** | $0.1228 |
| Wall time per program | 159s average, max 662s; ~70 min total across 3 parallel sessions | -- |

A ×3.9 gap in cost and ×6.4 in calls: noticeable, not an order of
magnitude. Input tokens grow because of the harness's system prompt on
every turn and re-sending history each turn.

**Beyond the protocol.** In 10 of 76 sessions the agent read the checker's
own source before repairing (the availability-map channel lists); in 11 it
read the checker script itself; in 12 it ran its own exploratory pandas
snippets against the database to look at date distributions. All 10
sessions that looked at the checker's internals ran longer than typical
(10-18 turns vs. 6-7) and all ended `CLEAN` (9 at iteration 1, one at
iteration 2 -- the same iteration-accounting case above). Direct invocation
gave the model no such channel: it only ever saw the code and the fixed
repair instruction. This is a real difference in run conditions, not a
protocol violation -- the skill's own steps were followed on the other 66
sessions. No session edited the state file or worked around the script's
safety refusal.

No revision of `SKILL.md` was needed between a one-program trial run and
the full run; the only edit was a paragraph clarifying that an `ERROR`
verdict should still trigger a repair attempt in the same pass.

**Verdict:** positive. All three registered conditions hold: McNemar
`p = 1.0` (0 vs. 1 discordant), CLEAN@5 = 100% ≥ 95.2%, and the one
disagreement with direct invocation is the already-known holdout case,
resolved in the skill's favor. The ×3.9 cost and ×6.4 call gap stay below
the order-of-magnitude threshold and are recorded as a practical packaging
limitation.

**Conclusion.** The "detect via witness/canary → repair → re-check on
held-out times" loop survives being packaged as a skill without losing
result quality: an agent on the same model, choosing the skill on its own,
reproduces and slightly exceeds direct invocation's CLEAN@k (75/76 already
at the first iteration, vs. 65/76 direct). Part of the k=1 gain comes from
the agent seeing the detector's column-level output and being able to read
the checker's own source -- a channel direct invocation never had, since
that run only ever saw the fixed repair text. The honest framing is
"no worse than," not "better than." Limitations: one model and one
database; the harness version used doesn't expose temperature or a
reasoning-token budget, so the comparison runs at provider defaults; the
harness's own skill directory is separate from this repository's, so
compatibility needs re-checking for other harnesses; the skill's sandbox
is stricter than the no-sandbox loader used for the no-LLM detection pass
(two sources failed to execute at iteration 0 as a result); one run per
program, no repeats.

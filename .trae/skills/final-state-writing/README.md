# Final-State Writing

English | [简体中文](README.zh-CN.md)

> **Write the end state, not the edit history.**

A Chinese-language public example captures the problem as **“Tomato and Eggs (without Dongpo Pork)”**.[^1]

The unwanted material is gone, but the history of correcting it still survives in the final text. In this project, we call that **correction residue**: text that remains only because an earlier draft was wrong.

## The problem

Final-State Writing is for a specific multi-turn writing failure mode:

```text
earlier draft
        ↓
user feedback changes the accepted state
        ↓
old material is removed
        ↓
traces of it remain in the final prose
```

Typical traces include rejected alternatives that linger, obsolete negations, correction-history narration, defensive disclaimers that no longer help the reader, explanations whose only purpose is to justify a correction, and answers to objections inherited only from earlier turns.

This is not a rule to delete every negative sentence. An absence can be useful final-state information. For example, in a deployment note, “The demo requires no external infrastructure” may be a useful operational property. The question is whether a sentence would still be useful if the final requirement had been known from the beginning. If so, it may belong in the final text.

## Before and after

```text
Draft:
We use Method A and Method B.

Feedback:
Use Method A only.

Correction residue:
We use Method A, not Method B.

Final state:
We use Method A.
```

```text
Draft:
We compare Method A, Method B, and a transformer baseline.

Later feedback:
The transformer run was invalid.

Correction residue:
The transformer baseline is excluded because it used incorrect preprocessing.

Final state:
We compare Method A and Method B.
```

## What it does

Final-State Writing treats later feedback as the authoritative definition of the current target. It rewrites the affected prose as though the accepted requirement had been known from the start, while preserving facts, tone, and useful detail.

It is intended for revisions and finalization after feedback or corrections. It is not a general humanization tool, an implementation-planning framework, or a substitute for ordinary first-draft writing.

## Validation

On a fixed correction-heavy challenge set using GPT-5.6-terra:

- Skill disabled: 20/21 outputs contained correction residue
- Skill available implicitly: 4/21 contained correction residue
- All 7 benchmark cases improved

A targeted release-regression gate achieved:

- task correctness: 12/12
- correction-residue failures under the final semantic criterion: 0/12

This was a deliberately difficult development benchmark designed around this failure mode. It is not a general error-rate estimate for all Codex or LLM writing tasks.

## Install

```bash
mkdir -p ~/.agents/skills
git clone https://github.com/intelland/final-state-writing \
  ~/.agents/skills/final-state-writing
```

Invoke it explicitly with:

```text
$final-state-writing
```

Codex may also choose the Skill implicitly when a task matches its description.

## Reference

[^1]: Adapted from a Chinese-language public example posted by [@songkeys on X](https://x.com/songkeys/status/2090416137720999992). “Tomato and Eggs (without Dongpo Pork)” is an English rendering of the cited Chinese example, not a claim that the wording was posted in English.
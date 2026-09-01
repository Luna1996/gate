---
name: "grill-me"
description: "Takes a loose idea and interviews you in rounds until you can commit to it. Invoke when user types /grill-me or wants to grill an idea/feature/decision before specifying it."
---

# grill-me

## What it does
`grill-me` takes a **loose idea** and interviews you until you can commit to it. You do not need a worked-out plan to start: producing one is what the session is for. It asks in **rounds**: each round is the whole **frontier** (every question whose prerequisites you have already settled), so you are never asked something that hinges on an answer it hasn't heard yet.

It is **stateless**. It writes no files and leaves no workspace behind. The only thing it leaves is a sharper version of the idea, in your own head.

## When to reach for it
You invoke this by typing `/grill-me`; the agent won't reach for it on its own. Start it in a **fresh conversation**, not on top of a plan you already had an agent write.

Reach for it as soon as you have an idea worth taking seriously (a feature, a product direction, a business call, a piece of writing), and long before you have worked out what it involves. Vagueness is not a reason to wait; it is the thing the session eats. If you can already specify the thing precisely, you don't need to grill it.

Leave plan mode off. Plan mode primes the agent to rush toward producing a plan, which is the opposite of staying in inquiry.

## How to ask questions (CRITICAL - use IDE native Q&A interface)
When grilling, **do not** print questions as plain text and wait. Instead, use the IDE's native Q&A interaction interface via the `AskUserQuestion` tool for every question you ask.

Rules for using `AskUserQuestion` during a grill session:
- Call `AskUserQuestion` once per question batch. A "round" may contain multiple questions; ask them as separate calls within the round, or batch independent questions into one `AskUserQuestion` call (up to 4 questions per call).
- Each question must have 2-4 mutually exclusive options. The first option can be the one you'd recommend, marked with "(Recommended)".
- Always allow the user to select "Other" - this is provided automatically, so users can type free-form answers including "I don't know".
- Keep option labels concise (1-5 words) and descriptions short.
- After the user answers, record the decision, then proceed to the next frontier question. If a user's answer opens new sub-questions, ask those next.
- Do not narrate the questions in text before calling `AskUserQuestion` - the tool already displays the question. A one-line framing sentence is fine, but the question itself goes into the tool.

This makes the session interactive, structured, and lets the user steer with one tap.

## It's a conversation, not an interview
The skill asks the questions, but **you** own the scope. That is the part people miss, and it separates a session that turns an idea into decisions from one that produces confident nonsense.

The failure mode is **passivity**: answering "agreed, agreed, agreed" for forty questions and coming out with a plan the agent wrote and you nodded at. It feels productive because it was long. Nothing was actually decided, and the result carries a certainty it hasn't earned.

Being active means steering. Push back on a question pitched beneath the fidelity you need. Say when the scope is drifting. Answer "I don't know" and mean it. This skill is built to aid an engineer, not to replace one: what comes out tracks the quality of your answers, not the number of questions asked.

The opposite error is real but rarer: staying in the interview so long you never reach code.

## Grillable and ungrillable
Some questions can be answered by talking. Others can't, and no amount of grilling will get you there.

"One long form or three pages?" and "how should this interaction feel?" are **ungrillable**: they need something to react to. When you hit one, stop grilling. Build the throwaway version with prototype, look at it, then come back and answer in one line.

Talking your way through an ungrillable question is where sessions balloon. The agent keeps rephrasing, you keep guessing, and the scope grows to fill the uncertainty.

## It's working if
- You disagree with something. A session with no pushback from you is a session you didn't need.
- Questions arrive in a few rounds rather than one long drip, and later rounds clearly build on what you said earlier.
- You end up somewhere you didn't expect, because a question surfaced a decision you had been making implicitly.
- At the end you could defend each choice to someone who wasn't there.

## Session flow
1. Ask the user what idea they want to grill. If they gave it in the opening message, use that.
2. Run the grilling in rounds. At the start of each round, briefly (one or two sentences) summarize the decisions locked in so far, then ask the frontier questions via `AskUserQuestion`.
3. Keep going until the frontier is empty: every branch visited, nothing left silently assumed.
4. When done, output a tight summary of every decision the user committed to, so they can hand the same conversation to `to-spec` or start building.

## Common questions
**How many questions should I expect, and how do I know when it ends?**
Count rounds, not questions. Forty-six questions across four rounds is an ordinary session. It ends when the frontier is empty.

**What if I genuinely don't know the answer?**
Say so (pick "Other" and type "I don't know"). A question you can't answer is usually a sign to prototype rather than to guess.

**Do I start a fresh session before writing the spec?**
No. The value of the session is the context you just built. Hand the same conversation straight to `to-spec`.

## Where it fits
`grill-me` is a **standalone you can run anywhere, on anything**. Being stateless is what makes it portable: no repo, no workspace, no setup, and no assumption that the idea is even about software. People point it at business decisions, at writing, at what to do next: anything that won't sit still in their head.

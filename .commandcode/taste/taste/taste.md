# Taste
- Prefers changes to be left uncommitted at the end of a task (working-tree diff for their own review), not auto-committed. Confidence: 0.6
- Prefers NOT to reformat or lint-fix pre-existing code outside the scope of the change: when `cargo fmt --check` / `clippy -D warnings` are already red on untouched files, keep the diff focused instead of doing a repo-wide reformat. Confidence: 0.7
- Wants their local environment/config restored after benchmarking (e.g. revert `video/vsync` back to its original value once measurement is done). Confidence: 0.6
- Documents performance investigations as dated rounds in `docs/editable-gigavoxel.md`, following the existing table format and methodology. Confidence: 0.65
- Benchmark methodology: same-session A/B at the same camera position and world state; compare GPU per-pass ms (`gate_dda_trace` etc.) rather than fps, because vsync masks framerate; use `GATE_BENCH=static` + `--features profile` and take means over a steady 2s window. Confidence: 0.65
- Prefers temporary diagnostic probes/instruments to be gated behind a `const` (zero cost when disabled, e.g. `LOD_DIAG = 0`) and removed or disabled after measurement, rather than deleting the diagnostic machinery. Confidence: 0.55
- Communicates in Chinese (Simplified) — including terse one-line bug reports — and expects replies (and doc/code comments) in Chinese too. Confidence: 0.6
- Reports visual/rendering bugs by attaching a screenshot and stating whether the symptom disappears with the feature disabled (e.g. "关闭GI时没有"), i.e. expects feature-toggle isolation as the first triage step. Confidence: 0.45
qqqqqq
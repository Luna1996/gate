"""Convert downloaded YouTube .vtt subtitle files to plain text (strip timestamps,
dedupe rolling caption repeats), then remove the .vtt. Reduces tokens when the
transcripts are later read for summarization."""
import re
import pathlib

D = pathlib.Path(__file__).parent / "transcripts"

for vtt in D.glob("*.vtt"):
  lines, seen = [], set()
  for raw in vtt.read_text(encoding="utf-8", errors="replace").splitlines():
    line = raw.strip()
    if (
      not line
      or line == "WEBVTT"
      or line.startswith(("Kind:", "Language:", "NOTE", "Identifier:"))
      or "-->" in line
      or re.match(r"^\d+$", line)
      or re.match(r"^[\d.,:]{2,}(\s*-->\s*[\d.,:]+)?$", line)
    ):
      continue
    line = re.sub(r"<[^>]+>", "", line).strip()
    if not line or line in seen:
      continue
    seen.add(line)
    lines.append(line)
  vtt.with_suffix(".txt").write_text(" ".join(lines), encoding="utf-8")
  vtt.unlink()
  print("stripped:", vtt.stem)

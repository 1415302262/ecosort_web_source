#!/usr/bin/env python3
"""Apply a restrained 9 pt style to Pandoc longtables.

Column labels are deliberately concise in the source Markdown. Keeping native
``longtable`` avoids the pathological over-scaling that can occur when xeCJK
line-break glue is measured inside an ``adjustbox``.
"""

from pathlib import Path


path = Path("ecosort_experiment_report.tex")
lines = path.read_text(encoding="utf-8").splitlines()
output = []
inside_table = False

for line in lines:
    if line.startswith(r"\begin{longtable}"):
        if inside_table:
            raise RuntimeError("Nested longtable is not supported")
        output.extend([
            r"\begingroup",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            line,
        ])
        inside_table = True
    elif inside_table and line == r"\end{longtable}":
        output.extend([
            line,
            r"\endgroup",
        ])
        inside_table = False
    else:
        output.append(line)

if inside_table:
    raise RuntimeError("Unclosed longtable")

path.write_text("\n".join(output) + "\n", encoding="utf-8")

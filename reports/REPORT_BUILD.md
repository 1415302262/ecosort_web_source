# EcoSort 报告构建说明

工作目录：仓库的 `reports/` 目录。

```bash
conda run -n cptac pandoc \
  ecosort_experiment_report.md \
  --from=gfm --to=latex --standalone --toc --no-highlight \
  --resource-path=. \
  --include-in-header=ecosort_report_header.tex \
  -V papersize=a4 -V geometry:margin=1.75cm \
  -V mainfont='Noto Serif CJK SC' \
  -V sansfont='Noto Sans CJK SC' \
  -V monofont='Noto Sans Mono CJK SC' \
  -V colorlinks=false \
  -o ecosort_experiment_report.tex

conda run -n cptac python fix_ecosort_latex.py

conda run -n cptac tectonic \
  --keep-logs --keep-intermediates ecosort_experiment_report.tex
```

交付文件：

- `ecosort_experiment_report.md`
- `ecosort_experiment_report.tex`
- `ecosort_experiment_report.pdf`
- `ecosort_experiment_report.log`
- `ecosort_report_header.tex`
- `EXPERIMENT_REPRO.md`（完整实验命令与 VLM 预算账本约束）

路由图表源文件位于 `ecosort_experiment_results/analysis/`。最终 Cascade 统计主图由
`scripts/plot_ecosort_cascade_portfolio.py` 生成至 `docs/portfolio/assets/`，交付
PNG/PDF/SVG/TIFF；SVG 保留可编辑文字。图表机器数据来自
`docs/portfolio/data/`，不包含 API key、原始 provider cache、账本或主机绝对路径。

完整 QA 结果、最终文件哈希和逐页检查记录见 `REPORT_QA.md`。

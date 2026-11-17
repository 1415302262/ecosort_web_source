# EcoSort 报告 QA

检查日期：2026-08-11

## 结果

- PDF：10 页，A4，1.75 cm 页边距。
- 编译：在 `cptac` 环境使用 Tectonic 完成；日志中无 `overfull`、`missing character`、invalid UTF-8、undefined control sequence 或字体替代警告。
- 文字：中文可搜索、可复制；共提取 17,433 个字符，并确认最终 Cascade Macro-F1 `96.337%` 可检索、旧的“实际调用 0 次”状态不存在。
- 字体：Noto Serif CJK / Noto Sans CJK 共 3 个 Type0 子集字体，全部嵌入；无 Type3 字体。
- 目录：已生成目录和 PDF bookmarks；正文采用手工章节号，没有重复自动编号。
- 视觉：逐页渲染检查 10 页；表格、3 张 routing trade-off 图、confusion matrix、Cascade 统计主图与代码块均无裁切、越界或比例失真。
- 图表：Cascade 主图交付 PNG/PDF/SVG/TIFF；SVG 保留文字节点，Nature 严格验证 14/14 通过、0 warning、0 fail。
- 统计：固定 911 样本上完成 10,000 次 paired bootstrap 与 exact McNemar test；Macro-F1 差值 95% CI 为 `[+3.328, +7.275]` pp，McNemar `p=7.054e-11`。
- VLM 安全回归：12/12 通过，其中 8 项覆盖 reservation/并发预算/partial 指标，4 项覆盖 Seed 响应解析、usage 提取和 Ark 图像像素上限预处理。
- 隐私：`docs/portfolio/` 未发现 API key、Bearer token、provider response ID、attempt ID、主机绝对路径或原始 cache/ledger。

## SHA256

- PDF：`03b9549f53fef1d5eac5fc740ded0c996b1041bc26290db18a7b93535464e2c6`
- LaTeX：`a564e5d46abcff758c1579574728c730618f012b23d2c733079b38a11ad566e0`
- Markdown：`2e47b20458e428a2fb7d6e8424d73d834128282dbff7bc2c69f2508ada39018e`

## 字体资源

构建机字体族：`Noto Serif CJK SC`、`Noto Sans CJK SC`、`Noto Sans Mono CJK SC`。Tectonic 会提示字体来自绝对系统路径；该提示不影响交付 PDF，因为最终字体已经嵌入。跨机器重建时需安装同名字体，或在构建命令中替换为可用 CJK 字体。

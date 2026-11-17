# EcoSort 求职材料

> 使用状态：**最终数字版。** 本地模型、Confidence 路由和 142 次 Doubao-Seed-2.0-Lite 真实 cascade 均已完成；下列数字来自排除确认精确重复后的固定 911 样本测试集。

## A. 一句话项目定位

面向边云协同场景的细粒度多模态分类级联评测框架：以本地 EfficientNet 完成低延迟主推理，通过不确定性估计自适应路由困难样本，并以有界、可缓存的 VLM fallback 评估效果—延迟—调用量 trade-off。

## B. 推荐简历标题

1. 大模型应用算法：**基于不确定性路由的 Local Model–VLM 级联评测框架**
2. 多模态应用：**面向细粒度垃圾识别的视觉模型与 VLM Fallback 评测**
3. CV 算法：**EfficientNet-B3 细粒度分类、校准路由与 Bad Case 闭环**

## C. 两条简历 Bullet

- 基于EfficientNet-B3构建14类细粒度垃圾识别模型，完成跨数据集重复审计与不确定性消融，排重测试集取得91.22%准确率、91.21% Macro-F1。
- 设计Confidence路由与Doubao-Seed-2.0-Lite级联，仅调用15.59%样本将Macro-F1提升至96.34%，净修正45例，系统P95延迟4.14秒。

## D. 一版极简简历

**EcoSort｜Local–VLM 自适应级联评测框架**：EfficientNet-B3 取得 91.21% Macro-F1；以 Confidence 路由 15.59% 困难样本至 Doubao-Seed-2.0-Lite，将整体 Macro-F1 提升至 96.34%，净修正 45 例。

## E. 面试材料

### 1. 为什么选择 EfficientNet-B3？

本轮不是重新做 backbone 搜索，而是复用仓库已选出的 V6 14 类 champion，保持本地模型固定，隔离研究 uncertainty routing 与 VLM fallback。B3 在当前 L20 单图实验中有 91.21% Macro-F1 和 65.62 ms end-to-end P95；没有新的 B0/B3 公平实验，因此只说“当前最佳已存 checkpoint”，不声称 B3 理论最优。训练期原 validation 含 1 个 train–validation 精确重复，可能影响 epoch 选择但目前无法回溯量化。

### 2. 为什么需要 VLM fallback？

本地模型 80 个错误中有大量对象语义与处置规则冲突，例如电池/电子垃圾、药品/塑料。真实实验中 VLM 修正 49 个 routed local error，仅引入 4 个新错误，说明把开放世界对象语义用于困难样本具有净收益。

### 3. Confidence、Margin、Entropy 有什么区别？

Confidence 只看 Top-1 概率；Margin 看 Top-1 与 Top-2 是否接近；Entropy 看整个类别分布是否分散。理论关注点不同，但在当前 checkpoint 的默认阈值下，A/B/C 拒绝集合完全相同，后两者没有实测增量。

### 4. Threshold 如何确定？

只在剔除确认重复后的 895 个 validation 样本上 sweep，再把选定阈值一次性应用到 911 个 test。0.85 在 validation 的候选调用率为 12.29%，错误召回 69.81%；test 上对应 15.59% 和 70.00%。没有用 test 反向调参。

### 5. 为什么不是所有样本直接调用 VLM？

本地模型已能正确处理 831/911 个样本，且单图 end-to-end P95 为 65.62 ms。全量调用会放大网络延迟、请求量与外部依赖；路由目标是把 VLM 集中到错误密度更高的子集，而不是默认云端全覆盖。

### 6. 如何衡量效果和成本 trade-off？

效果看 Overall Accuracy / Macro-F1 / Macro-Recall 和 VLM 修正、引入错误数；选择机制看 coverage、error-detection precision/recall；效率看 actual call rate、Mean/P50/P95 cascade latency 和 provider usage。没有真实价格配置时只报告 request/usage，不换算金额。

### 7. VLM 是否真的提高效果？

是。固定 Confidence-only 0.85 后真实调用 142/911 个样本，Macro-F1 从 91.21% 提升到 96.34%，Accuracy 从 91.22% 提升到 96.16%；修正 49 个错误、引入 4 个错误，净修正 45 个。代价是全链路 P95 从本地 65.62 ms 增至 4.14 s。

### 8. 当前系统最大的 Bad Case 是什么？

本轮复核了全部 7 个未修正错误、4 个 VLM 新增错误和 2 个 failure。主要问题是目录级标签与类别边界：多个 `tissue` 图像实际是干净废纸，`tea` 商品图同时包含包装和茶饮，`plastic_toy` 图含多种材质；另有 24/80 个本地错误在 confidence≥0.85 下不会触发 fallback。

## F. 是否替换药物脂溶性预测项目

**建议替换，尤其是投递“大模型应用算法 / AI 应用算法 / 多模态应用”岗位时。** EcoSort 已形成可审计的 Local Model → Uncertainty → Adaptive Routing → VLM Fallback → 效果/延迟/usage 闭环，且 15.59% 调用率带来 +5.13 个百分点 Macro-F1、净修正 45 例，比单一回归预测项目更贴近岗位关键词和系统设计追问。若投纯药物计算或计算化学岗位，则保留药物脂溶性项目更匹配。

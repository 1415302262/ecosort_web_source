# EcoSort 自动化与部署边界

本文区分两种交付：不产生云模型费用的静态求职展示，以及需要真实模型权重的在线推理服务。二者互不冒充。

## 1. 自动化闭环

本地与 CI 使用同一个无网络发布门禁：

```bash
conda run -n cptac bash scripts/release_check.sh
```

门禁会完成 Python 编译检查、部署脚本语法检查、20 项级联/缓存/预算回归测试、公开产物指标与 SHA256 校验、隐私扫描和 `git diff --check`。CI 不读取 API Key，不访问 VLM，也不会消耗免费额度。

合并到 `main` 后，GitHub Pages 工作流自动发布 `docs/portfolio/` 与实验报告 PDF。该展示完全静态，不需要 checkpoint、GPU 或云 API。

## 2. 在线推理的一致性门禁

线上服务只有同时满足以下条件，`/health` 才报告与已审计实验一致：

- checkpoint SHA256 为 `9ba817cdf0712b597e1eb322d722e130707d60d6733cf46d8d87149456946234`；
- 类别顺序与固定 14 类完全一致；
- 输入为 224 像素 letterbox；
- 路由规则为 `confidence < 0.85`。
- Seed 模型名、实验 prompt version 与规范化 42→14 mapping 签名一致。

仓库中的 134 字节 Git LFS pointer 不是模型权重。部署时必须拉取 LFS 对象，或把真实权重放在受控对象存储，并通过 Render secret `ECOSORT_MODEL_URL` 下载。启动脚本会先校验大小、LFS pointer 和 SHA256；校验失败时只保留静态页面与健康检查，`/predict` 不可用。

## 3. VLM 默认不调用

在线 VLM 的默认配置是：

```text
ECOSORT_VLM_EXECUTE=0
ECOSORT_VLM_MAX_CALLS=0
```

因此即使误部署也不会产生调用。确需开启时，必须由部署平台的 secret 管理器设置：

```text
ECOSORT_ARK_API_KEY=<secret>
ECOSORT_ARK_MODEL=doubao-seed-2-0-lite-260428
ECOSORT_VLM_EXECUTE=1
ECOSORT_VLM_MAX_CALLS=<明确正整数>
ECOSORT_VLM_STATE_DIR=<持久化目录>
```

每次请求在网络调用前写入持久化 ledger；相同图片、模型与 prompt 命中 cache 时不重复调用。失败或未映射输出保留本地预测。若托管平台没有持久磁盘，不应开启 VLM，因为重启会丢失预算账本。

## 4. 可观测性与安全默认值

- `/health`：模型是否加载、checkpoint hash、一致性门禁与 provider 配置状态；不返回 Key。
- `/metrics`：本地覆盖率、路由率、真实 provider attempt rate、Mean/P50/P95 延迟和 token 累计；不返回图片、cache key 或请求 ID。
- `ECOSORT_MAX_BATCH_SIZE` 默认 16，`ECOSORT_MAX_REQUEST_BYTES` 默认 12 MiB。
- `/reload_model` 默认关闭；只有显式设置 `ECOSORT_ALLOW_MODEL_RELOAD=1` 才启用。
- CORS 默认兼容演示环境，正式域名应通过 `ECOSORT_CORS_ORIGINS` 收紧。

## 5. 推荐的低人工维护方式

1. 日常只维护代码、公开聚合结果与报告，不提交 raw cache、ledger、API Key、原图和主机绝对路径。
2. PR 必须通过 `portfolio-ci`；由人工确认指标语义与账单边界后再合并。
3. 求职展示优先使用 GitHub Pages；它展示已经完成的 142 次真实调用实验，不会重复花费。
4. 在线 demo 默认只跑本地模型。只有确有面试演示需要且有持久预算账本时，才临时开启 Seed fallback。

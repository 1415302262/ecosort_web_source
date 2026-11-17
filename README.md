# EcoSort · 垃圾分类与纠错管理系统

建议本科毕设题目：**基于置信度路由与视觉语言模型协同的垃圾分类系统设计与实现**。

EcoSort 将本地 EfficientNet-B3、低置信度路由和可选视觉语言模型复核接入一个可追溯的 Web 系统，提供单图/批量识别、历史记录、纠错审核、统计和 CSV 导出。正式运行代码以本仓库 `main` 为准；模型、原始数据与私有运行状态独立管理。

## 功能

- 14 类细分类与四大类项目映射，统一 224 letterbox 和 `<0.85` 置信度路由。
- 单图/批量识别、参数与图片校验、逐项异常反馈。
- SQLite 记录、原图像素持久化、分页筛选、导出；按签名会话隔离访客记录。
- 用户纠错 → 管理员采纳/驳回 → 审核事件 → 已采纳纠错导出；原始模型输出始终保留。
- 云复核采用持久预算、缓存和失败保留本地预测；默认不开启付费调用。
- 固定实验结果页与业务统计分开；CI 检查业务流程、权限、级联与数据隔离。

## 已有固定实验

下表来自 2026 年 8 月固定实验，主测试集在确认精确重复排除后为 911 张。新增业务记录不改变这组实验结论。

| 方法 | Accuracy | Macro-F1 | 云调用率 | 全链路 P95 |
|---|---:|---:|---:|---:|
| 本地 EfficientNet-B3 | 91.218% | 91.208% | 0% | 65.6 ms |
| 本地 + 置信度路由 + VLM | 96.158% | 96.337% | 15.587% | 4.14 s |

142 次真实请求中，140 次成功映射、2 次失败保留本地；修正 49 例、引入 4 例，净修正 45 例。速度来自 NVIDIA L20，不能当作端侧实测。数据重复、标签边界、消融及其他限制见[原始实验报告](reports/ecosort_experiment_report.md)。

[静态实验页面](docs/portfolio/index.html) · [报告 PDF](reports/ecosort_experiment_report.pdf)

## 启动与访问

```bash
pip install -r requirements.txt
# 下载可信真实权重；仓库中的 Git LFS pointer 不是权重
export ECOSORT_MODEL_PATH=/absolute/path/to/best_model.pth
export ECOSORT_STATE_DIR=/private/persistent/state
export ECOSORT_ADMIN_TOKEN_FILE=/private/admin.token
export ECOSORT_VLM_EXECUTE=0
export ECOSORT_VLM_MAX_CALLS=0
python -m backend.wsgi
```

访问 `http://127.0.0.1:5000`。管理员口令至少 24 个字符；未配置时访客功能可用、管理员审核不可登录。生产使用 Linux 用户服务与 SSH 隧道，详见[部署与备份](docs/DEPLOYMENT.md)。本机 SSH 访问脚本：`scripts/open_server.ps1`。

网页与 API 采用同源会话。脚本调用须先获取 `/api/session` 的 Cookie 与 CSRF，再调用写入接口，详见 [API 文档](docs/API.md)。

## 测试

```bash
pip install -r requirements-ci.txt
bash scripts/release_check.sh
# 在 ML 环境验证数据加载器（不下载模型、不需要 GPU）
pytest -q tests/test_dataset_contract.py
```

Windows 开发测试子集和现场演示步骤见[验收清单](docs/ACCEPTANCE.md)。测试不发送真实 VLM 请求；Linux CI 另运行进程并发、调用预算与崩溃恢复回归。

## 论文与实验复现

- [选题、需求、架构与论文提纲](docs/THESIS_PLAN.md)
- [固定实验复现与新增数据构建](docs/REPRODUCIBILITY.md)
- [固定实验原复现记录](reports/EXPERIMENT_REPRO.md)
- [部署、更新、回滚与数据备份](docs/DEPLOYMENT.md)

训练依赖见 `requirements-training.txt`，V6 配置为 `configs/train_ecosort_v6_14class.yaml`。新建数据集使用 `scripts/build_dataset_from_mapping.py`：先排重再分层切分，不覆盖旧目录。缺少独立 test 目录或遇到损坏图片时明确报错，不静默复用训练数据或填充白图。

## 目录

```text
backend/             Web 工厂、持久化、ML 适配与前端
src/                 数据加载、模型、训练、级联策略与 VLM 适配
scripts/             数据审计、实验分析、发布、部署与备份
experiments/         训练与评估入口
tests/               业务、权限、数据与级联回归
configs/             模型、训练与类别映射配置
docs/portfolio/      已审计的公开聚合实验成果
reports/             既有实验报告与复现说明
runtime/             本地私有运行状态（Git 忽略）
```

原始数据、原图、模型、API Key、管理员口令、SQLite 和私有调用账本不得作为普通 Git 文件发布。历史研究工作区保留其数据与未提交代码，正式服务使用独立发布目录。

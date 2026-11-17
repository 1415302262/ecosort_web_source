# 正式运行、更新与备份

正式版本以 GitHub `main` 的提交 SHA 为准。原有研究工作区可能含数据、权重、训练日志和未提交研究代码；保留为研究存档，不对其执行 `reset --hard`、`clean` 或整目录覆盖。正式运行目录单独检出经测试的发布提交。

## 启动

Python 3.11 或 3.12，安装 `requirements.txt`。使用已审计的真实权重，Git LFS pointer 不是可运行权重。工作目录为仓库根：

```bash
export ECOSORT_MODEL_PATH=/absolute/path/to/trusted/best_model.pth
export ECOSORT_STATE_DIR=/absolute/path/to/private/persistent/state
export ECOSORT_ADMIN_TOKEN_FILE=/absolute/path/to/private/admin.token
export ECOSORT_VLM_EXECUTE=0
export ECOSORT_VLM_MAX_CALLS=0
export ECOSORT_RELEASE=$(git rev-parse HEAD)
python -m backend.wsgi
```

管理员口令至少 24 个字符，建议随机生成；仅服务器管理者可读。状态目录需长期保留，包含 SQLite、签名会话密钥及上传图片。默认不配置管理员口令时，审核不可登录。

Linux 用户服务安装器（会创建自己的 unit 和新配置；已有同名文件时拒绝覆盖）：

```bash
PYTHONPATH=. python scripts/install_server.py \
  --python /absolute/path/to/ml-environment/bin/python \
  --checkpoint /absolute/path/to/trusted/best_model.pth
systemctl --user status ecosort.service
```

服务使用 CPU、单 worker、4 个 HTTP 线程，绑定 `127.0.0.1:5187`，默认禁用云执行，避免占用研究 GPU或新增 API 费用。SQLite 使用默认 DELETE journal；优先将状态目录放在本机持久磁盘，若使用网络盘需保证可靠文件锁，并限定单主机运行；多主机部署应迁移至服务型数据库。

Windows 用户通过现有 SSH 配置访问：

```powershell
powershell -File scripts/open_server.ps1
# 如需登录审核台，在本人终端单独读取口令
powershell -File scripts/open_server.ps1 -ShowAdminToken
```

访问 `http://127.0.0.1:5187`。保持 SSH 控制台打开，Ctrl+C 关闭隧道，不停止服务器应用。也可直接手动运行 `ssh -L 5187:127.0.0.1:5187 -N oci-public` 后打开网址。

此演示默认仅通过 SSH 隧道访问。若公开部署，需另配 HTTPS、Secure Cookie、入口流量与存储配额、账号生命周期及数据留存政策；当前访客会话不是跨设备账号。Render 的免费临时磁盘不能满足识别记录持久化，需配置持久磁盘或其他数据库。Hugging Face adapter 仅为模型展示，不替代本项目完整 Web 业务系统。

## 版本更新与回滚

1. 记录当前 `git rev-parse HEAD`；确认正式运行目录 `git status --porcelain` 为空。
2. 拉取并检出已经通过 CI 的目标提交，禁止覆写脏研究工作区。
3. 将私有 `app.env` 中 `ECOSORT_RELEASE` 更新为目标 SHA，其他口令和状态路径保持不变。
4. `systemctl --user restart ecosort.service`，检查 `/health` 的 release、模型加载和审计一致性。
5. 运行识别与记录读取验收。若失败，检出上一步 SHA、恢复 release 值、重启；保留用户数据。遇到未来数据库 schema 变更需按迁移说明操作，不盲目降级。

## 数据备份

```bash
python scripts/backup_state.py --state-dir /path/to/state --output /new/path/to/private/backup
```

备份通过 SQLite backup API 创建数据库快照，再复制不可变图片和会话密钥。目标必须为新目录；源目录不被改写。备份包含私有图片及会话密钥，不得提交公开仓库。管理员口令文件需另行安全备份。恢复时停止本应用服务，将备份放到新的状态目录，更新 `ECOSORT_STATE_DIR` 后启动并复查。

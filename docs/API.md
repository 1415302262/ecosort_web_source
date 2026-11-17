# Web API

所有写入请求先 `GET /api/session` 获取签名会话 Cookie 与 `csrf_token`，后续携带 Cookie、`X-CSRF-Token` 和 `Content-Type: application/json`。网页已自动完成。服务采用同源访问，不开启跨站凭证调用。

| 方法与路径 | 用途 | 权限 |
|---|---|---|
| GET `/health` | 模型就绪、版本与审计一致性状态 | 公开 |
| GET `/model_info` | 固定类别、类别映射与已载入模型信息 | 公开 |
| GET `/api/session` | CSRF、角色与标签字典 | 当前会话 |
| POST `/predict` | `{image, format:"base64", filename}`，返回预测和 `record_id` | 当前会话 |
| POST `/batch_predict` | `{images:[base64,...]}`，逐项返回预测或 4xx 错误 | 当前会话 |
| GET `/api/records` | `page`、`limit`、`category`、`scope=mine` | 当前会话；`scope=all` 仅管理员 |
| GET `/api/records/<id>` | 原始输出与纠错状态 | 所有者或管理员 |
| GET `/api/records/<id>/image` | 去除上传 EXIF 的 JPEG 像素 | 所有者或管理员 |
| POST `/api/records/<id>/feedback` | `{expected_label, comment}` | 所有者或管理员 |
| GET `/api/stats` | 业务数量与分布 | 当前会话；`scope=all` 仅管理员 |
| GET `/api/records/export.csv` | 最多 10000 条记录的 CSV | 当前会话；`scope=all` 仅管理员 |
| POST `/api/admin/login` | `{token}`，服务器签名会话标记角色 | 部署配置的管理员口令 |
| POST `/api/admin/logout` | 清除当前会话管理员角色 | 当前会话 |
| GET `/api/reviews` | `status=pending/approved/rejected`、分页 | 管理员 |
| POST `/api/reviews/<id>/decision` | `{status:"approved"/"rejected",note,version}` | 管理员 |
| GET `/api/reviews/export.csv` | 已采纳纠错，保留原始预测与建议标签 | 管理员 |
| GET `/metrics` | 运行级联统计，不等同于测试集成绩 | 管理员 |

单张支持静态 JPG、PNG、WEBP；HTTP 请求默认上限 12 MiB，解码图像上限 1600 万像素，网页单文件上限 8 MiB。每批默认最多 16 张。分页 limit=1–100。

状态码：400 参数或图片无效；401 管理员口令错误；403 无权限、CSRF 或跨站写入；404 资源不存在或属于其他会话；409 重复纠错或审核已处理；413 请求、批次或图像像素超限；415 不支持的格式；503 模型未就绪。500 返回通用消息，具体异常仅写服务日志。

预测中的 `confidence` 为本地模型置信度；最终类别由 `class_name` / `final_prediction` 表示。VLM 成功与否、是否真实尝试、是否命中缓存记录在 `routing`。审核结果单独保存，不修改原始预测；同一记录仅接受一次纠错，一旦审核不允许再次改写。

兼容性：旧 `/detect_objects` 保留为启发式候选区域分类；不等同于目标检测。`/reload_model` 始终关闭，模型更新走受控部署。

签名会话持久保存在 HttpOnly、SameSite=Lax Cookie 中；会话密钥保存在状态目录，重启不变。HTTPS 反向代理部署应配置 `ECOSORT_COOKIE_SECURE=1`。管理员口令至少 24 个字符，生产建议使用部署脚本随机生成值，存储于权限受控文件，禁止提交 Git。

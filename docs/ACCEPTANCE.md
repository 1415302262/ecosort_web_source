# 毕设演示与验收

本文件是工程验收清单，不是尚未提供的学院评分标准。

## 自动测试

```bash
pip install -r requirements-ci.txt
bash scripts/release_check.sh

# 在已安装 torch / torchvision 的 ML 环境执行
pytest -q tests/test_dataset_contract.py
```

CI 分为无网络业务/级联回归和 CPU 数据加载合同两项。测试使用临时目录和合成图像；云提供方使用 fake adapter，不发送付费请求。旧 Gemini 实验脚本使用 Linux SIGALRM 超时，其进程级测试在 Linux CI/服务器执行。Windows 可先运行：

```powershell
python -m pytest tests/test_web_workflow.py tests/test_dataset_builder.py tests/test_cascade_runtime.py tests/test_seed_vlm_unit.py
```

## 真实模型验收（无云费用）

在 ML 环境指定可信 checkpoint，确认 SHA256、14 类顺序、224 letterbox 和 0.85 阈值一致；设置 `ECOSORT_VLM_EXECUTE=0`、`ECOSORT_VLM_MAX_CALLS=0`。启动后检查 `/health` 的 `model_loaded` 与 `runtime.audited_parity`。真实图片识别验证“模型能运行”；不以少量演示图片推导准确率。

1. 上传一张真实垃圾图片，观察分类、置信度、复核路径和记录。
2. 上传多张图；分别查看每张结果和原图。
3. 查看历史、分页、分类筛选并导出 CSV。
4. 为其中一条记录提交纠错；用管理员登录查看原图、采纳或驳回。
5. 重新查看记录：原模型预测仍保留，审核意见另列。
6. 查看业务统计与已采纳纠错导出；固定实验成绩在独立结果页。
7. 重启服务后复查同一会话记录及图片；新访客无法读取前一个访客的图片。
8. 云执行未开启或异常时保留本地输出，不宣称本轮调用了 VLM。

## 论文测试章需要保留的证据

- 提交 SHA、依赖环境、执行日期与完整测试结果。
- 需求编号与对应测试用例、预期和实际行为。
- 正常识别、异常输入、权限隔离、纠错审核和重启截图。
- 固定测试集报告、逐类指标、混淆矩阵、路由消融、调用量与延迟取舍。
- 数据来源说明、已知重复与近重复风险、失败案例和系统边界。

不应以“审核已采纳比例”冒充模型准确率；不把单次服务器速度写成端侧实测；不把 CI 绿色状态替代现场演示验收。

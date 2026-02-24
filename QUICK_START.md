#!/bin/bash
# EcoSort 快速启动脚本

echo "🚀 EcoSort 智能垃圾分类系统"
echo "=============================="
echo ""

# 检查环境
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 Conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 激活环境
echo "📦 激活 Conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecosort 2>/dev/null || {
    echo "⚠️  环境不存在，正在创建..."
    conda env create -f environment.yml
    conda activate ecosort
}

echo "✅ 环境已激活"
echo ""

# 菜单
echo "请选择操作:"
echo "1) 运行系统测试"
echo "2) 创建测试数据"
echo "3) 训练模型"
echo "4) 启动 API 服务"
echo "5) 查看项目文档"
echo "6) 退出"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "🧪 运行系统测试..."
        python scripts/test_system.py
        ;;
    2)
        echo ""
        echo "📊 创建测试数据..."
        read -p "每类样本数量 (默认100): " num_samples
        num_samples=${num_samples:-100}
        python scripts/create_dummy_data.py --num-samples $num_samples
        ;;
    3)
        echo ""
        echo "🎯 训练模型..."
        read -p "配置文件 (默认: configs/baseline_resnet50.yaml): " config
        config=${config:-configs/baseline_resnet50.yaml}
        read -p "数据目录 (默认: data/raw): " data_root
        data_root=${data_root:-data/raw}

        if [ ! -d "$data_root" ]; then
            echo "❌ 数据目录不存在: $data_root"
            echo "请先运行选项 2 创建测试数据"
            exit 1
        fi

        python experiments/train_baseline.py \
            --config $config \
            --data-root $data_root \
            --no-wandb
        ;;
    4)
        echo ""
        echo "🌐 启动 API 服务..."
        read -p "模型路径 (默认: checkpoints/baseline_resnet50/best_model.pth): " model_path
        model_path=${model_path:-checkpoints/baseline_resnet50/best_model.pth}

        if [ ! -f "$model_path" ]; then
            echo "❌ 模型文件不存在: $model_path"
            echo "请先运行选项 3 训练模型"
            exit 1
        fi

        python backend/app.py \
            --model-path $model_path \
            --model-type resnet \
            --port 5000
        ;;
    5)
        echo ""
        echo "📚 项目文档:"
        echo "  - README.md           项目概述"
        echo "  - COMPLETION_REPORT.md 完成报告"
        echo "  - docs/TECHNICAL_GUIDE.md  技术指南"
        echo "  - docs/USER_GUIDE.md      使用指南"
        echo "  - docs/PROJECT_SUMMARY.md 项目总结"
        echo ""
        read -p "按 Enter 键打开 README.md..."
        less README.md
        ;;
    6)
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "✅ 操作完成!"

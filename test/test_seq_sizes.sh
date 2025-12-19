#!/bin/bash
# 针对 seq_size=4096 配置的快速测试脚本

echo "=========================================="
echo "针对您的模型配置的测试"
echo "=========================================="
echo ""
echo "您的配置: hidden=2048, seq_size=4096, blocks=16"
echo "问题: seq_size=4096 太大，导致激活内存超限"
echo ""
echo "将测试三种解决方案:"
echo "  1. 原配置 + 小batch_size"
echo "  2. seq_size=2048 + 中等batch_size"
echo "  3. seq_size=1024 + 大batch_size"
echo ""

export PYTHONPATH=.
mkdir -p seq_test_results

# 测试1: 原配置，batch_size=8
echo "=========================================="
echo "测试 1/3: 原配置 (seq=4096) + batch_size=8"
echo "=========================================="
timeout 300 ./bin/calculon llm-optimal-execution \
  models/gpt3-1B.json \
  8 8 float32 \
  scripts/micro_benchmarks/nvidia/system.json \
  seq_test_results/output_seq4096_bs8.json \
  -m -t 1

if [ $? -eq 0 ] && [ -f "seq_test_results/output_seq4096_bs8.json" ]; then
    echo "✓ 成功!"
    sr=$(python -c "import json; print(json.load(open('seq_test_results/output_seq4096_bs8.json'))['0']['stats']['sample_rate'])" 2>/dev/null)
    echo "  样本率: $sr samples/sec"
    TEST1_SUCCESS=1
else
    echo "✗ 失败"
    TEST1_SUCCESS=0
fi

echo ""
sleep 2

# 测试2: seq_size=2048，batch_size=32
echo "=========================================="
echo "测试 2/3: seq_size=2048 + batch_size=32"
echo "=========================================="
timeout 300 ./bin/calculon llm-optimal-execution \
  gpt3-1B-seq2048.json \
  8 32 float32 \
  scripts/micro_benchmarks/nvidia/system.json \
  seq_test_results/output_seq2048_bs32.json \
  -m -t 1

if [ $? -eq 0 ] && [ -f "seq_test_results/output_seq2048_bs32.json" ]; then
    echo "✓ 成功!"
    sr=$(python -c "import json; print(json.load(open('seq_test_results/output_seq2048_bs32.json'))['0']['stats']['sample_rate'])" 2>/dev/null)
    echo "  样本率: $sr samples/sec"
    TEST2_SUCCESS=1
else
    echo "✗ 失败"
    TEST2_SUCCESS=0
fi

echo ""
sleep 2

# 测试3: seq_size=1024，batch_size=64
echo "=========================================="
echo "测试 3/3: seq_size=1024 + batch_size=64"
echo "=========================================="
timeout 300 ./bin/calculon llm-optimal-execution \
  gpt3-1B-seq1024.json \
  8 64 float32 \
  scripts/micro_benchmarks/nvidia/system.json \
  seq_test_results/output_seq1024_bs64.json \
  -m -t 1

if [ $? -eq 0 ] && [ -f "seq_test_results/output_seq1024_bs64.json" ]; then
    echo "✓ 成功!"
    sr=$(python -c "import json; print(json.load(open('seq_test_results/output_seq1024_bs64.json'))['0']['stats']['sample_rate'])" 2>/dev/null)
    echo "  样本率: $sr samples/sec"
    TEST3_SUCCESS=1
else
    echo "✗ 失败"
    TEST3_SUCCESS=0
fi

# 打印总结
echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo ""

success_count=$((TEST1_SUCCESS + TEST2_SUCCESS + TEST3_SUCCESS))

echo "成功的测试: $success_count / 3"
echo ""

if [ $success_count -gt 0 ]; then
    echo "性能对比:"
    echo ""
    
    if [ $TEST1_SUCCESS -eq 1 ]; then
        sr1=$(python -c "import json; print(json.load(open('seq_test_results/output_seq4096_bs8.json'))['0']['stats']['sample_rate'])" 2>/dev/null)
        echo "  原配置 (seq=4096, bs=8):    $sr1 samples/sec"
    fi
    
    if [ $TEST2_SUCCESS -eq 1 ]; then
        sr2=$(python -c "import json; print(json.load(open('seq_test_results/output_seq2048_bs32.json'))['0']['stats']['sample_rate'])" 2>/dev/null)
        echo "  seq=2048, bs=32:            $sr2 samples/sec"
    fi
    
    if [ $TEST3_SUCCESS -eq 1 ]; then
        sr3=$(python -c "import json; print(json.load(open('seq_test_results/output_seq1024_bs64.json'))['0']['stats']['sample_rate'])" 2>/dev/null)
        echo "  seq=1024, bs=64:            $sr3 samples/sec"
    fi
    
    echo ""
    echo "建议:"
    
    if [ $TEST1_SUCCESS -eq 1 ]; then
        echo "  • 如果需要处理4096长度的序列，使用原配置"
        echo "    但batch_size只能是8，训练效率较低"
    fi
    
    if [ $TEST2_SUCCESS -eq 1 ]; then
        echo "  • seq_size=2048是一个很好的折中方案"
        echo "    可以使用batch_size=32，训练效率更高"
    fi
    
    if [ $TEST3_SUCCESS -eq 1 ]; then
        echo "  • seq_size=1024可以使用最大的batch_size"
        echo "    训练最快，但序列长度受限"
    fi
    
else
    echo "所有测试都失败了！"
    echo ""
    echo "可能的原因:"
    echo "  1. 模型配置文件有其他问题"
    echo "  2. 系统配置不足"
    echo "  3. 需要更激进的优化"
    echo ""
    echo "建议:"
    echo "  1. 运行诊断工具:"
    echo "     python check_model_config.py models/gpt3-1B.json"
    echo ""
    echo "  2. 尝试更小的batch_size:"
    echo "     export PYTHONPATH=."
    echo "     ./bin/calculon llm-optimal-execution \\"
    echo "       models/gpt3-1B.json 4 2 float32 \\"
    echo "       scripts/micro_benchmarks/nvidia/system.json output.json -m"
    echo ""
    echo "  3. 查看错误日志获取更多信息"
fi

echo ""
echo "所有结果保存在: seq_test_results/"
echo "=========================================="
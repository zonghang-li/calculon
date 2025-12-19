#!/usr/bin/env python3
"""
Calculon 搜索失败诊断工具
找出为什么所有配置都被拒绝
"""

import sys
import json
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()

def diagnose(app_file, system_file, num_procs, max_batch, datatype):
    """诊断为什么找不到可行配置"""
    
    print("="*80)
    print("Calculon 配置失败诊断")
    print("="*80)
    print()
    
    # 读取配置
    with open(app_file) as f:
        app_cfg = json.load(f)
    with open(system_file) as f:
        sys_cfg = json.load(f)
    
    print(f"应用: {app_file}")
    print(f"系统: {system_file}")
    print(f"处理器: {num_procs}")
    print(f"最大 Batch: {max_batch}")
    print(f"数据类型: {datatype}")
    print()
    
    # 模型信息
    print("-"*80)
    print("模型配置")
    print("-"*80)
    hidden = app_cfg['hidden']
    ff = app_cfg['feedforward']
    seq = app_cfg['seq_size']
    heads = app_cfg['attn_heads']
    blocks = app_cfg['num_blocks']
    print(f"Hidden: {hidden}, FFN: {ff}, Seq: {seq}")
    print(f"Heads: {heads}, Blocks: {blocks}")
    print()
    
    # 系统信息
    print("-"*80)
    print("系统配置")
    print("-"*80)
    mem1_gib = sys_cfg['mem1']['GiB']
    mem2_gib = sys_cfg['mem2']['GiB']
    print(f"Mem1 (HBM): {mem1_gib} GiB")
    print(f"Mem2 (Host): {mem2_gib} GiB")
    print()
    
    # 检查效率曲线配置
    print("-"*80)
    print("效率曲线检查")
    print("-"*80)
    
    if datatype not in sys_cfg['matrix']:
        print(f"❌ 错误: 系统配置中没有 '{datatype}' 的矩阵处理器配置！")
        print(f"   系统配置中有: {list(sys_cfg['matrix'].keys())}")
        print()
        print("解决方案: 在 system.json 的 'matrix' 部分添加对应数据类型的配置")
        return
    
    matrix_cfg = sys_cfg['matrix'][datatype]
    print(f"✓ 找到 {datatype} 矩阵处理器配置")
    print(f"  Peak TFLOPS: {matrix_cfg['tflops']}")
    
    # 检查效率曲线结构
    eff_curve = matrix_cfg['gflops_efficiency']
    if isinstance(eff_curve, dict):
        print(f"  效率曲线类型: 形状感知 (shape-aware)")
        print(f"  Batch 数量: {len(eff_curve)}")
        
        # 检查 batch=1 的配置
        if 1 not in eff_curve:
            print(f"  ⚠️  警告: 没有 batch=1 的效率配置")
        else:
            batch1_bins = eff_curve[1]
            print(f"  Batch=1 的 GFLOPS bins: {len(batch1_bins)}")
            
            # 检查是否有空的 bin
            empty_bins = []
            for gflops, shapes in batch1_bins.items():
                if len(shapes) == 0:
                    empty_bins.append(gflops)
            
            if empty_bins:
                print(f"  ❌ 错误: 发现空的效率 bins!")
                print(f"     空的 GFLOPS bins: {empty_bins}")
                print()
                print("  这就是问题所在！")
                print("  在 processor.py 的 efficiency() 函数中:")
                print("    assert len(flops_bin) > 0, f\"This bin {flops/1e9} is empty...\"")
                print()
                print("  解决方案:")
                print("  1. 检查 system.json 中的 gflops_efficiency 配置")
                print("  2. 确保每个 GFLOPS bin 都有至少一个形状配置")
                print("  3. 或者删除空的 bins")
                return
            else:
                print(f"  ✓ 所有 bins 都有配置")
    else:
        print(f"  效率曲线类型: 简单 (size-only)")
        print(f"  Bins 数量: {len(eff_curve)}")
    
    print()
    
    # 尝试运行一个简单配置
    print("-"*80)
    print("测试最简单配置 (TP=1, PP=1, DP=1)")
    print("-"*80)
    
    try:
        sys.path.insert(0, '.')
        from calculon.llm import Llm
        from calculon.system import System
        
        app = Llm.Application(app_cfg)
        syst = System(sys_cfg)
        syst.set_datatype(datatype)
        
        # 最简单的配置
        exe_cfg = {
            'num_procs': 1,
            'tensor_par': 1,
            'pipeline_par': 1,
            'data_par': 1,
            'tensor_par_net': 0,
            'pipeline_par_net': 0,
            'data_par_net': 0,
            'batch_size': 1,
            'microbatch_size': 1,
            'datatype': datatype,
            'fused_activation': True,
            'qkv_packing': True,
            'grad_reduce_in_fp32': False,
            'attention_type': 'multihead',
            'activation_recompute': 'none',
            'pipeline_interleaving': 1,
            'optimizer_sharding': False,
            'tensor_par_comm_type': 'ar',
            'tensor_par_overlap': 'none',
            'seq_par_ag_redo': False,
            'data_par_overlap': False,
            'weight_offload': False,
            'activations_offload': False,
            'optimizer_offload': False,
            'training': True
        }
        
        model = Llm(app, logger)
        model.compile(syst, Llm.Execution.from_json(exe_cfg))
        model.run(syst)
        
        print("✓ 简单配置测试成功!")
        print(f"  内存需求: {model.get_mem_tier1_cap_req() / 1e9:.2f} GB")
        print(f"  可用内存: {mem1_gib} GB")
        print()
        
        if model.get_mem_tier1_cap_req() / 1e9 > mem1_gib:
            print("❌ 内存不足!")
            print("   即使是最简单的配置也超过了系统内存容量")
            print()
            print("解决方案:")
            print("  1. 减小模型大小")
            print("  2. 增加系统内存")
            print("  3. 启用 activation_recompute")
            print("  4. 使用 offload 到 Tier2")
        else:
            print("✓ 内存充足")
            print()
            print("如果简单配置可以运行但搜索失败，可能是:")
            print("  1. 效率曲线在某些 batch size 或形状上有问题")
            print("  2. 网络配置有问题")
            print("  3. 某些特定的并行配置触发了错误")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print()
        import traceback
        traceback.print_exc()
    
    print()
    print("="*80)

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("用法: python3 diagnose_search.py <app.json> <system.json> <num_procs> <max_batch> <datatype>")
        print()
        print("示例:")
        print("  python3 diagnose_search.py models/gpt-1B.json scripts/micro_benchmarks/nvidia/system.json 1024 4 float16")
        sys.exit(1)
    
    diagnose(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
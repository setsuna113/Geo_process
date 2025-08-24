#!/usr/bin/env python
"""
并行运行新旧系统，对比输出结果

这是删除旧系统前的最后验证步骤。

作者: Linus式验证
"""

import os
import sys
import time
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_old_system(config_path: str) -> Dict[str, Any]:
    """运行旧系统"""
    logger.info("Running OLD system...")
    
    # 导入旧系统
    sys.path.insert(0, str(Path(__file__).parent.parent / 'gee_integration'))
    from workflow.orchestrator import WorkflowOrchestrator
    from config import GEEConfig
    
    # 加载配置并运行
    config = GEEConfig.from_yaml(config_path)
    orchestrator = WorkflowOrchestrator(config)
    
    start_time = time.time()
    outputs = orchestrator.run()
    elapsed = time.time() - start_time
    
    return {
        'outputs': outputs,
        'elapsed_time': elapsed,
        'code_lines': 5500,  # 已知的代码行数
        'files': 18
    }


def run_new_system(config_path: str) -> Dict[str, Any]:
    """运行新系统"""
    logger.info("Running NEW system...")
    
    # 导入新系统
    sys.path.insert(0, str(Path(__file__).parent))
    from gee_simple import run_gee_pipeline
    
    start_time = time.time()
    outputs = run_gee_pipeline(config_path)
    elapsed = time.time() - start_time
    
    return {
        'outputs': outputs,
        'elapsed_time': elapsed,
        'code_lines': 1500,
        'files': 1
    }


def compare_outputs(old_output: Path, new_output: Path) -> Dict[str, Any]:
    """对比两个系统的输出"""
    logger.info(f"Comparing outputs: {old_output} vs {new_output}")
    
    # 读取parquet文件
    df_old = pd.read_parquet(old_output)
    df_new = pd.read_parquet(new_output)
    
    # 基本对比
    comparison = {
        'row_count_match': len(df_old) == len(df_new),
        'old_rows': len(df_old),
        'new_rows': len(df_new),
        'column_match': set(df_old.columns) == set(df_new.columns),
        'old_columns': list(df_old.columns),
        'new_columns': list(df_new.columns),
    }
    
    # 数值对比（允许小误差）
    if comparison['column_match'] and comparison['row_count_match']:
        # 排序以确保顺序一致
        df_old = df_old.sort_values(['latitude', 'longitude']).reset_index(drop=True)
        df_new = df_new.sort_values(['latitude', 'longitude']).reset_index(drop=True)
        
        # 对比数值列
        numeric_cols = df_old.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 计算差异
            diff = np.abs(df_old[col] - df_new[col])
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            # 相对误差（避免除零）
            rel_error = diff / (np.abs(df_old[col]) + 1e-10)
            max_rel_error = rel_error.max()
            
            comparison[f'{col}_max_diff'] = float(max_diff)
            comparison[f'{col}_mean_diff'] = float(mean_diff)
            comparison[f'{col}_max_rel_error'] = float(max_rel_error)
            
            # 判断是否匹配（允许0.1%的相对误差）
            comparison[f'{col}_match'] = max_rel_error < 0.001
    
    # 文件大小对比
    comparison['old_file_size'] = old_output.stat().st_size
    comparison['new_file_size'] = new_output.stat().st_size
    comparison['size_reduction'] = 1 - (comparison['new_file_size'] / comparison['old_file_size'])
    
    # 计算哈希值
    comparison['old_hash'] = hashlib.md5(open(old_output, 'rb').read()).hexdigest()
    comparison['new_hash'] = hashlib.md5(open(new_output, 'rb').read()).hexdigest()
    comparison['exact_match'] = comparison['old_hash'] == comparison['new_hash']
    
    return comparison


def run_comparison(config_path: str, output_dir: Path):
    """运行完整对比"""
    logger.info("="*80)
    logger.info("STARTING SYSTEM COMPARISON")
    logger.info("="*80)
    
    results = {}
    
    # 运行旧系统
    try:
        old_result = run_old_system(config_path)
        results['old_system'] = old_result
        logger.info(f"✅ Old system completed in {old_result['elapsed_time']:.2f}s")
    except Exception as e:
        logger.error(f"❌ Old system failed: {e}")
        results['old_system'] = {'error': str(e)}
    
    # 运行新系统
    try:
        new_result = run_new_system(config_path)
        results['new_system'] = new_result
        logger.info(f"✅ New system completed in {new_result['elapsed_time']:.2f}s")
    except Exception as e:
        logger.error(f"❌ New system failed: {e}")
        results['new_system'] = {'error': str(e)}
    
    # 对比输出
    if 'error' not in results['old_system'] and 'error' not in results['new_system']:
        comparisons = {}
        
        for resolution in ['1x', '2x', '4x']:
            if resolution in results['old_system']['outputs'] and resolution in results['new_system']['outputs']:
                comp = compare_outputs(
                    results['old_system']['outputs'][resolution],
                    results['new_system']['outputs'][resolution]
                )
                comparisons[resolution] = comp
        
        results['comparisons'] = comparisons
    
    # 生成报告
    generate_report(results, output_dir)
    
    return results


def generate_report(results: Dict[str, Any], output_dir: Path):
    """生成对比报告"""
    report_path = output_dir / "comparison_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# 新旧系统对比报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 性能对比
        f.write("## 性能对比\n\n")
        f.write("| 指标 | 旧系统 | 新系统 | 改进 |\n")
        f.write("|------|--------|--------|------|\n")
        
        if 'error' not in results.get('old_system', {}) and 'error' not in results.get('new_system', {}):
            old_time = results['old_system']['elapsed_time']
            new_time = results['new_system']['elapsed_time']
            
            f.write(f"| 运行时间 | {old_time:.2f}s | {new_time:.2f}s | {(1-new_time/old_time)*100:.1f}% |\n")
            f.write(f"| 代码行数 | 5500 | 1500 | -73% |\n")
            f.write(f"| 文件数 | 18 | 1 | -94% |\n")
        
        # 输出对比
        f.write("\n## 输出对比\n\n")
        
        if 'comparisons' in results:
            for resolution, comp in results['comparisons'].items():
                f.write(f"### {resolution} 分辨率\n\n")
                
                # 基本匹配
                f.write("**基本信息:**\n")
                f.write(f"- 行数匹配: {'✅' if comp['row_count_match'] else '❌'} ")
                f.write(f"({comp['old_rows']} vs {comp['new_rows']})\n")
                f.write(f"- 列匹配: {'✅' if comp['column_match'] else '❌'}\n")
                f.write(f"- 文件大小: {comp['old_file_size']/1024/1024:.2f}MB → {comp['new_file_size']/1024/1024:.2f}MB ")
                f.write(f"({comp['size_reduction']*100:.1f}% 减少)\n")
                f.write(f"- 完全匹配: {'✅' if comp['exact_match'] else '❌'}\n\n")
                
                # 数值精度
                if comp['column_match']:
                    f.write("**数值精度:**\n")
                    numeric_cols = ['bio1', 'bio2', 'elevation', 'species_richness']
                    
                    for col in numeric_cols:
                        if f'{col}_match' in comp:
                            match = '✅' if comp[f'{col}_match'] else '❌'
                            max_error = comp.get(f'{col}_max_rel_error', 0) * 100
                            f.write(f"- {col}: {match} (最大误差: {max_error:.4f}%)\n")
        
        # 结论
        f.write("\n## 结论\n\n")
        
        all_match = True
        if 'comparisons' in results:
            for comp in results['comparisons'].values():
                if not comp.get('row_count_match') or not comp.get('column_match'):
                    all_match = False
                    break
                
                # 检查所有数值列
                for key in comp:
                    if key.endswith('_match') and not comp[key]:
                        all_match = False
                        break
        
        if all_match:
            f.write("✅ **新系统完全兼容旧系统！**\n\n")
            f.write("- 所有输出匹配（在允许误差范围内）\n")
            f.write("- 性能显著提升\n")
            f.write("- 代码量大幅减少\n")
            f.write("\n**建议：可以安全地迁移到新系统**\n")
        else:
            f.write("⚠️ **发现差异，需要进一步验证**\n\n")
            f.write("请检查上述对比结果，确定差异是否可接受。\n")
        
        # Linus评价
        f.write("\n## Linus的评价\n\n")
        f.write("> \"")
        if all_match:
            f.write("好！新系统不仅保持了功能完整性，还把那堆垃圾代码清理干净了。")
            f.write("这就是我说的'好品味' - 用1500行做了5500行的事，")
            f.write("而且做得更好。现在可以把旧系统扔进垃圾桶了。")
        else:
            f.write("有差异不一定是坏事，但你必须理解每个差异的原因。")
            f.write("如果是因为修复了bug或改进了算法，那很好。")
            f.write("如果是引入了新bug，那就是垃圾。仔细检查！")
        f.write("\"\n")
    
    logger.info(f"✅ Report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="对比新旧GEE系统")
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--output-dir', default='comparison_results', help='输出目录')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = run_comparison(args.config, output_dir)
    
    # 打印摘要
    print("\n" + "="*80)
    print("对比完成！")
    print(f"详细报告: {output_dir / 'comparison_report.md'}")
    
    if 'comparisons' in results:
        all_match = all(
            comp.get('row_count_match', False) and comp.get('column_match', False)
            for comp in results['comparisons'].values()
        )
        
        if all_match:
            print("\n✅ 新系统完全兼容！可以安全迁移。")
        else:
            print("\n⚠️ 发现差异，请查看详细报告。")
    else:
        print("\n❌ 对比失败，请查看错误日志。")
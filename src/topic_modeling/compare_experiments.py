#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果对比分析工具
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse


def load_experiments(jsonl_path: str) -> List[Dict]:
    """加载所有实验结果"""
    experiments = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    return experiments


def create_comparison_table(experiments: List[Dict]) -> pd.DataFrame:
    """创建对比表格"""
    rows = []
    
    for exp in experiments:
        params = exp['parameters']
        metrics = exp['metrics']
        
        row = {
            'experiment_id': exp['experiment_id'],
            'timestamp': exp['timestamp'][:19],
            # UMAP参数
            'umap_n_neighbors': params['umap']['n_neighbors'],
            'umap_n_components': params['umap']['n_components'],
            'umap_min_dist': params['umap']['min_dist'],
            # HDBSCAN参数
            'hdbscan_min_cluster_size': params['hdbscan']['min_cluster_size'],
            'hdbscan_min_samples': params['hdbscan']['min_samples'],
            'hdbscan_selection': params['hdbscan']['cluster_selection_method'],
            # 聚类指标
            'silhouette_orig': metrics.get('silhouette_original'),
            'ch_score_orig': metrics.get('calinski_harabasz_original'),
            'db_score_orig': metrics.get('davies_bouldin_original'),
            'silhouette_umap': metrics.get('silhouette_umap'),
            # 主题模型指标
            'coherence_cv': metrics.get('coherence_cv'),
            'coherence_npmi': metrics.get('coherence_npmi'),
            'topic_diversity': metrics.get('topic_diversity'),
            # 分布指标
            'n_topics': metrics.get('n_topics'),
            'outlier_ratio': metrics.get('outlier_ratio'),
            'topic_size_cv': metrics.get('topic_size_cv'),
            # 概率质量
            'prob_mean': metrics.get('prob_mean'),
            'low_conf_ratio': metrics.get('low_confidence_ratio'),
            'reduce_outliers': params.get('bertopic', {}).get('reduce_outliers', False),
            'outlier_strategy': params.get('bertopic', {}).get('outlier_strategy'),
            'outlier_threshold': params.get('bertopic', {}).get('outlier_threshold'),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def rank_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """根据综合得分排序实验 - 修复版：处理空值"""
    # 评分配置：(权重, 方向) 1=越高越好, -1=越低越好
    # 注意：coherence_cv 可能为空，需要特殊处理
    scoring_config = {
        'silhouette_orig': (0.20, 1),      # 提高权重，因为coherence可能为空
        'coherence_cv': (0.25, 1),          # 主题一致性
        'topic_diversity': (0.15, 1),       # 多样性
        'outlier_ratio': (0.15, -1),        # 离群点比例
        'db_score_orig': (0.10, -1),        # Davies-Bouldin
        'low_conf_ratio': (0.10, -1),       # 低置信度比例
        'topic_size_cv': (0.05, -1)         # 主题大小变异系数
    }
    
    df_score = df.copy()
    total_score = np.zeros(len(df))
    total_weight_used = np.zeros(len(df))
    
    for col, (weight, direction) in scoring_config.items():
        if col in df.columns:
            values = df[col].copy()
            valid_mask = values.notna()
            
            if valid_mask.sum() > 0:
                # 只对有效值进行归一化
                valid_values = values[valid_mask]
                min_val, max_val = valid_values.min(), valid_values.max()
                
                if max_val > min_val:
                    # 归一化
                    normalized = (values - min_val) / (max_val - min_val)
                    if direction == -1:
                        normalized = 1 - normalized
                    
                    # 只对有效值累加得分
                    for i in range(len(df)):
                        if valid_mask.iloc[i]:
                            total_score[i] += weight * normalized.iloc[i]
                            total_weight_used[i] += weight
                else:
                    # 如果所有值相同，给中等分数
                    for i in range(len(df)):
                        if valid_mask.iloc[i]:
                            total_score[i] += weight * 0.5
                            total_weight_used[i] += weight
    
    # 归一化得分（按实际使用的权重）
    for i in range(len(df)):
        if total_weight_used[i] > 0:
            total_score[i] = total_score[i] / total_weight_used[i]
    
    df_score['composite_score'] = total_score
    df_score = df_score.sort_values('composite_score', ascending=False)
    
    return df_score


def print_comparison_report(df: pd.DataFrame, top_n: int = 3):
    """打印对比报告"""
    print("\n" + "=" * 100)
    print("实验对比报告 - EXPERIMENT COMPARISON REPORT")
    print("=" * 100)
    
    display_cols = [
        'experiment_id', 'hdbscan_selection', 
        'silhouette_orig', 'coherence_cv', 'topic_diversity',
        'n_topics', 'outlier_ratio', 'composite_score'
    ]
    
    available_cols = [c for c in display_cols if c in df.columns]
    
    print(f"\n--- 按综合得分排名 (显示全部 {len(df)} 个实验) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    df_display = df[available_cols].copy()
    df_display['experiment_id'] = df_display['experiment_id'].str[:45] + '...'
    print(df_display.to_string(index=False))
    
    print(f"\n{'=' * 100}")
    print(f"TOP {top_n} 实验详情")
    print("=" * 100)
    
    top_experiment_ids = []
    
    for i in range(min(top_n, len(df))):
        exp = df.iloc[i]
        top_experiment_ids.append(exp['experiment_id'])
        
        print(f"\n--- Rank {i+1} ---")
        print(f"实验ID: {exp['experiment_id']}")
        print(f"参数配置:")
        print(f"  UMAP: n_neighbors={exp.get('umap_n_neighbors')}, "
              f"n_components={exp.get('umap_n_components')}, "
              f"min_dist={exp.get('umap_min_dist')}")
        print(f"  HDBSCAN: min_samples={exp.get('hdbscan_min_samples')}, "
              f"selection={exp.get('hdbscan_selection')}")
        print(f"核心指标:")
        if exp.get('silhouette_orig') is not None:
            print(f"  Silhouette (原始空间): {exp['silhouette_orig']:.4f}")
        else:
            print(f"  Silhouette (原始空间): N/A")
        if exp.get('coherence_cv') is not None:
            print(f"  Topic Coherence (c_v): {exp['coherence_cv']:.4f}")
        else:
            print(f"  Topic Coherence (c_v): N/A")
        if exp.get('topic_diversity') is not None:
            print(f"  Topic Diversity: {exp['topic_diversity']:.4f}")
        print(f"  主题数: {exp.get('n_topics', 'N/A')}")
        outlier_ratio = exp.get('outlier_ratio')
        if outlier_ratio is not None:
            print(f"  离群点比例: {outlier_ratio:.1%}")
        else:
            print(f"  离群点比例: N/A")
        print(f"  综合得分: {exp.get('composite_score', 0):.4f}")
    
    # 指标解读
    print("\n" + "=" * 100)
    print("指标解读")
    print("=" * 100)
    print("  silhouette_orig: 原始空间轮廓系数，越接近1越好（>0.3为良好）")
    print("  coherence_cv: 主题一致性，越高越好（>0.4为良好）")
    print("  topic_diversity: 主题间词汇多样性，0-1之间，越高越好（>0.7为良好）")
    print("  outlier_ratio: 离群点比例，越低越好（<10%为良好）")
    print("  composite_score: 综合得分，基于以上指标加权计算（自动处理空值）")
    
    return top_experiment_ids


def main():
    parser = argparse.ArgumentParser(description='实验结果对比分析')
    parser.add_argument('--input', '-i', type=str, 
                        default='outputs/experiment_logs/experiment_summary.jsonl',
                        help='实验汇总文件路径')
    parser.add_argument('--output', '-o', type=str, 
                        default='outputs/comparison_report_sorted.csv',
                        help='对比报告输出路径')
    parser.add_argument('--base_dir', type=str,
                        default=str(Path(__file__).resolve().parents[2]),
                        help='项目根目录')
    parser.add_argument('--top_n', type=int, default=3,
                        help='输出排名前N的实验ID')
    parser.add_argument('--output_ids', type=str, default=None,
                        help='输出Top N实验ID的文件路径')
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    input_path = base_path / args.input
    output_path = base_path / args.output
    
    # 加载实验
    experiments = load_experiments(str(input_path))
    print(f"已加载 {len(experiments)} 个实验结果")
    
    # 创建对比表
    df = create_comparison_table(experiments)
    
    # 排序
    df_ranked = rank_experiments(df)
    
    # 输出报告
    top_experiment_ids = print_comparison_report(df_ranked, args.top_n)
    
    # 保存完整报告
    df_ranked.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n对比报告已保存至: {output_path}")
    
    # 保存Top N实验ID（供脚本使用）
    if args.output_ids:
        ids_path = base_path / args.output_ids
    else:
        ids_path = base_path / "outputs" / "top_experiment_ids.txt"
    
    with open(ids_path, 'w') as f:
        for exp_id in top_experiment_ids:
            f.write(exp_id + '\n')
    
    print(f"Top {args.top_n} 实验ID已保存至: {ids_path}")
    
    # 打印可直接使用的命令
    print("\n" + "=" * 100)
    print("可视化命令")
    print("=" * 100)
    for i, exp_id in enumerate(top_experiment_ids, 1):
        print(f"# Rank {i}:")
        print(f"python visualize_experiment.py --experiment_id \"{exp_id}\"")
        print()


if __name__ == "__main__":
    main()

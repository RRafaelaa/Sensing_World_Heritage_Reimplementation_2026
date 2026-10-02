#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic 实验可视化脚本 - 改进版 V3
- 使用 n-gram (1-3) + MaximalMarginalRelevance 作为 representation model
- 关键词：生成20个
- Representative Documents: 3篇，预览扩展至600字符
- 输出每篇文章的概率分布
- 修复：DataMapPlot 使用独特性关键词作为标签
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sklearn.metrics import silhouette_samples
from nltk.stem import PorterStemmer

# DataMapPlot
try:
    import datamapplot
    DATAMAPPLOT_AVAILABLE = True
except ImportError:
    DATAMAPPLOT_AVAILABLE = False
    print("⚠️  Warning: datamapplot not installed.")

warnings.filterwarnings('ignore')
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class ImprovedExperimentVisualizer:
    """改进版实验可视化器 V3"""
    
    def __init__(self, base_dir: str, experiment_id: str):
        self.base_dir = Path(base_dir)
        self.experiment_id = experiment_id
        self.output_dir = self.base_dir / "outputs"
        self.figure_dir = self.output_dir / "figures_improved_v3" / experiment_id
        self.cache_dir = self.output_dir / "model_cache"
        self.model_dir = self.base_dir / "models"
        
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据
        self.cache_data = None
        self.topic_model = None
        self.reduced_embeddings_2d = None
        self.probs_matrix = None
        self.embedding_model = None
        
        # 存储主题关键词
        self.topic_words = {}
        
        # 存储独特性标签
        self.unique_topic_labels = {}

        self.stemmer = PorterStemmer()
    
    def load_experiment(self) -> bool:
        """加载实验缓存和模型"""
        print(f"加载实验: {self.experiment_id}")
        
        # 加载缓存数据
        cache_path = self.cache_dir / f"cache_{self.experiment_id}.pkl"
        if not cache_path.exists():
            print(f"   ❌ 缓存文件不存在: {cache_path}")
            return False
        
        with open(cache_path, 'rb') as f:
            self.cache_data = pickle.load(f)
        
        print(f"   ✓ 缓存加载成功")
        
        # 加载BERTopic模型
        model_path = self.cache_dir / f"model_{self.experiment_id}"
        if not model_path.exists():
            print(f"   ❌ 模型文件不存在: {model_path}")
            return False
        
        # 加载embedding model
        model_local_path = str(self.model_dir / "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_local_path)
        
        self.topic_model = BERTopic.load(str(model_path), embedding_model=self.embedding_model)
        print(f"   ✓ 模型加载成功")
        
        # 计算2D降维
        print("   计算2D降维嵌入...")
        umap_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        self.reduced_embeddings_2d = umap_2d.fit_transform(self.cache_data['embeddings'])
        print(f"   ✓ 2D降维完成")
        
        # 处理概率矩阵
        self._process_probabilities()

        # 用 n-gram + MMR 重算topic（20个词）
        self._update_representation_with_ngram_mmr()
        
        # 移除 stemming 重复词
        self._remove_stem_duplicates()
        
        # 收集关键词（补充未处理的topic）
        self._collect_keywords()
        
        # 生成独特性标签并设置到模型
        self._generate_unique_labels()
        
        return True

    def _update_representation_with_ngram_mmr(self):
        """使用 n-gram (1-3) + MMR 重新生成主题关键词（20个词）"""
        print("   - 使用 n-gram (1-3) + MaximalMarginalRelevance 重新生成关键词...")
        
        # 设置 n-gram vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            min_df=5
        )
        
        # 链式表示模型：先用 KeyBERTInspired 保留 n-gram，再用 MMR 增加多样性
        representation_model = [
            KeyBERTInspired(top_n_words=40),  # 先选出更多候选词
            MaximalMarginalRelevance(diversity=0.3, top_n_words=25)
        ]
        
        docs = self.cache_data['docs']
        
        self.topic_model.update_topics(
            docs,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=20
        )
        
        print(f"     ✓ 已使用 n-gram(1-3) + MMR(diversity=0.3, top_n_words=20) 更新主题表示")

    def _remove_stem_duplicates(self):
        """移除 stemming 相同的重复词，保留权重最高的"""
        print("   - 移除 stemming 重复词...")
        
        topic_info = self.topic_model.get_topic_info()
        total_removed = 0
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue
            
            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                continue
            
            # 对每个词/短语进行 stemming
            seen_stems = {}  # stem_signature -> (word, weight)
            deduplicated = []
            
            for word, weight in topic_words:
                # 对短语中的每个词进行 stem，生成签名
                tokens = word.lower().split()
                stem_signature = tuple(self.stemmer.stem(token) for token in tokens)
                
                if stem_signature not in seen_stems:
                    seen_stems[stem_signature] = (word, weight)
                    deduplicated.append((word, weight))
                else:
                    total_removed += 1
                    # 已经有更高权重的版本，跳过
            
            # 更新存储（将在 _collect_keywords 中使用）
            self.topic_words[topic_id] = deduplicated
        
        print(f"     ✓ 共移除 {total_removed} 个重复词")

    
    def _collect_keywords(self):
        """收集各主题的关键词（补充未处理的topic）"""
        print("   - 收集主题关键词...")
        
        topic_info = self.topic_model.get_topic_info()
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            # 如果已经在 _remove_stem_duplicates 中处理过，跳过
            if topic_id in self.topic_words:
                continue
            
            if topic_id == -1:
                self.topic_words[topic_id] = []
                continue
            
            topic_words = self.topic_model.get_topic(topic_id)
            if topic_words:
                self.topic_words[topic_id] = topic_words
            else:
                self.topic_words[topic_id] = []
        
        n_topics = len([t for t in self.topic_words.keys() if t != -1])
        avg_words = sum(len(w) for t, w in self.topic_words.items() if t != -1) / n_topics if n_topics > 0 else 0
        print(f"     ✓ 收集完成: {n_topics} 个主题，平均 {avg_words:.1f} 个关键词/主题")
        
    # def _generate_unique_labels(self):
    #     """生成具有独特性的主题标签，并设置到模型"""
    #     print("   - 生成独特性主题标签...")
        
    #     # 收集所有topic的词集合，用于计算独特性
    #     all_topic_words = {}
    #     for tid, words in self.topic_words.items():
    #         if tid != -1 and words:
    #             all_topic_words[tid] = set(w[0].lower() for w in words)
        
    #     custom_labels = {}
        
    #     for topic_id, words in self.topic_words.items():
    #         if topic_id == -1:
    #             # outlier topic 标签
    #             custom_labels[topic_id] = "-1_Outliers"
    #             self.unique_topic_labels[topic_id] = "Outliers"
    #         elif words:
    #             # 计算每个词的独特性分数
    #             other_topics = [tid for tid in all_topic_words if tid != topic_id]
    #             unique_words = []
                
    #             for word, weight in words:
    #                 word_lower = word.lower()
    #                 # 统计这个词在多少其他topic中出现
    #                 overlap_count = sum(
    #                     1 for tid in other_topics 
    #                     if word_lower in all_topic_words.get(tid, set())
    #                 )
    #                 # 独特性分数 = 原始权重 / (1 + 出现在其他topic的次数)
    #                 # 这样在其他topic中出现越多，分数越低
    #                 uniqueness_score = weight / (1 + overlap_count)
    #                 unique_words.append((word, uniqueness_score, weight, overlap_count))
                
    #             # 按独特性分数排序，取前4个
    #             unique_words.sort(key=lambda x: x[1], reverse=True)
    #             top_unique = unique_words[:4]
                
    #             # 生成标签
    #             label_words = [w[0] for w in top_unique]
    #             label_str = " | ".join(label_words)
                
    #             # 存储用于可视化的标签
    #             self.unique_topic_labels[topic_id] = label_str
                
    #             # BERTopic的custom_labels格式需要带topic_id前缀
    #             custom_labels[topic_id] = f"{topic_id}_{label_str}"
                
    #             # 打印调试信息
    #             if len(all_topic_words) <= 15:  # 仅当topic数量不多时打印
    #                 overlap_info = ", ".join([f"{w[0]}({w[3]})" for w in top_unique])
    #                 print(f"       Topic {topic_id}: {label_str} (overlap counts: {overlap_info})")
    #         else:
    #             custom_labels[topic_id] = f"{topic_id}_Topic_{topic_id}"
    #             self.unique_topic_labels[topic_id] = f"Topic {topic_id}"
        
    #     # 设置自定义标签到模型
    #     self.topic_model.set_topic_labels(custom_labels)
        
    #     print(f"     ✓ 已生成并设置 {len(custom_labels)} 个独特性标签")


    # def _generate_unique_labels(self):
    #     """生成具有独特性的主题标签，并设置到模型"""
    #     print("   - 生成独特性主题标签...")
        
    #     # 收集所有topic的词集合，用于计算独特性
    #     all_topic_words = {}
    #     for tid, words in self.topic_words.items():
    #         if tid != -1 and words:
    #             all_topic_words[tid] = set(w[0].lower() for w in words)
        
    #     n_total_topics = len(all_topic_words)
    #     custom_labels = {}
        
    #     for topic_id, words in self.topic_words.items():
    #         if topic_id == -1:
    #             custom_labels[topic_id] = "-1_Outliers"
    #             self.unique_topic_labels[topic_id] = "Outliers"
    #         elif words:
    #             other_topics = [tid for tid in all_topic_words if tid != topic_id]
    #             unique_words = []
                
    #             for word, weight in words:
    #                 word_lower = word.lower()
    #                 # 统计这个词在多少其他topic中出现
    #                 overlap_count = sum(
    #                     1 for tid in other_topics 
    #                     if word_lower in all_topic_words.get(tid, set())
    #                 )
                    
    #                 # ===== 改动1: 更强的独特性惩罚 =====
    #                 # 方案A: 指数惩罚
    #                 uniqueness_score = weight * math.exp(-overlap_count * 0.5)
                
    #                 unique_words.append((word, uniqueness_score, weight, overlap_count))
                
    #             # ===== 改动2: 优先选择完全独特的词 =====
    #             # 先按 overlap_count 升序，再按 uniqueness_score 降序
    #             unique_words.sort(key=lambda x: (x[3], -x[1]))
                
    #             # ===== 改动3: 选择策略改进 =====
    #             # 优先选择 overlap_count=0 的词，不足再补充
    #             exclusive_words = [w for w in unique_words if w[3] == 0]
    #             non_exclusive_words = [w for w in unique_words if w[3] > 0]
                
    #             # 取前4个，优先用独占词
    #             top_unique = exclusive_words[:4]
    #             if len(top_unique) < 4:
    #                 remaining = 4 - len(top_unique)
    #                 top_unique.extend(non_exclusive_words[:remaining])
                
    #             # 生成标签
    #             label_words = [w[0] for w in top_unique]
    #             label_str = " | ".join(label_words)
                
    #             self.unique_topic_labels[topic_id] = label_str
    #             custom_labels[topic_id] = f"{topic_id}_{label_str}"
                
    #             if len(all_topic_words) <= 15:
    #                 overlap_info = ", ".join([f"{w[0]}({w[3]})" for w in top_unique])
    #                 print(f"       Topic {topic_id}: {label_str} (overlap: {overlap_info})")
    #         else:
    #             custom_labels[topic_id] = f"{topic_id}_Topic_{topic_id}"
    #             self.unique_topic_labels[topic_id] = f"Topic {topic_id}"
        
    #     self.topic_model.set_topic_labels(custom_labels)
    #     print(f"     ✓ 已生成并设置 {len(custom_labels)} 个独特性标签")

    def _generate_unique_labels(self):
        """直接用 representation model 输出的前4个词"""
        custom_labels = {}
        
        for topic_id, words in self.topic_words.items():
            if topic_id == -1:
                custom_labels[topic_id] = "-1_Outliers"
                self.unique_topic_labels[topic_id] = "Outliers"
            elif words:
                # 直接取前4个，不做额外处理
                label_words = [w[0] for w in words[:4]]
                label_str = " | ".join(label_words)
                self.unique_topic_labels[topic_id] = label_str
                custom_labels[topic_id] = f"{topic_id}_{label_str}"
            else:
                custom_labels[topic_id] = f"{topic_id}_Topic_{topic_id}"
                self.unique_topic_labels[topic_id] = f"Topic {topic_id}"
        
        self.topic_model.set_topic_labels(custom_labels)
    
    def _process_probabilities(self):
        """处理并存储概率矩阵"""
        print("   处理概率分布...")
        
        probs = self.cache_data.get('probs')
        topics = self.cache_data['topics']
        
        if probs is None:
            print("     ⚠️ 无概率数据")
            self.probs_matrix = None
            return
        
        topic_info = self.topic_model.get_topic_info()
        valid_topics = sorted([t for t in topic_info['Topic'] if t != -1])
        
        n_docs = len(topics)
        n_topics = len(valid_topics)
        
        if isinstance(probs, np.ndarray):
            if probs.ndim == 2:
                self.probs_matrix = probs
            else:
                self.probs_matrix = np.zeros((n_docs, n_topics))
                for i, (topic_id, prob) in enumerate(zip(topics, probs)):
                    if topic_id != -1 and topic_id in valid_topics:
                        col_idx = valid_topics.index(topic_id)
                        self.probs_matrix[i, col_idx] = prob if prob is not None else 0
        else:
            try:
                self.probs_matrix = np.array(probs)
                if self.probs_matrix.ndim == 1:
                    temp = np.zeros((n_docs, n_topics))
                    for i, (topic_id, prob) in enumerate(zip(topics, probs)):
                        if topic_id != -1 and topic_id in valid_topics:
                            col_idx = valid_topics.index(topic_id)
                            temp[i, col_idx] = float(prob) if prob is not None else 0
                    self.probs_matrix = temp
            except:
                self.probs_matrix = None
        
        if self.probs_matrix is not None:
            print(f"     ✓ 概率矩阵形状: {self.probs_matrix.shape}")
        else:
            print("     ⚠️ 概率矩阵处理失败")
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        print(f"\n生成可视化图表...")
        
        self._plot_topic_distribution()
        self._plot_silhouette()
        self._plot_topic_keywords()
        self._plot_topic_heatmap()
        self._plot_topics_over_time()
        self._plot_datamap()
        self._plot_documents_plotly()
        self._save_representative_docs_report()
        self._save_probability_distribution()
        
        print(f"\n✓ 所有可视化已保存至: {self.figure_dir}")
    
    def _plot_topic_distribution(self):
        """绘制主题分布图"""
        print("   - 生成主题分布图...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        topic_info = self.topic_model.get_topic_info()
        topic_info_valid = topic_info[topic_info['Topic'] != -1].copy()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_info_valid)))
        bars = ax.bar(range(len(topic_info_valid)), topic_info_valid['Count'], color=colors)
        
        ax.set_xlabel('Topic', fontsize=12)
        ax.set_ylabel('Number of Documents', fontsize=12)
        ax.set_title(f'Topic Distribution - {self.experiment_id[:40]}...', fontsize=11)
        ax.set_xticks(range(len(topic_info_valid)))
        ax.set_xticklabels([f"T{t}" for t in topic_info_valid['Topic']], rotation=45)
        
        for bar, count in zip(bars, topic_info_valid['Count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom', fontsize=10)
        
        n_outliers = topic_info[topic_info['Topic'] == -1]['Count'].sum()
        if n_outliers > 0:
            ax.text(0.98, 0.98, f'Outliers: {n_outliers}', 
                   transform=ax.transAxes, ha='right', va='top', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'topic_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_silhouette(self):
        """绘制轮廓系数图"""
        print("   - 生成轮廓系数图...")
        
        topics = np.array(self.cache_data['topics'])
        embeddings = self.cache_data['embeddings']
        
        non_outlier_mask = topics != -1
        topics_non_outlier = topics[non_outlier_mask]
        
        if len(set(topics_non_outlier)) <= 1:
            print("     ⚠️ 主题数不足，跳过轮廓系数图")
            return
        
        embeddings_non_outlier = embeddings[non_outlier_mask]
        silhouette_vals = silhouette_samples(embeddings_non_outlier, topics_non_outlier, metric='cosine')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_lower = 10
        unique_topics = sorted(set(topics_non_outlier))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_topics)))
        
        for i, topic in enumerate(unique_topics):
            topic_silhouette = silhouette_vals[topics_non_outlier == topic]
            topic_silhouette.sort()
            
            y_upper = y_lower + len(topic_silhouette)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, topic_silhouette,
                facecolor=colors[i], alpha=0.7,
                label=f'Topic {topic}'
            )
            y_lower = y_upper + 10
        
        avg_silhouette = silhouette_vals.mean()
        ax.axvline(x=avg_silhouette, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_silhouette:.3f}')
        
        ax.set_xlabel('Silhouette Coefficient', fontsize=11)
        ax.set_ylabel('Documents (grouped by topic)', fontsize=11)
        ax.set_title(f'Silhouette Analysis (Original Embedding Space)', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'silhouette_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_topic_keywords(self):
        """绘制主题关键词图"""
        print("   - 生成主题关键词图...")
        
        topic_info = self.topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]
        
        n_topics = len(topic_info)
        n_cols = 3
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(22, n_rows * 7))
        axes = axes.flatten() if n_topics > 1 else [axes]
        
        for idx, (_, row) in enumerate(topic_info.iterrows()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            topic_id = row['Topic']
            
            topic_words = self.topic_words.get(topic_id, [])
            if topic_words:
                words = [w[0] for w in topic_words]
                weights = [w[1] for w in topic_words]
                
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(words)))
                ax.barh(range(len(words)), weights, color=colors)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=8)
                ax.invert_yaxis()
                ax.set_title(f'Topic {topic_id} (n={row["Count"]}, {len(words)} keywords)', fontsize=11)
                ax.set_xlabel('Weight', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No keywords', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Topic {topic_id} (n={row["Count"]})', fontsize=11)
        
        for idx in range(len(topic_info), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Topic Keywords (n-gram 1-3 + MMR)', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'topic_keywords.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_topic_heatmap(self):
        """绘制主题相似性热力图"""
        print("   - 生成主题热力图...")
        
        try:
            fig = self.topic_model.visualize_heatmap(n_clusters=None, custom_labels=True)
            fig.write_html(str(self.figure_dir / 'topic_heatmap.html'))
        except Exception as e:
            print(f"     ⚠️ 热力图生成失败: {e}")
    
    def _plot_topics_over_time(self):
        """绘制主题随时间变化图"""
        print("   - 生成Topics over Time图...")
        
        years = self.cache_data.get('years')
        if years is None:
            print("     ⚠️ 没有年份数据，跳过")
            return
        
        docs = self.cache_data['docs']
        
        try:
            valid_years = []
            valid_docs = []
            for i, year in enumerate(years):
                if pd.notna(year) and isinstance(year, (int, float)):
                    valid_years.append(int(year))
                    valid_docs.append(docs[i])
            
            if len(valid_years) < 10:
                print("     ⚠️ 有效年份数据太少，跳过")
                return
            
            timestamps = [f"{y}-01-01" for y in valid_years]
            
            topics_over_time = self.topic_model.topics_over_time(
                valid_docs, 
                timestamps,
                nr_bins=min(20, len(set(valid_years)))
            )
            
            topics = self.cache_data['topics']
            n_display_topics = min(10, len(set(topics)) - 1)
            
            fig = self.topic_model.visualize_topics_over_time(
                topics_over_time, 
                top_n_topics=n_display_topics,
                custom_labels=True
            )
            fig.write_html(str(self.figure_dir / 'topics_over_time.html'))
            print(f"     ✓ Topics over Time图已保存")
            
        except Exception as e:
            print(f"     ⚠️ Topics over Time可视化失败: {e}")
    
    def _plot_datamap(self):
        """使用DataMapPlot生成文档点云图 - 使用独特性标签"""
        print("   - 生成DataMapPlot点云图...")
        
        if not DATAMAPPLOT_AVAILABLE:
            print("     ⚠️ DataMapPlot未安装，跳过")
            return
        
        docs = self.cache_data['docs']
        topics = self.cache_data['topics']
        
        # 为每个文档生成对应的标签
        topic_labels_per_doc = []
        for topic_id in topics:
            if topic_id in self.unique_topic_labels:
                topic_labels_per_doc.append(self.unique_topic_labels[topic_id])
            else:
                topic_labels_per_doc.append(f"Topic {topic_id}")
        
        topic_labels_per_doc = np.array(topic_labels_per_doc)
        
        try:
            # 方法1: 使用 custom_labels=True（依赖 set_topic_labels）
            fig = self.topic_model.visualize_document_datamap(
                docs,
                reduced_embeddings=self.reduced_embeddings_2d,
                title=f"Document Topic Map",
                custom_labels=True  # 使用通过 set_topic_labels 设置的标签
            )
            fig.savefig(
                str(self.figure_dir / 'datamap.png'),
                bbox_inches='tight',
                dpi=150
            )
            print(f"     ✓ DataMapPlot静态图已保存")
            
            # 交互式版本
            try:
                fig_interactive = self.topic_model.visualize_document_datamap(
                    docs,
                    reduced_embeddings=self.reduced_embeddings_2d,
                    custom_labels=True,
                    interactive=True
                )
                if hasattr(fig_interactive, 'write_html'):
                    fig_interactive.write_html(str(self.figure_dir / 'datamap_interactive.html'))
                elif hasattr(fig_interactive, 'save'):
                    fig_interactive.save(str(self.figure_dir / 'datamap_interactive.html'))
                print(f"     ✓ DataMapPlot交互式图已保存")
            except Exception as e:
                print(f"     ⚠️ 交互式DataMapPlot生成失败: {e}")
                
        except Exception as e:
            print(f"     ⚠️ DataMapPlot方法1失败: {e}")
            print("     尝试方法2: 直接使用 datamapplot 库...")
            
            # 方法2: 直接使用 datamapplot 库
            try:
                # 获取每个点的颜色（按topic分组）
                unique_topics = sorted(set(topics))
                n_colors = len(unique_topics)
                color_palette = plt.cm.Set3(np.linspace(0, 1, n_colors))
                topic_to_color_idx = {t: i for i, t in enumerate(unique_topics)}
                
                fig, ax = datamapplot.create_plot(
                    self.reduced_embeddings_2d,
                    topic_labels_per_doc,
                    title="Document Topic Map",
                    sub_title=f"Experiment: {self.experiment_id[:50]}",
                    label_font_size=10,
                    point_size=5,
                )
                fig.savefig(
                    str(self.figure_dir / 'datamap.png'),
                    bbox_inches='tight',
                    dpi=150
                )
                plt.close(fig)
                print(f"     ✓ DataMapPlot静态图已保存（方法2）")
                
            except Exception as e2:
                print(f"     ⚠️ DataMapPlot方法2也失败: {e2}")
    
    def _plot_documents_plotly(self):
        """使用Plotly生成文档可视化"""
        print("   - 生成Plotly文档可视化...")
        
        docs = self.cache_data['docs']
        
        try:
            fig = self.topic_model.visualize_documents(
                docs,
                reduced_embeddings=self.reduced_embeddings_2d,
                hide_document_hover=False,
                custom_labels=True  # 使用自定义标签
            )
            fig.write_html(str(self.figure_dir / 'documents_plotly.html'))
            print(f"     ✓ Plotly文档可视化已保存")
        except Exception as e:
            print(f"     ⚠️ Plotly文档可视化失败: {e}")
    
    def _save_representative_docs_report(self):
        """保存代表性文献报告 - 3篇，预览扩展至600字符"""
        print("   - 生成代表性文献报告...")
        
        docs = self.cache_data['docs']
        keys = self.cache_data['keys']
        titles = self.cache_data['titles']
        years = self.cache_data.get('years', [None] * len(docs))
        df = self.cache_data['df']
        topics = self.cache_data['topics']
        
        report_lines = ["# 各主题代表性文献\n\n"]
        report_lines.append(f"**实验ID:** `{self.experiment_id}`\n\n")
        report_lines.append("**配置说明:**\n")
        report_lines.append("- 关键词: n-gram(1-3) + MMR生成20个\n")
        report_lines.append("- 代表性文献: 3篇\n")
        report_lines.append("- 摘要预览: 600字符\n")
        report_lines.append("- 主题标签: 基于独特性选择（权重高且在其他主题中出现少的词）\n\n")
        report_lines.append("---\n\n")
        
        topic_info = self.topic_model.get_topic_info()
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue
            
            # 使用关键词
            topic_words = self.topic_words.get(topic_id, [])
            n_keywords = len(topic_words)
            keywords = ", ".join([w[0] for w in topic_words]) if topic_words else "N/A"
            
            # 获取独特性标签
            unique_label = self.unique_topic_labels.get(topic_id, f"Topic {topic_id}")
            
            report_lines.append(f"## Topic {topic_id}: {unique_label} ({row['Count']} documents)\n\n")
            report_lines.append(f"**All Keywords ({n_keywords}):**\n\n")
            report_lines.append(f"> {keywords}\n\n")
            report_lines.append("**Representative Documents:**\n\n")
            
            try:
                rep_docs = self.topic_model.get_representative_docs(topic_id)
                if rep_docs:
                    doc_count = 0
                    for doc_text in rep_docs[:3]:  # 保持3篇
                        for idx, orig_doc in enumerate(docs):
                            if doc_text in orig_doc or orig_doc in doc_text:
                                doc_count += 1
                                title = titles[idx] if titles else keys[idx]
                                year = years[idx] if years else "N/A"
                                abstract = df.iloc[idx].get('Abstract Note', '')
                                # 预览扩展至600字符
                                preview = abstract[:600] + '...' if len(abstract) > 600 else abstract
                                
                                prob = self._get_doc_probability(idx, topic_id)
                                prob_str = f"{prob:.4f}" if prob is not None else "N/A"
                                
                                report_lines.append(f"### {doc_count}. {title}\n\n")
                                report_lines.append(f"- **Key:** {keys[idx]}\n")
                                report_lines.append(f"- **Year:** {year}\n")
                                report_lines.append(f"- **Topic Probability:** {prob_str}\n")
                                report_lines.append(f"- **Abstract Preview:**\n\n")
                                report_lines.append(f"> {preview}\n\n")
                                break
                    
                    if doc_count == 0:
                        report_lines.append("_No representative documents found._\n\n")
                else:
                    report_lines.append("_No representative documents available._\n\n")
                    
            except Exception as e:
                report_lines.append(f"_Error retrieving representative documents: {e}_\n\n")
            
            report_lines.append("\n---\n\n")
        
        report_path = self.figure_dir / 'representative_docs.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"     ✓ 代表性文献报告已保存")
    
    def _get_doc_probability(self, doc_idx: int, topic_id: int) -> Optional[float]:
        """获取文档属于特定主题的概率"""
        if self.probs_matrix is None:
            return None
        
        topic_info = self.topic_model.get_topic_info()
        valid_topics = sorted([t for t in topic_info['Topic'] if t != -1])
        
        if topic_id not in valid_topics:
            return None
        
        try:
            col_idx = valid_topics.index(topic_id)
            if doc_idx < self.probs_matrix.shape[0] and col_idx < self.probs_matrix.shape[1]:
                return float(self.probs_matrix[doc_idx, col_idx])
        except:
            pass
        
        return None
    
    def _save_probability_distribution(self):
        """保存每篇文章属于各topic的概率分布"""
        print("   - 生成概率分布报告...")
        
        docs = self.cache_data['docs']
        keys = self.cache_data['keys']
        titles = self.cache_data['titles']
        topics = self.cache_data['topics']
        df = self.cache_data['df']
        
        topic_info = self.topic_model.get_topic_info()
        valid_topics = sorted([t for t in topic_info['Topic'] if t != -1])
        
        prob_data = {
            'Key': keys,
            'Title': titles,
            'Assigned_Topic': topics,
            'Topic_Label': [self.unique_topic_labels.get(t, f"Topic {t}") for t in topics],
        }
        
        if self.probs_matrix is not None:
            for i, topic_id in enumerate(valid_topics):
                if i < self.probs_matrix.shape[1]:
                    prob_data[f'Prob_Topic_{topic_id}'] = self.probs_matrix[:, i]
                else:
                    prob_data[f'Prob_Topic_{topic_id}'] = [0.0] * len(keys)
        
        prob_df = pd.DataFrame(prob_data)
        
        if self.probs_matrix is not None:
            assigned_probs = []
            second_probs = []
            second_topics = []
            
            for idx in range(len(topics)):
                topic_id = topics[idx]
                if topic_id == -1:
                    assigned_probs.append(0.0)
                    second_probs.append(0.0)
                    second_topics.append(-1)
                else:
                    if topic_id in valid_topics:
                        col_idx = valid_topics.index(topic_id)
                        assigned_probs.append(float(self.probs_matrix[idx, col_idx]))
                    else:
                        assigned_probs.append(0.0)
                    
                    probs_row = self.probs_matrix[idx]
                    sorted_indices = np.argsort(probs_row)[::-1]
                    if len(sorted_indices) > 1:
                        second_idx = sorted_indices[1]
                        second_probs.append(float(probs_row[second_idx]))
                        second_topics.append(valid_topics[second_idx] if second_idx < len(valid_topics) else -1)
                    else:
                        second_probs.append(0.0)
                        second_topics.append(-1)
            
            prob_df['Assigned_Probability'] = assigned_probs
            prob_df['Second_Topic'] = second_topics
            prob_df['Second_Probability'] = second_probs
            prob_df['Prob_Difference'] = np.array(assigned_probs) - np.array(second_probs)
        
        prob_excel_path = self.figure_dir / 'probability_distribution.xlsx'
        
        with pd.ExcelWriter(prob_excel_path, engine='openpyxl') as writer:
            prob_df.to_excel(writer, sheet_name='Full_Probabilities', index=False)
            
            summary_data = []
            for topic_id in valid_topics:
                topic_mask = np.array(topics) == topic_id
                n_docs = topic_mask.sum()
                
                if self.probs_matrix is not None and topic_id in valid_topics:
                    col_idx = valid_topics.index(topic_id)
                    topic_probs = self.probs_matrix[topic_mask, col_idx]
                    mean_prob = np.mean(topic_probs) if len(topic_probs) > 0 else 0
                    std_prob = np.std(topic_probs) if len(topic_probs) > 0 else 0
                    min_prob = np.min(topic_probs) if len(topic_probs) > 0 else 0
                    max_prob = np.max(topic_probs) if len(topic_probs) > 0 else 0
                else:
                    mean_prob = std_prob = min_prob = max_prob = 0
                
                topic_kw = self.topic_words.get(topic_id, [])
                keywords_str = ", ".join([w[0] for w in topic_kw[:10]])
                unique_label = self.unique_topic_labels.get(topic_id, f"Topic {topic_id}")
                
                summary_data.append({
                    'Topic': topic_id,
                    'Unique_Label': unique_label,
                    'N_Documents': n_docs,
                    'Keywords_Top10': keywords_str,
                    'Mean_Probability': mean_prob,
                    'Std_Probability': std_prob,
                    'Min_Probability': min_prob,
                    'Max_Probability': max_prob
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Topic_Summary', index=False)
            
            if 'Assigned_Probability' in prob_df.columns:
                low_conf_df = prob_df[prob_df['Assigned_Probability'] < 0.5].copy()
                low_conf_df = low_conf_df.sort_values('Assigned_Probability')
                low_conf_df.to_excel(writer, sheet_name='Low_Confidence_Docs', index=False)
            
            if 'Prob_Difference' in prob_df.columns:
                boundary_df = prob_df[prob_df['Prob_Difference'] < 0.1].copy()
                boundary_df = boundary_df.sort_values('Prob_Difference')
                boundary_df.to_excel(writer, sheet_name='Boundary_Documents', index=False)
        
        print(f"     ✓ 概率分布报告已保存: {prob_excel_path}")
        
        self._plot_probability_distribution(prob_df, valid_topics)
    
    def _plot_probability_distribution(self, prob_df: pd.DataFrame, valid_topics: List[int]):
        """绘制概率分布图"""
        print("   - 生成概率分布可视化...")
        
        if 'Assigned_Probability' not in prob_df.columns:
            print("     ⚠️ 无概率数据，跳过可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        ax1 = axes[0, 0]
        non_outlier_probs = prob_df[prob_df['Assigned_Topic'] != -1]['Assigned_Probability']
        ax1.hist(non_outlier_probs, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=non_outlier_probs.mean(), color='red', linestyle='--', 
                   label=f'Mean: {non_outlier_probs.mean():.3f}')
        ax1.set_xlabel('Assigned Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Assigned Probabilities')
        ax1.legend()
        
        ax2 = axes[0, 1]
        topic_probs = []
        topic_labels = []
        for topic_id in valid_topics:
            topic_mask = prob_df['Assigned_Topic'] == topic_id
            if topic_mask.sum() > 0:
                topic_probs.append(prob_df.loc[topic_mask, 'Assigned_Probability'].values)
                topic_labels.append(f'T{topic_id}')
        
        if topic_probs:
            bp = ax2.boxplot(topic_probs, labels=topic_labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(topic_probs)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax2.set_xlabel('Topic')
            ax2.set_ylabel('Probability')
            ax2.set_title('Probability Distribution by Topic')
            ax2.tick_params(axis='x', rotation=45)
        
        ax3 = axes[1, 0]
        if 'Prob_Difference' in prob_df.columns:
            non_outlier_diff = prob_df[prob_df['Assigned_Topic'] != -1]['Prob_Difference']
            ax3.hist(non_outlier_diff, bins=30, edgecolor='black', alpha=0.7, color='orange')
            ax3.axvline(x=0.1, color='red', linestyle='--', label='Threshold: 0.1')
            ax3.set_xlabel('Probability Difference (Assigned - Second)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Probability Differences')
            ax3.legend()
        
        ax4 = axes[1, 1]
        total_docs = len(prob_df)
        outliers = (prob_df['Assigned_Topic'] == -1).sum()
        low_conf = (prob_df['Assigned_Probability'] < 0.5).sum() if 'Assigned_Probability' in prob_df.columns else 0
        boundary = (prob_df['Prob_Difference'] < 0.1).sum() if 'Prob_Difference' in prob_df.columns else 0
        high_conf = total_docs - outliers - low_conf
        
        categories = ['High Confidence\n(prob≥0.5)', 'Low Confidence\n(prob<0.5)', 
                     'Boundary\n(diff<0.1)', 'Outliers']
        values = [max(0, high_conf), low_conf, boundary, outliers]
        colors_bar = ['green', 'orange', 'yellow', 'red']
        
        bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black')
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val}\n({val/total_docs*100:.1f}%)', ha='center', va='bottom', fontsize=9)
        ax4.set_ylabel('Number of Documents')
        ax4.set_title('Document Classification Quality')
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'probability_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     ✓ 概率分布可视化已保存")


def main():
    parser = argparse.ArgumentParser(description='BERTopic 实验可视化 - 改进版 V3')
    parser.add_argument('--base_dir', type=str,
                        default=str(Path(__file__).resolve().parents[2]),
                        help='项目根目录')
    parser.add_argument('--experiment_id', '-e', type=str, required=True,
                        help='实验ID')
    args = parser.parse_args()
    
    visualizer = ImprovedExperimentVisualizer(args.base_dir, args.experiment_id)
    
    if visualizer.load_experiment():
        visualizer.generate_all_visualizations()
    else:
        print("❌ 加载实验失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

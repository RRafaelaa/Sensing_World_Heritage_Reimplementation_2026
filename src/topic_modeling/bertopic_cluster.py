#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic 文献聚类分析
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from umap import UMAP
from hdbscan import HDBSCAN

# Gensim for coherence calculation
try:
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️  Warning: gensim not installed. Topic coherence will not be calculated.")

warnings.filterwarnings('ignore')
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class ImprovedBERTopicClusterer:
    """改进版 BERTopic 文献聚类器 V3"""
    
    def __init__(self, base_dir: str, args: argparse.Namespace):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "outputs"
        self.figure_dir = self.output_dir / "figures"
        self.model_dir = self.base_dir / "models"
        self.log_dir = self.output_dir / "experiment_logs"
        self.cache_dir = self.output_dir / "model_cache"
        
        # 创建目录
        for dir_path in [self.output_dir, self.figure_dir, self.log_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 保存参数
        self.args = args
        # Download from Hugging Face by default. A local model directory can be
        # supplied with --model_path for offline runs.
        self.model_path = args.model_path or "sentence-transformers/all-MiniLM-L6-v2"
        self.skip_visualization = getattr(args, 'skip_visualization', False)
        self.metrics_only = getattr(args, 'metrics_only', False)
        
        # 生成实验ID
        self.experiment_id = self._generate_experiment_id()
        
        # 数据存储
        self.df = None
        self.docs = None
        self.keys = None
        self.titles = None
        self.years = None
        self.embeddings = None
        self.umap_embeddings = None
        self.reduced_embeddings_2d = None
        self.tokenized_docs = None
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.embedding_model = None
    
    def _generate_experiment_id(self) -> str:
        """生成唯一的实验ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_str = (f"umap_n{self.args.umap_n_neighbors}_c{self.args.umap_n_components}_"
                      f"d{self.args.umap_min_dist}_hdb_c{self.args.hdbscan_min_cluster_size}_"
                      f"s{self.args.hdbscan_min_samples}_{self.args.hdbscan_cluster_selection}")
        if self.args.reduce_outliers:
            params_str += (f"_ro_{self.args.outlier_strategy}"
                           f"_t{self.args.outlier_threshold}")
        else:
            params_str += "_ro_off"
        return f"{timestamp}_{params_str}"
    
    def _get_params_dict(self) -> Dict:
        """获取当前参数字典"""
        return {
            'umap': {
                'n_neighbors': self.args.umap_n_neighbors,
                'n_components': self.args.umap_n_components,
                'min_dist': self.args.umap_min_dist,
                'metric': 'cosine'
            },
            'hdbscan': {
                'min_cluster_size': self.args.hdbscan_min_cluster_size,
                'min_samples': self.args.hdbscan_min_samples,
                'cluster_selection_method': self.args.hdbscan_cluster_selection,
                'metric': 'euclidean'
            },
            'bertopic': {
                'nr_topics': self.args.n_topics,
                'top_n_words': 20,
                'ngram_range': '(1, 3)',
                'representation_model': 'KeyBERTInspired + MMR',
                'calculate_probabilities': True,
                'reduce_outliers': self.args.reduce_outliers,
                'outlier_strategy': self.args.outlier_strategy if self.args.reduce_outliers else None,
                'outlier_threshold': self.args.outlier_threshold if self.args.reduce_outliers else None,
            }
        }
    
    def load_data(self, data_path: str, sheet_name: str = "meta_data") -> pd.DataFrame:
        """加载数据 - 拼接 Title + Abstract Note + All Keywords"""
        print(f"[1/7] 加载数据: {data_path}")
        
        file_path = Path(data_path)
        if not file_path.is_absolute():
            file_path = self.data_dir / file_path
        if file_path.suffix.lower() == '.csv':
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 选取需要的列
        required_cols = ['Key', 'Title', 'Abstract Note', 'All Keywords', 'Publication Year', 'Author']
        available_cols = [col for col in required_cols if col in self.df.columns]
        self.df = self.df[available_cols].copy()
        
        # 过滤无效数据
        self.df = self.df.dropna(subset=['Abstract Note'])
        self.df = self.df[self.df['Abstract Note'].str.strip() != '']
        self.df = self.df.reset_index(drop=True)
        
        # 拼接文本：Title + Abstract Note + All Keywords
        def combine_text(row):
            parts = []
            if pd.notna(row.get('Title')) and str(row['Title']).strip():
                parts.append(str(row['Title']).strip())
            if pd.notna(row.get('Abstract Note')) and str(row['Abstract Note']).strip():
                parts.append(str(row['Abstract Note']).strip())
            if pd.notna(row.get('All Keywords')) and str(row['All Keywords']).strip():
                parts.append(str(row['All Keywords']).strip())
            return ' '.join(parts)
        
        self.df['Combined_Text'] = self.df.apply(combine_text, axis=1)
        
        self.docs = self.df['Combined_Text'].tolist()
        self.keys = self.df['Key'].tolist()
        self.titles = self.df['Title'].tolist() if 'Title' in self.df.columns else self.keys
        
        # 处理年份
        if 'Publication Year' in self.df.columns:
            self.df['Publication Year'] = pd.to_numeric(self.df['Publication Year'], errors='coerce')
            self.years = self.df['Publication Year'].tolist()
        else:
            self.years = None
        
        print(f"   ✓ 有效文档数量: {len(self.docs)}")
        print(f"   ✓ 文本拼接: Title + Abstract Note + All Keywords")
        return self.df
    
    def generate_embeddings(self) -> np.ndarray:
        """生成文档嵌入"""
        cache_path = None
        if self.args.embeddings_cache:
            cache_path = Path(self.args.embeddings_cache)
            if not cache_path.is_absolute():
                cache_path = self.base_dir / cache_path
            if cache_path.exists():
                self.embeddings = np.load(cache_path)
                if self.embeddings.shape[0] != len(self.docs):
                    raise ValueError(
                        f"Embedding cache has {self.embeddings.shape[0]} rows; "
                        f"input has {len(self.docs)} documents"
                    )
                print(f"[2/7] 从缓存加载嵌入: {cache_path}")
                return self.embeddings

        print(f"[2/7] 加载本地模型并生成嵌入: {self.model_path}")
        
        self.embedding_model = SentenceTransformer(self.model_path)
        self.embeddings = self.embedding_model.encode(
            self.docs, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.embeddings)
            print(f"   ✓ 嵌入缓存已保存: {cache_path}")
        
        print(f"   ✓ 嵌入维度: {self.embeddings.shape}")
        return self.embeddings
    
    def _tokenize_docs(self) -> List[List[str]]:
        """预处理文档用于coherence计算"""
        print("[3/7] 预处理文档...")
        
        self.tokenized_docs = []
        for doc in self.docs:
            tokens = doc.lower().split()
            tokens = [t.strip('.,!?;:()[]{}"\'-') for t in tokens]
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2 and t.isalpha()]
            self.tokenized_docs.append(tokens)
        
        print(f"   ✓ 文档分词完成")
        return self.tokenized_docs
    
    def build_topic_model(self) -> BERTopic:
        """构建并训练BERTopic模型"""
        print(f"[4/7] 构建BERTopic模型 (目标主题数: {self.args.n_topics})")
        print(f"   UMAP参数: n_neighbors={self.args.umap_n_neighbors}, "
              f"n_components={self.args.umap_n_components}, min_dist={self.args.umap_min_dist}")
        print(f"   HDBSCAN参数: min_cluster_size={self.args.hdbscan_min_cluster_size}, "
              f"min_samples={self.args.hdbscan_min_samples}, method={self.args.hdbscan_cluster_selection}")
        
        # UMAP模型
        umap_model = UMAP(
            n_neighbors=self.args.umap_n_neighbors,
            n_components=self.args.umap_n_components,
            min_dist=self.args.umap_min_dist,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN模型
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.args.hdbscan_min_cluster_size,
            min_samples=self.args.hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method=self.args.hdbscan_cluster_selection,
            prediction_data=True
        )
        
        # 向量化模型 - N-gram改为(1, 3)
        vectorizer_model = CountVectorizer(
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3)
        )
        
        # c-TF-IDF模型
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        
        # 链式representation model - 增加 top_n_words 到 20
        keybert_model = KeyBERTInspired(top_n_words=20)
        mmr_model = MaximalMarginalRelevance(diversity=0.3)
        representation_model = [keybert_model, mmr_model]
        
        print(f"   ✓ Representation Model: KeyBERTInspired(top_n_words=20) + MMR(diversity=0.3)")
        
        # 构建BERTopic模型 - top_n_words 设为 20
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            nr_topics=self.args.n_topics,
            top_n_words=20,
            calculate_probabilities=True,
            verbose=True
        )
        
        # 训练模型
        self.topics, self.probs = self.topic_model.fit_transform(
            self.docs, 
            embeddings=self.embeddings
        )

        # The paper reports BERTopic outlier reduction with the distributions
        # strategy and threshold 0.1. Keep it optional so the matched ablation
        # without reassignment can be reproduced from the same script.
        if self.args.reduce_outliers:
            self.topics = self.topic_model.reduce_outliers(
                self.docs,
                self.topics,
                probabilities=self.probs,
                strategy=self.args.outlier_strategy,
                threshold=self.args.outlier_threshold,
            )
            self.topic_model.update_topics(
                self.docs,
                topics=self.topics,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                representation_model=representation_model,
            )
        
        # 获取UMAP降维后的embeddings
        self.umap_embeddings = self.topic_model.umap_model.transform(self.embeddings)
        
        # 打印主题分布
        topic_info = self.topic_model.get_topic_info()
        print(f"\n   主题分布:")
        for _, row in topic_info.iterrows():
            print(f"   Topic {row['Topic']:2d}: {row['Count']:3d} 篇文档 - {row['Name'][:60]}...")
        
        return self.topic_model
    
    def get_representative_documents(self) -> Dict[int, List[Dict]]:
        """获取每个主题的代表性文献"""
        print("[5/7] 提取代表性文献...")
        
        representative_docs = {}
        topic_info = self.topic_model.get_topic_info()
        
        for topic_id in topic_info['Topic']:
            if topic_id == -1:
                continue
            
            try:
                rep_docs = self.topic_model.get_representative_docs(topic_id)
                if rep_docs:
                    docs_info = []
                    for doc_text in rep_docs[:5]:
                        for idx, orig_doc in enumerate(self.docs):
                            if doc_text in orig_doc or orig_doc in doc_text:
                                docs_info.append({
                                    'key': self.keys[idx],
                                    'title': self.titles[idx] if self.titles else self.keys[idx],
                                    'year': self.years[idx] if self.years else None,
                                    'abstract_preview': self.df.iloc[idx].get('Abstract Note', '')[:200] + '...'
                                })
                                break
                    representative_docs[topic_id] = docs_info
            except Exception as e:
                print(f"   ⚠️  Topic {topic_id} 获取代表性文献失败: {e}")
                representative_docs[topic_id] = []
        
        total_rep_docs = sum(len(docs) for docs in representative_docs.values())
        print(f"   ✓ 共提取 {total_rep_docs} 篇代表性文献")
        
        return representative_docs
    
    def evaluate_clustering(self) -> Dict[str, Any]:
        """计算综合评估指标"""
        print("[6/7] 评估聚类质量")
        
        metrics = {}
        
        non_outlier_mask = np.array(self.topics) != -1
        n_non_outliers = non_outlier_mask.sum()
        topics_non_outlier = np.array(self.topics)[non_outlier_mask]
        n_unique_topics = len(set(topics_non_outlier))
        
        # 聚类指标
        print("\n   --- 聚类指标 ---")
        
        if n_non_outliers > 1 and n_unique_topics > 1:
            embeddings_non_outlier = self.embeddings[non_outlier_mask]
            
            metrics['silhouette_original'] = float(silhouette_score(
                embeddings_non_outlier, topics_non_outlier, metric='cosine'
            ))
            metrics['calinski_harabasz_original'] = float(calinski_harabasz_score(
                embeddings_non_outlier, topics_non_outlier
            ))
            metrics['davies_bouldin_original'] = float(davies_bouldin_score(
                embeddings_non_outlier, topics_non_outlier
            ))
            
            umap_non_outlier = self.umap_embeddings[non_outlier_mask]
            metrics['silhouette_umap'] = float(silhouette_score(
                umap_non_outlier, topics_non_outlier
            ))
            
            print(f"   Silhouette (原始空间, cosine): {metrics['silhouette_original']:.4f}")
            print(f"   Silhouette (UMAP空间, 仅参考): {metrics['silhouette_umap']:.4f}")
            print(f"   Calinski-Harabasz (原始空间): {metrics['calinski_harabasz_original']:.2f}")
            print(f"   Davies-Bouldin (原始空间): {metrics['davies_bouldin_original']:.4f}")
        else:
            metrics['silhouette_original'] = None
            metrics['silhouette_umap'] = None
            metrics['calinski_harabasz_original'] = None
            metrics['davies_bouldin_original'] = None
            print("   ⚠️  主题数不足，无法计算聚类指标")
        
        # 主题模型指标
        print("\n   --- 主题模型指标 ---")
        
        metrics['topic_diversity'] = self._calculate_topic_diversity()
        print(f"   Topic Diversity: {metrics['topic_diversity']:.4f}")
        
        if GENSIM_AVAILABLE and self.tokenized_docs:
            coherence_scores = self._calculate_topic_coherence()
            metrics.update(coherence_scores)
            if metrics.get('coherence_cv') is not None:
                print(f"   Topic Coherence (c_v): {metrics['coherence_cv']:.4f}")
            else:
                print("   ⚠️  Topic Coherence (c_v): 计算失败")
        else:
            metrics['coherence_cv'] = None
            metrics['coherence_npmi'] = None
        
        # 分布指标
        print("\n   --- 分布指标 ---")
        
        topic_counts = Counter(self.topics)
        n_outliers = topic_counts.get(-1, 0)
        n_topics = len([t for t in topic_counts.keys() if t != -1])
        
        metrics['n_topics'] = n_topics
        metrics['n_outliers'] = n_outliers
        metrics['outlier_ratio'] = n_outliers / len(self.topics)
        metrics['total_docs'] = len(self.topics)
        
        non_outlier_sizes = [c for t, c in topic_counts.items() if t != -1]
        if non_outlier_sizes:
            metrics['topic_size_mean'] = float(np.mean(non_outlier_sizes))
            metrics['topic_size_std'] = float(np.std(non_outlier_sizes))
            metrics['topic_size_cv'] = metrics['topic_size_std'] / metrics['topic_size_mean'] if metrics['topic_size_mean'] > 0 else 0
            metrics['topic_size_min'] = int(min(non_outlier_sizes))
            metrics['topic_size_max'] = int(max(non_outlier_sizes))
        
        print(f"   有效主题数: {n_topics}")
        print(f"   离群点数量: {n_outliers} ({metrics['outlier_ratio']:.1%})")
        
        # 概率质量指标
        prob_metrics = self._calculate_probability_metrics()
        metrics.update(prob_metrics)
        
        return metrics
    
    def _calculate_topic_diversity(self, top_n: int = 20) -> float:
        """计算主题多样性 - 使用20个词"""
        topic_info = self.topic_model.get_topic_info()
        all_words = []
        
        for topic_id in topic_info['Topic']:
            if topic_id == -1:
                continue
            topic_words = self.topic_model.get_topic(topic_id)
            if topic_words:
                words = [w[0] for w in topic_words[:top_n]]
                all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words)
        return float(diversity)
    
    def _calculate_topic_coherence(self) -> Dict[str, Optional[float]]:
        """计算主题一致性 - 修复版：只使用单词，过滤n-gram"""
        if not GENSIM_AVAILABLE or not self.tokenized_docs:
            return {"coherence_cv": None, "coherence_npmi": None}
        
        # 构建字典
        dictionary = Dictionary(self.tokenized_docs)
        
        # 获取字典中的所有有效词汇
        valid_vocab = set(dictionary.token2id.keys())
        
        topic_words = []
        topic_info = self.topic_model.get_topic_info()
        
        for topic_id in topic_info['Topic']:
            if topic_id == -1:
                continue
            words = self.topic_model.get_topic(topic_id)
            if words:
                # 只保留单词（非n-gram），且必须在字典中
                filtered_words = []
                for w in words[:20]:  # 取前20个
                    word = w[0].lower()
                    # 检查是否是单词（不含空格）且在字典中
                    if ' ' not in word and word in valid_vocab:
                        filtered_words.append(word)
                    elif ' ' in word:
                        # 对于n-gram，尝试提取其中的单词
                        for subword in word.split():
                            subword = subword.strip()
                            if subword in valid_vocab and subword not in filtered_words:
                                filtered_words.append(subword)
                
                # 确保每个主题至少有5个有效词
                if len(filtered_words) >= 5:
                    topic_words.append(filtered_words[:10])  # 使用前10个有效词计算coherence
        
        if len(topic_words) < 2:
            print("   ⚠️  有效主题词过少，无法计算coherence")
            return {"coherence_cv": None, "coherence_npmi": None}
        
        results = {}
        
        # 计算 c_v coherence
        try:
            coherence_cv = CoherenceModel(
                topics=topic_words,
                texts=self.tokenized_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            results["coherence_cv"] = float(coherence_cv.get_coherence())
        except Exception as e:
            print(f"   ⚠️  c_v coherence计算失败: {e}")
            results["coherence_cv"] = None
        
        # 计算 c_npmi coherence（可选，不参与排名）
        try:
            coherence_npmi = CoherenceModel(
                topics=topic_words,
                texts=self.tokenized_docs,
                dictionary=dictionary,
                coherence='c_npmi'
            )
            results["coherence_npmi"] = float(coherence_npmi.get_coherence())
        except Exception as e:
            results["coherence_npmi"] = None
        
        return results
    
    def _calculate_probability_metrics(self) -> Dict[str, Any]:
        """计算概率相关的质量指标"""
        metrics = {}
        
        if self.probs is None:
            return {"prob_mean": None, "prob_std": None, "low_confidence_ratio": None}
        
        assigned_probs = self._get_assigned_probabilities()
        non_outlier_mask = np.array(self.topics) != -1
        non_outlier_probs = assigned_probs[non_outlier_mask]
        
        if len(non_outlier_probs) > 0:
            metrics["prob_mean"] = float(np.mean(non_outlier_probs))
            metrics["prob_std"] = float(np.std(non_outlier_probs))
            metrics["low_confidence_ratio"] = float(np.mean(non_outlier_probs < 0.5))
        else:
            metrics["prob_mean"] = None
            metrics["prob_std"] = None
            metrics["low_confidence_ratio"] = None
        
        return metrics
    
    def _get_assigned_probabilities(self) -> np.ndarray:
        """获取每个文档被分配到其topic的概率"""
        assigned_probs = np.zeros(len(self.topics))
        
        for i, topic_id in enumerate(self.topics):
            if topic_id == -1:
                assigned_probs[i] = 0.0
            else:
                if isinstance(self.probs, np.ndarray):
                    if self.probs.ndim == 1:
                        assigned_probs[i] = float(self.probs[i]) if self.probs[i] is not None else 0.0
                    elif self.probs.ndim == 2:
                        topic_info = self.topic_model.get_topic_info()
                        valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
                        if topic_id in valid_topics:
                            col_idx = valid_topics.index(topic_id)
                            if col_idx < self.probs.shape[1]:
                                assigned_probs[i] = float(self.probs[i, col_idx])
                            else:
                                assigned_probs[i] = float(np.max(self.probs[i]))
                        else:
                            assigned_probs[i] = 0.0
                else:
                    try:
                        if hasattr(self.probs[i], '__len__'):
                            assigned_probs[i] = float(max(self.probs[i]))
                        else:
                            assigned_probs[i] = float(self.probs[i]) if self.probs[i] is not None else 0.0
                    except:
                        assigned_probs[i] = 0.0
        
        return assigned_probs
    
    def save_cache(self) -> str:
        """保存模型和数据缓存用于后续可视化"""
        cache_path = self.cache_dir / f"cache_{self.experiment_id}.pkl"
        
        cache_data = {
            'experiment_id': self.experiment_id,
            'docs': self.docs,
            'keys': self.keys,
            'titles': self.titles,
            'years': self.years,
            'embeddings': self.embeddings,
            'topics': self.topics,
            'probs': self.probs,
            'df': self.df,
            'params': self._get_params_dict()
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # 保存BERTopic模型
        model_path = self.cache_dir / f"model_{self.experiment_id}"
        self.topic_model.save(str(model_path), serialization="pickle", save_ctfidf=True)
        
        print(f"   ✓ 缓存已保存: {cache_path}")
        return str(cache_path)
    
    def save_results(self, representative_docs: Dict = None) -> str:
        """保存聚类结果"""
        print("[7/7] 保存聚类结果")
        
        output_filename = f"cluster_results_{self.experiment_id}.xlsx"
        
        result_df = self.df.copy()
        result_df['Topic'] = self.topics
        
        assigned_probs = self._get_assigned_probabilities()
        result_df['Topic_Probability'] = assigned_probs
        
        topic_info = self.topic_model.get_topic_info()
        topic_labels = dict(zip(topic_info['Topic'], topic_info['Name']))
        result_df['Topic_Label'] = result_df['Topic'].map(topic_labels)
        
        # 获取20个关键词
        def get_keywords(topic_id):
            if topic_id == -1:
                return "Outlier"
            words = self.topic_model.get_topic(topic_id)
            return ", ".join([w[0] for w in words[:20]]) if words else ""
        
        result_df['Topic_Keywords'] = result_df['Topic'].apply(get_keywords)
        result_df['Is_Outlier'] = result_df['Topic'] == -1
        
        output_path = self.output_dir / output_filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name='Cluster_Results', index=False)
            topic_info.to_excel(writer, sheet_name='Topic_Overview', index=False)
            
            for topic_id in sorted(set(self.topics)):
                if topic_id == -1:
                    sheet_name = 'Outliers'
                else:
                    sheet_name = f'Topic_{topic_id}'
                
                topic_docs = result_df[result_df['Topic'] == topic_id][
                    ['Key', 'Title', 'Author', 'Publication Year', 'Topic_Probability']
                ]
                topic_docs.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 保存代表性文献
            if representative_docs:
                rep_rows = []
                for topic_id, docs in representative_docs.items():
                    for doc in docs:
                        rep_rows.append({
                            'Topic': topic_id,
                            'Key': doc.get('key'),
                            'Title': doc.get('title'),
                            'Year': doc.get('year'),
                            'Abstract_Preview': doc.get('abstract_preview')
                        })
                if rep_rows:
                    pd.DataFrame(rep_rows).to_excel(writer, sheet_name='Representative_Docs', index=False)
        
        print(f"   ✓ 结果已保存至: {output_path}")
        return str(output_path)
    
    def save_experiment_log(self, metrics: Dict, representative_docs: Dict = None) -> str:
        """保存实验日志"""
        log_entry = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'parameters': self._get_params_dict(),
            'metrics': metrics,
            'topic_distribution': {},
            'cache_path': None if self.metrics_only else str(self.cache_dir / f"cache_{self.experiment_id}.pkl"),
            'model_path': None if self.metrics_only else str(self.cache_dir / f"model_{self.experiment_id}")
        }
        
        topic_info = self.topic_model.get_topic_info()
        for _, row in topic_info.iterrows():
            log_entry['topic_distribution'][f"topic_{row['Topic']}"] = {
                'count': int(row['Count']),
                'name': row['Name'][:100]
            }
        
        log_path = self.log_dir / f"experiment_{self.experiment_id}.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        summary_path = self.log_dir / "experiment_summary.jsonl"
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        print(f"   ✓ 实验日志已保存至: {log_path}")
        return str(log_path)
    
    def run(self, data_path: str = "topic_modeling/meta_data.csv") -> Dict:
        """运行完整的聚类分析流程"""
        print("=" * 70)
        print("BERTopic 文献聚类分析 - 改进版 V3")
        print(f"实验ID: {self.experiment_id}")
        if self.skip_visualization:
            print("模式: 仅计算指标（跳过可视化）")
        print("=" * 70)
        
        self.load_data(data_path)
        self.generate_embeddings()
        self._tokenize_docs()
        self.build_topic_model()
        representative_docs = {} if self.metrics_only else self.get_representative_documents()
        metrics = self.evaluate_clustering()
        if not self.metrics_only:
            self.save_cache()
            self.save_results(representative_docs)
        self.save_experiment_log(metrics, representative_docs)
        
        print("=" * 70)
        print("✓ 聚类分析完成!")
        print("=" * 70)
        
        return metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='BERTopic 文献聚类分析 - 改进版 V3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--base_dir', type=str,
                        default=str(Path(__file__).resolve().parents[2]),
                        help='项目根目录')
    parser.add_argument('--data_file', type=str, default='topic_modeling/meta_data.csv',
                        help='data/ 下的 CSV/XLSX 输入文件，或绝对路径')
    parser.add_argument('--model_path', type=str, default=None,
                        help='嵌入模型路径')
    parser.add_argument('--embeddings_cache', type=str,
                        default='outputs/cache/embeddings_all_minilm_l6_v2.npy',
                        help='复用文档嵌入的 .npy 缓存；设为空字符串可禁用')
    parser.add_argument('--n_topics', type=lambda x: x if x == 'auto' else int(x), default='auto',
                        help='目标主题数，或 auto')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='跳过可视化（用于批量实验）')
    parser.add_argument('--metrics_only', action='store_true',
                        help='只保存指标日志，不保存每个配置的模型缓存和结果工作簿')
    
    # UMAP 参数
    parser.add_argument('--umap_n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors参数')
    parser.add_argument('--umap_n_components', type=int, default=5,
                        help='UMAP n_components参数')
    parser.add_argument('--umap_min_dist', type=float, default=0.0,
                        help='UMAP min_dist参数')
    
    # HDBSCAN 参数
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=10,
                        help='HDBSCAN min_cluster_size参数')
    parser.add_argument('--hdbscan_min_samples', type=int, default=5,
                        help='HDBSCAN min_samples参数')
    parser.add_argument('--hdbscan_cluster_selection', type=str, default='eom',
                        choices=['eom', 'leaf'],
                        help='HDBSCAN cluster_selection_method参数')

    # Outlier reduction reported in the manuscript.
    parser.add_argument('--reduce_outliers', action='store_true',
                        help='使用 BERTopic reduce_outliers 重新分配部分离群文档')
    parser.add_argument('--outlier_strategy', type=str, default='distributions',
                        choices=['distributions', 'probabilities', 'c-tf-idf', 'embeddings'],
                        help='BERTopic 离群点重分配策略')
    parser.add_argument('--outlier_threshold', type=float, default=0.1,
                        help='离群点重分配阈值')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    clusterer = ImprovedBERTopicClusterer(
        base_dir=args.base_dir,
        args=args
    )
    
    metrics = clusterer.run(args.data_file)
    return metrics


if __name__ == "__main__":
    main()

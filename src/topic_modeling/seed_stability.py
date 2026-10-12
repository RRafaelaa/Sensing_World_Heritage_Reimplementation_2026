#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_stability.py —— 随机种子稳定性检验（minor revision 补丁）

与 bertopic_cluster.py 的关系：
  - 继承 ImprovedBERTopicClusterer，复用 load_data / generate_embeddings /
    _tokenize_docs / evaluate_clustering；本脚本保留原始词干去重规则
  - 只重写建模步骤：加入 seed，以及 legacy / standard 两种主题表示口径
  - 不写 experiment_summary.jsonl，不写 model_cache，输出隔离在 outputs/seed_stability/

Step 1（replicate，legacy 口径）:
  用论文最终参数 + seed=42 复刻，并与已保存的实验日志 / cache 逐项核对：
  指标、每主题文档数、每主题关键词（逐词）、每篇文档归属。
  legacy 口径 = reduce_outliers 之后仅 update_topics(docs, topics=new_topics)
  （BERTopic 退回默认：unigram、不去停用词、top_n_words=10、无 representation_model）

Step 2（seeds，standard 口径）:
  固定同一组参数，多随机种子重跑（只变 UMAP random_state），所有 run 统一使用
  n-gram(1-3) + KeyBERTInspired→MMR、top_n_words=20、stem 去重。
  比较主题数、离群比例、主题内容、文档归属；seed=42 为参考 run。
  ⚠ Step 2 的 coherence/diversity 与论文中的 0.6736/0.3889 不是同一口径，不可直接对比。
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from umap import UMAP
from hdbscan import HDBSCAN

# 复用初稿流程
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bertopic_cluster import ImprovedBERTopicClusterer  # noqa: E402

warnings.filterwarnings('ignore')

DEFAULT_BASE_DIR = str(Path(__file__).resolve().parents[2])

# 论文最终采用的模型参数（Appendix B）
FINAL_PARAMS = dict(
    n_topics='auto',
    umap_n_neighbors=10,
    umap_n_components=8,
    umap_min_dist=0.1,
    hdbscan_min_cluster_size=10,
    hdbscan_min_samples=5,
    hdbscan_cluster_selection='eom',
    reduce_outliers=True,
    outlier_strategy='distributions',
    outlier_threshold=0.1,
)

METRIC_KEYS = [
    'n_topics', 'n_outliers', 'outlier_ratio',
    'silhouette_original', 'davies_bouldin_original', 'calinski_harabasz_original',
    'coherence_cv', 'coherence_npmi', 'topic_diversity', 'topic_size_cv',
    'prob_mean', 'low_confidence_ratio',
]


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def env_info() -> Dict[str, str]:
    """记录关键包版本，供审稿回复中的 reproducibility 说明使用"""
    from importlib.metadata import version, PackageNotFoundError
    info = {'python': sys.version.split()[0]}
    for pkg in ['bertopic', 'umap-learn', 'hdbscan', 'scikit-learn', 'numpy',
                'sentence-transformers', 'torch', 'gensim', 'numba']:
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = 'n/a'
    return info


def load_summary_entries(base_dir: Path) -> List[Dict]:
    path = base_dir / 'outputs' / 'experiment_logs' / 'experiment_summary.jsonl'
    if not path.exists():
        return []
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def params_match(entry: Dict, p: Dict) -> bool:
    try:
        u, h, ro = entry['parameters']['umap'], entry['parameters']['hdbscan'], entry['parameters']['reduce_outliers']
        return (u['n_neighbors'] == p['umap_n_neighbors']
                and u['n_components'] == p['umap_n_components']
                and np.isclose(float(u['min_dist']), p['umap_min_dist'])
                and h['min_cluster_size'] == p['hdbscan_min_cluster_size']
                and h['min_samples'] == p['hdbscan_min_samples']
                and h['cluster_selection_method'] == p['hdbscan_cluster_selection']
                and bool(ro.get('enabled')) == p['reduce_outliers']
                and (not p['reduce_outliers'] or (
                    ro.get('strategy') == p['outlier_strategy']
                    and np.isclose(float(ro.get('threshold')), p['outlier_threshold']))))
    except (KeyError, TypeError):
        return False


def find_reference(base_dir: Path, params: Dict, ref_id: Optional[str],
                   target_cv: Optional[float]) -> Optional[Dict]:
    """定位论文最终 run 的日志条目（优先 experiment_id，否则按参数 + coherence 最接近）"""
    entries = load_summary_entries(base_dir)
    if ref_id:
        for e in entries:
            if e['experiment_id'] == ref_id:
                return e
        p = base_dir / 'outputs' / 'experiment_logs' / f'experiment_{ref_id}.json'
        if p.exists():
            return json.load(open(p, 'r', encoding='utf-8'))
        print(f"❌ 未找到 experiment_id={ref_id} 的日志")
        return None

    cands = [e for e in entries if params_match(e, params)]
    if not cands:
        print("❌ experiment_summary.jsonl 中没有与最终参数匹配的实验")
        return None
    print(f"   与最终参数匹配的实验共 {len(cands)} 个：")
    for e in cands:
        m = e['metrics']
        print(f"     {e['experiment_id']}  cv={m.get('coherence_cv')}  "
              f"outlier={m.get('outlier_ratio')}  n_topics={m.get('n_topics')}")
    if target_cv is not None:
        cands.sort(key=lambda e: abs((e['metrics'].get('coherence_cv') or 9) - target_cv))
    else:
        cands.sort(key=lambda e: e['timestamp'], reverse=True)
    print(f"   → 选用: {cands[0]['experiment_id']}")
    return cands[0]


def load_cache(base_dir: Path, exp_id: str) -> Optional[Dict]:
    p = base_dir / 'outputs' / 'model_cache' / f'cache_{exp_id}.pkl'
    if not p.exists():
        print(f"   ⚠️ cache 不存在: {p}")
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def topic_label(words, k=5) -> str:
    return " | ".join(w[0] for w in words[:k]) if words else ""


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


# ----------------------------------------------------------------------------
# 单次 run 的结果容器
# ----------------------------------------------------------------------------
class SeedRun:
    def __init__(self, seed, mode, raw_topics, topics, metrics, keywords, elapsed):
        self.seed = seed
        self.mode = mode
        self.raw_topics = np.asarray(raw_topics)      # reduce_outliers 之前（HDBSCAN 原始）
        self.topics = np.asarray(topics)              # reduce_outliers 之后（最终归属）
        self.metrics = metrics
        self.keywords = keywords                      # {topic_id: [(word, weight), ...]}
        self.elapsed = elapsed

    @property
    def raw_outlier_ratio(self):
        return float(np.mean(self.raw_topics == -1))


# ----------------------------------------------------------------------------
# 复用初稿流程的 Runner
# ----------------------------------------------------------------------------
class SeedStabilityRunner(ImprovedBERTopicClusterer):

    def __init__(self, base_dir: str, params: Dict, data_file: str,
                 standard_top_n_words: int = 20, legacy_stem_dedup: bool = False):
        args = SimpleNamespace(model_path=None, skip_visualization=True,
                               embeddings_cache=None, data_file=data_file, **params)
        super().__init__(base_dir, args)
        self.standard_top_n_words = standard_top_n_words
        self.legacy_stem_dedup = legacy_stem_dedup
        self.stemmer = PorterStemmer()

    def _remove_stem_duplicates(self):
        """Remove duplicate topic terms using the original model's stemming rule."""
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop_words = set(ENGLISH_STOP_WORDS)
        for _, row in self.topic_model.get_topic_info().iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue

            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                continue

            seen_stems = {}
            deduplicated = []
            for word, weight in topic_words:
                tokens = word.lower().split()
                non_stop_tokens = [t for t in tokens if t not in stop_words]
                if len(non_stop_tokens) == 0:
                    continue
                if len(tokens) > 1 and (tokens[0] in stop_words or tokens[-1] in stop_words):
                    continue

                stem_signature = tuple(self.stemmer.stem(token) for token in tokens)
                if stem_signature not in seen_stems:
                    seen_stems[stem_signature] = (word, weight)
                    deduplicated.append((word, weight))

            if hasattr(self.topic_model, 'topic_representations_'):
                self.topic_model.topic_representations_[topic_id] = deduplicated

    # ---- 数据与嵌入只准备一次 ----
    def prepare(self, embeddings_source: str, cache: Optional[Dict]) -> Dict:
        info = {}
        self.load_data(self.args.data_file)

        if cache is not None:
            same_keys = list(cache.get('keys', [])) == list(self.keys)
            info['keys_match_cache'] = same_keys
            if not same_keys:
                print("   ⚠️ 当前数据的 Key 序列与 cache 不一致（数据文件可能变动），文档级对比将不可靠")

        if embeddings_source == 'cache':
            if cache is None or 'embeddings' not in cache:
                raise RuntimeError("--embeddings_source cache 但 cache 中无 embeddings")
            print(f"[2/7] 使用 cache 中的嵌入（跳过重新编码）")
            self.embedding_model = SentenceTransformer(self.model_path)
            self.embeddings = np.asarray(cache['embeddings'])
            info['embedding_max_abs_diff_vs_cache'] = 0.0
        else:
            self.generate_embeddings()
            if cache is not None and 'embeddings' in cache \
                    and np.shape(cache['embeddings']) == self.embeddings.shape:
                d = float(np.max(np.abs(np.asarray(cache['embeddings']) - self.embeddings)))
                info['embedding_max_abs_diff_vs_cache'] = d
                print(f"   嵌入与 cache 的最大绝对差: {d:.2e}"
                      + ("" if d < 1e-5 else "  ⚠️ 差异偏大，若 Step1 不能复刻请改用 --embeddings_source cache"))

        self._tokenize_docs()
        return info

    # ---- 模型组件（与 bertopic_cluster.build_topic_model 一致，仅 seed 可变）----
    def _components(self, seed: int):
        umap_model = UMAP(n_neighbors=self.args.umap_n_neighbors,
                          n_components=self.args.umap_n_components,
                          min_dist=self.args.umap_min_dist,
                          metric='cosine', random_state=seed)
        hdbscan_model = HDBSCAN(min_cluster_size=self.args.hdbscan_min_cluster_size,
                                min_samples=self.args.hdbscan_min_samples,
                                metric='euclidean',
                                cluster_selection_method=self.args.hdbscan_cluster_selection,
                                prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words='english', min_df=2, max_df=0.95,
                                           ngram_range=(1, 3))
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        representation_model = [KeyBERTInspired(top_n_words=40),
                                MaximalMarginalRelevance(diversity=0.3, top_n_words=25)]
        return umap_model, hdbscan_model, vectorizer_model, ctfidf_model, representation_model

    # ---- 单次拟合 ----
    def fit_one(self, seed: int, mode: str) -> SeedRun:
        assert mode in ('legacy', 'standard')
        t0 = time.time()
        np.random.seed(seed)
        umap_model, hdbscan_model, vectorizer_model, ctfidf_model, representation_model = self._components(seed)

        model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            nr_topics=None if self.args.n_topics == 'auto' else int(self.args.n_topics),
            top_n_words=20,
            calculate_probabilities=True,
            verbose=False,
        )
        raw_topics, probs = model.fit_transform(self.docs, embeddings=self.embeddings)
        topics = list(raw_topics)

        if self.args.reduce_outliers:
            new_topics = model.reduce_outliers(self.docs, raw_topics,
                                               strategy=self.args.outlier_strategy,
                                               threshold=self.args.outlier_threshold)
            if mode == 'legacy':
                # 历史行为：不传任何其他参数 → BERTopic 退回默认表示
                model.update_topics(self.docs, topics=new_topics)
            else:
                # 统一口径：n-gram(1-3) + c-TF-IDF + KeyBERTInspired→MMR, top_n_words=20
                model.update_topics(self.docs, topics=new_topics,
                                    top_n_words=self.standard_top_n_words,
                                    vectorizer_model=vectorizer_model,
                                    ctfidf_model=ctfidf_model,
                                    representation_model=representation_model)
            topics = new_topics

        self.topic_model = model
        self.topics = topics
        self.probs = probs
        self.umap_embeddings = model.umap_model.transform(self.embeddings)

        if mode == 'standard' or (mode == 'legacy' and self.legacy_stem_dedup):
            self._remove_stem_duplicates()

        metrics = self.evaluate_clustering()   # 与初稿完全相同的指标代码

        keywords = {}
        for tid in model.get_topic_info()['Topic']:
            if tid != -1:
                keywords[int(tid)] = [(w, float(s)) for w, s in (model.get_topic(tid) or [])]

        return SeedRun(seed, mode, raw_topics, topics, metrics, keywords, time.time() - t0)


# ----------------------------------------------------------------------------
# Step 1：复刻核对
# ----------------------------------------------------------------------------
def step1_replicate(runner: SeedStabilityRunner, ref_entry: Dict, cache: Optional[Dict],
                    prep_info: Dict, out_dir: Path, seed: int) -> Dict:
    print("\n" + "=" * 70)
    print(f"STEP 1  复刻论文最终模型（legacy 口径, seed={seed}）")
    print(f"        参考实验: {ref_entry['experiment_id']}")
    print("=" * 70)

    run = runner.fit_one(seed, mode='legacy')
    report = {'reference_experiment_id': ref_entry['experiment_id'], 'seed': seed,
              'mode': 'legacy', 'legacy_stem_dedup': runner.legacy_stem_dedup,
              'prep_info': prep_info, 'elapsed_sec': run.elapsed}

    # 1) 指标
    rows = []
    all_exact = True
    for k in METRIC_KEYS:
        paper = ref_entry['metrics'].get(k)
        rep = run.metrics.get(k)
        if paper is None or rep is None:
            status = 'N/A'
            diff = None
        else:
            diff = float(rep) - float(paper)
            status = 'EXACT' if abs(diff) < 1e-4 else ('CLOSE' if abs(diff) < 1e-2 else 'MISMATCH')
            if status != 'EXACT':
                all_exact = False
        rows.append({'metric': k, 'paper_log': paper, 'replicated': rep, 'diff': diff, 'status': status})
    metric_df = pd.DataFrame(rows)
    report['metrics'] = rows

    # 2) 每主题文档数
    dist_rows = []
    dist_ok = True
    rep_counts = Counter(run.topics.tolist())
    for key, v in sorted(ref_entry.get('topic_distribution', {}).items(),
                         key=lambda kv: int(kv[0].split('_')[-1])):
        tid = int(key.split('_')[-1])
        rc = rep_counts.get(tid, 0)
        ok = (rc == v['count'])
        dist_ok &= ok
        dist_rows.append({'topic': tid, 'paper_count': v['count'], 'replicated_count': rc, 'match': ok})
    for tid in rep_counts:
        if f"topic_{tid}" not in ref_entry.get('topic_distribution', {}):
            dist_ok = False
            dist_rows.append({'topic': tid, 'paper_count': 0, 'replicated_count': rep_counts[tid], 'match': False})
    report['topic_distribution'] = dist_rows

    # 3) 关键词逐词核对（对比对象 = 日志中的 legacy 关键词，不是 KeyBERT+MMR 词组）
    kw_rows = []
    kw_ok = True
    ref_kw = ref_entry.get('topic_keywords', {})
    for key in sorted(ref_kw.keys(), key=lambda s: int(s.split('_')[-1])):
        tid = int(key.split('_')[-1])
        paper_words = [d['word'] for d in ref_kw[key]]
        paper_w = np.array([d['weight'] for d in ref_kw[key]])
        rep_words = [w for w, _ in run.keywords.get(tid, [])]
        rep_w = np.array([s for _, s in run.keywords.get(tid, [])])
        exact = (paper_words == rep_words)
        kw_ok &= exact
        wdiff = float(np.max(np.abs(paper_w - rep_w))) if exact and len(paper_w) else None
        kw_rows.append({'topic': tid, 'n_paper': len(paper_words), 'n_replicated': len(rep_words),
                        'exact_order_match': exact,
                        'set_jaccard': jaccard(set(paper_words), set(rep_words)),
                        'max_weight_diff': wdiff,
                        'paper_words': ", ".join(paper_words),
                        'replicated_words': ", ".join(rep_words)})
    report['keywords'] = kw_rows

    # 4) 文档归属（对比 cache 里的 topics）
    assign = {}
    if cache is not None and 'topics' in cache and prep_info.get('keys_match_cache', True):
        paper_topics = np.asarray(cache['topics'])
        if len(paper_topics) == len(run.topics):
            same = paper_topics == run.topics
            assign = {'n_docs': int(len(same)), 'n_identical': int(same.sum()),
                      'identical_ratio': float(same.mean()),
                      'ari': float(adjusted_rand_score(paper_topics, run.topics)),
                      'n_changed_docs': int((~same).sum())}
            if (~same).any():
                idx = np.where(~same)[0]
                assign['changed_examples'] = [
                    {'idx': int(i), 'key': runner.keys[i], 'paper': int(paper_topics[i]),
                     'replicated': int(run.topics[i])} for i in idx[:20]]
    report['assignments'] = assign
    assign_ok = bool(assign) and assign.get('identical_ratio', 0) == 1.0

    verdict = 'EXACT' if (all_exact and dist_ok and kw_ok and assign_ok) else 'NOT EXACT'
    report['verdict'] = verdict

    # ---- 输出 ----
    with open(out_dir / 'step1_replication_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    pd.DataFrame(kw_rows).to_csv(out_dir / 'step1_keywords_check.csv', index=False, encoding='utf-8-sig')

    lines = [f"# Step 1 复刻核对报告\n\n",
             f"- 参考实验: `{ref_entry['experiment_id']}`\n",
             f"- 复刻口径: legacy（reduce_outliers 后仅 `update_topics(docs, topics=new_topics)`；"
             f"stem 去重={'on' if runner.legacy_stem_dedup else 'off'}）, seed={seed}\n",
             f"- 嵌入来源/差异: {prep_info}\n",
             f"- **结论: {verdict}**\n\n",
             "## 指标\n\n| metric | paper_log | replicated | diff | status |\n|---|---|---|---|---|\n"]
    for r in rows:
        lines.append(f"| {r['metric']} | {fmt(r['paper_log'])} | {fmt(r['replicated'])} | "
                     f"{fmt(r['diff'], 6)} | {r['status']} |\n")
    lines.append("\n## 每主题文档数\n\n| topic | paper | replicated | match |\n|---|---|---|---|\n")
    for r in dist_rows:
        lines.append(f"| {r['topic']} | {r['paper_count']} | {r['replicated_count']} | {'✓' if r['match'] else '✗'} |\n")
    lines.append("\n## 关键词（legacy 口径 vs 日志）\n\n| topic | exact | set_jaccard | max_weight_diff | paper | replicated |\n|---|---|---|---|---|---|\n")
    for r in kw_rows:
        lines.append(f"| {r['topic']} | {'✓' if r['exact_order_match'] else '✗'} | {r['set_jaccard']:.2f} | "
                     f"{fmt(r['max_weight_diff'], 6)} | {r['paper_words']} | {r['replicated_words']} |\n")
    lines.append("\n## 文档归属 vs cache\n\n")
    lines.append(f"```\n{json.dumps(assign, indent=2, ensure_ascii=False)}\n```\n" if assign
                 else "_cache 不可用或 Key 不一致，跳过_\n")
    with open(out_dir / 'step1_replication_report.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("\n" + metric_df.to_string(index=False))
    print(f"\n主题分布一致: {dist_ok} | 关键词逐词一致: {kw_ok} | 文档归属一致: {assign_ok}")
    print(f"STEP 1 结论: {verdict}")
    if verdict != 'EXACT':
        print("  提示: 若嵌入差异>1e-5 → 试 --embeddings_source cache；"
              "若关键词只差停用词 → 试 --legacy_stem_dedup；"
              "若指标不同但关键词一致 → 早期指标代码可能不同，把报告发我。")
    return report


# ----------------------------------------------------------------------------
# Step 2：多种子
# ----------------------------------------------------------------------------
def match_to_reference(ref_topics: np.ndarray, run_topics: np.ndarray):
    ref_ids = sorted(set(ref_topics.tolist()) - {-1})
    run_ids = sorted(set(run_topics.tolist()) - {-1})
    ref_sets = {r: set(np.where(ref_topics == r)[0]) for r in ref_ids}
    run_sets = {s: set(np.where(run_topics == s)[0]) for s in run_ids}
    J = np.zeros((len(ref_ids), len(run_ids)))
    for i, r in enumerate(ref_ids):
        for j, s in enumerate(run_ids):
            J[i, j] = jaccard(ref_sets[r], run_sets[s])
    mapping = {}
    if J.size:
        ri, ci = linear_sum_assignment(-J)
        mapping = {ref_ids[i]: run_ids[j] for i, j in zip(ri, ci) if J[i, j] > 0}
    return ref_ids, run_ids, ref_sets, run_sets, J, mapping


def step2_seeds(runner: SeedStabilityRunner, seeds: List[int], ref_seed: int, mode: str,
                cache: Optional[Dict], prep_info: Dict, out_dir: Path) -> Dict:
    print("\n" + "=" * 70)
    print(f"STEP 2  多随机种子重跑（{mode} 口径）  seeds={seeds}  参考 seed={ref_seed}")
    print("=" * 70)

    runs: Dict[int, SeedRun] = {}
    for s in seeds:
        print(f"\n>>> seed={s}")
        runs[s] = runner.fit_one(s, mode=mode)
        m = runs[s].metrics
        print(f"    n_topics={m.get('n_topics')}  outlier(raw→final)={runs[s].raw_outlier_ratio:.4f}→"
              f"{m.get('outlier_ratio'):.4f}  cv={fmt(m.get('coherence_cv'))}  "
              f"div={fmt(m.get('topic_diversity'))}  sil={fmt(m.get('silhouette_original'))}  "
              f"[{runs[s].elapsed:.0f}s]")

    ref = runs[ref_seed]
    ref_topics = ref.topics
    paper_topics = None
    if cache is not None and prep_info.get('keys_match_cache', True) and len(cache.get('topics', [])) == len(ref_topics):
        paper_topics = np.asarray(cache['topics'])

    n_docs = len(ref_topics)
    ref_ids = sorted(set(ref_topics.tolist()) - {-1})
    ref_out = set(np.where(ref_topics == -1)[0])

    # 逐 seed 汇总
    run_rows, match_rows = [], []
    mapped_all: Dict[int, np.ndarray] = {}
    for s, run in runs.items():
        m = run.metrics
        _, run_ids, ref_sets, run_sets, J, mapping = match_to_reference(ref_topics, run.topics)
        inv = {v: k for k, v in mapping.items()}
        mapped = np.array([inv.get(t, 1000 + t) if t != -1 else -1 for t in run.topics])
        mapped_all[s] = mapped

        both_non_out = (ref_topics != -1) & (run.topics != -1)
        run_out = set(np.where(run.topics == -1)[0])
        row = {
            'seed': s, 'is_reference': s == ref_seed,
            'n_topics': m.get('n_topics'),
            'outlier_ratio_raw_hdbscan': run.raw_outlier_ratio,
            'outlier_ratio_final': m.get('outlier_ratio'),
            'coherence_cv': m.get('coherence_cv'), 'coherence_npmi': m.get('coherence_npmi'),
            'topic_diversity': m.get('topic_diversity'),
            'silhouette_original': m.get('silhouette_original'),
            'davies_bouldin_original': m.get('davies_bouldin_original'),
            'topic_size_cv': m.get('topic_size_cv'),
            'prob_mean': m.get('prob_mean'), 'low_confidence_ratio': m.get('low_confidence_ratio'),
            'ari_vs_ref_all': adjusted_rand_score(ref_topics, run.topics),
            'nmi_vs_ref_all': normalized_mutual_info_score(ref_topics, run.topics),
            'ari_vs_ref_nonoutlier_both': (adjusted_rand_score(ref_topics[both_non_out], run.topics[both_non_out])
                                           if both_non_out.sum() > 1 else np.nan),
            'doc_agreement_after_matching': float(np.mean(mapped == ref_topics)),
            'n_ref_topics_matched': len(mapping),
            'n_ref_topics_matched_J>=0.5': int(sum(1 for r, t in mapping.items()
                                                    if J[ref_ids.index(r), run_ids.index(t)] >= 0.5)),
            'n_unmatched_new_topics': len(run_ids) - len(mapping),
            'ref_outliers_still_outlier': (len(ref_out & run_out) / len(ref_out)) if ref_out else np.nan,
            'run_outliers_were_ref_outlier': (len(ref_out & run_out) / len(run_out)) if run_out else np.nan,
            'ari_vs_paper_cache': (adjusted_rand_score(paper_topics, run.topics) if paper_topics is not None else np.nan),
            'elapsed_sec': run.elapsed,
        }
        run_rows.append(row)

        # 逐参考主题匹配
        for i, r in enumerate(ref_ids):
            t = mapping.get(r)
            j = run_ids.index(t) if t is not None else None
            jac = J[i, j] if j is not None else 0.0
            inter = len(ref_sets[r] & run_sets[t]) if t is not None else 0
            ref_kw = set(w for w, _ in ref.keywords.get(r, [])[:10])
            run_kw = set(w for w, _ in run.keywords.get(t, [])[:10]) if t is not None else set()
            # split/merge 诊断：覆盖该参考主题 ≥20% 文档的 run 主题数；匹配主题中来自 ≥20% 的参考主题数
            n_split = int(sum(1 for s2 in run_ids if len(ref_sets[r] & run_sets[s2]) / len(ref_sets[r]) >= 0.2))
            n_merge = (int(sum(1 for r2 in ref_ids if len(run_sets[t] & ref_sets[r2]) / len(run_sets[t]) >= 0.2))
                       if t is not None else 0)
            match_rows.append({
                'seed': s, 'ref_topic': r, 'ref_size': len(ref_sets[r]),
                'matched_run_topic': t, 'matched_size': len(run_sets[t]) if t is not None else 0,
                'jaccard_docs': jac,
                'recall_ref_docs_kept': inter / len(ref_sets[r]),
                'precision_matched': (inter / len(run_sets[t])) if t is not None else 0.0,
                'ref_docs_to_outlier': len(ref_sets[r] & run_out) / len(ref_sets[r]),
                'keyword_jaccard_top10': jaccard(ref_kw, run_kw),
                'n_run_topics_covering_ref_>=20%': n_split,
                'n_ref_topics_in_matched_>=20%': n_merge,
                'ref_label': topic_label(ref.keywords.get(r, [])),
                'matched_label': topic_label(run.keywords.get(t, [])) if t is not None else '',
            })

    run_df = pd.DataFrame(run_rows)
    match_df = pd.DataFrame(match_rows)
    non_ref = [s for s in seeds if s != ref_seed]

    # 每参考主题的跨种子稳定性
    topic_rows = []
    for r in ref_ids:
        sub = match_df[(match_df.ref_topic == r) & (match_df.seed != ref_seed)]
        topic_rows.append({
            'ref_topic': r, 'ref_label_top5': topic_label(ref.keywords.get(r, [])),
            'ref_size': int((ref_topics == r).sum()),
            'jaccard_mean': sub.jaccard_docs.mean(), 'jaccard_min': sub.jaccard_docs.min(),
            'jaccard_max': sub.jaccard_docs.max(),
            'recall_mean': sub.recall_ref_docs_kept.mean(),
            'to_outlier_mean': sub.ref_docs_to_outlier.mean(),
            'keyword_jaccard_mean': sub.keyword_jaccard_top10.mean(),
            f'n_seeds_matched_J>=0.5 (of {len(non_ref)})': int((sub.jaccard_docs >= 0.5).sum()),
            f'n_seeds_matched_J>=0.7 (of {len(non_ref)})': int((sub.jaccard_docs >= 0.7).sum()),
            'n_seeds_split': int((sub['n_run_topics_covering_ref_>=20%'] > 1).sum()),
            'n_seeds_merged': int((sub['n_ref_topics_in_matched_>=20%'] > 1).sum()),
        })
    topic_df = pd.DataFrame(topic_rows)

    # 文档级稳定性
    doc_df = pd.DataFrame({'Key': runner.keys, 'Title': runner.titles, 'ref_topic': ref_topics})
    if paper_topics is not None:
        doc_df.insert(2, 'paper_cache_topic', paper_topics)
    for s in seeds:
        doc_df[f'seed{s}_raw'] = runs[s].topics
        doc_df[f'seed{s}_mapped'] = mapped_all[s]
    stable = np.zeros(n_docs, dtype=int)
    for s in non_ref:
        stable += (mapped_all[s] == ref_topics).astype(int)
    doc_df['n_seeds_same_as_ref'] = stable
    doc_df['stable_ratio'] = stable / max(len(non_ref), 1)
    doc_df['ref_label'] = [topic_label(ref.keywords.get(t, [])) if t != -1 else 'Outliers' for t in ref_topics]

    non_out_mask = ref_topics != -1
    summary = {
        'mode': mode, 'seeds': seeds, 'ref_seed': ref_seed, 'n_docs': n_docs,
        'n_topics': {'ref': int(len(ref_ids)), 'min': int(run_df.n_topics.min()),
                     'max': int(run_df.n_topics.max()),
                     'distribution': {str(k): int(v) for k, v in Counter(run_df.n_topics).items()}},
        'outlier_ratio_final': {'ref': float(ref.metrics['outlier_ratio']),
                                'mean': float(run_df.outlier_ratio_final.mean()),
                                'std': float(run_df.outlier_ratio_final.std(ddof=0)),
                                'min': float(run_df.outlier_ratio_final.min()),
                                'max': float(run_df.outlier_ratio_final.max())},
        'outlier_ratio_raw_hdbscan': {'mean': float(run_df.outlier_ratio_raw_hdbscan.mean()),
                                      'std': float(run_df.outlier_ratio_raw_hdbscan.std(ddof=0))},
        'coherence_cv': {'mean': float(run_df.coherence_cv.mean()), 'std': float(run_df.coherence_cv.std(ddof=0))},
        'topic_diversity': {'mean': float(run_df.topic_diversity.mean()), 'std': float(run_df.topic_diversity.std(ddof=0))},
        'silhouette_original': {'mean': float(run_df.silhouette_original.mean()), 'std': float(run_df.silhouette_original.std(ddof=0))},
        'ari_vs_ref_all': {'mean': float(run_df[~run_df.is_reference].ari_vs_ref_all.mean()),
                           'std': float(run_df[~run_df.is_reference].ari_vs_ref_all.std(ddof=0)),
                           'min': float(run_df[~run_df.is_reference].ari_vs_ref_all.min())},
        'ari_vs_ref_nonoutlier_both': {'mean': float(run_df[~run_df.is_reference].ari_vs_ref_nonoutlier_both.mean()),
                                       'std': float(run_df[~run_df.is_reference].ari_vs_ref_nonoutlier_both.std(ddof=0))},
        'doc_agreement_after_matching': {'mean': float(run_df[~run_df.is_reference].doc_agreement_after_matching.mean()),
                                         'min': float(run_df[~run_df.is_reference].doc_agreement_after_matching.min())},
        'docs_stable_in_all_seeds_ratio': float(np.mean(stable == len(non_ref))),
        'docs_stable_in_>=80%_seeds_ratio': float(np.mean(stable >= np.ceil(0.8 * len(non_ref)))),
        'nonoutlier_docs_stable_in_all_seeds_ratio': float(np.mean(stable[non_out_mask] == len(non_ref))),
        'ref_seed_ari_vs_paper_cache': (float(adjusted_rand_score(paper_topics, ref_topics)) if paper_topics is not None else None),
        'env': env_info(),
    }

    # ---- 输出 ----
    run_df.to_csv(out_dir / 'seed_runs.csv', index=False, encoding='utf-8-sig')
    match_df.to_csv(out_dir / 'topic_matching.csv', index=False, encoding='utf-8-sig')
    topic_df.to_csv(out_dir / 'topic_stability_summary.csv', index=False, encoding='utf-8-sig')
    doc_df.to_csv(out_dir / 'doc_assignments.csv', index=False, encoding='utf-8-sig')
    with open(out_dir / 'step2_keywords.json', 'w', encoding='utf-8') as f:
        json.dump({str(s): {str(t): kw for t, kw in runs[s].keywords.items()} for s in seeds},
                  f, indent=2, ensure_ascii=False)
    with open(out_dir / 'step2_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    show_cols = ['seed', 'n_topics', 'outlier_ratio_raw_hdbscan', 'outlier_ratio_final', 'coherence_cv',
                 'topic_diversity', 'silhouette_original', 'ari_vs_ref_all', 'ari_vs_ref_nonoutlier_both',
                 'doc_agreement_after_matching', 'n_ref_topics_matched_J>=0.5', 'n_unmatched_new_topics',
                 'ref_outliers_still_outlier', 'ari_vs_paper_cache']
    lines = [f"# Step 2 随机种子稳定性汇总（{mode} 口径）\n\n",
             f"- 固定参数: {FINAL_PARAMS}\n",
             f"- seeds: {seeds}（参考 seed={ref_seed}）；随机性仅来自 UMAP random_state\n",
             f"- 参考 run 与论文 cache 归属的 ARI: {fmt(summary['ref_seed_ari_vs_paper_cache'])}\n",
             "- ⚠ 本步所有关键词为 n-gram(1-3)+KeyBERTInspired→MMR、top_n_words=20、stem 去重；"
             "coherence/diversity 数值与论文正文（legacy 默认表示）不是同一口径，只在本表内部横向比较。\n\n",
             "## 逐 seed\n\n",
             run_df[show_cols].to_markdown(index=False, floatfmt='.4f'), "\n\n",
             "## 汇总\n\n```\n", json.dumps({k: v for k, v in summary.items() if k != 'env'}, indent=2, ensure_ascii=False),
             "\n```\n\n",
             "## 每个参考主题的跨种子稳定性\n\n",
             topic_df.to_markdown(index=False, floatfmt='.3f'), "\n\n",
             "## 环境\n\n```\n", json.dumps(summary['env'], indent=2), "\n```\n"]
    with open(out_dir / 'step2_summary.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("\n" + run_df[show_cols].to_string(index=False))
    print("\n" + topic_df.to_string(index=False))
    return summary


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='BERTopic 随机种子稳定性检验（minor revision 补丁）',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    p.add_argument('--data_file', type=str, default='topic_modeling/meta_data.csv')
    p.add_argument('--step', choices=['1', '2', 'all'], default='2')
    p.add_argument('--ref_experiment_id', type=str, default=None,
                   help='论文最终 run 的 experiment_id；不填则按最终参数 + coherence 自动定位')
    p.add_argument('--target_coherence', type=float, default=0.6736,
                   help='自动定位时用于挑选的论文 coherence_cv 值')
    p.add_argument('--ref_seed', type=int, default=42, help='论文使用的随机种子 / Step2 的参考 run')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   help='Step2 种子列表（ref_seed 若不在其中会自动加入）')
    p.add_argument('--embeddings_source', choices=['fresh', 'cache'], default='fresh',
                   help='fresh=重新编码并与 cache 比对；cache=直接用论文 run 的嵌入')
    p.add_argument('--legacy_stem_dedup', action='store_true',
                   help='Step1 legacy 口径下是否也做 stem 去重（默认不做；若关键词只差停用词可开启）')
    p.add_argument('--step2_mode', choices=['standard', 'legacy'], default='standard')
    p.add_argument('--standard_top_n_words', type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = base_dir / 'outputs' / 'seed_stability' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BERTopic 随机种子稳定性检验")
    print(f"输出目录: {out_dir}")
    print(f"固定参数: {FINAL_PARAMS}")
    print("=" * 70)

    ref_entry = find_reference(base_dir, FINAL_PARAMS, args.ref_experiment_id, args.target_coherence)
    cache = load_cache(base_dir, ref_entry['experiment_id']) if ref_entry else None
    if args.step in ('1', 'all') and ref_entry is None:
        print("❌ Step 1 需要参考实验日志，退出"); sys.exit(1)

    runner = SeedStabilityRunner(str(base_dir), FINAL_PARAMS, args.data_file,
                                 standard_top_n_words=args.standard_top_n_words,
                                 legacy_stem_dedup=args.legacy_stem_dedup)
    prep_info = runner.prepare(args.embeddings_source, cache)
    prep_info['embeddings_source'] = args.embeddings_source

    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'final_params': FINAL_PARAMS,
                   'reference_experiment_id': ref_entry['experiment_id'] if ref_entry else None,
                   'prep_info': prep_info, 'env': env_info()}, f, indent=2, ensure_ascii=False)

    if args.step in ('1', 'all'):
        step1_replicate(runner, ref_entry, cache, prep_info, out_dir, seed=args.ref_seed)

    if args.step in ('2', 'all'):
        seeds = list(args.seeds)
        if args.ref_seed not in seeds:
            seeds.insert(0, args.ref_seed)
        step2_seeds(runner, seeds, args.ref_seed, args.step2_mode, cache, prep_info, out_dir)

    print("\n" + "=" * 70)
    print("完成。输出文件:")
    if args.step in ('1', 'all'):
        print(f"  {out_dir / 'step1_replication_report.md'}")
    if args.step in ('2', 'all'):
        print(f"  {out_dir / 'seed_runs.csv'}")
        print(f"  {out_dir / 'doc_assignments.csv'}")
        print(f"  {out_dir / 'topic_matching.csv'}")
        print(f"  {out_dir / 'topic_stability_summary.csv'}")
    print("=" * 70)


if __name__ == '__main__':
    main()

# %% Decoding analysis for neural data
# %% 导入所需模块
from itertools import combinations
from loaddata import *
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
sns.set_theme(context="paper", style="whitegrid")
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# 沿用之前配置
# cfg = ExpConfig("..\config\m71.json")
# %% 颜色和可视化配置
CLASS_COLORS = {
    3: "#7f7f7f",   # 背景色
    1: "#d62728",   # 类别1
    2: "#ffbf00",   # 类别2
    0: "#1f77b4",   # 类别3
}
PAIR_COLORS = {
    (1, 3): "#c55d5d",
    (2, 3): "#d8b94f",
    (0, 3): "#6aa3d8",
    (1, 2): "#ff7f0e",  # 类别 1 VS 2
    (0, 1): "#b065d9",  # 类别 1 VS 0
    (0, 2): "#2ca02c",  # 类别 2 VS 0
}
def _get_class_color(label):
    return CLASS_COLORS.get(int(label), "#555555")
def _get_pair_color(pair):
    try:
        key = tuple(sorted(int(p) for p in pair))
    except Exception:
        key = tuple(sorted(pair))
    return PAIR_COLORS.get(key, '#4d4d4d')
def _style_single_axis(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(1.2)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=10)
# %% 底层功能函数的封装
# ========== Fisher信息 ===========
def _split_by_class(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples must match labels")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("Fisher information currently supports exactly two classes")
    class_a, class_b = classes
    group_a = X[y == class_a]
    group_b = X[y == class_b]
    if group_a.size == 0 or group_b.size == 0:
        raise ValueError("Each class must contain at least one sample")
    return group_a, group_b
def fisher_information_univariate(group_a, group_b, epsilon=1e-6):
    """计算单变量Fisher信息"""
    mean_diff = np.mean(group_a, axis=0) - np.mean(group_b, axis=0)
    var_sum = np.var(group_a, axis=0, ddof=1) + np.var(group_b, axis=0, ddof=1)
    return (mean_diff ** 2) / (var_sum + epsilon)
def fisher_information_multivariate(group_a, group_b, shrinkage=1e-3):
    """计算多变量Fisher信息"""
    def _cov(data):
        if data.shape[0] <= 1:
            return np.zeros((data.shape[1], data.shape[1]))
        return np.cov(data, rowvar=False, ddof=1)
    cov_a = _cov(group_a)
    cov_b = _cov(group_b)
    pooled = ((group_a.shape[0] - 1) * cov_a + (group_b.shape[0] - 1) * cov_b)
    denom = max(group_a.shape[0] + group_b.shape[0] - 2, 1)
    pooled = pooled / denom
    pooled += shrinkage * np.eye(pooled.shape[0])
    mean_diff = np.mean(group_a, axis=0) - np.mean(group_b, axis=0)
    try:
        inv_cov = np.linalg.inv(pooled)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(pooled)
    return float(mean_diff @ inv_cov @ mean_diff)
def Fisher_information(X, y, epsilon=1e-6, mode="univariate", shrinkage=1e-3):
    """
    计算单变量Fisher信息或多变量Fisher信息
    mode: 'univariate' 单变量
          'multivariate' 多变量
    """
    mode = (mode or "univariate").lower()
    if mode not in {"univariate", "multivariate"}:
        raise ValueError(f"无效的 mode: {mode}")
    group_a, group_b = _split_by_class(X, y)

    if mode == "univariate":
        return fisher_information_univariate(group_a, group_b, epsilon=epsilon)
    return fisher_information_multivariate(group_a, group_b, shrinkage=shrinkage)
# %% 主程序函数
# ========== 分类准确率 ===========
def classify_by_timepoints(segments, labels, cfg=cfg, window_size=5, step_size=1):
    """
    ????????????????????????
    """
    n_trials, _, n_timepoints = segments.shape
    assert len(labels) == n_trials, "??????????"
    time_points = []
    accuracies = []
    accuracy_std = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_folds = cv.get_n_splits()
    for t in range(n_timepoints):
        timepoint_data = segments[:, :, t]
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(timepoint_data)
        clf = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
        scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring='accuracy')
        accuracies.append(scores.mean())
        accuracy_std.append(scores.std(ddof=1))
        time_points.append((t - cfg.exp_info["t_stimulus"]) / 4)
    return np.array(accuracies), np.array(time_points), np.array(accuracy_std), n_folds
# ========== Fisher信息 ===========
def FI_by_timepoints(segments, labels, cfg=cfg, reduction="mean", epsilon=1e-6, mode="univariate", shrinkage=1e-3):
    """
    计算每个时间点上的Fisher信息
    参数:
    segments: (trials, neurons, timepoints)
    labels: 每个trial的标签
    reduction: 'mean' / 'sum' / 'none'，默认为'mean'，仅在mode='univariate'时有效
    mode: 'univariate'，表示单变量分析，或'multivariate'，表示多变量分析
    fisher_dict: dict，包含以下键值对:
    time_points: 每个时间点的时间戳
    """
    segments = np.asarray(segments)
    labels = np.asarray(labels)
    mode = (mode or "univariate").lower()
    if segments.ndim != 3:
        raise ValueError("segments 必须是 (trials, neurons, timepoints) 形式的三维数组?")
    if mode not in {"univariate", "multivariate"}:
        raise ValueError(f"无效的 mode: {mode}")
    
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Fisher information currently supports exactly two classes")
    
    n_trials, n_neurons, n_timepoints = segments.shape
    time_points = np.array([(t - cfg.exp_info["t_stimulus"]) / 4 for t in range(n_timepoints)])
    
    def _reduce(matrix):
        if reduction in (None, "none"):
            return matrix
        if reduction == "mean":
            return matrix.mean(axis=1)
        if reduction == "sum":
            return matrix.sum(axis=1)
        raise ValueError(f"无效的 reduction 方法: {reduction}")
   
    if mode == "multivariate" and reduction not in (None, "none"):
        raise ValueError("mode='multivariate' does not support reduction aggregation")
   
    fisher_dict = {}
    for pair in combinations(unique_labels, 2):
        mask = np.isin(labels, pair)
        if mask.sum() == 0:
            continue
        pair_segments = segments[mask]
        pair_labels = labels[mask]

        if mode == "univariate":
            fi_matrix = np.zeros((n_timepoints, n_neurons))
            for t in range(n_timepoints):
                fi_matrix[t] = Fisher_information(pair_segments[:, :, t], pair_labels, epsilon=epsilon, mode=mode)
            fisher_dict[pair] = _reduce(fi_matrix)
        else:
            fi_series = np.zeros(n_timepoints)
            
            # 定义PCA。基于当前pair的样本数
            n_samples_pair = pair_labels.shape[0]
            n_components_target = int(n_samples_pair / 2) - 1
            if n_components_target < 1:
                n_components_target = 1
            _pca = PCA(n_components=n_components_target)

            for t in range(n_timepoints):
                # 先从 (n_samples, n_neurons) 降维到 (n_samples, n_components)
                X_reduced = _pca.fit_transform(pair_segments[:, :, t])
                
                # 在降维后的数据上计算FI
                fi_series[t] = Fisher_information(
                    X_reduced, 
                    pair_labels, 
                    epsilon=epsilon, 
                    mode="multivariate", 
                    shrinkage=shrinkage
                )
            fisher_dict[pair] = fi_series
    return fisher_dict, time_points
# ========== Fisher信息 ===========
def FI_by_timepoints_v2(segments, labels, cfg=cfg, reduction="mean", epsilon=1e-6, 
                        mode="univariate", shrinkage=1e-3, 
                        balance_samples=True, random_state=42):
    """
    计算每个时间点上的Fisher信息
    v3版: 
    1. 修正了 multivariate 模式下的 PCA 降维
    2. 增加了 balance_samples 选项以通过下采样平衡类别
    """
    
    segments = np.asarray(segments)
    labels = np.asarray(labels)
    mode = (mode or "univariate").lower()
    
    # ... (所有前置检查代码保持不变) ...
    if segments.ndim != 3:
        raise ValueError("segments 必须是 (trials, neurons, timepoints) 形式的三维数组?")
    if mode not in {"univariate", "multivariate"}:
        raise ValueError(f"无效的 mode: {mode}")
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Fisher information currently supports exactly two classes")
        
    n_trials, n_neurons, n_timepoints = segments.shape
    time_points = np.array([(t - cfg.exp_info["t_stimulus"]) / 4 for t in range(n_timepoints)])
    
    def _reduce(matrix):
        if reduction in (None, "none"):
            return matrix
        if reduction == "mean":
            return matrix.mean(axis=1)
        if reduction == "sum":
            return matrix.sum(axis=1)
        raise ValueError(f"无效的 reduction 方法: {reduction}")
        
    if mode == "multivariate" and reduction not in (None, "none"):
        raise ValueError("mode='multivariate' does not support reduction aggregation")
        
    # 初始化一个可复现的随机数生成器
    rng = np.random.default_rng(random_state)
    
    fisher_dict = {}
    
    for pair in combinations(unique_labels, 2):
        mask = np.isin(labels, pair)
        if mask.sum() == 0:
            continue
            
        pair_segments_all = segments[mask]
        pair_labels_all = labels[mask]
        
        # --- *** 新增：样本均衡逻辑 *** ---
        if balance_samples:
            class_a, class_b = pair
            
            # 1. 找到两个类别的索引 (相对于 'pair_labels_all')
            indices_a = np.where(pair_labels_all == class_a)[0]
            indices_b = np.where(pair_labels_all == class_b)[0]
            
            # 2. 找到最小的样本数
            n_min = min(len(indices_a), len(indices_b))
            if n_min == 0:
                continue # 如果有一个类没有样本，跳过
                
            # 3. 从每个类别中随机抽取 n_min 个样本
            balanced_indices_a = rng.choice(indices_a, size=n_min, replace=False)
            balanced_indices_b = rng.choice(indices_b, size=n_min, replace=False)
            
            # 4. 合并索引，得到均衡后的数据
            balanced_indices = np.concatenate([balanced_indices_a, balanced_indices_b])
            
            # 5. 更新用于后续计算的变量
            pair_segments = pair_segments_all[balanced_indices]
            pair_labels = pair_labels_all[balanced_indices]
        else:
            # 如果不均衡，就使用所有数据
            pair_segments = pair_segments_all
            pair_labels = pair_labels_all
        # --- *** 样本均衡结束 *** ---

        
        if mode == "univariate":
            # --- 单变量模式 (无需PCA) ---
            fi_matrix = np.zeros((n_timepoints, n_neurons))
            for t in range(n_timepoints):
                fi_matrix[t] = Fisher_information(
                    pair_segments[:, :, t], 
                    pair_labels, 
                    epsilon=epsilon, 
                    mode="univariate"
                )
            fisher_dict[pair] = _reduce(fi_matrix)
            
        else:
            # --- 多变量模式 (需要PCA) ---
            fi_series = np.zeros(n_timepoints)
            
            # 定义PCA。基于均衡后（或原始）的样本数
            n_samples_pair = pair_labels.shape[0]
            n_components_target = int(n_samples_pair / 2) - 1
            if n_components_target < 1:
                n_components_target = 1
                
            # 检查 n_features 是否会小于 n_components_target
            # (虽然在这里 n_neurons 很大，但以防万一)
            if n_neurons < n_components_target:
                n_components_target = n_neurons
                
            _pca = PCA(n_components=n_components_target)
            
            for t in range(n_timepoints):
                X_t = pair_segments[:, :, t]
                
                # 检查此时间点的方差是否为0 (在某些预处理中可能发生)
                if np.var(X_t) < 1e-10:
                    fi_series[t] = 0.0
                    continue
                
                X_reduced = _pca.fit_transform(X_t)
                
                fi_series[t] = Fisher_information(
                    X_reduced, 
                    pair_labels, 
                    epsilon=epsilon, 
                    mode="multivariate", 
                    shrinkage=shrinkage
                )
            fisher_dict[pair] = fi_series
            
    return fisher_dict, time_points

# ========== Fisher信息随着神经元数量增加的变化 ===========
def FI_by_neuron_count(segments, labels, n_neurons_step = 5):
    """计算Fisher信息随着神经元数量增加的变化"""
    neuron_counts = np.arange(n_neurons_step, segments.shape[1] + 1, n_neurons_step)
    fi_values = []
    for n_neurons in neuron_counts:
        selected_segments = segments[(labels == 1) | (labels == 2), :n_neurons, 10:14]
        selected_labels = labels[(labels == 1) | (labels == 2)]
        X = selected_segments.reshape(selected_segments.shape[0], -1)

        # 进行维度检查
        n_samples = selected_segments.shape[0]
        n_features = X.shape[1]
        if n_features <= n_samples/2 - 1:
            X_reduce = X
        else:
            _pca = PCA(n_components=int(n_samples/2 - 1))
            X_reduce = _pca.fit_transform(X)
        fi_values.append(Fisher_information(X_reduce, selected_labels, mode="multivariate"))
    fi_values = np.asarray(fi_values, dtype=float)    
    return neuron_counts, fi_values
# =========== pca 可视化 ============
# %% 绘制主成分
# ========= 可视化 ============
# =========== PCA+t-SNE可视化 ===========
def _pca_tsne_visualization(segments, labels):
    """PCA and t-SNE visualization"""

    segments = np.asarray(segments)
    labels = np.asarray(labels)
    n_trials, n_neurons, _ = segments.shape
    window = 4
    start_idx = 14
    segment_peak = segments[:, :, start_idx:start_idx + window]
    segments_reshaped = segment_peak.reshape(n_trials, n_neurons * window)
    pca = PCA()
    segments_pca = pca.fit_transform(segments_reshaped)
    tsne = TSNE(n_components=2, random_state=10, perplexity=30)
    embedding = tsne.fit_transform(segments_pca)
    df = pd.DataFrame({
        "t-SNE 1": embedding[:, 0],
        "t-SNE 2": embedding[:, 1],
        "Class": labels.astype(int),
    })
    palette = {int(k): v for k, v in CLASS_COLORS.items()}
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    sns.scatterplot(
        data=df,
        x="t-SNE 1",
        y="t-SNE 2",
        hue="Class",
        palette=palette,
        s=48,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    _style_single_axis(ax)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('t-SNE of PCA features', fontsize=13)
    ax.legend(frameon=False, fontsize=9, title="Class", loc='best')
    fig.tight_layout()
    plt.show()
    return True
# =========== 绘制分类准确率随着时间变化的变化 ===========
def _visualize_accuracies(time_points, accuracies, accuracy_std, n_folds, chance_level=0.4, title="Classification Accuracy Over Time"):
    """Plot classification accuracy over time"""
    tp = np.asarray(time_points)
    acc = np.asarray(accuracies)
    std = np.asarray(accuracy_std)
    sem = std / np.sqrt(max(n_folds, 1))
    ci_low = np.clip(acc - 1.96 * sem, 0.0, 1.0)
    ci_high = np.clip(acc + 1.96 * sem, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    line = sns.lineplot(x=tp, y=acc, color="#355C7D", linewidth=2.4, ax=ax, label="Accuracy", ci=None)
    ax.fill_between(tp, ci_low, ci_high, color="#355C7D", alpha=0.18, label="95% CI")
    ax.axvline(x=0, color="#aa3a3a", linestyle="--", linewidth=1.2, label="Stimulus onset")
    if chance_level is not None:
        ax.hlines(chance_level, tp.min(), tp.max(), colors="#8c8c8c", linestyles=":", linewidth=1.1, label="Chance level")
    if chance_level is not None and n_folds > 1:
        sem_safe = np.where(sem == 0, np.nan, sem)
        t_stats = (acc - chance_level) / sem_safe
        p_vals = stats.t.sf(np.abs(t_stats), df=n_folds - 1) * 2
        significant = np.isfinite(p_vals) & (p_vals < 0.05)
        if significant.any():
            ax.scatter(tp[significant], acc[significant], color="#1b1b1b", s=26, zorder=5, label="p < 0.05")
    peak_idx = np.nanargmax(acc)
    peak_text = f"Peak {acc[peak_idx]:.2f} at {tp[peak_idx]:.2f}s"
    ax.text(
        0.02,
        0.92,
        peak_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f6f6f6", edgecolor="#d8d8d8"),
    )
    _style_single_axis(ax)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlabel("Time course (s)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(tp.min(), tp.max())
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    plt.show()
    return fig, ax
# =========== 绘制Fisher信息随着时间变化的变化 ===========
def _visualize_fisher_information(fisher_dict, time_points):
    """Visualize Fisher information over time"""
    records = []
    for pair, fi_values in fisher_dict.items():
        pair_label = f"Class {int(pair[0])} vs {int(pair[1])}" if len(pair) == 2 else str(pair)
        values = np.asarray(fi_values, dtype=float)
        for t, value in zip(time_points, values):
            records.append({"Time (s)": t, "Fisher information": value, "Pair": pair_label})
    if not records:
        return False
    data = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    sns.lineplot(
        data=data,
        x="Time (s)",
        y="Fisher information",
        hue="Pair",
        palette="tab10",
        linewidth=2.0,
        ax=ax,
    )
    ax.axvline(x=0, color="#aa3a3a", linestyle="--", linewidth=1.2, label="Stimulus onset")
    for pair_label, group in data.groupby("Pair"):
        max_idx = group["Fisher information"].astype(float).idxmax()
        if not np.isnan(group.loc[max_idx, "Fisher information"]):
            ax.annotate(
                f"Peak {group.loc[max_idx, 'Fisher information']:.2f}",
                xy=(group.loc[max_idx, "Time (s)"], group.loc[max_idx, "Fisher information"]),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
            )
    _style_single_axis(ax)
    ax.set_xlabel("Time course (s)", fontsize=11)
    ax.set_ylabel("Fisher information", fontsize=11)
    ax.set_title("Fisher information over time", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    plt.show()
    return True
# =========== 绘制Fisher信息随着神经元数量增加的变化 ===========
def _visualize_fi_by_neuron_count(neuron_counts, fi_values):
    """绘制Fisher信息随着神经元数量增加的变化"""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    sns.regplot(
        x=neuron_counts,
        y=fi_values,
        scatter_kws={"s": 48, "color": "#355C7D", "edgecolor": "white"},
        # line_kws={"color": "#F67280"},
        ci=95,
        ax=ax,
    )
    rho, p_val = stats.spearmanr(neuron_counts, fi_values)
    ax.text(
        0.02,
        0.92,
        f"Spearman rho = {rho:.2f}\np = {p_val:.3g}",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f6f6f6", edgecolor="#d8d8d8"),
    )
    _style_single_axis(ax)
    ax.set_xlabel("Number of neurons", fontsize=11)
    ax.set_ylabel("Fisher information", fontsize=11)
    ax.set_title("Fisher information vs. neuron count", fontsize=13)
    fig.tight_layout()
    plt.show()
    return True
# %% 主流程
if __name__ == "__main__":
    print("Decoding analysis")
    # %% 读取数据
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data()
    segments, labels, neuron_pos_rr = preprocess_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    # %% 计算时间点上的分类准确率
    accuracies, time_points, accuracy_std, n_folds = classify_by_timepoints(segments, labels)
    # 可视化
    _visualize_accuracies(time_points, accuracies, accuracy_std, n_folds)
    # %% 对数据进行PCA+t-SNE可视化
    _pca_tsne_visualization(segments, labels)
    # %% 计算时间点上的Fisher信息
    fisher_dict, time_points_fi = FI_by_timepoints_v2(segments, labels, mode="multivariate", reduction=None)
    # 可视化    
    _visualize_fisher_information(fisher_dict, time_points_fi)
    # %% 计算时间点上的Fisher信息
    fisher_dict, time_points_fi = FI_by_timepoints_v2(segments, labels, mode="univariate", reduction="mean")
    # 可视化    
    _visualize_fisher_information(fisher_dict, time_points_fi)
    # %% Fisher信息随着神经元数量的变化
    neuron_counts, fi_values = FI_by_neuron_count(segments, labels)
    _visualize_fi_by_neuron_count(neuron_counts, fi_values)
    # %% 部分信息分解
    # 用于计算部分信息分解的神经元
    segments_pid = segments[:, :, 10:14]
    # 计算唯一信息，冗余信息，剩余信息
    import dit
    from dit.pid import PID_BROJA # BROJA 是一种常用的PID计算方法

    def calculate_pid_core(X1_discrete, X2_discrete, Y_target):
        """
        计算三个离散变量的PID（底层函数）。
        
        参数:
        X1_discrete (array-like): 源1 (n_samples,)
        X2_discrete (array-like): 源2 (n_samples,)
        Y_target (array-like): 目标 (n_samples,)
        
        返回:
        dict: 包含 'redundancy', 'unique_X1', 'unique_X2', 'synergy' 的字典
        """
        
        # 1. 将数据打包成 dit 库要求的格式
        #    (list of tuples, e.g., [(x1, x2, y), (x1, x2, y), ...])
        try:
            data_tuples = list(zip(X1_discrete, X2_discrete, Y_target))
            
            # 2. 从数据中创建联合概率分布
            dist = dit.Distribution.from_data(data_tuples)
            
            # 3. 命名变量（可选，但推荐）
            dist.set_rv_names(['X1', 'X2', 'Y'])
            
            # 4. 初始化PID计算器 (我们使用 BROJA 算法)
            #    PID_BROJA(dist, sources, target)
            pid = PID_BROJA(dist, ['X1', 'X2'], 'Y')
            
            # 5. 提取结果
            result = {
                "redundancy": pid.get_atom(('X1', 'X2'), ('Y',)),
                "unique_X1": pid.get_atom(('X1',), ('Y',)),
                "unique_X2": pid.get_atom(('X2',), ('Y',)),
                "synergy": pid.get_atom(('X1', 'X2'), ('X1', 'X2'), ('Y',)),
            }
            return result
            
        except Exception as e:
            print(f"PID 计算失败: {e}")
            # 在数据完全一样（例如全0）时，dit 可能会报错
            return {"redundancy": 0, "unique_X1": 0, "unique_X2": 0, "synergy": 0}
      
    from itertools import combinations
    def analyze_pid_for_groups(segments, labels, 
                            group1_indices, group2_indices, 
                            pair=(1, 2), 
                            time_window=(10, 14), 
                            n_bins=4,
                            balance_samples=True,
                            random_state=42):
        """
        高层封装函数：计算两个神经元群体(Group)之间的PID。
        
        参数:
        segments (array): 原始数据 (n_trials, n_neurons, n_timepoints)
        labels (array): 标签 (n_trials,)
        group1_indices (list or array): 群体1的神经元索引
        group2_indices (list or array): 群体2的神经元索引
        pair (tuple): 你感兴趣的标签对，例如 (1, 2)
        time_window (tuple): (start, end) 时间窗口
        n_bins (int): 离散化时的分箱数量
        balance_samples (bool): 是否如下采样以平衡样本
        random_state (int): 随机种子
        """
        
        # --- 1. 数据准备 (使用你修正后的样本均衡逻辑) ---
        mask = np.isin(labels, pair)
        if mask.sum() == 0:
            raise ValueError(f"没有找到 {pair} 的数据")
            
        pair_segments_all = segments[mask]
        pair_labels_all = labels[mask]
        rng = np.random.default_rng(random_state)
        
        if balance_samples:
            class_a, class_b = pair
            indices_a = np.where(pair_labels_all == class_a)[0]
            indices_b = np.where(pair_labels_all == class_b)[0]
            n_min = min(len(indices_a), len(indices_b))
            if n_min == 0:
                raise ValueError(f"类别 {pair} 至少有一个样本数为0")
                
            balanced_indices_a = rng.choice(indices_a, size=n_min, replace=False)
            balanced_indices_b = rng.choice(indices_b, size=n_min, replace=False)
            balanced_indices = np.concatenate([balanced_indices_a, balanced_indices_b])
            
            segments_final = pair_segments_all[balanced_indices]
            labels_final = pair_labels_all[balanced_indices]
        else:
            segments_final = pair_segments_all
            labels_final = pair_labels_all
            
        # 目标 Y 已经是离散的了
        Y_target = labels_final
        
        # --- 2. 提取并处理源 (X1, X2) ---
        
        # 提取时间窗口
        segments_window = segments_final[:, :, time_window[0]:time_window[1]]
        
        # 计算每个群体的平均活动 (在神经元和时间上都取平均)
        # X1: (n_samples,)
        X1_continuous = np.mean(segments_window[:, group1_indices, :], axis=(1, 2))
        # X2: (n_samples,)
        X2_continuous = np.mean(segments_window[:, group2_indices, :], axis=(1, 2))

        # --- 3. 离散化 (关键步骤) ---
        # 我们使用分位数分箱 (qcut)，它比等距分箱更稳健
        # duplicates='drop' 会在数据点相同时自动合并箱子
        try:
            X1_discrete = pd.qcut(X1_continuous, q=n_bins, labels=False, duplicates='drop')
            X2_discrete = pd.qcut(X2_continuous, q=n_bins, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"分箱失败 (可能数据方差为0): {e}")
            # 如果数据全一样，分箱会失败
            X1_discrete = np.zeros(len(X1_continuous), dtype=int)
            X2_discrete = np.zeros(len(X2_continuous), dtype=int)

        # --- 4. 计算PID ---
        return calculate_pid_core(X1_discrete, X2_discrete, Y_target)
    
    pid_single = analyze_pid_for_groups(
    segments_pid, labels,
    group1_indices=[5],   # 神经元 5
    group2_indices=[10],  # 神经元 10
    pair=(1, 2),
    time_window=(10, 14), # 使用你之前发现的响应窗口
    n_bins=4              # 4个分箱（例如，低、中、高、极高）
)
# %%

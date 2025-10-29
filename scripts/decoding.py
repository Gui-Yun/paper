# %% Decoding analysis for neural data
# %% 导入所需模块
from itertools import combinations
from loaddata import *
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# 沿用之前配置
cfg = ExpConfig("..\config\m91.json")

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
def classify_by_timepoints(segments, labels,cfg=cfg, window_size=5, step_size=1):
    """
    计算每个时间点上的分类准确率
    """
    n_trials, n_neurons, n_timepoints = segments.shape
    assert len(labels) == n_trials, "标签数组长度与试验数量不匹配"
    time_points = []
    accuracies = []
    # 遍历每个时间点
    for t in range(n_timepoints):
        # 获取当前时间点的所有数据
        timepoint_data = segments[:, :, t]  # (trials, neurons)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(timepoint_data)
        clf = SVC(kernel='rbf', class_weight='balanced',C=1.0,gamma='scale')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring='accuracy')
        accuracy = scores.mean()
        time_points.append((t-cfg.exp_info["t_stimulus"])/4)
        accuracies.append(accuracy)
        if t % 5 == 0:  # 每隔5个时间点输出一次结果
           # print(f"时间点{t}: 分类准确率{accuracy:.3f}")
           pass # 不需要输出

    return np.array(accuracies), np.array(time_points)
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
            for t in range(n_timepoints):
                fi_series[t] = Fisher_information(
                    pair_segments[:, :, t], pair_labels, epsilon=epsilon, mode=mode, shrinkage=shrinkage
                )
            fisher_dict[pair] = fi_series

    return fisher_dict, time_points
# ========== Fisher信息随着神经元数量增加的变化 ===========
# =========== pca 可视化 ============
# %% 绘制主成分
# ========= 可视化 ============
def pca_tsne_visualization(segments, labels):
    """将数据降维到2D并可视化 PCA+t-SNE 结果"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    segments = np.asarray(segments)
    labels = np.asarray(labels)

    n_trials, n_neurons, _ = segments.shape
    windows = 4
    start_idx = 14
    segment_peak = segments[:, :, start_idx:start_idx + windows]
    segments_reshaped = segment_peak.reshape(n_trials, n_neurons * windows)

    pca = PCA()
    segments_pca = pca.fit_transform(segments_reshaped)
    tsne = TSNE(n_components=2, random_state=10, perplexity=30)
    segments_tsne = tsne.fit_transform(segments_pca)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    plotted = False
    for lbl in sorted(np.unique(labels)):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(
            segments_tsne[mask, 0],
            segments_tsne[mask, 1],
            s=38,
            color=_get_class_color(lbl),
            alpha=0.78,
            linewidth=0.6,
            edgecolors='white',
            label=f"Class {int(lbl)}"
        )
        plotted = True

    _style_single_axis(ax)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('t-SNE of PCA Features', fontsize=13)
    if plotted:
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    plt.show()

def plot_accuracies(time_points, accuracies, chance_level=0.4, title="Classification Accuracy Over Time"):
    """绘制每个时间点的分类准确率曲线"""
    tp = np.asarray(time_points)
    acc = np.asarray(accuracies)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(tp, acc, color='#1b6ca8', linewidth=2.2, label='Accuracy')

    ax.axvline(x=0, color='#aa3a3a', linestyle='--', linewidth=1.4, label='Stimulus onset')
    if chance_level is not None:
        ax.hlines(chance_level, tp.min(), tp.max(), colors='#8c8c8c', linestyles=':', linewidth=1.2, label='Chance level')

    _style_single_axis(ax)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_xlabel('Time Course (s)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(tp.min(), tp.max())

    ax.legend(frameon=False, fontsize=9, loc='lower right')

    fig.tight_layout()
    plt.show()
    return fig, ax

def visualize_fisher_information(fisher_dict, time_points):
    """绘制每个时间点的Fisher信息曲线"""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    global_max = 0.0

    for pair in fisher_dict.keys():
        fi_array = np.asarray(fisher_dict[pair], dtype=float)
        if fi_array.size == 0:
            continue
        global_max = max(global_max, float(np.nanmax(fi_array)))
        color = _get_pair_color(pair)
        label = f"Class {int(pair[0])} vs {int(pair[1])}"
        ax.plot(time_points, fi_array, color=color, linewidth=2.0, label=label)

    ax.axvline(x=0, color='#aa3a3a', linestyle='--', linewidth=1.4, label='Stimulus onset')

    _style_single_axis(ax)
    ax.set_xlabel('Time Course (s)', fontsize=11)
    ax.set_ylabel('Fisher Information', fontsize=11)
    ax.set_title('Fisher Information Over Time', fontsize=13)

    if global_max > 0:
        ax.set_ylim(0, global_max * 1.15)

    ax.legend(frameon=False, fontsize=9, loc='upper right')

    fig.tight_layout()
    plt.show()

# %% 主流程
if __name__ == "__main__":
    print("Decoding analysis")
    # %% 读取数据
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data()
    segments, labels, neuron_pos_rr = preprocess_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    # %% 计算时间点上的分类准确率
    accuracies, time_points = classify_by_timepoints(segments, labels)
    # 绘制结果
    plot_accuracies(time_points, accuracies)
    # %% 对数据进行PCA+t-SNE可视化
    pca_tsne_visualization(segments, labels)
    # %% 计算时间点上的Fisher信息
    fisher_dict, time_points_fi = FI_by_timepoints(segments, labels, mode="multivariate", reduction=None)
    # 可视化    
    visualize_fisher_information(fisher_dict, time_points_fi)
    # %% 计算时间点上的Fisher信息
    fisher_dict, time_points_fi = FI_by_timepoints(segments, labels, mode="univariate", reduction="mean")
    # 可视化    
    visualize_fisher_information(fisher_dict, time_points_fi)
    # %% Fisher信息随着神经元数量的变化
    n_neurons_step = 5
    fi = []
    for n_neurons in range(n_neurons_step, segments.shape[1]+1, n_neurons_step):
        selected_segments = segments[(labels == 1) | (labels == 2), :n_neurons, 10:14]
        selected_labels = labels[(labels == 1) | (labels == 2)]
        X = selected_segments.reshape(selected_segments.shape[0], -1)
        fi.append(Fisher_information(X, selected_labels, mode="multivariate"))

    # %%可视化
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(
        np.arange(n_neurons_step, segments.shape[1]+1, n_neurons_step),
        fi,
        color='#2a7f5f',
        linewidth=2.2
    )   

# %%

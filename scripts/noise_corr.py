# %% 导入必要的模块
from network import *
import pandas as pd
import seaborn as sns
from scipy import stats

# %% 功能函数
# =========== 破坏时间同步性 ==============
def destroy_time_synchronization(segments, labels, time_range=None, shuffle_neurons=None):
    """Destroy the time synchronization of the segments."""
    segments_destroyed = segments.copy()
    if time_range is not None:
        start, end = time_range
        segments_destroyed = segments_destroyed[:, :, start:end]
    # 如果shuffle_neurons为None，则默认使用所有神经元
    if shuffle_neurons is None:
        shuffle_neurons = segments_destroyed.shape[1]
    
    selected_neurons = np.random.choice(segments.shape[1], size=shuffle_neurons, replace=False)
    for cls in np.unique(labels):
        mask = labels == cls
        selected_segments = segments_destroyed[mask, :, :]
        # 在每个神经元上分别混洗，以破坏同步性
        for neuron in selected_neurons:
            selected_segments[:, neuron, :] = selected_segments[np.random.permutation(selected_segments.shape[0]), neuron, :]
        segments_destroyed[mask, :, :] = selected_segments
    return segments_destroyed
# %% 主要分析函数
# =========== 计算噪音相关矩阵 ==============
def Noise_Corr(segments, labels=None, class_filter=None, time_range=None):
    # 计算平均响应
    if class_filter is not None:
        if labels is None:
            raise ValueError("labels are required when class_filter is set")
        labels = np.asarray(labels)
        if np.isscalar(class_filter):
            mask = labels == class_filter
        else:
            mask = np.isin(labels, class_filter)
        if mask.sum() == 0:
            raise ValueError(f"No trials found for class filter {class_filter}")
        segments = segments[mask]   

    mean_response = np.mean(segments, axis=0)   
    noise_segments = segments - mean_response

    data = reshape_segments(noise_segments, time_range=time_range)
    noise_corr = np.corrcoef(data)
    noise_corr = np.nan_to_num(noise_corr, nan=0.0)
    np.fill_diagonal(noise_corr, 1.0)
    return noise_corr

# =========== 破坏时间同步性后计算网络指标 ==============
def shuffle_network(segments, labels, n_step=10, n_max=None) -> (list|list):
    """Shuffle the network and calculate the Fisher information and efficiency."""
    fi = []
    efficiency = []

    if n_max is None:
        n_max = segments.shape[1]

    for n in range(5, n_max, n_step):
        shuffle_segments = destroy_time_synchronization(segments, labels, time_range=None, shuffle_neurons=n)
        shuffle_segments = shuffle_segments[labels!=0, :, 14:18]
        y = np.array([labels[i] for i in range(len(labels)) if labels[i]!=0])   
        X = shuffle_segments.reshape(shuffle_segments.shape[0], -1)
        fi_shuffle = Fisher_information(X, y, mode="multivariate")

        _, graph, __ = construct_correlation_network(shuffle_segments, labels, class_filter=None, time_range=None, zscore=False, threshold=None, top_k=0.05, weighted=False, absolute=False)
        efficiency_shuffle = nx.global_efficiency(graph)
        fi.append(fi_shuffle)
        efficiency.append(efficiency_shuffle)
    return fi, efficiency

# =========== 破坏时间同步性后计算网络指标 ==============
def shuffle_network_by_condition(segments, labels, n_step=10, n_max=None) -> (list|list):
    """Shuffle the network and calculate the Fisher information and efficiency."""
    efficiencies = {}

    if n_max is None:
        n_max = segments.shape[1]
    for cls in np.unique(labels):
        efficiencies[cls] = []
        for n in range(5, n_max, n_step):
            shuffle_segments = destroy_time_synchronization(segments, labels, time_range=None, shuffle_neurons=n)
            shuffle_segments = shuffle_segments[:, :, 14:18]

            _, graph, __ = construct_correlation_network(shuffle_segments, labels, class_filter=cls, time_range=None, zscore=False, threshold=None, top_k=0.05, weighted=False, absolute=False)
            efficiency_shuffle = nx.global_efficiency(graph)
            efficiencies[cls].append(efficiency_shuffle)
    return efficiencies
# =========== 噪音相关性矩阵在类别间有无差异(已废弃)==============
def _noise_corr_classification(segments, labels):
    """Classify the noise correlation matrix using SVM."""
    # 噪音相关性矩阵在类别间有无差异
    mean_response = {} # 每个类别的平均响应
    noise_corr_by_class = {} # 每个类别的噪音相关性矩阵
    for cls in np.unique(labels):
        mean_response[cls] = np.mean(segments[labels==cls], axis=0)
    # 对每个trial计算一个噪音相关性矩阵，然后用SVM做分类
    noise_corrs = np.zeros((segments.shape[0], segments.shape[1], segments.shape[1]))
    for trial in range(segments.shape[0]):
        noise_segments = segments[trial] - mean_response[labels[trial]]
        noise_corr_trial = np.corrcoef(noise_segments.reshape(noise_segments.shape[0], -1))
        noise_corrs[trial] = noise_corr_trial   
    X = noise_corrs.reshape(noise_corrs.shape[0], -1)
    y = np.array([labels[i] for i in range(len(labels))])
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    # 用pca降维
    from sklearn.decomposition import PCA

    pca = PCA(n_components=7)
    X_pca = pca.fit_transform(X_scaled)
    # 用SVM做分类
    clf = SVC(kernel='rbf', class_weight='balanced',C=1.0,gamma='scale')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_pca, y, cv=cv, scoring='accuracy')
    accuracy = scores.mean()
    print(f"Noise Correlation Matrix Classification Accuracy: {accuracy:.3f}")

# %% 可视化函数
# =========== 绘制相关性和噪音相关的成对分布图 ==============
def _plot_noise_corr_pairwise_distribution(noise_corr, corr_matrix):
    noise_corr = np.asarray(noise_corr, dtype=float)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    if noise_corr.shape != corr_matrix.shape:
        raise ValueError("noise_corr and corr_matrix must have the same shape")

    idx = np.triu_indices_from(noise_corr, k=1)
    noise_vals = noise_corr[idx]
    signal_vals = corr_matrix[idx]

    data = pd.DataFrame({"Noise correlation": noise_vals, "Signal correlation": signal_vals})

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sns.histplot(
        data=data,
        x="Noise correlation",
        y="Signal correlation",
        bins=60,
        cmap="mako",
        cbar=True,
        cbar_kws={"label": "Count"},
        pmax=0.96,
        ax=ax,
    )

    if noise_vals.size > 1:
        r_val, p_val = stats.pearsonr(noise_vals, signal_vals)
        ax.text(
            0.02,
            0.96,
            f"Pearson r = {r_val:.2f}\np = {p_val:.3g}",
            transform=ax.transAxes,
            fontsize=9,
            color="#333333",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f6f6f6", edgecolor="#d8d8d8"),
        )

    ax.set_title("Noise vs. signal correlation density", fontsize=13)
    ax.set_xlabel("Noise correlation", fontsize=11)
    ax.set_ylabel("Signal correlation", fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    return True

# =========== 绘制各类别效率下降曲线 ==============
def _plot_efficiency_by_condition(efficiencies, num, n_step):
    """Plot the efficiency by condition."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cls, eff_list in efficiencies.items():
        x = list(range(5, num, n_step))
        ax.plot(
            x,
            eff_list,
            label=f"Class {cls}",
            marker="o",
            linewidth=2,
        )
    ax.set_xlabel("Number of shuffled neurons", fontsize=11)
    ax.set_ylabel("Global Efficiency", fontsize=11)
    ax.set_title("Network efficiency vs. shuffled neurons\nby condition", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()
# =========== 绘制破坏时间同步性后网络指标的曲线 ==============
def _plot_shuffle_network_curve(fi, efficiency, num, n_step):
    counts = np.arange(5, num, n_step)
    if len(fi) != len(counts) or len(efficiency) != len(counts):
        raise ValueError("fi and efficiency lengths must match the sweep defined by num and n_step")

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 4.6), sharex=True)

    # Fisher information 子图
    ax_fi = axes[0]
    sns.lineplot(
        x=counts,
        y=fi,
        marker="o",
        linewidth=2.0,
        color="#355C7D",
        ax=ax_fi,
    )
    if counts.size > 1:
        rho_fi, p_fi = stats.spearmanr(counts, fi)
        ax_fi.text(
            0.02,
            0.05,
            f"Spearman rho={rho_fi:.2f}\np={p_fi:.3g}",
            transform=ax_fi.transAxes,
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#d0d0d0"),
        )
    ax_fi.set_title("Fisher information vs. shuffled neurons", fontsize=13)
    ax_fi.set_xlabel("Number of shuffled neurons", fontsize=11)
    ax_fi.set_ylabel("Fisher information", fontsize=11)
    sns.despine(ax=ax_fi)

    # Global efficiency 子图
    ax_eff = axes[1]
    sns.lineplot(
        x=counts,
        y=efficiency,
        marker="o",
        linewidth=2.0,
        color="#F67280",
        ax=ax_eff,
    )
    if counts.size > 1:
        rho_eff, p_eff = stats.spearmanr(counts, efficiency)
        ax_eff.text(
            0.02,
            0.05,
            f"Spearman rho={rho_eff:.2f}\np={p_eff:.3g}",
            transform=ax_eff.transAxes,
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#d0d0d0"),
        )
    ax_eff.set_title("Global efficiency vs. shuffled neurons", fontsize=13)
    ax_eff.set_xlabel("Number of shuffled neurons", fontsize=11)
    ax_eff.set_ylabel("Global efficiency", fontsize=11)
    sns.despine(ax=ax_eff)

    fig.tight_layout()
    plt.show()
    return True
# %% 主流程
if __name__ == "__main__":
    print("=== 关于噪声和相关性的分析 ===")
    # %% 加载数据
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data()
    segments, labels, neuron_pos_rr = preprocess_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    # %% 构建网络
    corr_matrix, corr_graph, summary = construct_correlation_network(
        segments,
        labels=labels,
        class_filter=None,
        time_range=None,
        zscore=False,
        threshold=None,
        top_k=0.05, 
        weighted=False,
        absolute=False,
    )
    # %% 计算噪声和相关性的关系
    noise_corr = Noise_Corr(segments, labels, class_filter=None, time_range=None)
    plot_correlation_matrix(noise_corr, title="Noise Correlation Matrix")
    plot_correlation_matrix(corr_matrix, title="Overall Correlation Matrix")
    # %% 构建噪声相关网络
    noise_graph = build_correlation_graph(noise_corr, threshold=None, top_k=0.05, weighted=False, absolute=False)
    plot_correlation_graph(noise_graph, neuron_pos=neuron_pos_rr, title="Noise Correlation Network")
    # %% 计算度，聚类中心性， 特征向量中心性等指标的神经元分布
    # 计算总体网络的度，聚类中心性， 特征向量中心性等指标的神经元分布
    degree_values, clustering_values, eigenvector_values = compute_network_metrics(corr_graph)
    plot_network_metric_distributions(degree_values, clustering_values, eigenvector_values)
    rich_club_coeffs = nx.rich_club_coefficient(corr_graph, normalized=True)
    plot_rich_club_coefficient(rich_club_coeffs)
    # 计算噪声相关网络的度，聚类中心性， 特征向量中心性等指标的神经元分布
    degree_values, clustering_values, eigenvector_values = compute_network_metrics(noise_graph)
    plot_network_metric_distributions(degree_values, clustering_values, eigenvector_values)
    rich_club_coeffs = nx.rich_club_coefficient(noise_graph, normalized=True)
    plot_rich_club_coefficient(rich_club_coeffs)
    # %% 分条件进行
    nx_result = {}
    for cls in np.unique(labels):
        noise_corr = Noise_Corr(segments, labels, class_filter=cls, time_range=None)
        noise_graph = build_correlation_graph(noise_corr, threshold=None, top_k=0.05, weighted=False, absolute=False)
        summary = correlation_network_summary(noise_graph)
        efficiency = nx.global_efficiency(noise_graph)
        modularity = nx.algorithms.community.modularity(
            noise_graph,
            nx.algorithms.community.greedy_modularity_communities(noise_graph),
        )   
        nx_result[cls] = {
            "summary": summary,
            "efficiency": efficiency,
            "modularity": modularity,
        }
    # print(nx_result)
    plot_network_metrics_by_class(nx_result)
    # %% 绘制噪音相关和相关性的成对分布图
    _plot_noise_corr_pairwise_distribution(noise_corr, corr_matrix)
    # %%
    num = 250
    n_step = 10
    fi, efficiency = shuffle_network(segments, labels, n_max = num, n_step = n_step)
    _plot_shuffle_network_curve(fi, efficiency, num, n_step)
 
    # %% 分条件查看效率的下降情况
    efficiencies = shuffle_network_by_condition(segments, labels, n_max = num, n_step = n_step)
    # 可视化
    _plot_efficiency_by_condition(efficiencies, num, n_step)
    # %%  噪音相关性的主要来源
    

    
# %%

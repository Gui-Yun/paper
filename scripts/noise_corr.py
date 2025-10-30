# %% 导入必要的模块
from network import *
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

# =========== 噪音相关性矩阵在类别间有无差异(已废弃)==============
def _noise_corr_classification(segments, labels):
    """Classify the noise correlation matrix using SVM."""
        # %% 噪音相关性矩阵在类别间有无差异
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
def plot_noise_corr_pairwise_distribution(noise_corr, corr_matrix):
    fig, ax = plt.subplots(figsize=(10, 10))
    # x为噪音相关值，y为对应的信号相关，绘制计数
    counts, x_edges, y_edges = np.histogram2d(noise_corr.flatten(), corr_matrix.flatten(), bins=100)
    im = ax.imshow(counts, cmap="bwr", vmin=0, vmax=np.max(counts))
    ax.set_title("Noise Correlation vs Signal Correlation")
    ax.set_xlabel("Noise Correlation")
    ax.set_ylabel("Signal Correlation")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    plt.show()
    return True

# =========== 绘制破坏时间同步性后网络指标的曲线 ==============
def _plot_shuffle_network_curve(fi, efficiency, num, n_step):
    # 在三张子图上绘制三条曲线
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.4))
    axs[0].plot(range(5, num, n_step), fi, marker='o', label='Fisher Information')
    axs[0].legend(frameon=False, fontsize=9, loc='upper right')
    axs[0].set_xlabel('Number of Shuffled Neurons', fontsize=11)
    axs[0].set_ylabel('Fisher Information', fontsize=11)
    axs[0].set_title('Fisher Information vs Number of Shuffled Neurons', fontsize=13)
    axs[0].grid(True)
    axs[1].plot(range(5, num, n_step), efficiency, marker='o', label='Efficiency')
    axs[1].legend(frameon=False, fontsize=9, loc='upper right')
    axs[1].set_xlabel('Number of Shuffled Neurons', fontsize=11)
    axs[1].set_ylabel('Efficiency', fontsize=11)
    axs[1].set_title('Efficiency vs Number of Shuffled Neurons', fontsize=13)
    axs[1].grid(True)
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
    plot_noise_corr_pairwise_distribution(noise_corr, corr_matrix)
    # %%
    num = 250
    n_step = 10
    fi, efficiency = shuffle_network(segments, labels, n_max = num, n_step = n_step)
    _plot_shuffle_network_curve(fi, efficiency, num, n_step)
 


# %%

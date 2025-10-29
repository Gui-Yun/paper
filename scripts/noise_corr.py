# %% 导入必要的模块
from network import *
# %% 功能函数

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


# %%

# %% Network Analysis
from decoding import *
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# %% ========= helpers =========
def _style_axis(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="both", which="major", labelsize=9)
def reshape_segments(segments, time_range=None):
    """Convert segments to shape (neurons, samples) for correlation analysis."""
    segments = np.asarray(segments)
    if segments.ndim != 3:
        raise ValueError("segments must be shaped (trials, neurons, timepoints)")

    if time_range is not None:
        start, end = time_range
        segments = segments[:, :, start:end]

    _, neurons, _ = segments.shape
    return segments.transpose(1, 0, 2).reshape(neurons, -1)

# %% ========= core functions =========
# =========== 计算相关矩阵 ============
def compute_correlation_matrix(segments, labels=None, class_filter=None, time_range=None, zscore=True):
    """Return neuron x neuron correlation matrix with optional trial/time selection."""
    segments = np.asarray(segments)

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

    data = reshape_segments(segments, time_range=time_range)

    if zscore:
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + 1e-9
        data = (data - mean) / std

    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr
# ========= 构建相关网络 =========
def build_correlation_graph(corr_matrix, threshold=None, top_k=None, weighted=True, absolute=True, weight_attr="weight"):
    """Convert correlation matrix to a NetworkX graph."""
    corr = np.asarray(corr_matrix)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr_matrix must be square")

    n_nodes = corr.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            value = float(corr[i, j])
            score = abs(value) if absolute else value
            edges.append((score, i, j, value))

    if threshold is not None:
        edges = [edge for edge in edges if edge[0] >= threshold]

    edges.sort(key=lambda e: e[0], reverse=True)
    total_edges = len(edges)
    if top_k is not None:
        edges = edges[: int(top_k * total_edges)]

    for _, i, j, value in edges:
        if weighted:
            graph.add_edge(i, j, **{weight_attr: value})
        else:
            graph.add_edge(i, j)

    return graph
# ========= 相关网络统计 =========
def correlation_network_summary(graph):
    """Return basic statistics for the correlation graph."""
    if graph.number_of_nodes() == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "mean_degree": 0.0,
            "largest_component": 0,
            "avg_clustering": 0.0,
            "global_efficiency": 0.0,
            "transitivity": 0.0,
        }

    degrees = [deg for _, deg in graph.degree()]
    components = [len(c) for c in nx.connected_components(graph)]
    largest_component = max(components) if components else 0
    avg_clustering = nx.average_clustering(graph, weight="weight") if graph.number_of_nodes() > 1 else 0.0
    global_eff = nx.global_efficiency(graph)
    trans = nx.transitivity(graph)

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "largest_component": largest_component,
        "avg_clustering": float(avg_clustering),
        "global_efficiency": float(global_eff),
        "transitivity": float(trans),
    }
# ========= 构建相关网络的流水线 =========
def construct_correlation_network(segments, labels=None, class_filter=None, time_range=None, zscore=True, threshold=0.5, top_k=None, weighted=True, absolute=True):
    """Pipeline: compute correlation matrix, build graph, and summarise."""
    corr_matrix = compute_correlation_matrix(
        segments,
        labels=labels,
        class_filter=class_filter,
        time_range=time_range,
        zscore=zscore,
    )

    graph = build_correlation_graph(
        corr_matrix,
        threshold=threshold,
        top_k=top_k,
        weighted=weighted,
        absolute=absolute,
    )

    summary = correlation_network_summary(graph)
    return corr_matrix, graph, summary
# =========== 计算网络指标 =========
def compute_network_metrics(graph):
    """Compute degree, clustering coefficient, and eigenvector centrality for each node."""
    degrees = dict(graph.degree())
    clustering = nx.clustering(graph)
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    degree_values = np.array(list(degrees.values()))
    clustering_values = np.array(list(clustering.values()))
    eigenvector_values = np.array(list(eigenvector.values()))
    return degree_values, clustering_values, eigenvector_values

# =========== 计算每个条件的网络指标 ============
def compute_network_metrics_by_class(segments, labels):
    """Compute network metrics for each class."""
    nx_result = {}
    for cls in np.unique(labels):
        corr_matrix, corr_graph, summary = construct_correlation_network(
            segments,
            labels=labels,
            class_filter=cls,
            time_range=None,
            zscore=False,
            threshold=None,
            top_k=0.05,
            weighted=False,
            absolute=False,
        )
        # 效率，模块化
        efficiency = nx.global_efficiency(corr_graph)
        modularity = nx.algorithms.community.modularity(
            corr_graph,
            nx.algorithms.community.greedy_modularity_communities(corr_graph),
        )   
        # 保存结果
        nx_result[cls] = {
            "corr_matrix": corr_matrix,
            "corr_graph": corr_graph,
            "summary": summary,
            "efficiency": efficiency,
            "modularity": modularity,
        }
    return nx_result
# %% ========= plotting functions =========
def plot_correlation_matrix(corr_matrix, title="Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(corr_matrix, cmap="bwr", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Neuron index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")
    plt.show()
    return True
# ============ 绘制网络图 ============
def plot_correlation_graph(graph, neuron_pos=None, title="Correlation Network"):
    # 结合neuron位置绘制网络图
    assert neuron_pos.shape[1] == graph.number_of_nodes(), "neuron_pos shape does not match graph nodes"
    pos_dict = {i: (neuron_pos[0, i], neuron_pos[1, i]) for i in range(neuron_pos.shape[1])}
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(
        graph,
        pos=pos_dict,
        node_size=20,
        node_color="blue",
        edge_color="gray",
        with_labels=False,
        alpha=0.7,
    )
    plt.title("Overall Correlation Network")
    plt.axis("off")
    plt.show()

# ============ 分布图绘制 ============
def plot_network_metric_distributions(degree_values, clustering_values, eigenvector_values):
    # 绘制度分布图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(degree_values, bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    _style_axis(ax)
    plt.show()
    # 绘制聚类中心性分布图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clustering_values, bins=30, color="lightgreen", edgecolor="black")
    ax.set_title("Clustering Coefficient Distribution")
    ax.set_xlabel("Clustering Coefficient")
    ax.set_ylabel("Count")
    _style_axis(ax)
    plt.show()
    # 绘制特征向量中心性分布图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(eigenvector_values, bins=30, color="salmon", edgecolor="black")
    ax.set_title("Eigenvector Centrality Distribution")
    ax.set_xlabel("Eigenvector Centrality")
    ax.set_ylabel("Count")
    _style_axis(ax)
    plt.show()
# =========== 富人俱乐部 ============
def plot_rich_club_coefficient(rich_club_coeffs):
    # 可视化富人俱乐部系数
    degrees = sorted(rich_club_coeffs.keys())
    coeffs = [rich_club_coeffs[d] for d in degrees]
    plt.figure(figsize=(6, 4))
    plt.plot(degrees, coeffs, marker="o", linestyle="-")
    plt.title(f"Rich-Club Coefficient")
    plt.xlabel("Degree")
    plt.ylabel("Rich-Club Coefficient")
    plt.grid()
    plt.show()
    return True
# ========== 可视化对比图 ===========
def plot_network_metrics_by_class(nx_result, metrics = ["largest_component", "avg_clustering", "global_efficiency", "transitivity"]):
    # 由于每个指标的值域不一致，需要归一化处理
    for metric in metrics:
        all_values = [nx_result[cls]["summary"][metric] for cls in nx_result.keys()]
        mean_val = np.mean(all_values)
        min_val = np.min(all_values)
        std_val = np.std(all_values)
        for cls in nx_result.keys():
            value = nx_result[cls]["summary"][metric]
            if std_val > 1e-6:
                norm_value = value / mean_val
            else:
                norm_value = 0.0
            nx_result[cls]["summary"][f"{metric}_norm"] = norm_value
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.2
    for i, cls in enumerate(sorted(nx_result.keys())):
        values = [nx_result[cls]["summary"][f"{metric}_norm"] for metric in metrics]
        ax.bar(x + i * width, values, width=width, label=f"Class {cls}")
    ax.set_xticks(x + width * (len(nx_result) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=45, ha="right")    
    ax.set_ylabel("Value")
    ax.set_title("Network Metrics by Class")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return  True
# %% ========= main analysis =========
if __name__ == "__main__":
    print("==== network analysis ====")
    # %% load and preprocess data
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data()
    segments, labels, neuron_pos_rr = preprocess_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    # %% 全数据的网络指标
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
    print("Overall network summary: {summary}")
    # 可视化相关矩阵和网络
    plot_correlation_matrix(corr_matrix, title="Overall Correlation Matrix")
    plot_correlation_graph(corr_graph, neuron_pos=neuron_pos_rr, title="Overall Correlation Network")
    # %% 计算度，聚类中心性， 特征向量中心性等指标的神经元分布
    degree_values, clustering_values, eigenvector_values = compute_network_metrics(corr_graph)
    plot_network_metric_distributions(degree_values, clustering_values, eigenvector_values)
    rich_club_coeffs = nx.rich_club_coefficient(corr_graph, normalized=True)
    plot_rich_club_coefficient(rich_club_coeffs)
    # %% 每个条件的网络指标
    nx_result = compute_network_metrics_by_class(segments, labels)
    plot_network_metrics_by_class(nx_result)

# %%

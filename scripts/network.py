# %% Network Analysis
from decoding import *
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from scipy import stats

sns.set_theme(context="paper", style="whitegrid")
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
    # corr_matrix = np.asarray(corr_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sns.heatmap(
        corr_matrix,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    # ax.set_xlabel("Neuron index", fontsize=11)
    ax.set_ylabel("Neuron index", fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    return True
# ============ 绘制网络图 ============
def plot_correlation_graph(graph, neuron_pos=None, title="Correlation Network"):
    if neuron_pos is None:
        raise ValueError("neuron_pos is required for plotting the spatial layout")
    assert neuron_pos.shape[1] == graph.number_of_nodes(), "neuron_pos shape does not match graph nodes"

    pos_dict = {i: (float(neuron_pos[0, i]), float(neuron_pos[1, i])) for i in range(neuron_pos.shape[1])}
    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
    degree_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-6)

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    edge_weights = np.array([abs(graph[u][v].get("weight", 1.0)) for u, v in graph.edges()], dtype=float)
    if edge_weights.size:
        edge_norm = edge_weights / (edge_weights.max() + 1e-6)
        nx.draw_networkx_edges(
            graph,
            pos=pos_dict,
            ax=ax,
            edge_color="#b0b8c5",
            width=0.4 + 2.6 * edge_norm,
            alpha=0.25 + 0.6 * edge_norm,
        )
    else:
        nx.draw_networkx_edges(
            graph,
            pos=pos_dict,
            ax=ax,
            edge_color="#b0b8c5",
            width=0.8,
            alpha=0.25,
        )

    node_sizes = 60 + 220 * degree_norm
    nx.draw_networkx_nodes(
        graph,
        pos=pos_dict,
        ax=ax,
        node_size=node_sizes,
        node_color=degree_norm,
        cmap=cmap,
        linewidths=0.6,
        edgecolors="#f8f8f8",
    )

    ax.set_title(title, fontsize=13)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=degrees.min(), vmax=degrees.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Node degree", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    plt.show()

# ============ 分布图绘制 ============
def plot_network_metric_distributions(degree_values, clustering_values, eigenvector_values):
    metric_data = [
        ("Degree", np.asarray(degree_values, dtype=float)),
        ("Clustering coefficient", np.asarray(clustering_values, dtype=float)),
        ("Eigenvector centrality", np.asarray(eigenvector_values, dtype=float)),
    ]

    fig, axes = plt.subplots(1, len(metric_data), figsize=(12.5, 4.2))
    axes = np.atleast_1d(axes)
    palette = sns.color_palette("crest", n_colors=len(metric_data))

    for (name, values), color, ax in zip(metric_data, palette, axes):
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        bins = min(30, max(int(values.size / 4), 8))
        sns.histplot(values, bins=bins, stat="density", color=color, alpha=0.28, edgecolor="none", ax=ax)
        sns.kdeplot(values, fill=True, color=color, linewidth=1.8, alpha=0.6, ax=ax)
        ax.axvline(np.median(values), color=color, linestyle="--", linewidth=1.2, label="Median")
        ax.set_title(f"{name} distribution", fontsize=12)
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        _style_axis(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    plt.show()
    return True
# =========== 富人俱乐部 ============
def plot_rich_club_coefficient(rich_club_coeffs):
    degrees = np.array(sorted(rich_club_coeffs.keys()), dtype=float)
    coeffs = np.array([rich_club_coeffs[d] for d in degrees], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sns.lineplot(x=degrees, y=coeffs, ax=ax, color="#355C7D", marker="o", linewidth=2.1)
    ax.fill_between(degrees, coeffs, color="#355C7D", alpha=0.18)

    slope, intercept, r_value, p_value, _ = stats.linregress(degrees, coeffs)
    ax.text(
        0.02,
        0.92,
        f"Pearson r = {r_value:.2f}\np = {p_value:.3g}",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f6f6f6", edgecolor="#d8d8d8"),
    )

    ax.set_xlabel("Degree", fontsize=11)
    ax.set_ylabel("Rich-club coefficient", fontsize=11)
    ax.set_title("Rich-Club Coefficient Profile", fontsize=13)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    return True
# ========== 可视化对比图 ===========
def plot_network_metrics_by_class(nx_result, metrics=None):
    if metrics is None:
        metrics = ["largest_component", "avg_clustering", "global_efficiency", "transitivity"]

    records = []
    for metric in metrics:
        values = np.array([nx_result[cls]["summary"][metric] for cls in nx_result.keys()], dtype=float)
        mean_val = values.mean() if values.size else 0.0
        for cls in nx_result.keys():
            value = nx_result[cls]["summary"][metric]
            norm_value = value / mean_val if mean_val else 0.0
            records.append({"Class": cls, "Metric": metric, "Normalized": norm_value})

    data = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.pointplot(
        data=data,
        x="Metric",
        y="Normalized",
        hue="Class",
        dodge=0.25,
        markers="o",
        linestyles="-",
        palette="deep",
        ax=ax,
    )

    ax.axhline(1.0, color="#8a8a8a", linestyle="--", linewidth=1.0, label="Global mean")
    ax.set_ylabel("Normalized value", fontsize=11)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_title("Network metrics by class (normalized)", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(frameon=False, fontsize=9, loc="best")
    sns.despine(ax=ax)

    spreads = data.groupby("Metric")["Normalized"].agg(["max", "min"])
    if not spreads.empty:
        span_metric = (spreads["max"] - spreads["min"]).idxmax()
        span_value = (spreads["max"] - spreads["min"]).max()
        ax.text(
            0.02,
            0.05,
            f"Largest spread: {span_metric} ?={span_value:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f4f4", edgecolor="#d8d8d8"),
        )

    fig.tight_layout()
    plt.show()
    return True
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

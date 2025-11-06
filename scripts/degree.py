# %% 关于度的分析   
# ==========================================
# 导入必要的库以及脚本内容
from network import *
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(context="paper", style="whitegrid")
# %% 功能函数

# %% 主要分析函数
# =========== Fisher信息和度的关系==============
# =========== Fisher信息和度的关系==============ni
def Fi_by_degree(degrees, segments) -> (list|list):
    fi_degree = []
    fi_random = []
    for n in range(5, segments.shape[1], 10):
        selected_segments = segments[labels!=1, :, 14:18]
        # 度最高的n个神经元
        selected_neurons = sorted(degrees, key=degrees.get, reverse=True)[:n]
        selected_segments_degree = selected_segments[:, selected_neurons, :]
        # 随机挑选作为对照
        selected_neurons = np.random.choice(list(degrees.keys()), size=n, replace=False)
        selected_segments_random = selected_segments[:, selected_neurons, :]
        # 选择度最高的神经元
        y = np.array([labels[i] for i in range(len(labels)) if labels[i]==0 or labels[i]==2])
        # 计算Fisher信息
        X_degree = selected_segments_degree.reshape(selected_segments_degree.shape[0], -1)
        X_random = selected_segments_random.reshape(selected_segments_random.shape[0], -1)

        _pca = PCA(n_components=min(int(len(y)/2),X_degree.shape[1]))
        X_degree_reduce = _pca.fit_transform(X_degree)
        X_random_reduce = _pca.fit_transform(X_random)
        fi_degree.append(Fisher_information(X_degree_reduce, y, mode="multivariate"))
        fi_random.append(Fisher_information(X_random_reduce, y, mode="multivariate"))
    return fi_degree, fi_random 

# ============== 单神经元的Fisher信息和度的关系==============
def Fi_Single_neuron(segments, labels, degrees) -> bool:
    X = segments[labels!=0, :, :]
    y = np.array([labels[i] for i in range(len(labels)) if labels[i]==1 or labels[i]==2])
    fi_neuron = np.max(Fisher_information(X,y,mode="univariate"), axis=1)
    neuron_degrees = np.array([degrees[i] for i in range(segments.shape[1])])
    
    _visualize_fi_single_neuron(fi_neuron, neuron_degrees)
    return True

# %% 可视化函数
# ============== 度分布与Fisher信息的关系==============
def _visualize_fi_by_degree(fi_degree, fi_random):
    fi_degree = np.asarray(fi_degree, dtype=float)
    fi_random = np.asarray(fi_random, dtype=float)
    neuron_counts = np.arange(5, 5 + 10 * len(fi_degree), 10)

    data = pd.DataFrame({
        "Neurons": np.tile(neuron_counts, 2),
        "Fisher Information": np.concatenate([fi_degree, fi_random]),
        "Selection": np.repeat(["Top-degree", "Random"], len(neuron_counts))
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    palette = {"Top-degree": "#355C7D", "Random": "#F67280"}
    sns.lineplot(
        data=data,
        x="Neurons",
        y="Fisher Information",
        hue="Selection",
        style="Selection",
        markers=True,
        dashes=False,
        linewidth=2.1,
        markeredgecolor="white",
        markeredgewidth=0.8,
        palette=palette,
        ax=ax
    )

    ax.set_xlabel("Number of Selected Neurons", fontsize=11)
    ax.set_ylabel("Fisher Information", fontsize=11)
    ax.set_title("Fisher Information of Degree-based vs Random Selections", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    differences = fi_degree - fi_random
    annotation = ""
    if differences.size > 1:
        try:
            ci = stats.bootstrap((differences,), np.mean, confidence_level=0.95, n_resamples=2000, method="basic")
            mean_diff = differences.mean()
            ci_low, ci_high = ci.confidence_interval
            annotation += f"Delta FI = {mean_diff:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}])\n"
        except Exception:
            pass
        _t_stat, p_val = stats.ttest_rel(fi_degree, fi_random, nan_policy="omit")
        annotation += f"Paired t-test p = {p_val:.3g}"
    elif differences.size == 1:
        annotation += f"Delta FI = {differences[0]:.3f}"

    if annotation:
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=9,
            color="#333333",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#d0d0d0")
        )

    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    return True

# ============== 单神经元的Fisher信息和度的关系==============
def _visualize_fi_single_neuron(fi_neuron, neuron_degrees):
    fi_neuron = np.asarray(fi_neuron, dtype=float)
    neuron_degrees = np.asarray(neuron_degrees, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sns.regplot(
        x=neuron_degrees,
        y=fi_neuron,
        scatter_kws={"s": 40, "alpha": 0.55, "color": "#355C7D", "edgecolor": "white"},
        line_kws={"color": "#F67280"},
        ci=95,
        ax=ax
    )

    ax.set_xlabel("Neuron Degree", fontsize=11)
    ax.set_ylabel("Fisher Information", fontsize=11)
    ax.set_title("Fisher Information vs. Neuron Degree", fontsize=13)

    rho, p_val = stats.spearmanr(neuron_degrees, fi_neuron, nan_policy="omit")
    ax.text(
        0.02,
        0.95,
        f"Spearman rho = {rho:.2f}\np = {p_val:.3g}",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f5f5", edgecolor="#d0cdd4")
    )

    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    return True 

# %% 主程序
if __name__ == "__main__":
    print("=== 关于度的分析 ===")
    # %% 加载数据
    # %% load and preprocess data
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
    # %% 计算度分布
    degrees = dict(corr_graph.degree())
    # degrees = nx.eigenvector_centrality(corr_graph, max_iter=1000)
    fi_degree, fi_random = Fi_by_degree(degrees, segments)
    _visualize_fi_by_degree(fi_degree, fi_random)
    # %% 
    # 计算单神经元的Fisher信息与度的关系
    Fi_Single_neuron(segments, labels, degrees)

# %%

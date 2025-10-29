# %% 关于度的分析   
# ==========================================
# 导入必要的库以及脚本内容
from network import *
# %% 功能函数

# %% 主要分析函数
# =========== Fisher信息和度的关系 ==============
def Fi_by_degree(degrees, segments) -> (list|list):
    fi_degree = []
    fi_random = []
    for n in range(5, segments.shape[1], 10):
        selected_segments = segments[labels!=1, :, 14:18]
        # 度最高的n个神经元
        selected_neurons = sorted(degrees, key=degrees.get, reverse=True)[:n]
        selected_segments_degree = selected_segments[:, selected_neurons, :]
        # 随机挑选作为对比
        selected_neurons = np.random.choice(list(degrees.keys()), size=n, replace=False)
        selected_segments_random = selected_segments[:, selected_neurons, :]
        # 选择度最高的神经元
        y = np.array([labels[i] for i in range(len(labels)) if labels[i]==0 or labels[i]==2])
        # 计算Fisher信息
        X_degree = selected_segments_degree.reshape(selected_segments_degree.shape[0], -1)
        X_random = selected_segments_random.reshape(selected_segments_random.shape[0], -1)
        fi_degree.append(Fisher_information(X_degree, y, mode="multivariate"))
        fi_random.append(Fisher_information(X_random, y, mode="multivariate"))
    return fi_degree, fi_random 

# ============== 单神经元的Fisher信息和度的关系 ==============
def Fi_Single_neuron(segments, labels, degrees) -> bool:
    X = segments[labels!=0, :, :]
    y = np.array([labels[i] for i in range(len(labels)) if labels[i]==1 or labels[i]==2])
    fi_neuron = np.max(Fisher_information(X,y,mode="univariate"), axis=1)
    neuron_degrees = np.array([degrees[i] for i in range(segments.shape[1])])
    
    visualize_fi_single_neuron(fi_neuron, neuron_degrees)
    return True

# %% 可视化函数
# ============== 度分布与Fisher信息的关系 ==============
def visualize_fi_by_degree(fi_degree, fi_random):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(range(5, segments.shape[1], 10), fi_degree, marker='o', label='Top Degree Neurons')
    ax.plot(range(5, segments.shape[1], 10), fi_random, marker='x', label='Random Neurons')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_xlabel('Number of Top Degree Neurons', fontsize=11)
    ax.set_ylabel('Fisher Information', fontsize=11)
    ax.set_title('Fisher Information vs Number of Top Degree Neurons', fontsize=13)
    ax.grid(True)
    plt.show()
    return True

# ============== 单神经元的Fisher信息和度的关系 ==============
def visualize_fi_single_neuron(fi_neuron, neuron_degrees):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(neuron_degrees, fi_neuron)
    # 回归
    z = np.polyfit(neuron_degrees, fi_neuron, 1)
    p = np.poly1d(z)
    ax.plot(neuron_degrees, p(neuron_degrees), color='red')
    ax.set_xlabel('Neuron Degree', fontsize=11)
    ax.set_ylabel('Fisher Information', fontsize=11)
    ax.set_title('Fisher Information vs Neuron Degree', fontsize=13)
    ax.grid(True)
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
    visualize_fi_by_degree(fi_degree, fi_random)
    # %% 
    # 计算单神经元的Fisher信息与度的关系
    Fi_Single_neuron(segments, labels, degrees)

# %%

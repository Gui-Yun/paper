# %% 基础的数据读取 + 简单分析
# 导入必要的库
import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %% 定义配置
class ExpConfig:
    def __init__(self, file_path = None):
        # 加载配置文件
        if file_path is not None:
            try:
                self.load_config(file_path)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                self.set_default_config()
        else:
            self.set_default_config()
        self.preprocess_cfg = {
            'preprocess': True,
            'win_size' : 150
        }

    def load_config(self, file_path):
        # 从文件加载配置
        # 如果不是json
        if not file_path.endswith('.json'):
            raise NotImplementedError("目前仅支持JSON格式的配置文件")
        # 解析配置数据
        import json
        with open(file_path, 'r') as f:
            config_data = json.load(f)  

        # 检查必要字段
        required_keys = ['DATA_PATH']
        missing = [k for k in required_keys if k not in config_data]
        if missing:
            raise KeyError(f"配置文件缺少字段: {', '.join(missing)}")
        
        # 赋值配置
        self.data_path = config_data.get("DATA_PATH")
        self.data_path = self.data_path.replace(
            "C:\\Users\\76629", 
            os.environ.get('USERPROFILE')
        )
        self.trial_info = config_data.get("TRIAL_INFO", {})
        self.exp_info = config_data.get("EXP_INFO")


    def set_default_config(self):
        # 设置默认配置
        # 数据路径
        self.data_path = "C:\\Users\\76629\\OneDrive\\brain\\Micedata\\M65_0816"
        self.data_path = self.data_path.replace(
            "C:\\Users\\76629", 
            os.environ.get('USERPROFILE')
        )
        # 试次信息
        self.trial_info = {
            "TRIAL_START_SKIP": 0,
            "TOTAL_TRIALS": 176
        }
        # 刺激参数
        self.exp_info = {
            "t_stimulus": 10,
            "l_stimulus": 20,
            "l_trials": 40,
            "IPD":5.0,
            "ISI":5.0
        }


cfg = ExpConfig("..\config\\m71.json")

# %% 预处理相关函数定义(通用)
# 从matlab改过来的，经过检查应该无误
def process_trigger(txt_file, IPD=cfg.exp_info["IPD"], ISI=cfg.exp_info["ISI"], fre=None, min_sti_gap=4.0):
    """
    处理触发文件，修改自step1x_trigger_725right.m
    
    参数:
    txt_file: str, txt文件路径
    IPD: float, 刺激呈现时长(s)，默认2s
    ISI: float, 刺激间隔(s)，默认6s
    fre: float, 相机帧率Hz，None则从相机触发时间自动估计
    min_sti_gap: float, 相邻刺激"2"小于此间隔(s)视作同一次（用于去重合并），默认5s
    
    返回:
    dict: 包含start_edge, end_edge, stimuli_array的字典
    """
    
    # 读入文件
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0])
                    ch_str = parts[1]
                    abs_ts = float(parts[2]) if len(parts) >= 3 else None
                    data.append((time_val, ch_str, abs_ts))
                except ValueError:
                    continue
    
    if not data:
        raise ValueError("未能从文件中读取到有效数据")
    
    # 解析数据
    times, channels, abs_timestamps = zip(*data)
    times = np.array(times)
    
    # 转换通道为数值，非数值的设为NaN
    ch_numeric = []
    valid_indices = []
    for i, ch_str in enumerate(channels):
        try:
            ch_val = float(ch_str)
            ch_numeric.append(ch_val)
            valid_indices.append(i)
        except ValueError:
            continue
    
    if not valid_indices:
        raise ValueError("未找到有效的数值通道数据")
    
    # 只保留有效数据
    t = times[valid_indices]
    ch = np.array(ch_numeric)
    
    # 相机帧与刺激起始时间
    cam_t_raw = t[ch == 1]
    print(len(cam_t_raw))
    sti_t_raw = t[ch == 2]
    print(len(sti_t_raw))
    if len(cam_t_raw) == 0:
        raise ValueError("未检测到相机触发(值=1)")
    if len(sti_t_raw) == 0:
        raise ValueError("未检测到刺激触发(值=2)")
    
    # 去重/合并：将时间靠得很近的"2"视作同一次刺激
    sti_t = np.sort(sti_t_raw)
    if len(sti_t) > 0:
        keep = np.ones(len(sti_t), dtype=bool)
        for i in range(1, len(sti_t)):
            if (sti_t[i] - sti_t[i-1]) < min_sti_gap:
                keep[i] = False  # 合并到前一个
        sti_t = sti_t[keep]
    
    # 帧率估计或使用给定值
    if fre is None:
        dt = np.diff(cam_t_raw)
        fre = 1 / np.median(dt)  # 用相机帧时间戳的中位间隔

    IPD_frames = max(1, round(IPD * fre))
    isi_frames = round((IPD + ISI) * fre)
    
    # 把每个刺激时间映射到最近的相机帧索引
    cam_t = cam_t_raw.copy()
    nFrames = len(cam_t)
    start_edge = np.zeros(len(sti_t), dtype=int)
    
    for k in range(len(sti_t)):
        idx = np.argmin(np.abs(cam_t - sti_t[k]))
        start_edge[k] = idx
    
    end_edge = start_edge + IPD_frames - 1
    
    # 边界裁剪，避免越界
    valid = (start_edge >= 0) & (end_edge < nFrames) & (start_edge <= end_edge)
    start_edge = start_edge[valid]
    end_edge = end_edge[valid]
    
    # 尾段完整性检查（与旧逻辑一致）
    if len(start_edge) >= 2:
        d = np.diff(start_edge)
        while len(d) > 0 and d[-1] not in [isi_frames-1, isi_frames, isi_frames+1, isi_frames+2]:
            # 丢掉最后一个可疑的刺激段
            start_edge = start_edge[:-1]
            end_edge = end_edge[:-1]
            if len(start_edge) >= 2:
                d = np.diff(start_edge)
            else:
                break
    
    # 生成0/1刺激数组（可视化/保存用）
    stimuli_array = np.zeros(nFrames)
    for i in range(len(start_edge)):
        stimuli_array[start_edge[i]:end_edge[i]+1] = 1
    
    # 保存结果到mat文件
    save_path = os.path.join(os.path.dirname(txt_file), 'visual_stimuli_with_label.mat')
    scipy.io.savemat(save_path, {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array
    })
    
    return {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array,
        'camera_frames': len(cam_t),
        'stimuli_count': len(start_edge)
    }
# ========== RR神经元筛选函数 ========== 
def rr_selection(trials, labels, t_stimulus=cfg.exp_info["t_stimulus"], l=cfg.exp_info["l_stimulus"], alpha_fdr=0.05, alpha_level=0.05, reliability_threshold=0.7, snr_threshold=0.8, effect_size_threshold=0.5, response_ratio_threshold=0.6):
    """
    快速RR神经元筛选
    优化策略:
    1. 向量化计算替代循环
    2. 简化统计检验（t检验替代Mann-Whitney U）
    3. 批量处理所有神经元
    """
    import time
    start_time = time.time()
    
    print("使用快速RR筛选算法...")
    
    # 过滤有效数据
    valid_mask = (labels == 1) | (labels == 2)
    valid_trials = trials[valid_mask]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_trials.shape
    
    # 定义时间窗口
    baseline_pre = np.arange(0, t_stimulus)
    baseline_post = np.arange(t_stimulus + l, n_timepoints)
    stimulus_window = np.arange(t_stimulus, t_stimulus + l)
    
    print(f"处理 {n_trials} 个试次, {n_neurons} 个神经元")
    
    # 1. 响应性检测 - 向量化计算
    # 计算基线和刺激期的平均值
    baseline_pre_mean = np.mean(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_mean = np.mean(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的平均
    baseline_mean = (baseline_pre_mean + baseline_post_mean) / 2
    
    stimulus_mean = np.mean(valid_trials[:, :, stimulus_window], axis=2)  # (trials, neurons)
    
    # 简化的响应性检测：基于效应大小和标准误差
    baseline_pre_std = np.std(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_std = np.std(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的标准差
    baseline_std = (baseline_pre_std + baseline_post_std) / 2
    
    stimulus_std = np.std(valid_trials[:, :, stimulus_window], axis=2)
    
    # Cohen's d效应大小
    pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2)
    effect_size = np.abs(stimulus_mean - baseline_mean) / (pooled_std + 1e-8)
    
    # 响应性标准：平均效应大小 > 阈值 且 至少指定比例试次有响应
    response_ratio = np.mean(effect_size > effect_size_threshold, axis=0)
    enhanced_neurons = np.where((response_ratio > response_ratio_threshold) & 
                              (np.mean(stimulus_mean > baseline_mean, axis=0) > response_ratio_threshold))[0].tolist()

    # 2. 可靠性检测 - 简化版本
    # 计算每个神经元在每个试次的信噪比
    signal_strength = np.abs(stimulus_mean - baseline_mean)
    noise_level = baseline_std + 1e-8
    snr = signal_strength / noise_level
    
    # 可靠性：指定比例的试次信噪比 > 阈值
    reliability_ratio = np.mean(snr > snr_threshold, axis=0)
    reliable_neurons = np.where(reliability_ratio >= reliability_threshold)[0].tolist()
    
    # 3. 最终RR神经元
    rr_neurons = list(set(enhanced_neurons) & set(reliable_neurons))
    
    elapsed_time = time.time() - start_time
    print(f"快速RR筛选完成，耗时: {elapsed_time:.2f}秒")
    
    return rr_neurons
#  ========== 数据分割函数 ========== 
def segment_neuron_data(neuron_data, trigger_data, label, pre_frames=cfg.exp_info["t_stimulus"], post_frames=cfg.exp_info["l_trials"]-cfg.exp_info["t_stimulus"]):
    """
    改进的数据分割函数
    
    参数:
    pre_frames: 刺激前的帧数（用于基线）
    post_frames: 刺激后的帧数（用于反应）
    baseline_correct: 是否进行基线校正 (ΔF/F)
    """
    total_frames = pre_frames + post_frames
    segments = np.zeros((len(trigger_data), neuron_data.shape[1], total_frames))
    labels = []

    for i in range(len(trigger_data)): # 遍历每个触发事件
        start = trigger_data[i] - pre_frames
        end = trigger_data[i] + post_frames
        # 边界检查
        if start < 0 or end >= neuron_data.shape[0]:
            print(f"警告: 第{i}个刺激的时间窗口超出边界，跳过")
            continue
        segment = neuron_data[start:end, :]
        segments[i] = segment.T
        labels.append(label[i])
    labels = np.array(labels)
    return segments, labels

# %% 实际功能函数
# ========== 加载数据 ==============================
def load_data(data_path = cfg.data_path, start_idx=cfg.trial_info["TRIAL_START_SKIP"], end_idx=cfg.trial_info["TRIAL_START_SKIP"] + cfg.trial_info["TOTAL_TRIALS"]):
    '''
    加载神经数据、位置数据、触发数据和刺激数据
    '''
    ######### 读取神经数据 #########
    print("开始处理数据...")
    mat_file = os.path.join(data_path, 'wholebrain_output.mat')
    if not os.path.exists(mat_file):
        raise ValueError(f"未找到神经数据文件: {mat_file}")
    try:
        data = h5py.File(mat_file, 'r')
    except Exception as e:
        raise ValueError(f"无法读取mat文件: {mat_file}，错误信息: {e}")

    # 检查关键数据集是否存在
    if 'whole_trace_ori' not in data or 'whole_center' not in data:
        raise ValueError("mat文件缺少必要的数据集（'whole_trace_ori' 或 'whole_center'）")

    # ==========神经数据================
    neuron_data = data['whole_trace_ori']
    # 转化成numpy数组
    neuron_data = np.array(neuron_data)
    print(f"原始神经数据形状: {neuron_data.shape}")
    
    # 只做基本的数据清理：移除NaN和Inf
    neuron_data = np.nan_to_num(neuron_data, nan=0.0, posinf=0.0, neginf=0.0)
    neuron_pos = data['whole_center']
    # 检查和处理neuron_pos维度
    if len(neuron_pos.shape) != 2:
        raise ValueError(f"neuron_pos 应为2D数组，实际为: {neuron_pos.shape}")
    
    # 灵活处理不同维度的neuron_pos
    if neuron_pos.shape[0] > 2:
        # 标准格式 (4, n)，提取前两维
        neuron_pos = neuron_pos[0:2, :]
    elif neuron_pos.shape[0] == 2:
        # 已经是2维，直接使用
        print(f"检测到2维neuron_pos格式: {neuron_pos.shape}")
    else:
        raise ValueError(f"不支持的neuron_pos维度: {neuron_pos.shape[0]}，期望为2、3或4维")

    trigger_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    trigger_data = process_trigger(trigger_files[0])
    
    # 刺激数据
    stimulus_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    stimulus_data = pd.read_csv(stimulus_files[0])
    # 转化成numpy数组
    stimulus_data = np.array(stimulus_data)
    
    # 保持指定试验数，去掉首尾 - 对触发数据和刺激数据同时处理
    start_edges = trigger_data['start_edge'][start_idx:end_idx]
    stimulus_data = stimulus_data[0:end_idx - start_idx, :]
    
    return neuron_data, neuron_pos, start_edges, stimulus_data 
# ========== 预处理， 去除负值神经元 + 矫正 + 分割trial ==================
def preprocess_data(neuron_data, neuron_pos, start_edge, stimulus_data, cfg=cfg):

    # =========== 第一步 提取仅有正值的神经元==================
    # 带负值的神经元索引
    mask = np.any(neuron_data <= 0, axis=0)   # 每列是否存在 <=0
    keep_idx = np.where(~mask)[0]

    # 如果 neuron_pos 与 neuron_data 的列对齐，则同步删除对应列
    if neuron_pos.shape[1] == neuron_data.shape[1]:
        # 从数据中删除这些列
        neuron_data = neuron_data[:, keep_idx]
        neuron_pos = neuron_pos[:, keep_idx]
    else:
        raise ValueError(f"警告: neuron_pos 列数({neuron_pos.shape[1]}) 与 neuron_data 列数({neuron_data.shape[1]}) 不匹配，未修改 neuron_pos")
    
    from scipy import ndimage
    # =========== 第二步 预处理 ===========================
    if cfg.preprocess_cfg["preprocess"]:
        win_size = cfg.preprocess_cfg["win_size"]
        if win_size % 2 == 0:
            win_size += 1
        T, N = neuron_data.shape
        F0_dynamic = np.zeros((T, N), dtype=float)
        for i in range(N):
            # ndimage.percentile_filter 输出每帧的窗口百分位值
            F0_dynamic[:, i] = ndimage.percentile_filter(neuron_data[:, i], percentile=8, size=win_size, mode='reflect')
        # 通常取每个神经元动态基线的中位数或逐帧使用（此处返回按神经元取中位数的 F0）
        F0 = np.median(F0_dynamic, axis=0)
    # 计算 dF/F（逐帧）
    # dff = (neuron_data - F0[np.newaxis, :]) / F0[np.newaxis, :]
    dff = (neuron_data - F0_dynamic) / F0_dynamic
    # =========== 可视化：随机挑选神经元对比原始信号和dF/F ==================
    # n_samples = min(4, N)  # 最多展示6个神经元
    # sample_indices = np.random.choice(N, size=n_samples, replace=False)
    
    # fig, axes = plt.subplots(n_samples, 2, figsize=(30, 3*n_samples))
    # if n_samples == 1:
    #     axes = axes.reshape(1, -1)
    
    # time_axis = np.arange(T)
    
    # for i, neuron_idx in enumerate(sample_indices):
    #     # 左侧：原始信号
    #     axes[i, 0].plot(time_axis, neuron_data[:, neuron_idx], 'b-', linewidth=0.8)
    #     axes[i, 0].set_ylabel('Raw Fluorescence', fontsize=10)
    #     axes[i, 0].set_title(f'Neuron {neuron_idx} - Original Signal', fontsize=11)
    #     axes[i, 0].grid(True, alpha=0.3)
        
    #     # 右侧：dF/F信号
    #     axes[i, 1].plot(time_axis, dff[:, neuron_idx], 'r-', linewidth=0.8)
    #     axes[i, 1].set_ylabel('dF/F', fontsize=10)
    #     axes[i, 1].set_title(f'Neuron {neuron_idx} - dF/F Signal', fontsize=11)
    #     axes[i, 1].grid(True, alpha=0.3)
    #     axes[i, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
    #     # 只在最后一行显示x轴标签
    #     if i == n_samples - 1:
    #         axes[i, 0].set_xlabel('Time (frames)', fontsize=10)
    #         axes[i, 1].set_xlabel('Time (frames)', fontsize=10)
    #     else:
    #         axes[i, 0].set_xticklabels([])
    #         axes[i, 1].set_xticklabels([])
    
    # plt.tight_layout()# %%

    # =========== 第三步 分割神经数据 =====================================
    labels = reclassify(stimulus_data)
    segments, labels = segment_neuron_data(dff, start_edge, labels)
    
    # ============= 第四步 筛选rr神经元
    rr_neurons = rr_selection(segments, np.array(labels))
    segments = segments[:, rr_neurons, :]
    neuron_pos = neuron_pos[:, rr_neurons]

    return segments, labels, neuron_pos

# %% 特殊函数（和刺激类型等相关）
def reclassify(stimulus_data):
    '''
    刺激重新分类函数
    '''
    # 示例：假设刺激数据的第一列是原始类别
    new_labels = []
    for stim in stimulus_data:
        if stim[0] == 1 and stim[1] <= 0.2:
            new_labels.append(1)  # 类别1
        elif stim[0] == 2 and stim[1] <= 0.2:
            new_labels.append(2)  # 类别2
        elif stim[1] == 1.0:
            new_labels.append(3)  # 类别3
        else:
            new_labels.append(0)  # 其他类别
    return np.array(new_labels)
# ===
# === 计算相邻元素的差值 ===
def adjacent_differences(lst):
    # 处理空列表或只有一个元素的情况
    if len(lst) < 2:
        return []
    
    differences = []
    # 遍历列表，计算相邻元素的差值
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i-1]
        differences.append(diff)
    
    return differences
# %% 可视化相关函数定义
def _rr_distribution_plot(neuron_pos, neuron_pos_rr, cfg=cfg):
    """RR neuron distribution plot"""
    from tifffile import imread

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    brain_img = imread(cfg.data_path + "/whole_brain_3d.tif")
    mid_slice = brain_img[brain_img.shape[0] // 2, :, :].astype(float)
    mid_slice = mid_slice / np.nanmax(mid_slice)
    ax.imshow(mid_slice, cmap="Greys", alpha=0.35)

    sns.scatterplot(
        x=neuron_pos[1, :],
        y=neuron_pos[0, :],
        s=18,
        color="#9fb3c8",
        alpha=0.35,
        edgecolor="none",
        ax=ax,
        label="All neurons",
    )
    sns.scatterplot(
        x=neuron_pos_rr[1, :],
        y=neuron_pos_rr[0, :],
        s=32,
        color="#F67280",
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
        label="RR neurons",
    )

    ax.set_title('RR neuron spatial distribution', fontsize=13)
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_aspect('equal')
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    return True
# =================可视化RR神经元响应=====================
def _plot_rr_responses(segments, labels, n=20, cfg=cfg):
    """RR neuron response plot"""
    n_samples = min(n, segments.shape[1])
    if n_samples == 0:
        return False
    sample_indices = np.random.choice(segments.shape[1], size=n_samples, replace=False)
    time_axis = np.arange(segments.shape[2])
    class_ids = sorted(np.unique(labels))
    palette = sns.color_palette('tab10', n_colors=len(class_ids))
    color_map = {cls: palette[i] for i, cls in enumerate(class_ids)}

    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.6 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, neuron_idx in zip(axes, sample_indices):
        for cls in class_ids:
            traces = segments[labels == cls, neuron_idx, :]
            if traces.size == 0:
                continue
            mean_trace = np.mean(traces, axis=0)
            sem_trace = stats.sem(traces, axis=0, nan_policy='omit')
            ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=color_map[cls], alpha=0.18)
            ax.plot(time_axis, mean_trace, color=color_map[cls], linewidth=1.6, label=f'Class {int(cls)}')
        ax.axvline(x=cfg.exp_info["t_stimulus"], color="#aa3a3a", linestyle="--", linewidth=1.0)
        ax.set_title(f'Neuron {neuron_idx}', fontsize=10)
        ax.set_ylim(-0.3, 1.3)

    for ax in axes[len(sample_indices):]:
        ax.axis('off')

    handles, labels_legend = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_legend, frameon=False, loc='upper center', ncol=len(handles))
    for ax in axes[:len(sample_indices)]:
        sns.despine(ax=ax)
        ax.tick_params(labelsize=8)

    fig.text(0.5, 0.02, 'Time (frames)', ha='center', fontsize=11)
    fig.text(0.02, 0.5, 'dF/F', va='center', rotation='vertical', fontsize=11)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    plt.show()
    return True
# %% =============  主程序逻辑 =============================
if __name__ == "__main__":
    print("开始运行主程序")
    # %% 数据处理
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data()

    segments, labels, neuron_pos_rr = preprocess_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    # %% 可视化RR神经元分布
    _rr_distribution_plot(neuron_pos, neuron_pos_rr)
    # %% 可视化RR神经元响应
    _plot_rr_responses(segments, labels, n=20)


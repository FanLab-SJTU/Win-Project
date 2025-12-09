import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ssm
import pandas as pd
import seaborn as sns

state_names = [
    'win_center',   # Win decision → tube
    'win_dm',       # Win tube → interaction
    'win_tube',     # Win interaction → tube_end
    'win_return',   # Win tube_end → trial_end
    'go_center',    # Win push
    'go_dm',        # Go decision → tube
    'go_tube',      # Go tube
    'go_return',    # Go tube_end → trial_end
    'empty'         # default
]
palette = sns.xkcd_palette([
    "yellow",     # win_cen
    "orange",     # win_dm
    "red",        # win_tube
    "pink",       # win_return
    "cyan",       # go_center
    "green",      # go_dm
    "blue",       # go_tube
    "sky blue",   # go_return
    "grey"        # DM
])
state_colors = dict(zip(state_names, palette))

## input behavioral data
save_dir= "/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample results/"
tag_top = pd.read_csv('/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/DEMO win day topview.csv')
WRG_posterior_probabilities = "/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/DEMO_state_probability.csv"

# ----------- General time axis ----------- # *** key region
split_indices = tag_top[tag_top['Behavior'] == 't'].index
split_times = tag_top.loc[split_indices, 'Start (s)'].values
time_step = 1 / 10
time_start = 0
time_end = max(split_times)
time_axis = np.arange(time_start, time_end, time_step)

# Behavior annotation
win_trials = []
go_trials = []
for i in range(len(split_indices) - 1):
    start_time = split_times[i]
    end_time = split_times[i + 1]
    trial_behaviors = tag_top[
        (tag_top['Start (s)'] > start_time) & 
        (tag_top['Start (s)'] < end_time)
    ]
    if len(trial_behaviors[trial_behaviors['Behavior'] == 'w']) == 1:
        win_trials.append((start_time, end_time))
    elif len(trial_behaviors[trial_behaviors['Behavior'] == 'g']) == 1:
        go_trials.append((start_time, end_time))

win_trials_df = pd.DataFrame(win_trials, columns=['Start Time (s)', 'End Time (s)'])
go_trials_df = pd.DataFrame(go_trials, columns=['Start Time (s)', 'End Time (s)'])
ws_times = win_trials_df['Start Time (s)'].values   # win start
wt_times = tag_top.loc[tag_top['Behavior'] == 'i', 'Start (s)'].values   # win turn
wi_times = tag_top.loc[tag_top['Behavior'] == 'w', 'Start (s)'].values   # win in
wo_times = tag_top.loc[tag_top['Behavior'] == 'c', 'Start (s)'].values   # win out
we_times = win_trials_df['End Time (s)'].values   # win end

gs_times = go_trials_df['Start Time (s)'].values   # go start
gt_times = tag_top.loc[tag_top['Behavior'] == 'q', 'Start (s)'].values   # go turn
gi_times = tag_top.loc[tag_top['Behavior'] == 'g', 'Start (s)'].values   # go in
go_times = tag_top.loc[tag_top['Behavior'] == 'v', 'Start (s)'].values   # go out
ge_times = go_trials_df['End Time (s)'].values   # go start

# win fragments
win_centers = []    # win center: win start -> win turn
for ws in ws_times:
    wt_after_ws = wt_times[wt_times > ws]
    if len(wt_after_ws) > 0:
        win_centers.append((ws, wt_after_ws[0]))
win_dms = []    # win dm: win turn -> win in
for wt in wt_times:
    wi_after_wt = wi_times[wi_times > wt]
    if len(wi_after_wt) > 0:
        win_dms.append((wt, wi_after_wt[0]))
win_tubes = []      # win tube: win in -> win out
for wi in wi_times:
    wo_after_wi = wo_times[wo_times > wi]
    if len(wo_after_wi) > 0:
        win_tubes.append((wi, wo_after_wi[0]))
win_returns = []      # win return: win out -> win end
for wo in wo_times:
    we_after_wo = we_times[we_times > wo]
    if len(we_after_wo) > 0:
        win_returns.append((wo, we_after_wo[0]))
        
# go fragments
go_centers = []    # go center: go start -> go turn
for gs in gs_times:
    gt_after_gs = gt_times[gt_times > gs]
    if len(gt_after_gs) > 0:
        go_centers.append((gs, gt_after_gs[0]))
go_dms = []    # go dm: go turn -> go in
for gt in gt_times:
    gi_after_gt = gi_times[gi_times > gt]
    if len(gi_after_gt) > 0:
        go_dms.append((gt, gi_after_gt[0]))
go_tubes = []      # go tube: go in -> go out
for gi in gi_times:
    go_after_gi = go_times[go_times > gi]
    if len(go_after_gi) > 0:
        go_tubes.append((gi, go_after_gi[0]))
go_returns = []      # go return: go out -> go end
for go in go_times:
    ge_after_go = ge_times[ge_times > go]
    if len(ge_after_go) > 0:
        go_returns.append((go, ge_after_go[0]))
        
# Construct behavior matrix
behavior_matrix = {
    'Win Trial': np.zeros_like(time_axis),
    'Win Center': np.zeros_like(time_axis),
    'Win DM': np.zeros_like(time_axis),
    'Win Tube': np.zeros_like(time_axis),
    'Win Return': np.zeros_like(time_axis),
    'Go Trial': np.zeros_like(time_axis),
    'Go Center': np.zeros_like(time_axis),
    'Go DM': np.zeros_like(time_axis),
    'Go Tube': np.zeros_like(time_axis),
    'Go Return': np.zeros_like(time_axis)
}
for start, end in win_trials:
    behavior_matrix['Win Trial'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in win_centers:
    behavior_matrix['Win Center'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in win_dms:
    behavior_matrix['Win DM'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in win_tubes:
    behavior_matrix['Win Tube'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in win_returns:
    behavior_matrix['Win Return'][(time_axis >= start) & (time_axis < end)] = 1

for start, end in go_trials:
    behavior_matrix['Go Trial'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in go_centers:
    behavior_matrix['Go Center'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in go_dms:
    behavior_matrix['Go DM'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in go_tubes:
    behavior_matrix['Go Tube'][(time_axis >= start) & (time_axis < end)] = 1
for start, end in go_returns:
    behavior_matrix['Go Return'][(time_axis >= start) & (time_axis < end)] = 1

behaviors = ['Go Return', 'Go Tube', 'Go DM', 'Go Center', 'Go Trial', 
             'Win Return', 'Win Tube', 'Win DM', 'Win Center', 'Win Trial']
beha_colors = ['purple', 'blue', 'green', 'cyan', 'blue', 
               'pink', 'red', 'orange', 'yellow', 'red']
alphas = [0.8,0.8,0.8,0.8,0.3,
          0.8,0.8,0.8,0.8,0.3]

from rSLDS_plot import visualize_behavior_timeline
fig, ax = visualize_behavior_timeline(
    time_axis,
    behavior_matrix,
    behaviors,
    beha_colors,
    alphas,
    save_dir
)

# state ground truth
code_win_cen    = 0  # Win trial: start to turn
code_win_dm     = 1  # Win trial: turn to in
code_win_tube   = 2  # Win trial: in to out
code_win_return = 3  # Win trial: out to end
code_go_cen     = 4  # Go trial: start to turn
code_go_dm      = 5  # Go trial: turn to in
code_go_tube    = 6  # Go trial: in to out
code_go_return  = 7  # Go trial: out to end
code_empty      = np.NAN  # NAN

z_gt = np.full_like(time_axis, code_empty)
# win codes
for win_center_start, win_center_end in win_centers:
    z_gt[(time_axis >= win_center_start) & (time_axis < win_center_end)] = code_win_cen
for win_dm_start, win_dm_end in win_dms:
    z_gt[(time_axis >= win_dm_start) & (time_axis < win_dm_end)] = code_win_dm
for win_tube_start, win_tube_end in win_tubes:
    z_gt[(time_axis >= win_tube_start) & (time_axis < win_tube_end)] = code_win_tube
for win_return_start, win_return_end in win_returns:
    z_gt[(time_axis >= win_return_start) & (time_axis < win_return_end)] = code_win_return    
# go codes
for go_center_start, go_center_end in go_centers:
    z_gt[(time_axis >= go_center_start) & (time_axis < go_center_end)] = code_go_cen
for go_dm_start, go_dm_end in go_dms:
    z_gt[(time_axis >= go_dm_start) & (time_axis < go_dm_end)] = code_go_dm
for go_tube_start, go_tube_end in go_tubes:
    z_gt[(time_axis >= go_tube_start) & (time_axis < go_tube_end)] = code_go_tube
for go_return_start, go_return_end in go_returns:
    z_gt[(time_axis >= go_return_start) & (time_axis < go_return_end)] = code_go_return 

fig, ax = plt.subplots(figsize=(18, 2))
ax.step(time_axis, z_gt, where='post', color='black')
ax.set_yticks(range(len(state_names)))
ax.set_yticklabels(state_names, fontsize=10)
ax.set_xlabel('Time (s)')
ax.set_title('Fine-Grained Ground Truth States')
ax.set_ylim(-0.5, len(state_names)-0.5)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
GT_save_path = f"{save_dir}/Raw behavior ground truth.pdf"
plt.savefig(GT_save_path, format="pdf")

# simple state ground truth 这个方案只区分 win trial 0 和 go trial 1
code_simple_win = 0
code_simple_go  = 1
palette_simple = sns.xkcd_palette([
    "red",      # win_state
    "blue"      # go_state
])
z_gt_simple = np.full_like(time_axis, code_simple_go)
for wt_start, wt_end in win_trials:
    mask = (time_axis >= wt_start) & (time_axis < wt_end)
    z_gt_simple[mask] = code_simple_win
for gt_start, gt_end in go_trials:
    mask = (time_axis >= gt_start) & (time_axis < gt_end)
    z_gt_simple[mask] = code_simple_go

#%% Neural Activity
from rSLDS_plot import visualize_channel_poisson
from rSLDS_data_processing import infer_units_names

Timedelay_top = 37.188625 # in vivo - top view time delay
FS_SPIKE = 40000.0
channel_files = [
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel1 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel2 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel2 Unit2.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel5 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel5 Unit2.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel6 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel6 Unit2.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel6 Unit3.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel8 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel9 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel10 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel10 Unit2.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel11 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel13 Unit1.mat",
"/Users/shuimuqinghua/Desktop/rSLDS/rSLDS_project Github/Sample data/240718 0425-2 win-2 Channel16 Unit1.mat"
]

units_names = infer_units_names(channel_files)
print("Inferred units_names:")
for u in units_names:
    print("  ", u)

num_channels = len(channel_files)
bin_edges = np.concatenate([time_axis, [time_axis[-1] + time_step]])  # 右闭边界
Poisson_activity = np.zeros((num_channels, len(time_axis)), dtype=int)

for i, fpath in enumerate(channel_files):
    unit_key = units_names[i]
    print(f"[{i+1}/{num_channels}] Loading {unit_key} ...")
    mdict = loadmat(fpath)
    if unit_key not in mdict:
        # 友好报错：展示可用的 key
        keys_preview = [k for k in mdict.keys() if not k.startswith("__")]
        raise KeyError(f"在 {os.path.basename(fpath)} 中找不到键 '{unit_key}'。可用键：{keys_preview}")
    unit_vec = mdict[unit_key].flatten()
    # 时间轴校正：把样本点转秒，并加上与 topview 的对齐延迟
    spike_times = unit_vec / FS_SPIKE + Timedelay_top
    # 裁剪到分析窗口
    mask = (spike_times >= time_start) & (spike_times < time_end)
    spike_times = spike_times[mask]
    # 直方计数 -> 每个 time bin 的发放数
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    Poisson_activity[i, :] = counts
print("Activity counts shape:", Poisson_activity.shape)
fig, axes = visualize_channel_poisson(Poisson_activity, time_axis, save_dir)

#%% Desample
from rSLDS_data_processing import merge_intervals
from rSLDS_data_processing import subtract_intervals
from rSLDS_data_processing import complement_intervals
from rSLDS_data_processing import extract_segments_with_indices
from rSLDS_plot import plot_segments_single_row

win_durations = [end - start for start, end in win_tubes]
go_durations  = [end - start for start, end in go_tubes]
tube_durations = win_durations + go_durations
mean_win_duration = np.mean(win_durations) if len(win_durations) > 0 else 0.0
mean_go_duration  = np.mean(go_durations)  if len(go_durations)  > 0 else 0.0
mean_tube_duration = np.mean(tube_durations)  if len(tube_durations)  > 0 else 0.0
print(f"mean_win_duration = {mean_win_duration:.3f} s,  mean_go_duration = {mean_go_duration:.3f} s, mean_tube_durations = {mean_tube_duration:.3f}")

DS_RATE   = 1.0    # 剩余区间降采样频率（秒）
DT        = time_axis[1] - time_axis[0]
manual_remove_intervals = [ # 手动指定那些要完全删去的区间：
    (0, split_times[2]), # !!!! baseline [0] 默认不加入前两个 trial 也就是第 3 个 t 之前的时间均删除 / force trial delete
]
        
interest_ext = []
for s, e in win_tubes:
    s2 = max(time_axis[0], s - mean_tube_duration / 2)
    e2 = min(time_axis[-1], e + mean_tube_duration / 2)
    interest_ext.append((s2, e2))
for s, e in go_tubes:
    s2 = max(time_axis[0], s - mean_tube_duration / 2)   # 统一保留 tube 长度 /2
    e2 = min(time_axis[-1], e + mean_tube_duration / 2)
    interest_ext.append((s2, e2))
interest_ints = merge_intervals(interest_ext)
interest_ints = subtract_intervals(interest_ints, manual_remove_intervals)  # manual_remove

# 构建 Baseline（降采样）区间 = 全域补集 - manual_remove
full_ints     = [(time_axis[0], time_axis[-1])]
baseline_ints = complement_intervals(interest_ints, time_axis[0], time_axis[-1]) # 全域关于兴趣的补集
baseline_ints = subtract_intervals(baseline_ints, manual_remove_intervals)       # 全域关于兴趣的补集 - manual_remove

# 采样：Interest 保原分辨率；Not interest 按 DS_RATE 降采样
interest_idx = np.concatenate([ # Interest 部分：直接取所有时间点的索引
    np.arange(
        np.searchsorted(time_axis, s),
        np.searchsorted(time_axis, e)
    )
    for s,e in interest_ints
])
interest_idx = np.unique(interest_idx)  # 保证有序且不重复

ds_bin = int(np.round(DS_RATE / DT))    # Not interest 部分：按窗口平均再取整
baseline_idx = []
for s,e in baseline_ints:
    i0 = np.searchsorted(time_axis, s)
    i1 = np.searchsorted(time_axis, e)
    for start in range(i0, i1, ds_bin):
        end = min(start + ds_bin, i1)
        # 用窗口内平均再取整
        # 对神经活动，各通道平均再 round
        # 但这里先记录中心点索引，后面再做真正的重采样
        center = (start + end) // 2
        baseline_idx.append(center)
        
baseline_idx = np.array(sorted(set(baseline_idx)))
all_idx = np.concatenate([interest_idx, baseline_idx])
all_idx = np.unique(all_idx)
time_axis_ds = time_axis[all_idx]   # 构建重新采样后的数据

# Poisson Activity: interest copy, not interest +-5idx mean->round
Desampled_Poisson_Activity = np.zeros((Poisson_activity.shape[0], len(all_idx)))
for i, idx in enumerate(all_idx):
    # 判断 idx 是否在 baseline_idx，若是 baseline 再重算
    if idx in baseline_idx:
        # 找对应窗口
        # 找最近的 baseline_ints，让 idx 落在该窗口内
        for s,e in baseline_ints:
            i0 = np.searchsorted(time_axis, s)
            i1 = np.searchsorted(time_axis, e)
            if i0 <= idx < i1:
                # 对应窗口段
                start = max(i0, idx - ds_bin//2)
                end   = min(i1, idx + ds_bin//2)
                # 平均再 round
                Desampled_Poisson_Activity[:, i] = np.round(Poisson_activity[:, start:end].mean(axis=1))
                break
    else:
        # interest 段保原值
        Desampled_Poisson_Activity[:, i] = Poisson_activity[:, idx]
Desampled_Poisson_Activity = Desampled_Poisson_Activity.astype(int)

# Behavior ground truth: all_idx grab
Desampled_State = z_gt[all_idx].astype(int)
Desampled_State_simple = z_gt_simple[all_idx].astype(int)

z8 = np.asarray(Desampled_State).copy()      # 1D
label_map = {0:"win_center", 1:"win_dm", 2:"win_tube", 3:"win_return", 4:"go_center", 5:"go_dm", 6:"go_tube", 7:"go_return"}
label_colors = {0:"yellow", 1:"orange", 2:"red", 3:"pink", 4:"cyan", 5:"green", 6:"blue", 7:"purple"}

# plot desampled behavior
behavior_segments = extract_segments_with_indices(z8, label_map=label_map)
fig, ax = plot_segments_single_row(
    behavior_segments, label_colors, save_dir)
plt.show()

#%% Fit an rSLDS with its default initialization, using Laplace-EM with a structured variational posterior
# rSLDS parameters
D_obs = num_channels        # 可观测维度
D_latent = 4                # 连续隐变量维度
K = 3                       # 离散隐变量数量
y = Desampled_Poisson_Activity.T
z = Desampled_State


# rSLDS perematers: model structure
rslds = ssm.SLDS(
    D_obs, 
    K, 
    D_latent,
    # transitions="recurrent",
    transitions="recurrent_only",   # 连续潜变量在空间中划分决策边界，轨迹走到不同分区就自动切换离散模式, 不存在外界输入
    # dynamics="gaussian",          # 当你怀疑各潜在分量之间噪声本身也有相关性（非独立）时 能够刻画维度之间任意的线性耦合和噪声相关性 最复杂 数据量要大
    dynamics="diagonal_gaussian",   # 可捕捉不同潜在维度之间的线性耦合 但噪声协方差被强制为对角阵 表示不同维度的随机扰动彼此独立 复杂度适中
    # emissions="gaussian_orthog",
    # emissions="poisson_orthog",            # neural activity as poissonal count distribution 显然从数据结构看 poisson 是最优的 ***key point
    emissions="poisson",
    single_subspace=True            # 所有状态的隐变量共享一个子空间
)
# rSLDS initiation
activity_pca = PCA(D_latent).fit(y)
x_init = activity_pca.transform(y)
tags = [ {"z_init": z} ]    # z ground truth as initial value
tags[0]["x_init"] = x_init  # set activity PC as initial value
rslds.initialize(y, tags = tags)
rslds_bbvi = rslds_lem = rslds
print("Model initialized.")

# Fit an rSLDS using Laplace-EM with a structured variational posterior
q_elbos_lem, q_lem = rslds_lem.fit(
    y,
    # masks=[mask2],                                # 根据行为重要性和持续时间加权
    method="laplace_em",                            # E-步（Laplace 近似）关于连续潜变量 x_{1:T} 部分，进行二阶泰勒展开，以高斯分布来近似条件后验
    variational_posterior="structured_meanfield",   # 同时拥有平滑连续轨迹和连贯离散状态 
    initialize=False,
    num_iters=40,                                   # 观察 ELBO 收敛情况决定
    alpha=0.05                                      # 默认 α=0.0 代表每次 M-步完全替换参数 可尝试动量式部分更新 0.05-0.2
)
print("Model fitted with Laplace-EM.")

xhat_lem = q_lem.mean_continuous_states[0]  # shape (T, D_latent)
# rslds.permute(find_permutation(z, rslds.most_likely_states(xhat_lem, y)))
zhat_lem = rslds_lem.most_likely_states(xhat_lem, y)    # 推断离散潜变量
scaler = StandardScaler(with_mean=True, with_std=True)
xhat_zscore = scaler.fit_transform(xhat_lem)     # shape (T, D_latent)
pca_full = PCA(n_components=D_latent)  # 可取更高维便于看解释方差
xhat_pca_full = pca_full.fit_transform(xhat_zscore)        # shape (T, ncomp)
xhat_pca2 = xhat_pca_full[:, :2]                      # 用前2维做轨迹分析

os.makedirs(save_dir, exist_ok=True)
df_latent_pca2 = pd.DataFrame(xhat_pca2, columns=["pc1", "pc2"])
df_latent_pca2.insert(0, "time_idx", np.arange(len(xhat_pca2)))
df_latent_pca2.to_csv(os.path.join(save_dir, "latent_pca2.csv"), index=False)

exp_var = pca_full.explained_variance_ratio_
df_pca_exp = pd.DataFrame({
    "pc": [f"PC{i+1}" for i in range(len(exp_var))],
    "explained_variance_ratio": exp_var,
    "cumulative": exp_var.cumsum()
})
df_pca_exp.to_csv(os.path.join(save_dir, "latent_pca_explained_variance.csv"), index=False)

print("[Info] PCA done:",
      f"top-2 PCs explain {exp_var[:2].sum():.3f} variance (z-scored space).")

# ELBO 收敛曲线 ELBO 是拟合过程中用来度量后验近似和模型证据下界的目标函数 看相同条件下相对是否更大
plt.figure()
# plt.plot(q_elbos_bbvi, label="BBVI")
plt.plot(q_elbos_lem[1:], label="Laplace-EM")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("ELBO")
ELBO_path = os.path.join(save_dir, "ELBO")
plt.savefig(ELBO_path)

# 6 type behavior simplify: 0 win_dm, 1 win_tube, 3 win_return, 5 go_dm, 6 go_tube, 7 go_return
from rSLDS_plot import plot_desampled_trajectories_2d
z8 = np.asarray(Desampled_State).copy()      # 1D
fig, ax = plot_desampled_trajectories_2d(
    xhat_pca2, behavior_segments, label_colors, save_dir)
plt.show()
from rSLDS_data_processing import save_behavior_and_latents
save_behavior_and_latents(behavior_segments, xhat_lem, save_dir)

#%% 1D field flow visualization & inferred state
from rSLDS_plot import plot_latents_and_states_3row
plot_latents_and_states_3row(
    time_axis=time_axis_ds,
    x_latent=xhat_pca2,            # (T,2)
    z_labels=z8,                   # 0..7 的行为标签
    zhat=zhat_lem,                 # 模型推断的离散隐状态（0..K-1）
    state_names=["state A","state B","state C"],
    state_colors=["#e74c3c","#3498db","#2ecc71"],
    label_map=label_map,           # {int: "win_dm", ...}
    label_colors=label_colors,     # {int: "#color", ...}
    save_dir=save_dir,
    filename="latents_pc12_and_states_raster.pdf",
    smooth_win=4,
    alpha=0.18
)

#%% mean behavior trace
from rSLDS_plot import plot_all_behavior_mean_paths
results_all = plot_all_behavior_mean_paths(
    xhat=xhat_pca2,
    seg_df=behavior_segments,     # 或 z6 版的 df
    label_map=label_map,
    label_colors=label_colors,       # 支持 int 或 name 做 key
    behaviors=list(range(8)),        # 0..7 八类
    use_label="label",               # 按 id 取段
    n_points=256,
    save_dir=save_dir,
    filename="all_behaviors_mean_paths.pdf",   # ← 你自定义的 pdf 名称
    store_data=True,
    data_prefix="all_behaviors_mean_paths"     # 数据文件前缀
)

#%% 2D square size
from Local_Field_Flow_helper import plot_trial_loop_areas
from Local_Field_Flow_helper import plot_trial_area_comparison
from Local_Field_Flow_helper import analyze_looparea_posterior_correlation

df_areas = plot_trial_loop_areas(
    xhat=xhat_pca2,
    seg_df=behavior_segments,     # 或 z6 版 DF；需包含 win/go 的 dm/tube/return 段
    save_dir=save_dir,
    csv_name="trial_loop_areas.csv",
    use_label="label_name"           # 如果只有数字列，先在 DF 中加一列 label_name
)

df_areas_norm = plot_trial_area_comparison(
    df_areas=df_areas,
    save_dir=save_dir,
    filename_pdf="trial_area_comparison_norm.pdf",
    csv_name="trial_loop_areas_with_norm.csv"
)

df_combined, df_corr = analyze_looparea_posterior_correlation(
    df_areas=df_areas,
    posterior_csv=WRG_posterior_probabilities,
    save_dir=save_dir,
    filename_scatter="looparea_vs_posterior_scatter.pdf",
    filename_corr="looparea_posterior_corr.pdf",
    csv_name="looparea_with_posterior.csv"
)
    
#%% Ring dynamic analysis
from Local_Field_Flow_helper import analyze_trials_ring_dynamics
# 如果有时间轴（和 xhat 对齐），可传入 time_axis_ds；否则不传，默认 dt=1
RD_results = analyze_trials_ring_dynamics(
    xhat=xhat_pca2,
    seg_df=behavior_segments,             # 需包含 label_name/start_idx/end_idx
    save_dir=save_dir,
    filename_panel="win_go_ring_panel.pdf",  # 2×2 面板：原始+均值 & 向量场+均值
    filename_bars="win_go_dynamics_bars.pdf",
    n_points=1024,
    time_axis=None                           # 或 time_axis_ds
)

#%% rSLDS trajectory quantification by DTW (Directed curve distance)
# DTW calculate neural activity sequence distance
from rSLDS_DTWhelper import extract_trajectory_behavior
all_trajs, all_labels, all_names, all_starts = extract_trajectory_behavior(xhat_lem, behavior_segments) # extract 2D traj
N = len(all_trajs)
print(f"#segments = {N}")

from rSLDS_DTWhelper import pairwise_distance_matrix
metric = "dtw"
sc_band_ratio = 0.10
Dmat = pairwise_distance_matrix(all_trajs, sc_band_ratio=sc_band_ratio)
print("Distance matrix:", Dmat.shape)

### sort by behavior type
label_order = [0, 1, 2, 3, 4, 5, 6, 7]  # win_dm, win_tube, win_return, go_dm, go_tube, go_return
order_key  = {lab: i for i, lab in enumerate(label_order)}

# 优先行为类型 其次发生次序
label_rank = np.array([order_key.get(int(l), 999) for l in all_labels], dtype=int)
starts_1d  = np.asarray(all_starts, dtype=float)

# 使用 lexsort：keys 的“最后一个”为主键 → 传入顺序 (次键, 主键)
perm = np.lexsort((starts_1d, label_rank)).astype(int)
assert perm.ndim == 1 and perm.shape[0] == len(all_labels), f"perm must be 1D of length N; got {perm.shape}"

# 基于 perm 重排矩阵与信息
D_sorted      = Dmat[np.ix_(perm, perm)]
labels_sorted = all_labels[perm]
names_sorted  = [all_names[i] for i in perm]
starts_sorted = starts_1d[perm]
print("perm shape:", perm.shape, perm.dtype)
print("D_sorted shape:", D_sorted.shape)

from rSLDS_DTWhelper import groupwise_mean_distances
mean_d = groupwise_mean_distances(D_sorted, labels_sorted, label_order)
name = {0:"win_center", 1:"win_dm", 2:"win_tube", 3:"win_return", 4:"go_center", 5:"go_dm", 6:"go_tube", 7:"go_return"}
print("\nGroupwise mean distances (DTW avg):")
for (a, b), v in mean_d.items():
    if not np.isnan(v):
        print(f"  {name[a]} vs {name.get(b, a)}: {v:.3f}")

from rSLDS_plot import plot_distance_heatmap_scaled
fig, ax = plot_distance_heatmap_scaled(
    D_sorted, labels_sorted, label_order, save_dir,
    use_log=False,
    vmin=0
)
plt.show()

from rSLDS_data_processing import save_dtw_matrices_to_csv
paths = save_dtw_matrices_to_csv(
    D_sorted=D_sorted,
    labels_sorted=labels_sorted,
    label_order=label_order,
    name_map=label_map,
    save_dir=save_dir,
    dtw_matrix_csv="DTW_matrix_labeled.csv",
    dtw_group_csv="DTW_groupwise_mean.csv"
)
        
## hierarchy tree based on DTW distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

D = np.array(D_sorted, float)  # 或者用你未重排的 Dmat
finite = np.isfinite(D)
if not finite.all():
    # 用有限值的“最大值”顶帽替换 NaN/Inf
    max_fin = np.nanmax(D[finite])
    D[~finite] = max_fin

# linkage 需要“condensed”上三角向量
# squareform 要求对角线为 0 且矩阵对称
np.fill_diagonal(D, 0.0)
D = 0.5*(D + D.T)
condensed = squareform(D, checks=False)

# -------- 1) 层次聚类（average/complete 都可试） --------
Z = linkage(condensed, method="complete")  # "complete" 更保守，"single" 更容易链化

name_map = label_map

fig, ax = plt.subplots(figsize=(12, 5))
dendro = dendrogram(
    Z,
    labels=[name_map[int(x)] for x in labels_sorted],  # 只是显示名
    leaf_rotation=90,
    leaf_font_size=8,
    truncate_mode="level",
    above_threshold_color="#444444",
    link_color_func=lambda k: "#444444"
)

# 正确的着色：按聚类后的“叶子顺序”来上色
leaves_order = dendro["leaves"]              # 这是聚类后叶子在“输入序列中的索引”
tick_texts   = ax.get_xmajorticklabels()     # 这些 tick 的顺序与 leaves_order 一一对应

for txt, idx in zip(tick_texts, leaves_order):
    lab_val = int(labels_sorted[idx])        # 找到该叶子对应的行为标签
    txt.set_color(label_colors.get(lab_val, "#999999"))

ax.set_title("Hierarchical clustering of behavior segments")
ax.set_ylabel("Linkage distance")
plt.tight_layout()
plt.show()
HC_save_path = f"{save_dir}/DTW distance hierarchical clustering.pdf"
plt.savefig(HC_save_path, format="pdf")

# 这里演示按簇数切（比如切成 6 簇，可自行调整）：
n_clusters = 6
cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")
beh_names = [name_map[int(l)] for l in labels_sorted]
df = pd.DataFrame({"cluster": cluster_ids, "behavior": beh_names})
summary = df.groupby(["cluster", "behavior"]).size().unstack(fill_value=0)
print(summary)

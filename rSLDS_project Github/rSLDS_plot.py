import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

def visualize_behavior_timeline(time_axis,
                                behavior_matrix,
                                behaviors,
                                beha_colors,
                                alphas,
                                save_dir,
                                figsize=(18, 3),
                                title='Behavior Timeline: Multiple Events'):
    """
    time_axis : 1D array of floats
        The common time axis.
    behavior_matrix : dict
        Keys are behavior names, values are 0/1 arrays of same length as time_axis.
    behaviors : list of str
        Ordered list of behavior names to plot (one per row).
    beha_colors : list of colors
        Same length as behaviors, the fill color for each behavior.
    alphas : list of floats
        Same length as behaviors, the alpha for each behavior.
    figsize : tuple, optional
        Figure size passed to plt.subplots.
    title : str, optional
        Plot title.
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    y_centers = np.arange(len(behaviors))
    band_height = 1
    gap = 0

    for i, (behavior, color, alpha) in enumerate(zip(behaviors, beha_colors, alphas)):
        arr = behavior_matrix[behavior]
        padded = np.concatenate([[0], arr, [0]])
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends   = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(
                time_axis[s], time_axis[e-1],
                ymin=(y_centers[i] + gap/2) / len(behaviors),
                ymax=(y_centers[i] + band_height + gap/2) / len(behaviors),
                color=color, alpha=alpha, linewidth=0
            )

    # horizontal separator lines
    for i in range(len(behaviors) + 1):
        ax.axhline(i / len(behaviors), color='gray', linewidth=0.8, alpha=0.25)

    ax.set_ylim(0, 1)
    ax.set_yticks([])

    # labels on the left
    for i, label in enumerate(behaviors):
        ax.text(
            -5,
            (i + 0.5) / len(behaviors),
            label,
            va='center', ha='right',
            color=beha_colors[i],
            fontsize=10
        )

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(time_axis[0], time_axis[-1])
    plt.tight_layout()
    save_path = f"{save_dir}/Raw behavior raster plot.pdf"
    plt.savefig(save_path, format="pdf")
    return fig, ax


def visualize_ground_truth_states(time_axis, z_gt, title='Ground Truth z'):
    """
    Plot discrete state z over time with steps.
    """
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.step(time_axis, z_gt, where='post', color='black')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Win', 'Go', 'DM'])
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def visualize_channel_poisson(activity_scaled, time_axis, save_dir,
                              figsize=None):
    """
    Plot z-scored neural activity for each channel over time.

    Parameters
    ----------
    activity_scaled : array-like, shape (num_channels, num_timepoints)
        Z-scored activity per channel.
    time_axis : array-like, shape (num_timepoints,)
        Time stamps corresponding to activity samples.
    figsize : tuple, optional
        Figure size; defaults to (18, 2 * num_channels).

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, numpy.ndarray of Axes
        The figure and axes for further customization.
    """
    num_channels = activity_scaled.shape[0]
    if figsize is None:
        figsize = (18, 2 * num_channels)
    fig, axes = plt.subplots(num_channels, 1, figsize=figsize, sharex=True)
    for i in range(num_channels):
        ax = axes[i] if num_channels > 1 else axes
        ax.plot(time_axis, activity_scaled[i, :], 
                label=f'Channel {i+1}', linewidth=0.8)
        ax.set_ylabel('Poisson #')
        ax.legend(loc='upper right')
    # x-label on last subplot
    if num_channels > 1:
        axes[-1].set_xlabel('Time (s)')
    else:
        axes.set_xlabel('Time (s)')
    plt.tight_layout()
    save_path = f"{save_dir}/Poisson units.pdf"
    plt.savefig(save_path, format="pdf")
    return fig, axes


def plot_observations(z, y, ax=None, ls="-", lw=1):

    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=state_colors[z[start] % len(state_colors)],
                    alpha=1.0)
    return ax


### rSLDS trajectory plots ###
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


def plot_trajectory_ori(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=state_colors[z[start] % len(state_colors)],
                alpha=1.0)
    return ax


def plot_trajectory(z, x, ax=None):
    """
    Plot trajectory using discrete points:
    - Color by discrete state
    - Alpha increases with time
    """
    z = np.asarray(z)
    x = np.asarray(x)
    T = len(z)
    ax.axis("equal")

    # 时间归一化到 0~1，用于 alpha 显示
    alphas = np.linspace(0.1, 1.0, T)

    for t in range(T):
        # if z[t] == 0:
        #     color = "red"
        # else: color = "blue"
        ax.plot(x[t, 0], x[t, 1],
                marker='o', markersize=1,
                color=palette[z[t]],
                # color=color,
                alpha=0.6
                )

    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_title("Trajectory (discrete states & time fading)")
    return ax


def animate_trajectory(z, x, ax=None, interval=30):
    """
    Animate a 2D trajectory with discrete-state coloring and time-fading alpha.
    (Clears & re-plots each frame to avoid mismatched array lengths.)
    """
    z = np.asarray(z)
    x = np.asarray(x)
    T = len(z)
    assert x.shape[1] == 2

    # Create a fresh figure/axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    # caller can set title on ax
    
    def update(frame):
        ax.clear()
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        # fade alphas from 0.1 to 1.0
        alphas = np.linspace(0.1, 1.0, frame + 1)
        # colors per state
        cols = [palette[state] for state in z[:frame + 1]]
        # scatter all points up to `frame`
        ax.scatter(x[:frame+1, 0],
                   x[:frame+1, 1],
                   c=cols,
                   alpha=alphas,
                   s=4)
        return ()

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=interval,
        blit=False,   # must be False when clearing
    )
    return anim


def animate_trajectory_3d(z, x, interval, figsize=(6,6)):
    """
    Animate a 3D trajectory with discrete-state coloring and time-fading alpha.
    
    Parameters
    ----------
    z : array-like, shape (T,)
        Discrete state labels for each timepoint.
    x : array-like, shape (T, 3)
        Continuous latent positions in 3D.
    interval : int, optional
        Delay between frames in milliseconds.
    figsize : tuple, optional
        Figure size.
    
    Returns
    -------
    anim : FuncAnimation
        The matplotlib animation object.
    """
    z = np.asarray(z)
    x = np.asarray(x)
    T = len(z)
    assert x.shape[1] == 3, "x must be T×3 for 3D animation"
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("Latent Dim 3")
    
    # Precompute alphas
    alphas_all = np.linspace(0.1, 1.0, T)
    
    # Determine fixed axes limits
    lims = np.vstack([x.min(axis=0), x.max(axis=0)])
    for dim, (mn, mx) in enumerate(lims.T):
        getattr(ax, f"set_{['x','y','z'][dim]}lim")((mn-0.1*(mx-mn), mx+0.1*(mx-mn)))
    
    def update(frame):
        ax.clear()
        # redraw axes labels + limits
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_zlabel("Latent Dim 3")
        for dim, (mn, mx) in enumerate(lims.T):
            getattr(ax, f"set_{['x','y','z'][dim]}lim")((mn-0.1*(mx-mn), mx+0.1*(mx-mn)))
        
        # plot points up to current frame
        for t in range(frame+1):
            ax.scatter(
                x[t,0], x[t,1], x[t,2],
                s=6,
                color=state_colors[z[t]],
                alpha=alphas_all[t]
            )
        return fig,
    
    anim = FuncAnimation(
        fig, update,
        frames=T,
        interval=interval,
        blit=False
    )
    return anim


### plot dynamics ###
def plot_most_likely_dynamics(model,
    xlim=(-10, 10), ylim=(-10, 10), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    assert model.D == 4
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    # z = np.argmax(xy.dot(model.transitions.Rs.T), axis=1)
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=palette[k % len(palette)], alpha=0.2)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    return ax


def plot_most_likely_dynamics_new(model,
                              xlim=(-4, 4), ylim=(-4, 4),
                              nxpts=20, nypts=20,
                              alpha=0.8, ax=None, figsize=(3, 3)):
    """
    Visualize most likely dynamics in 2D latent space for recurrent_only transitions.
    """
    K = model.K
    assert model.D == 2, "This function only handles 2D latents"
    
    # 构建网格
    x1 = np.linspace(xlim[0], xlim[1], nxpts)
    x2 = np.linspace(ylim[0], ylim[1], nypts)
    X1, X2 = np.meshgrid(x1, x2)
    xy = np.column_stack((X1.ravel(), X2.ravel()))  # (nxpts*nypts, 2)

    # 计算每个网格点上的最可能离散状态 z
    trans = model.transitions
    # logits = xy @ Rs^T + r (若 r 不存在就省略)
    logits = xy.dot(trans.Rs.T)
    if hasattr(trans, 'r'):
        logits = logits + trans.r  # 偏置
    z_grid = np.argmax(logits, axis=1)

    # 对每个网格点，用对应状态的 A 矩阵计算 dx = A x
    dX = np.zeros_like(xy)
    for k in range(K):
        A = model.dynamics.As[k]
        # 找出网格上预测为状态 k 的点
        mask = (z_grid == k)
        if mask.any():
            dX[mask] = (xy[mask] @ A.T)

    U = dX[:,0].reshape(X1.shape)
    V = dX[:,1].reshape(X2.shape)

    # 绘制箭头场
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(X1, X2, U, V, angles="xy", alpha=alpha)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    return ax

# 降采样 6 行为可视化 单时间轴
def plot_segments_single_row(df, label_colors, save_dir):
    if df.empty:
        print("No segments to plot.")
        return None, None

    fig, ax = plt.subplots(figsize=(16, 2.6))

    # 画彩色时间带
    for _, row in df.iterrows():
        ax.axvspan(
            row["start_time"], row["end_time"],
            ymin=0.25, ymax=0.75,
            color=label_colors.get(row["label"], "#999999"),
            alpha=0.9, linewidth=0
        )

    # 轴样式
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlim(0, df["end_time"].iloc[-1])
    ax.set_xlabel("Index")
    ax.set_title("Behavior timeline (single row)")

    # 图例
    order = []
    seen = set()
    for _, r in df.iterrows():
        if r["label_name"] not in seen:
            order.append((r["label"], r["label_name"]))
            seen.add(r["label_name"])
    handles = [Patch(facecolor=label_colors.get(lbl, "#999999"), label=name) for lbl, name in order]
    ax.legend(handles=handles, loc="upper right", ncol=3, frameon=False)
    plt.tight_layout()
    save_path = f"{save_dir}/Desampled behavior segment.pdf"
    plt.savefig(save_path, format="pdf")
    return fig, ax

def plot_desampled_trajectories_2d(x_latent, df_segments, label_colors, save_dir,
                                   title="Neural trajectories by behavior (2D)",
                                   linewidth=1.2, alpha=0.7, show_start_dots=True):
    """
    在同一张 2D 图上绘制每段行为对应的轨迹（按时间顺序连线）
    兼容两种字段命名：
      - 优先使用 start_idx/end_idx
      - 若不存在则使用 start_time/end_time（在索引即时间的设定下等价）
    需要列：label, label_name
    """
    # 字段兼容：取出 s/e
    if {"start_idx","end_idx"}.issubset(df_segments.columns):
        s_col, e_col = "start_idx", "end_idx"
    elif {"start_time","end_time"}.issubset(df_segments.columns):
        s_col, e_col = "start_time", "end_time"
    else:
        raise ValueError("df_segments 缺少 start_idx/end_idx 或 start_time/end_time")

    if not {"label","label_name"}.issubset(df_segments.columns):
        raise ValueError("df_segments 缺少 label / label_name 列")

    assert x_latent.ndim == 2 and x_latent.shape[1] == 2, "x_latent 必须是 T×2"

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    legend_handles = {}

    for _, seg in df_segments.iterrows():
        s = int(seg[s_col]); e = int(seg[e_col])
        if e <= s:
            continue
        lab   = int(seg["label"])
        lname = seg["label_name"]
        xy = x_latent[s:e, :]  # (len,2)

        ax.plot(xy[:,0], xy[:,1],
                lw=linewidth, alpha=alpha,
                color=label_colors.get(lab, "#999999"))
        if show_start_dots:
            ax.plot(xy[0,0], xy[0,1], 'o', ms=3,
                    alpha=min(1.0, alpha+0.2),
                    color=label_colors.get(lab, "#999999"))
        if lname not in legend_handles:
            legend_handles[lname] = Patch(facecolor=label_colors.get(lab, "#999999"), label=lname)

    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(alpha=0.2)
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="best", frameon=False, ncol=2)
    plt.tight_layout()
    save_path = f"{save_dir}/rSLDS trajectory.pdf"
    plt.savefig(save_path, format="pdf")
    return fig, ax


# DTW distance heatmap
def plot_distance_heatmap_scaled(
    D, labels_sorted, label_order, save_dir,
    title="Pairwise distances (grouped by behavior)",
    use_log=True,                  # 是否对距离做 log10 变换
    clip_percentiles=(5, 95),      # 按分位数裁剪色阶
    vmin=None, vmax=None,          # 也可手动给定数值范围（在变换后空间）
    cmap="viridis"
):
    """
    D: 已按段重排后的距离矩阵 (N×N)
    use_log=True -> 显示 log10(D)，并按 clip_percentiles 选择色阶范围
    若 vmin/vmax 提供，则优先使用（注意：这里是“变换后”的数值范围）
    """
    D = np.asarray(D, float)
    N = D.shape[0]
    assert D.shape[1] == N

    # ---- 1) 取上三角非对角的样本做尺度估计 ----
    mask_off = ~np.eye(N, dtype=bool)
    vals = D[mask_off]
    vals = vals[np.isfinite(vals)]

    # 变换
    if use_log:
        eps = 1e-12
        T = np.log10(np.maximum(D, eps))
        tvals = np.log10(np.maximum(vals, eps))
        cblabel = "log10 directed curve distance"
    else:
        T = D.copy()
        tvals = vals.copy()
        cblabel = "directed curve distance"

    # ---- 2) 计算色阶范围 ----
    if vmin is None or vmax is None:
        lo, hi = np.percentile(tvals, clip_percentiles)
        # 保障 lo < hi
        if lo >= hi:
            lo, hi = np.min(tvals), np.max(tvals)
    else:
        lo, hi = vmin, vmax

    print(f"[Heatmap scale] use_log={use_log}, range=({lo:.3f}, {hi:.3f}) in transformed space")

    # ---- 3) 画图 ----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(T, interpolation='nearest', cmap=cmap, origin='upper', vmin=lo, vmax=hi)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cblabel)

    # ---- 4) 类别边界与刻度 ----
    pos = 0
    tick_pos, tick_lab = [], []
    name_map = {0:"win_center", 1:"win_dm", 2:"win_tube", 3:"win_return", 4:"go_center", 5:"go_dm", 6:"go_tube", 7:"go_return"}
    for lab in label_order:
        cnt = int(np.sum(labels_sorted == lab))
        if cnt > 0:
            a, b = pos, pos + cnt
            # 边框
            ax.plot([a-0.5, b-0.5], [a-0.5, a-0.5], color='w', lw=1.5)
            ax.plot([a-0.5, b-0.5], [b-0.5, b-0.5], color='w', lw=1.5)
            ax.plot([a-0.5, a-0.5], [a-0.5, b-0.5], color='w', lw=1.5)
            ax.plot([b-0.5, b-0.5], [a-0.5, b-0.5], color='w', lw=1.5)
            # 刻度
            mid = a + cnt/2 - 0.5
            tick_pos.append(mid); tick_lab.append(name_map[lab])
            pos = b

    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, rotation=45, ha='right')
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
    ax.set_title(title)
    plt.tight_layout()
    save_path = f"{save_dir}/DTW distance matrix heatmap.pdf"
    plt.savefig(save_path, format="pdf")
    return fig, ax

# ---- helper: from a label sequence (z) build contiguous segments (start_idx, end_idx, label) ----
def _contiguous_segments_from_labels(z):
    """把序列 z 中值相同的连续段分解为 [(i0, i1, label), ...]（左闭右开）"""
    z = np.asarray(z)
    segs = []
    if z.size == 0:
        return segs
    i0 = 0
    for i in range(1, len(z)):
        if z[i] != z[i-1]:
            segs.append((i0, i, int(z[i-1])))
            i0 = i
    segs.append((i0, len(z), int(z[-1])))
    return segs

def _runs_from_labels(z, target):
    """返回 z==target 的所有连续区间索引对 [(i0, i1), ...]（左闭右开）"""
    z = np.asarray(z)
    runs, i0 = [], None
    for i, val in enumerate(z):
        if val == target and i0 is None:
            i0 = i
        elif val != target and i0 is not None:
            runs.append((i0, i))
            i0 = None
    if i0 is not None:
        runs.append((i0, len(z)))
    return runs

def plot_latents_with_behavior_shading(
    time_axis, x_latent, z_labels,
    label_map, label_colors,
    smooth_win=None, alpha=0.15,
    ylim=None, title_suffix="", axes=None, figsize=(16, 6)
):
    """
    上下两行分别绘制 PC1 / PC2；背景按行为着色，时间轴一致但 y 轴不共享。
    - time_axis: (T,)
    - x_latent:  (T,2)
    - z_labels:  (T,) int in {0..}
    - label_map: {int: "name"}
    - label_colors: {int: color-string}  注意这里 key=label(int)
    """
    t = np.asarray(time_axis)
    X = np.asarray(x_latent)
    z = np.asarray(z_labels).astype(int)
    assert X.shape[0] == t.shape[0], f"Length mismatch: {X.shape[0]} vs {t.shape[0]}"
    assert X.shape[1] >= 2, "x_latent 必须至少包含 2 个维度用于 PC1/PC2 可视化"

    # 平滑仅用于可视化
    if smooth_win and smooth_win > 1:
        Xv = np.column_stack([
            uniform_filter1d(X[:, 0], size=smooth_win, mode='nearest'),
            uniform_filter1d(X[:, 1], size=smooth_win, mode='nearest')
        ])
    else:
        Xv = X[:, :2]

    segs = _contiguous_segments_from_labels(z)

    # 布局
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig = axes[0].figure
        assert len(axes) == 2, "axes 必须是长度为 2 的轴列表"

    # 逐轴绘制
    for d, ax in enumerate(axes):
        # 行为阴影
        for i0, i1, lbl in segs:
            x0, x1 = t[i0], t[i1-1] if i1-1 < len(t) else t[-1]
            ax.axvspan(x0, x1, color=label_colors.get(lbl, "#cccccc"), alpha=alpha, linewidth=0)

        # PC 曲线
        ax.plot(t, Xv[:, d], lw=1.2, color='k')
        ax.set_ylabel(f"Latent PC{d+1}")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, axis='y')

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"1D latent trajectories with behavior shading{(' - ' + title_suffix) if title_suffix else ''}")

    # 图例（只放第一行）
    present = sorted({int(lbl) for _, _, lbl in segs})
    handles = [Patch(facecolor=label_colors.get(lbl, "#cccccc"), edgecolor='none',
                     alpha=alpha, label=label_map.get(lbl, str(lbl)))
               for lbl in present]
    if handles:
        axes[0].legend(handles=handles, loc='upper left', ncol=4, fontsize=9, frameon=True)

    return fig, axes

# =========================
# 1) 工具函数：按段对齐并求平均轨迹 + 置信带
# =========================
import os
from scipy.interpolate import interp1d

def _resample_segment(seg_xy, n_points=80):
    """把一条 (n,2) 轨迹按进度 [0,1] 线性插值到固定长度 n_points。"""
    if len(seg_xy) < 2:
        return None
    t = np.linspace(0, 1, len(seg_xy))
    new_t = np.linspace(0, 1, n_points)
    f0 = interp1d(t, seg_xy[:, 0], kind="linear")
    f1 = interp1d(t, seg_xy[:, 1], kind="linear")
    return np.stack([f0(new_t), f1(new_t)], axis=1)  # (n_points,2)

def compute_behavior_mean_path(xhat, seg_df, behavior, use_label="auto",
                               n_points=80, ci_mode="sem", ci_alpha=0.95,
                               min_len=3):
    """
    计算某个行为的平均轨迹与置信带。
    Parameters
    ----------
    xhat : (T,2) np.ndarray
    seg_df : pd.DataFrame  含 start_idx, end_idx, 和 label 或 label_name
    behavior : int 或 str  行为 id 或名
    use_label : 'auto' | 'label' | 'label_name'
    n_points : int 对齐到多少点
    ci_mode : 'sem' | 'std' | 'percentile'
    ci_alpha : float CI 置信度，仅对 'percentile' 有意义（如 0.95）
    min_len : int 段最小点数
    Returns
    -------
    dict: {
        'mean': (n_points,2),
        'lower': (n_points,2), 'upper': (n_points,2),
        'all_resampled': (n_seg, n_points, 2),
        'n_segments': int
    }
    """
    # 选段
    if use_label == "auto":
        if "label" in seg_df.columns and isinstance(behavior, (int, np.integer)):
            sel = seg_df["label"] == int(behavior)
        elif "label_name" in seg_df.columns and isinstance(behavior, str):
            sel = seg_df["label_name"] == behavior
        else:
            raise ValueError("请设置 use_label='label' 或 'label_name'，并与 behavior 类型匹配。")
    elif use_label == "label":
        sel = seg_df["label"] == int(behavior)
    else:
        sel = seg_df["label_name"] == behavior

    segs = [(int(r.start_idx), int(r.end_idx)) for r in seg_df.loc[sel].itertuples(index=False)]

    resampled = []
    for (i0, i1) in segs:
        if i1 - i0 < min_len:
            continue
        seg_xy = xhat[i0:i1]
        rs = _resample_segment(seg_xy, n_points=n_points)
        if rs is not None:
            resampled.append(rs)

    if not resampled:
        return None

    arr = np.stack(resampled, axis=0)  # (n_seg, n_points, 2)
    mean_traj = arr.mean(axis=0)
    if ci_mode == "sem":
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ci = 1.96 * sem  # 近似 95% CI
        lower = mean_traj - ci
        upper = mean_traj + ci
    elif ci_mode == "std":
        sd = arr.std(axis=0, ddof=1)
        lower = mean_traj - sd
        upper = mean_traj + sd
    elif ci_mode == "percentile":
        lo = (1 - ci_alpha) / 2 * 100
        hi = (1 + ci_alpha) / 2 * 100
        lower = np.percentile(arr, lo, axis=0)
        upper = np.percentile(arr, hi, axis=0)
    else:
        raise ValueError("ci_mode 必须是 'sem' | 'std' | 'percentile'")

    return dict(mean=mean_traj, lower=lower, upper=upper,
                all_resampled=arr, n_segments=arr.shape[0])

# =========================
# 2) 主函数：8类代表性路径 + 置信带，合在一张图
#    并把图和数据保存到指定目录
# =========================    
def plot_all_behavior_mean_paths(
    xhat, seg_df,
    label_map, label_colors,
    behaviors=None, use_label="label",
    n_points=80, ci_mode="sem", ci_alpha=1,
    save_dir=None, filename="mean_paths_all.pdf",
    store_data=True, data_prefix="mean_paths",
    annotate_starts=True,
    start_label_map=None
):
    """
    在一张图上绘制全部行为的“代表性平均路径 + 置信带”，并在平均路径起点打点并标注。
    - annotate_starts: 是否标注起点
    - start_label_map: 行为名 -> 起点文字 的映射（未匹配到的行为不标注）
    """

    # 默认起点标注映射（中文）
    if start_label_map is None:
        start_label_map = {
            "win_tube":   "win in",
            "win_dm":     "win turn",
            "win_return": "win out",
            "go_tube":    "go in",
            "go_dm":      "go turn",
            "go_return":  "go out",
            # 如需也标注 return，可自行添加：
            "win_center": "win start",
            "go_center":  "go start",
        }

    # 准备行为列表
    if behaviors is None:
        behaviors = list(label_map.keys())  # 默认用 id 列表
    # 行为显示名与颜色
    beh_display, beh_colors = [], []
    for beh in behaviors:
        # 显示名：按 use_label 决定
        if use_label == "label":
            name = label_map.get(int(beh), str(beh))
        else:
            name = str(beh)
        # 颜色：既支持用 int 作 key，也支持用 name 作 key
        col = (label_colors.get(beh, None)
               if beh in label_colors
               else label_colors.get(name, None))
        if col is None:
            col = "C0"
        beh_display.append(name)
        beh_colors.append(col)

    # 计算每类平均轨迹
    results = {}
    for beh in behaviors:
        res = compute_behavior_mean_path(
            xhat=xhat,
            seg_df=seg_df,
            behavior=beh,
            use_label=("label" if use_label=="label" else "label_name"),
            n_points=n_points,
            ci_mode=ci_mode,
            ci_alpha=ci_alpha
        )
        results[beh] = res

    # 画图
    plt.figure(figsize=(8.5, 7.5))
    ax = plt.gca()

    # 统一坐标范围（包含所有原始轨迹）
    xs, ys = [], []
    for beh in behaviors:
        res = results.get(beh, None)
        if res is None:
            continue
        arr = res["all_resampled"]   # (n_seg, n_points, 2)
        xs.append(arr[:,:,0].ravel())
        ys.append(arr[:,:,1].ravel())

    if xs and ys:
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        pad = 0.8  # 给边界留一点空隙
        ax.set_xlim(xs.min()-pad, xs.max()+pad)
        ax.set_ylim(ys.min()-pad, ys.max()+pad)

    for beh, name, col in zip(behaviors, beh_display, beh_colors):
        res = results.get(beh, None)
        if res is None:
            continue
        mu = res["mean"]

        # 直接画原始轨迹（不要用 all_resampled）
        sel = seg_df["label"] == beh if use_label=="label" else seg_df["label_name"]==name
        for row in seg_df[sel].itertuples(index=False):
            seg_xy = xhat[int(row.start_idx):int(row.end_idx)]
            if len(seg_xy) > 1:
                ax.plot(seg_xy[:,0], seg_xy[:,1], color=col, alpha=0.08, lw=1)

        # 平均轨迹：高亮
        ax.plot(mu[:,0], mu[:,1], color=col, lw=2.1, label=f"{name} (n={res['n_segments']})")

        # 起点标注（保留之前逻辑）
        if annotate_starts and len(mu) >= 1:
            x0, y0 = mu[0,0], mu[0,1]
            ax.scatter([x0],[y0], s=48, color=col, edgecolor="k", zorder=5)
            key = name.lower().replace(" ", "_")
            text = start_label_map.get(key, None)
            if text:
                ax.text(x0, y0, f" {text}", fontsize=9, color="k",
                        ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                        zorder=6)

    ax.set_title("Representative mean paths with confidence regions (all behaviors)")
    ax.set_xlabel("Latent Dim 1"); ax.set_ylabel("Latent Dim 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()

    # 保存图 & 数据
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[Saved figure] {fig_path}")

        if store_data:
            # 1) 合并 npz：每类行为存 mean/lo/up/all_resampled
            npz_dict = {}
            for beh, name in zip(behaviors, beh_display):
                res = results.get(beh, None)
                if res is None: 
                    continue
                key = str(beh) if use_label=="label" else name
                npz_dict[f"{key}_mean"] = res["mean"]
                npz_dict[f"{key}_lower"] = res["lower"]
                npz_dict[f"{key}_upper"] = res["upper"]
                npz_dict[f"{key}_all"] = res["all_resampled"]
            npz_path = os.path.join(save_dir, f"{data_prefix}.npz")
            np.savez_compressed(npz_path, **npz_dict)
            print(f"[Saved data npz] {npz_path}")

            # 2) 每类行为单独 csv（mean + sem 或 CI 宽度）
            for beh, name in zip(behaviors, beh_display):
                res = results.get(beh, None)
                if res is None: 
                    continue
                mu = res["mean"]; lo = res["lower"]; up = res["upper"]
                df = pd.DataFrame({
                    "u": np.linspace(0,1,len(mu)),
                    "mean_x": mu[:,0], "mean_y": mu[:,1],
                    "lower_x": lo[:,0], "lower_y": lo[:,1],
                    "upper_x": up[:,0], "upper_y": up[:,1],
                })
                csv_name = f"{data_prefix}_{name}.csv"
                csv_path = os.path.join(save_dir, csv_name)
                df.to_csv(csv_path, index=False)
                print(f"[Saved CSV] {csv_path}")

    plt.show()
    return results  # 返回所有行为的结果（可用于后续统计）

# return -> dm
def visualize_return_dm_keypoints(
    xhat, seg_df, 
    save_dir, filename="return_to_dm_keypoints.pdf",
    use_label="label_name",              # 'label_name' 或 'label'
    name_map=None,                       # 若 use_label='label' 时传 {id: name}
    colors=None,                         # 自定义关键点颜色
    plot_raw=True, raw_alpha=0.07, raw_lw=0.9,  # 是否叠加原始“整段 trial”轨迹
    ms=36                                # 散点大小
):
    """
    在同一张图上可视化：
      1) win_return / go_return 的终止点 (end_idx-1)
      2) win_dm / go_dm 的起始点 (start_idx)
    并叠加“整段 trial（dm->tube->return）”的原始轨迹：
       - win trials：浅红
       - go  trials：浅蓝

    约定：
      - seg_df 至少包含列：start_idx、end_idx、(label_name 或 label)
      - xhat 为 (T,2) 的连续隐变量
    """
    assert {"start_idx","end_idx"}.issubset(seg_df.columns), "seg_df 需含 start_idx/end_idx"

    # -------- 基本颜色（关键点） --------
    if colors is None:
        colors = {
            "win_return": "#f4a6c6",  # 粉/浅红
            "go_return":  "#7b3294",  # 紫
            "win_dm":     "#ff7f0e",  # 橙
            "go_dm":      "#2ca02c",  # 绿
        }
    faint_win = "#e74c3c"  # 浅红（原始win轨迹）
    faint_go  = "#3498db"  # 浅蓝（原始go轨迹)

    # -------- 按名称筛选的适配 --------
    if use_label == "label_name":
        assert "label_name" in seg_df.columns, "use_label='label_name' 但 seg_df 无该列"
        lab_series = seg_df["label_name"].astype(str)
        def sel(name): 
            return seg_df[lab_series == name]
        get_name = lambda v: str(v)
    else:
        assert "label" in seg_df.columns, "use_label='label' 但 seg_df 无该列"
        if name_map is None:
            raise ValueError("use_label='label' 时请提供 name_map={id:'name',...}")
        inv_map = {v:k for k,v in name_map.items()}
        def sel(name):
            if name not in inv_map:
                return seg_df.iloc[0:0]
            return seg_df[seg_df["label"] == inv_map[name]]
        get_name = lambda i: name_map.get(int(i), str(i))

    # -------- 小工具：从 df 里抽“整段 trial 的索引区间” --------
    def _extract_trials_ranges(df, kind, use_label="label_name"):
        """返回 list[dict(i0,i1)]，每个元素是一段 trial（dm->tube->return）的全局索引范围"""
        df2 = df.copy().sort_values("start_idx").reset_index(drop=True)
        if use_label == "label_name":
            names = df2["label_name"].astype(str).values
            def is_name(i, nm): return names[i] == nm
        else:
            if name_map is None:
                raise ValueError("use_label='label' 时必须提供 name_map")
            id2name = {int(k):v for k,v in name_map.items()}
            labs = df2["label"].astype(int).values
            def is_name(i, nm): 
                return id2name.get(int(labs[i]), None) == nm

        want_dm, want_tube, want_ret = f"{kind}_dm", f"{kind}_tube", f"{kind}_return"
        out = []
        i, n = 0, len(df2)
        while i < n:
            if is_name(i, want_dm):
                dm = df2.iloc[i]
                j = i + 1
                while j < n and not is_name(j, want_tube):
                    j += 1
                if j >= n: break
                tube = df2.iloc[j]
                k = j + 1
                while k < n and not is_name(k, want_ret):
                    k += 1
                if k >= n: break
                ret = df2.iloc[k]
                i0, i1 = int(dm.start_idx), int(ret.end_idx)
                if 0 <= i0 < i1 <= len(xhat):
                    out.append(dict(i0=i0, i1=i1))
                    i = k + 1
                else:
                    i += 1
            else:
                i += 1
        return out

    # -------- 收集四类关键点 --------
    pts = {}
    # return 的终止点：end_idx-1
    for name in ["win_return", "go_return"]:
        tmp = sel(name)
        idx = (tmp["end_idx"].astype(int) - 1).clip(lower=0).values
        idx = idx[idx < len(xhat)]
        pts[name] = xhat[idx] if len(idx) > 0 else np.zeros((0,2))

    # dm 的起始点：start_idx
    for name in ["win_dm", "go_dm"]:
        tmp = sel(name)
        idx = tmp["start_idx"].astype(int).values
        idx = idx[(idx >= 0) & (idx < len(xhat))]
        pts[name] = xhat[idx] if len(idx) > 0 else np.zeros((0,2))

    # -------- 画图 --------
    plt.figure(figsize=(7.6, 7.0))
    ax = plt.gca()

    # 叠加“整段 trial”的原始轨迹（按类型上色）
    if plot_raw:
        win_trials = _extract_trials_ranges(seg_df, kind="win", use_label=use_label)
        go_trials  = _extract_trials_ranges(seg_df, kind="go",  use_label=use_label)

        for tr in win_trials:
            i0, i1 = tr["i0"], tr["i1"]
            seg = xhat[i0:i1]
            if len(seg) > 1:
                ax.plot(seg[:,0], seg[:,1], color=faint_win, alpha=raw_alpha, lw=raw_lw)

        for tr in go_trials:
            i0, i1 = tr["i0"], tr["i1"]
            seg = xhat[i0:i1]
            if len(seg) > 1:
                ax.plot(seg[:,0], seg[:,1], color=faint_go, alpha=raw_alpha, lw=raw_lw)

    # 关键点散点（return=圆，dm=三角）
    handles, labels = [], []
    order = ["win_return","go_return","win_dm","go_dm"]
    mstyles = {"win_return":"o", "go_return":"o", "win_dm":"^", "go_dm":"^"}  # 圆=return，三角=dm
    for name in order:
        arr = pts.get(name, np.zeros((0,2)))
        if len(arr) == 0:
            continue
        h = ax.scatter(arr[:,0], arr[:,1], s=ms, c=colors.get(name,"C0"),
                       edgecolor="k", linewidths=0.6, marker=mstyles.get(name,"o"), zorder=4)
        handles.append(h)
        labels.append(name)

    # 坐标范围：优先依据 trial 轨迹；若未绘制，则依据关键点
    all_x, all_y = [], []
    if plot_raw:
        # 把刚才收集的 trial 段拼起来计算范围
        for tr in (win_trials + go_trials):
            i0, i1 = tr["i0"], tr["i1"]
            seg = xhat[i0:i1]
            if len(seg) > 0:
                all_x.append(seg[:,0]); all_y.append(seg[:,1])
    if not all_x:  # 如果没有 trial 轨迹，退化为关键点范围
        for arr in pts.values():
            if len(arr) > 0:
                all_x.append(arr[:,0]); all_y.append(arr[:,1])

    if all_x and all_y:
        X = np.concatenate(all_x); Y = np.concatenate(all_y)
        pad = 0.6
        ax.set_xlim(X.min()-pad, X.max()+pad)
        ax.set_ylim(Y.min()-pad, Y.max()+pad)

    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_title("Return endpoints & DM startpoints\n(faint raw trials: win=red, go=blue)")
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=9)
    plt.tight_layout()

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {out_path}")
    
## desect state-dependent dynamics
import numpy as np
from numpy.linalg import lstsq, eigvals
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from ssm.util import find_permutation
from scipy.optimize import linear_sum_assignment

# ---------------------------
# 1) 推断离散状态，并按 GT 顺序重排（可选）
# ---------------------------
def infer_discrete_states(rslds, xhat, y=None, z_gt=None, verbose=True):
    """
    用模型推断离散状态；若提供 z_gt，则尝试对齐。
    出现不合法 perm 时自动回退为恒等排列。
    """
    zhat = rslds.most_likely_states(xhat, y) if y is not None else rslds.most_likely_states(xhat)

    return zhat

# ---------------------------
# 2) 按状态拟合线性动力学  x_{t+1} = A_k x_t + c_k
#    仅使用“状态内连续步”的样本对
def fit_statewise_linear_dynamics(xhat, zhat, K, ridge=0.0, min_pairs=None):
    """
    在 xhat 的当前维度 D 上对每个离散状态 k 拟合:
        X_{t+1} = A_k X_t + c_k
    仅使用“状态内连续”的样本对 {(t) | z_t=k 且 z_{t+1}=k}。
    返回 params[k] = {"A":(D,D), "c":(D,), "eig":(D,), "n_pairs":N}

    参数:
      - ridge: float, 岭正则系数（>0 更稳，尤其样本少时）
      - min_pairs: 至少需要的样本对数量；默认 max(3, D+1)
    """
    T, D = xhat.shape
    if min_pairs is None:
        min_pairs = max(3, D + 1)

    params = {}
    for k in range(K):
        idx = np.where((zhat[:-1] == k) & (zhat[1:] == k))[0]
        N = len(idx)
        if N < min_pairs:
            params[k] = dict(
                A=np.full((D, D), np.nan),
                c=np.full(D, np.nan),
                eig=np.full(D, np.nan, dtype=complex),
                n_pairs=N
            )
            continue

        X_t   = xhat[idx, :]       # (N, D)
        X_tp1 = xhat[idx + 1, :]   # (N, D)

        # 设计矩阵 Phi: [X_t, 1]
        Phi = np.hstack([X_t, np.ones((N, 1), dtype=X_t.dtype)])  # (N, D+1)

        if ridge and ridge > 0:
            # 一次性解多输出的岭回归：Theta = (Phi^T Phi + λI)^-1 Phi^T X_tp1
            G = Phi.T @ Phi                          # (D+1, D+1)
            G.flat[::G.shape[0] + 1] += float(ridge) # 加到对角
            Theta = np.linalg.solve(G, Phi.T @ X_tp1)  # (D+1, D)
        else:
            # 普通最小二乘（多输出），等价于对每个输出维度分别 lstsq
            Theta, _, _, _ = np.linalg.lstsq(Phi, X_tp1, rcond=None)  # (D+1, D)

        # 拆回 A, c
        # Theta 形状是 (D+1, D)；前 D 行是 A^T，最后一行是 c^T
        A = Theta[:D, :].T    # (D, D)
        c = Theta[D, :].T     # (D,)

        # 特征值（D 个）
        try:
            lam = np.linalg.eigvals(A)  # (D,)
        except np.linalg.LinAlgError:
            lam = np.full(D, np.nan, dtype=complex)

        params[k] = dict(A=A, c=c, eig=lam, n_pairs=N)

    return params

# ---------------------------
# 3) 基础可视化：散点/轨迹按“推断状态”上色
# ---------------------------
def plot_points_by_inferred_state(xhat, zhat, palette, state_names=None,
                                  save_path=None, title="Inferred states on latent space"):
    plt.figure(figsize=(6,6))
    K = len(palette)
    for k in range(K):
        pts = xhat[zhat==k]
        if len(pts)==0: continue
        label = state_names[k] if state_names is not None else f"state{k}"
        plt.scatter(pts[:,0], pts[:,1], s=6, color=palette[k], alpha=0.55, label=label)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25); plt.legend(fontsize=8)
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.show()

# ---------------------------
# 4) 向量场绘制工具：x_{t+1}-x_t 的场（离散时间）
# ---------------------------
def quiver_field(ax, A, c, bounds, density=25, color="0.5", alpha=0.6,
                 scale=1.0, U=None):
    """
    绘制离散线性动力学的方向场。
    - 若 A, c 是高维 (DxD, D)，而可视化在2D，则需传入 U (D×2 的投影矩阵)。
    - 若 U=None 且 A 是 2×2，则直接绘制。
    """
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, density)
    ys = np.linspace(y_min, y_max, density)
    Xg, Yg = np.meshgrid(xs, ys)
    P = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # (N, 2)

    if A.shape[0] > 2 and U is not None:
        # 高维投影到2D
        A_2d = U.T @ A @ U
        c_2d = U.T @ c
        vec = (P @ A_2d.T) + c_2d - P
    elif A.shape == (2, 2):
        # 已经是2D
        vec = (P @ A.T) + c - P
    else:
        raise ValueError(f"A 的形状 {A.shape} 与可视化维度不匹配，请传入投影矩阵 U")

    Uq = vec[:, 0].reshape(Xg.shape)
    Vq = vec[:, 1].reshape(Yg.shape)
    ax.quiver(Xg, Yg, Uq, Vq, color=color, alpha=alpha,
              angles="xy", scale_units="xy", scale=scale)

# ---------------------------
# 5) 每个状态单独子图：散点 + 本状态向量场
# ---------------------------
def plot_statewise_fields(xhat, zhat, params, palette, state_names=None,
                          save_path=None, density=25, scale=0.5, U=None):
    """
    每个状态单独子图：散点 + 本状态向量场
    - 若 A 是高维 (D×D) 且可视化在2D，则传入 U (D×2) 做投影
    """
    K = len(palette)
    pad = 0.4
    x_min, x_max = xhat[:,0].min()-pad, xhat[:,0].max()+pad
    y_min, y_max = xhat[:,1].min()-pad, xhat[:,1].max()+pad
    bounds = (x_min, x_max, y_min, y_max)

    ncols = min(4, K)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).reshape(-1)

    for k in range(K):
        ax = axes[k]
        pts = xhat[zhat==k]
        label = state_names[k] if state_names is not None else f"state{k}"
        ax.scatter(pts[:,0], pts[:,1], s=6, color=palette[k], alpha=0.55, label=f"{label} (n={len(pts)})")

        A = params[k]["A"]; c = params[k]["c"]
        if A is not None and np.all(np.isfinite(A)):
            # 关键：把高维 (A,c) 投影到 2D（若 A 不是 2×2）
            if A.shape != (2,2):
                if U is None:
                    raise ValueError(f"A 的形状 {A.shape} 与可视化维度不匹配，请传入投影矩阵 U (D×2)")
                # 投影：A_2d = U^T A U, c_2d = U^T c
                A_2d = U.T @ A @ U
                c_2d = U.T @ c
                quiver_field(ax, A_2d, c_2d, bounds, density=density,
                             color=palette[k], alpha=0.55, scale=scale)
            else:
                quiver_field(ax, A, c, bounds, density=density,
                             color=palette[k], alpha=0.55, scale=scale)

        ax.set_title(label); ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=8, loc="best")

    for j in range(K, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.show()

# ---------------------------
# 6) 综合图：单张坐标轴，点按状态着色；
#    每个状态区域内（用凸包近似）叠加“本状态”的向量场
# ---------------------------
def plot_composite_field_masked(xhat, zhat, params, palette, state_names=None,
                                save_path=None, density=25, scale=0.5, hull_alpha=0.08, U=None):
    """
    单图综合：点按状态着色；每个状态区域（凸包）内叠加该状态的方向场
    - 若 A 是高维 (D×D) 且可视化在2D，则传入 U (D×2) 做投影
    """
    pad = 0.4
    x_min, x_max = xhat[:,0].min()-pad, xhat[:,0].max()+pad
    y_min, y_max = xhat[:,1].min()-pad, xhat[:,1].max()+pad
    bounds = (x_min, x_max, y_min, y_max)

    fig, ax = plt.subplots(1,1, figsize=(6.5,6.5))
    K = len(palette)

    # 背景散点
    for k in range(K):
        pts = xhat[zhat==k]
        if len(pts)==0: continue
        label = state_names[k] if state_names is not None else f"state{k}"
        ax.scatter(pts[:,0], pts[:,1], s=6, color=palette[k], alpha=0.5, label=label)

    # 每个状态：凸包区域 + 该状态的方向场
    for k in range(K):
        pts = xhat[zhat==k]
        if len(pts) < 10:
            continue
        try:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            path = Path(poly)
            ax.fill(poly[:,0], poly[:,1], color=palette[k], alpha=hull_alpha, lw=0)

            A = params[k]["A"]; c = params[k]["c"]
            if A is not None and np.all(np.isfinite(A)):
                if A.shape != (2,2):
                    if U is None:
                        raise ValueError(f"A 的形状 {A.shape} 与可视化维度不匹配，请传入投影矩阵 U (D×2)")
                    A_2d = U.T @ A @ U
                    c_2d = U.T @ c
                    # 仅在该状态凸包内部绘制
                    quiver_field(ax, A_2d, c_2d, bounds, density=density,
                                 color=palette[k], alpha=0.65, scale=scale, mask_path=path)
                else:
                    quiver_field(ax, A, c, bounds, density=density,
                                 color=palette[k], alpha=0.65, scale=scale, mask_path=path)
        except Exception:
            # 凸包失败时，全局绘制
            A = params[k]["A"]; c = params[k]["c"]
            if A is not None and np.all(np.isfinite(A)):
                if A.shape != (2,2):
                    if U is None:
                        raise ValueError(f"A 的形状 {A.shape} 与可视化维度不匹配，请传入投影矩阵 U (D×2)")
                    A_2d = U.T @ A @ U
                    c_2d = U.T @ c
                    quiver_field(ax, A_2d, c_2d, bounds, density=density,
                                 color=palette[k], alpha=0.45, scale=scale)
                else:
                    quiver_field(ax, A, c, bounds, density=density,
                                 color=palette[k], alpha=0.45, scale=scale)

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    ax.set_title("State-dependent vector fields (masked)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.show()

# ===========================
# ===== 主调用（示例） =======
# ===========================
# xhat = xhat_lem 或者你 PCA 后的 xhat_for_analysis（推荐）
# y 可传 None；z_gt 可传 Desampled_State 或简化版用作顺序对齐
def run_inferred_state_dynamics_plots(
    xhat, rslds, save_dir,
    y=None, z_gt=None,
    state_names=None, palette=None,
    density=25, scale=0.5,
    pca=None,             # 可选：若外部已经做了 PCA，可传一个已fit的PCA对象
    verbose=True
):
    """
    高维拟合 + 2D 可视化的一站式入口：
    - 拟合/推断：始终在 xhat 的原始维度 D 上完成（D>=2均可）
    - 可视化：若 D>2，则内部做 PCA->2D（或用传入的 pca）得到 x2d 与投影矩阵 U；再把 (A,c) 投影到 2D 画场
    - 若 D==2，直接用 2D 数据与 2x2 的 (A,c)

    返回：dict(zhat, perm, params, x2d, pca)
    """
    import os
    import numpy as np
    from sklearn.decomposition import PCA

    os.makedirs(save_dir, exist_ok=True)
    T, D = xhat.shape

    # 1) 推断离散状态（顺序对齐有兜底，不会因 perm 报错而中断）
    zhat = infer_discrete_states(rslds, xhat, y=y, z_gt=z_gt, verbose=verbose)
    K = int(np.nanmax(zhat)) + 1

    # 2) 颜色与名字的兜底
    if palette is None or len(palette) < K:
        from matplotlib.cm import get_cmap
        cmap = get_cmap("tab10")
        palette = [cmap(i % 10) for i in range(K)]
    if state_names is None or len(state_names) < K:
        state_names = [f"state{k}" for k in range(K)]

    # 3) 在“高维”上按状态拟合 LDS（关键：fit 在原始 D 维）
    params = fit_statewise_linear_dynamics(xhat, zhat, K, ridge=1e-3)

    # 4) 准备 2D 可视化坐标 x2d 与投影矩阵 U（若 D>2 则PCA；若 D==2 则直传）
    if D > 2:
        if pca is None:
            pca = PCA(n_components=2).fit(xhat)
        x2d = pca.transform(xhat)[:, :2]
        U = pca.components_.T[:, :2]    # (D,2) 高维->2D 的列正交投影
        if verbose:
            evr = getattr(pca, "explained_variance_ratio_", None)
            if evr is not None:
                print(f"[PCA] top-2 explain {evr[:2].sum():.3f} variance.")
    else:
        x2d = xhat
        U = None  # 不需要投影

    # 5) 图1：推断状态着色（仅用于展示模型无监督分配）
    plot_points_by_inferred_state(
        x2d, zhat, palette, state_names=state_names,
        save_path=os.path.join(save_dir, "inferred_states_scatter.pdf"),
        title="Inferred discrete states on 2D latent"
    )

    # 6) 图2：每个状态单独子图：散点 + 本状态向量场（自动投影 A,c）
    plot_statewise_fields(
        xhat=x2d, zhat=zhat, params=params,
        palette=palette, state_names=state_names,
        save_path=os.path.join(save_dir, "statewise_vector_fields.pdf"),
        density=density, scale=scale, U=U
    )

    # 7) 图3：综合图（单轴）：按状态着色；各自凸包内绘制本状态向量场（自动投影 A,c）
    plot_composite_field_masked(
        xhat=x2d, zhat=zhat, params=params,
        palette=palette, state_names=state_names,
        save_path=os.path.join(save_dir, "composite_state_dependent_fields.pdf"),
        density=density, scale=scale, hull_alpha=0.08, U=U
    )

    return dict(zhat=zhat, params=params, x2d=x2d, pca=pca)

## return -> dm
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel, wilcoxon

def _nn_to_set(src_pts, dst_pts):
    """返回：每个 src 点到 dst 集合的最近邻距离（向量，长度=len(src_pts)）"""
    if len(src_pts)==0 or len(dst_pts)==0:
        return np.array([])
    D = cdist(src_pts, dst_pts)   # (Ns, Nd)
    return D.min(axis=1)

def _avgdist_to_set(src_pts, dst_pts):
    """返回每个 src 点到 dst 集合的平均距离"""
    if len(src_pts)==0 or len(dst_pts)==0:
        return np.array([])
    D = cdist(src_pts, dst_pts)   # (Ns, Nd)
    return D.mean(axis=1)         # 每个 return 点到 dm 群的平均距离

def _sig_star(p):
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))

def compare_return_to_dm_paired(
    xhat, seg_df, save_dir,
    filename="return_to_dm_paired.pdf",
    use_label="label_name",
    random_state=0
):
    """
    两组配对比较：
      1) win_return→win_dm  vs win_return→go_dm
      2) go_return →win_dm  vs go_return →go_dm
    - 距离=每个 return 点到目标 dm 集合的最近邻距离（paired）
    - 导出图与 CSV
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    if use_label not in seg_df.columns:
        raise ValueError(f"seg_df 中找不到列 {use_label}，请检查。")

    # ---- 取索引 ----
    T = len(xhat)
    def _idx(series):
        idx = series.astype(int).values
        return idx[(idx>=0) & (idx<T)]

    idx_WR_end = _idx(seg_df.loc[seg_df[use_label]=="win_return", "end_idx"])
    idx_WD_sta = _idx(seg_df.loc[seg_df[use_label]=="win_dm",     "start_idx"])
    idx_GR_end = _idx(seg_df.loc[seg_df[use_label]=="go_return",  "end_idx"])
    idx_GD_sta = _idx(seg_df.loc[seg_df[use_label]=="go_dm",      "start_idx"])

    WR = xhat[idx_WR_end] if len(idx_WR_end) else np.empty((0,2))
    WD = xhat[idx_WD_sta] if len(idx_WD_sta) else np.empty((0,2))
    GR = xhat[idx_GR_end] if len(idx_GR_end) else np.empty((0,2))
    GD = xhat[idx_GD_sta] if len(idx_GD_sta) else np.empty((0,2))

    # ---- 平均距离 ----
    wr_to_wd = _avgdist_to_set(WR, WD)
    wr_to_gd = _avgdist_to_set(WR, GD)
    gr_to_wd = _avgdist_to_set(GR, WD)
    gr_to_gd = _avgdist_to_set(GR, GD)

    # 与源集长度对齐（若目标为空则得到空向量）
    n_wr = len(WR)
    n_gr = len(GR)

    # ---- 配对统计（仅当两向量同长且长度>=2）----
    def paired_stats(a, b):
        if len(a) == len(b) and len(a) >= 2:
            # 配对 t 检验
            t_stat, t_p = ttest_rel(a, b, nan_policy="omit")
            # Wilcoxon（需非全相等 & 长度>=2）
            try:
                w_stat, w_p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                w_stat, w_p = np.nan, np.nan
        else:
            t_stat = t_p = w_stat = w_p = np.nan
        return dict(t_p=t_p, w_p=w_p)

    stats_WR = paired_stats(wr_to_wd, wr_to_gd)
    stats_GR = paired_stats(gr_to_wd, gr_to_gd)

    # ---- 图：两面板配对比较 ----
    fig, axes = plt.subplots(1, 2, figsize=(5, 5), sharey=True)

    def panel(ax, a, b, title, labels, colors):
        """a vs b：画柱 + 配对散点 + 连线 + 显著性"""
        # 均值柱
        means = [np.nanmean(a) if len(a)>0 else np.nan,
                 np.nanmean(b) if len(b)>0 else np.nan]
        sems  = [np.nanstd(a, ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan,
                 np.nanstd(b, ddof=1)/np.sqrt(len(b)) if len(b)>1 else np.nan]
        x = np.array([0,1], float)
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.8, capsize=5)

        # 配对散点 + 连线
        m = min(len(a), len(b))
        jit = (rng.rand(m)-0.5)*0.12
        ax.scatter(x[0]+jit, a[:m], s=18, color=colors[0], alpha=0.75, zorder=3)
        ax.scatter(x[1]+jit, b[:m], s=18, color=colors[1], alpha=0.75, zorder=3)
        # 连线
        for i in range(m):
            ax.plot([x[0]+jit[i], x[1]+jit[i]], [a[i], b[i]], color="gray", alpha=0.35, lw=1)

        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")

        # 显著性（配对）
        st = paired_stats(a[:m], b[:m])
        if np.isfinite(st["t_p"]):
            star = _sig_star(st["t_p"])
            ytop = np.nanmax(np.r_[a[:m], b[:m]]) if m>0 else 0.0
            ytop = ytop * 1.08 + (0.02 if ytop==0 else 0)
            ax.plot([x[0], x[1]], [ytop, ytop], color="k", lw=1)
            ax.text(np.mean(x), ytop*1.01, f"{star}\n(p_t={st['t_p']:.2g}; p_w={st['w_p']:.2g})",
                    ha="center", va="bottom", fontsize=9)

    panel(
        axes[0], wr_to_wd, wr_to_gd,
        title=f"WR→WD vs WR→GD (n={n_wr})",
        labels=["WR→WD", "WR→GD"],
        colors=["#c0392b", "#ff7675"]     # 深红 vs 浅红
    )
    panel(
        axes[1], gr_to_wd, gr_to_gd,
        title=f"GR→WD vs GR→GD (n={n_gr})",
        labels=["GR→WD", "GR→GD"],
        colors=["#1f618d", "#5dade2"]     # 深蓝 vs 浅蓝
    )

    axes[0].set_ylabel("Nearest distance (return → dm)")
    plt.tight_layout()
    fig_path = os.path.join(save_dir, filename)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {fig_path}")

    # ---- CSV 导出 ----
    # WR 明细
    df_wr = pd.DataFrame({
        "wr_to_win_dm": wr_to_wd,
        "wr_to_go_dm":  wr_to_gd,
        "delta(win - go)": wr_to_wd - wr_to_gd
    })
    csv_wr = os.path.join(save_dir, "WR_return_to_dm_paired.csv")
    df_wr.to_csv(csv_wr, index=False)
    print(f"[Saved] {csv_wr}")

    # GR 明细
    df_gr = pd.DataFrame({
        "gr_to_win_dm": gr_to_wd,
        "gr_to_go_dm":  gr_to_gd,
        "delta(win - go)": gr_to_wd - gr_to_gd
    })
    csv_gr = os.path.join(save_dir, "GR_return_to_dm_paired.csv")
    df_gr.to_csv(csv_gr, index=False)
    print(f"[Saved] {csv_gr}")

    # 汇总 p 值
    df_p = pd.DataFrame([
        dict(group="WR", n=n_wr, p_t=stats_WR["t_p"], p_w=stats_WR["w_p"]),
        dict(group="GR", n=n_gr, p_t=stats_GR["t_p"], p_w=stats_GR["w_p"]),
    ])
    csv_p = os.path.join(save_dir, "return_to_dm_paired_pvalues.csv")
    df_p.to_csv(csv_p, index=False)
    print(f"[Saved] {csv_p}")

    return dict(
        n_WR=n_wr, n_GR=n_gr,
        wr_to_wd=wr_to_wd, wr_to_gd=wr_to_gd,
        gr_to_wd=gr_to_wd, gr_to_gd=gr_to_gd,
        p_WR_t=stats_WR["t_p"], p_WR_w=stats_WR["w_p"],
        p_GR_t=stats_GR["t_p"], p_GR_w=stats_GR["w_p"],
        fig_path=fig_path, csv_wr=csv_wr, csv_gr=csv_gr, csv_p=csv_p
    )

## inferred state
def _runs_from_labels(labels, target):
    """
    给定 1D 标签序列 labels，找出等于 target 的所有连续区间，返回 [(start_idx, end_idx), ...]（左闭右开）
    """
    labels = np.asarray(labels)
    idx = np.where(labels == target)[0]
    if idx.size == 0:
        return []
    # 找断点
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(g[0], g[-1] + 1) for g in groups]  # 右边界+1

def plot_inferred_state_raster(
    time_axis,
    zhat,
    K=3,
    state_names=None,      # list[str]，长度 K
    colors=None,           # list[str] 或 list[tuple]，长度 K
    height=0.8,            # 每一行条带高度
    hgap=0.25,             # 行间距
    alpha=1.0,             # 颜色透明度
    lw=0.6,                # 区段边框线宽（0 表示无边框）
    ax=None, figsize=(14, 3.5)
):
    """
    在时间轴上画 K 行的 raster：每行对应一个离散隐状态；该行在该状态活跃时段以该状态颜色填充。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    time_axis = np.asarray(time_axis).ravel()
    zhat = np.asarray(zhat).ravel()
    assert time_axis.ndim == 1 and zhat.ndim == 1 and len(time_axis) == len(zhat), "time_axis 与 zhat 需等长的一维数组"
    assert K >= int(zhat.max()) + 1, "K 小于 zhat 中的最大状态编号+1"

    if state_names is None:
        state_names = [f"state{k}" for k in range(K)]
    if colors is None:
        default = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        colors = [default[k % len(default)] for k in range(K)]

    y_ticks, y_ticklabels = [], []
    for row, k in enumerate(range(K)):
        y_center = (K - 1 - row) * (1.0 + hgap)
        y_bottom = y_center - height / 2.0

        runs = _runs_from_labels(zhat, k)
        for (i0, i1) in runs:
            t_start = time_axis[i0]
            t_end = time_axis[i1] if i1 < len(time_axis) else time_axis[-1]
            width = float(t_end - t_start)
            if width <= 0:
                continue
            rect = plt.Rectangle(
                (t_start, y_bottom), width, height,
                facecolor=colors[k], edgecolor=(colors[k] if lw > 0 else "none"),
                linewidth=lw, alpha=alpha
            )
            ax.add_patch(rect)

        y_ticks.append(y_center)
        y_ticklabels.append(state_names[k])

    ax.set_ylim(- (height / 2.0), (K - 1) * (1.0 + hgap) + (height / 2.0))
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Inferred discrete states (raster)")
    ax.grid(axis="x", alpha=0.25)

    return fig, ax

# 1D latent & behavior & inferred state
def plot_latents_and_states_3row(
    time_axis,
    x_latent,
    z_labels,
    zhat,
    state_names,
    state_colors,
    label_map,
    label_colors,
    save_dir,
    filename="latents_pc12_and_states_raster.pdf",
    smooth_win=4,
    alpha=0.15,
    figsize=(16, 8)
):
    """
    三行子图：PC1（着色）、PC2（着色）、推断状态raster。三行时间轴严格对齐。
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.8]}
    )

    # 上两行：PC1、PC2（方案A）
    plot_latents_with_behavior_shading(
        time_axis=time_axis,
        x_latent=x_latent,
        z_labels=z_labels,
        label_map=label_map,
        label_colors=label_colors,
        smooth_win=smooth_win,
        alpha=alpha,
        axes=axes[:2],            # 传入两个子图
        figsize=(16, 6),
        title_suffix=""
    )
    axes[0].set_title("Latent PC1 with behavior shading")
    axes[1].set_title("Latent PC2 with behavior shading")

    # 底行：推断状态 raster
    plot_inferred_state_raster(
        time_axis=time_axis,
        zhat=zhat,
        K=len(state_names),
        state_names=state_names,
        colors=state_colors,
        ax=axes[2]
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {save_path}")
    return fig, axes

## posterior vs loop area

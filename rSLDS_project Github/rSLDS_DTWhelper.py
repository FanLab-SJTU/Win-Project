import numpy as np

# extract sagmental trajectory from behavior sequence
def extract_trajectory_behavior(x_latent, df):
    trajs = []
    labels = []
    names  = []
    starts = []
    for _, seg in df.iterrows():
        s, e = int(seg["start_idx"]), int(seg["end_idx"])
        if e > s:
            trajs.append(x_latent[s:e, :])
            labels.append(int(seg["label"]))
            names.append(seg["label_name"])
            starts.append(s)
    return trajs, np.array(labels), names, np.array(starts)

### DTW core algorithm
def _dtw_core(P, Q, band):
    m, n = len(P), len(Q)
    INF = 1e18
    D = np.full((m+1, n+1), INF, float)
    L = np.zeros((m+1, n+1), int)
    D[0,0] = 0.0
    for i in range(1, m+1):
        jmin = max(1, i - band)
        jmax = min(n, i + band)
        if jmin > jmax:  # 这一行也很关键
            continue
        Pi = P[i-1]
        for j in range(jmin, jmax+1):
            cost = float(np.linalg.norm(Pi - Q[j-1]))
            a = D[i-1, j]   + cost
            b = D[i,   j-1] + cost
            c = D[i-1, j-1] + cost
            if a <= b and a <= c:
                D[i,j] = a; L[i,j] = L[i-1,j]   + 1
            elif b <= a and b <= c:
                D[i,j] = b; L[i,j] = L[i,  j-1] + 1
            else:
                D[i,j] = c; L[i,j] = L[i-1,j-1] + 1
    return D[m,n], L[m,n]

def dtw_avg_distance(P, Q, band=None, sc_band_ratio=0.10, auto_fallback=True):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    m, n = len(P), len(Q)
    if m == 0 or n == 0 or not np.isfinite(P).all() or not np.isfinite(Q).all():
        return np.nan
    if band is None:
        Lmax = max(m, n)
        band = int(max(1, sc_band_ratio * Lmax))
    band = max(int(band), abs(m - n))  # 保证可达

    Dmn, Lmn = _dtw_core(P, Q, band)
    if not (np.isfinite(Dmn) and Lmn > 0) and auto_fallback:
        Dmn, Lmn = _dtw_core(P, Q, max(m, n))  # 无带
    if not (np.isfinite(Dmn) and Lmn > 0):
        return np.nan
    return Dmn / Lmn

def pairwise_distance_matrix(trajs, sc_band_ratio=0.10):
    N = len(trajs)
    D = np.zeros((N, N), float)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_avg_distance(trajs[i], trajs[j],
                                 band=None, sc_band_ratio=sc_band_ratio,
                                 auto_fallback=True)
            D[i,j] = D[j,i] = d
    np.fill_diagonal(D, 0.0)
    return D


# 组内/组间平均距离
def groupwise_mean_distances(D_sorted, labels_sorted, label_order):
    res = {}
    idx = {lab: np.where(labels_sorted == lab)[0] for lab in label_order}
    for i, a in enumerate(label_order):
        Ia = idx[a]
        # 组内
        if len(Ia) >= 2:
            tri = D_sorted[np.ix_(Ia, Ia)]
            iu = np.triu_indices_from(tri, 1)
            res[(a, a)] = float(tri[iu].mean()) if iu[0].size > 0 else np.nan
        else:
            res[(a, a)] = np.nan
        # 组间
        for b in label_order[i+1:]:
            Ib = idx[b]
            res[(a, b)] = float(D_sorted[np.ix_(Ia, Ib)].mean()) if len(Ia) > 0 and len(Ib) > 0 else np.nan
    return res
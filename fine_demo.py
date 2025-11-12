import torch
import tensorly as tl
from tensorly.decomposition import tucker

# =============== 配置 ===============
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tl.set_backend('pytorch')
torch.manual_seed(0)

# 数据规模（demo 可设为 2e5；真实可到 1e6+，注意显存）
N = 200_000
VOXEL_SIZE = 5
TOP_DENSE_PERCENT = 0.3
KEEP_RATIO_PER_CLUSTER = 0.9
ALPHA_SPATIAL = 0.3
BETA_IMPORT   = 0.4
GAMMA_LOW_RANK= 0.3
MAX_POINTS_FOR_TUCKER = 1024
RANK_IMPORT   = 3      # Tucker 第一维（3个重要度通道）秩
RANK_POINT    = 16     # 点维秩
RANK_FEATURE  = 3      # 特征维秩（<= 特征数，这里特征=4）

# =============== 1) 构造示例数据（你的工程里替换为真实数据） ===============
coords = (torch.rand(N, 3, device=device) * 100.0)
for center in [(20,20,20), (50,60,40), (80,30,70)]:
    idx = torch.randint(0, N, (N//20,), device=device)
    coords[idx] = torch.tensor(center, device=device) + torch.randn_like(coords[idx]) * 1.0

# 三个重要度通道：梯度 / 不透明度 / 访问次数（示例随机）
grad_imp   = torch.rand(N, device=device)
opacity_imp= torch.rand(N, device=device)
access_cnt = torch.rand(N, device=device)
# 叠成 [N,3]
importance3 = torch.stack([grad_imp, opacity_imp, access_cnt], dim=1)

# =============== 2) 体素聚类（GPU 上快速定位高密区域） ===============
def voxel_hash(coords, voxel_size):
    voxel_idx = torch.floor(coords / voxel_size).to(torch.int64)      # [N,3]
    base = voxel_idx.min(dim=0).values
    voxel_idx_shift = voxel_idx - base
    keys = (voxel_idx_shift[:, 0].to(torch.int64) << 42) \
         + (voxel_idx_shift[:, 1].to(torch.int64) << 21) \
         +  voxel_idx_shift[:, 2].to(torch.int64)
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    return voxel_idx, unique_keys, inv

voxel_idx, voxel_keys, inv = voxel_hash(coords, VOXEL_SIZE)
num_voxels = voxel_keys.numel()
counts = torch.bincount(inv, minlength=num_voxels)

k = max(1, int(num_voxels * TOP_DENSE_PERCENT))
top_counts, top_voxel_ids = torch.topk(counts, k, largest=True)
dense_voxel_mask = torch.zeros(num_voxels, dtype=torch.bool, device=device)
dense_voxel_mask[top_voxel_ids] = True

points_in_dense = dense_voxel_mask[inv]  # [N] bool
dense_coords = coords[points_in_dense]                # [Nd,3]
dense_importance3 = importance3[points_in_dense]      # [Nd,3]
dense_inv = inv[points_in_dense]                      # [Nd]

print(f"[Info] 总点数={N}, 体素数={num_voxels}, 密集体素数={k}, 密集体素内点数={dense_coords.shape[0]}")

# =============== 3) 每个密集体素内：Tucker(通道×点×特征) + 三分量融合 ===============
final_scores = torch.zeros(N, device=device)
dense_voxel_ids = torch.unique(dense_inv)

for vid in dense_voxel_ids:
    idx = torch.nonzero(dense_inv == vid, as_tuple=False).squeeze(1)
    global_idx = torch.nonzero(points_in_dense, as_tuple=False).squeeze(1)[idx]
    if idx.numel() == 0:
        continue

    pts = dense_coords[idx]                # [m,3]
    imp3 = dense_importance3[idx]          # [m,3] 三个通道

    # —— 局部归一化坐标 ——
    pmin = pts.min(dim=0).values
    pmax = pts.max(dim=0).values
    rng = (pmax - pmin).clamp_min(1e-6)
    pts_norm = (pts - pmin) / rng          # [m,3]

    # —— 重要度三通道分开归一化（按通道 min-max） ——
    ch_min = imp3.min(dim=0).values        # [3]
    ch_max = imp3.max(dim=0).values        # [3]
    imp3_norm = (imp3 - ch_min) / (ch_max - ch_min + 1e-8)   # [m,3]

    # —— 空间稀疏性分数（KNN 平均距离） ——
    m = pts_norm.shape[0]
    if m > 4096:
        ref_idx = torch.randint(0, m, (4096,), device=device)
        ref = pts_norm[ref_idx]
        d = torch.cdist(pts_norm, ref)         # [m,4096]
        knn = 8
        spatial_score, _ = d.topk(knn, largest=False)
        spatial_score = spatial_score.mean(dim=1)   # [m]
    elif m > 2:
        d = torch.cdist(pts_norm, pts_norm)         # [m,m]
        knn = min(8, max(1, m-1))
        spatial_score, _ = d.topk(knn+1, largest=False)  # 含自身
        spatial_score = spatial_score[:, 1:].mean(dim=1)  # 去掉自身
    else:
        spatial_score = torch.zeros(m, device=device)

    s_sp = (spatial_score - spatial_score.min()) / (spatial_score.max()-spatial_score.min()+1e-8)

    # —— 构造三通道特征张量：X.shape = [3, m, 4]
    # 每个通道的特征 = [x,y,z(归一化), 该通道的重要度]
    feats_c = [torch.cat([pts_norm, imp3_norm[:, c:c+1]], dim=1) for c in range(3)]  # 3 × [m,4]
    feats_stack = torch.stack(feats_c, dim=0)  # [3, m, 4]

    # —— 采样用于 Tucker（控制显存），点维采样对所有通道一致 ——
    X = feats_stack  # [3, m, 4]

    # if m > MAX_POINTS_FOR_TUCKER:  # 可删
    #     samp_idx = torch.randint(0, m, (MAX_POINTS_FOR_TUCKER,), device=device)
    #     X = feats_stack[:, samp_idx, :]      # [3, Ms, 4]
    # else:
    #     X = feats_stack                      # [3, m, 4]

    # —— Tucker 分解：ranks = [r_import, r_point, r_feature] ——
    rI = min(RANK_IMPORT,  X.shape[0])   # ≤3
    rP = min(RANK_POINT,   X.shape[1])
    rF = min(RANK_FEATURE, X.shape[2])   # ≤4
    try:
        core, factors = tucker(X, ranks=[rI, rP, rF])
    except TypeError:
        core, factors = tucker(X, rank=[rI, rP, rF])

    # 因子矩阵：factors[0]=通道(U_importance) [3,rI]；factors[1]=点 [M,rP]；factors[2]=特征(U_feature) [4,rF]
    U_feature = factors[2]   # [4, rF]

    # —— 低秩分数（对三个通道分别投影后再平均） ——
    # 对全体点（未采样）计算，更稳健
    lr_list = []
    for c in range(3):
        low_rank_vec = feats_c[c] @ U_feature     # [m, rF]
        lr_list.append(torch.norm(low_rank_vec, dim=1))  # [m]
    s_lr = torch.stack(lr_list, dim=1).mean(dim=1)       # [m] 三通道平均
    s_lr = (s_lr - s_lr.min()) / (s_lr.max()-s_lr.min()+1e-8)

    # —— 重要度汇总分数（把三通道的重要度做简单融合；也可换成加权和/最大值） ——
    s_imp = imp3_norm.mean(dim=1)   # [m] 也可用 .amax(dim=1).values

    # —— 融合三类分数 ——
    score = ALPHA_SPATIAL * s_sp + BETA_IMPORT * s_imp + GAMMA_LOW_RANK * s_lr  # [m]

    # —— 在该体素内保留前 KEEP_RATIO_PER_CLUSTER ——
    keep_m = max(1, int(m * KEEP_RATIO_PER_CLUSTER))
    _, keep_local = torch.topk(score, k=keep_m, largest=True)
    final_scores[global_idx[keep_local]] = score[keep_local]

# 仅保留 final_scores > 0 的点（被选中）
global_mask = final_scores > 0
dense_keep_mask = final_scores > 0
kept_coords = dense_coords[dense_keep_mask]
kept_importance3 = dense_importance3[dense_keep_mask]

print(f"[Result] 密集体素内点: {dense_coords.shape[0]} → 保留: {kept_coords.shape[0]}")

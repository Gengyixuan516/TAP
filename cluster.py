import torch
import tensorly as tl
from tensorly.decomposition import tucker

# =============== 配置 ===============
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tl.set_backend('pytorch')  # 让 tensorly 使用 PyTorch 后端
torch.manual_seed(0)

# 数据规模（demo 可设为 2e5；真实可到 1e6+，注意显存）
N = 200_000
VOXEL_SIZE = 5         # 体素边长（按你的坐标尺度调整）
TOP_DENSE_PERCENT = 0.3  # 仅保留最密的前 30% 体素
KEEP_RATIO_PER_CLUSTER = 0.9  # 在每个密集簇里保留 90%
ALPHA_SPATIAL = 0.3      # 空间稀疏性权重
BETA_IMPORT = 0.4        # 已知重要度权重
GAMMA_LOW_RANK = 0.3     # 低秩分数权重
MAX_POINTS_FOR_TUCKER = 1024  # 每个簇用于分解的最多采样点数
RANK_POINT = 16          # Tucker 点模式秩（越大越细，显存越高）
RANK_FEATURE = 3         # Tucker 特征模式秩（<= 特征维）

# =============== 1) 构造示例数据（你的工程里替换为真实数据） ===============
# 坐标范围假设在 [0, 100)^3，生成一些高密子团 + 噪声
coords = (torch.rand(N, 3, device=device) * 100.0)
# 造几个高密团
for center in [(20,20,20), (50,60,40), (80,30,70)]:
    idx = torch.randint(0, N, (N//20,), device=device)
    coords[idx] = torch.tensor(center, device=device) + torch.randn_like(coords[idx]) * 1.0

# 已知每点“重要度”分数（真实工程里用你的分数，如透明度累计/访问次数等）
importance = torch.rand(N, device=device)

# =============== 2) 体素聚类（GPU 上快速定位高密区域） ===============
def voxel_hash(coords, voxel_size):
    """把坐标映射到体素整数网格，并返回各点的体素索引与唯一体素表"""
    voxel_idx = torch.floor(coords / voxel_size).to(torch.int64)      # [N,3]
    # 将 3D 体素索引压成唯一键（避免溢出，转成字符串成本高，这里用位移hash）
    # 为稳妥，先把索引移到非负区间（以最小值为基准）
    base = voxel_idx.min(dim=0).values
    voxel_idx_shift = voxel_idx - base
    # 位移哈希（假设索引范围不极端大）
    keys = (voxel_idx_shift[:, 0].to(torch.int64) << 42) \
         + (voxel_idx_shift[:, 1].to(torch.int64) << 21) \
         +  voxel_idx_shift[:, 2].to(torch.int64)
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    return voxel_idx, unique_keys, inv  # inv: 每个点对应的体素ID（0..num_vox-1）

voxel_idx, voxel_keys, inv = voxel_hash(coords, VOXEL_SIZE)
num_voxels = voxel_keys.numel()

# 统计每个体素的点数（密度）
counts = torch.bincount(inv, minlength=num_voxels)

# 选择密度最高的前 p% 体素
k = max(1, int(num_voxels * TOP_DENSE_PERCENT))
top_counts, top_voxel_ids = torch.topk(counts, k, largest=True)
dense_voxel_mask = torch.zeros(num_voxels, dtype=torch.bool, device=device)
dense_voxel_mask[top_voxel_ids] = True

# 标记哪些点位于“密集体素”中
points_in_dense = dense_voxel_mask[inv]  # [N] bool
dense_coords = coords[points_in_dense]
dense_importance = importance[points_in_dense]
dense_inv = inv[points_in_dense]         # 在密集体素子集内的体素ID（仍指向全局ID）

print(f"[Info] 总点数={N}, 体素数={num_voxels}, 密集体素数={k}, "
      f"密集体素内点数={dense_coords.shape[0]}")

# =============== 3) 在每个密集体素内，用 Tucker 提取特征子空间并评分 ===============
# 我们对每个簇（体素）做一次小规模 Tucker：
#   - 点特征 = [归一化局部坐标 (3) + 已知重要度 (1)] → F=4
#   - 构建张量形状 (1, M, F)，M为采样点数
#   - Tucker ranks: [1, r_point, r_feature]，只学习 feature 模式的子空间 U_feature
#   - 任意点 x 的低秩分数 = || x @ U_feature ||_2
#
# 这样不需要为每个点学习系数，计算开销很小，且可拓展到大规模

# 为每个点预留最终打分
final_scores = torch.zeros(dense_coords.shape[0], device=device)

# 取出密集体素的唯一ID，遍历（可按点数排序、或分批处理以控显存）
dense_voxel_ids = torch.unique(dense_inv)

for vid in dense_voxel_ids.tolist():
    # 取该体素内的点索引（在 dense_* 的子集坐标系里）
    idx = torch.nonzero(dense_inv == vid, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        continue

    pts = dense_coords[idx]        # [m,3]
    imp = dense_importance[idx]    # [m]

    # —— 局部归一化（平衡尺度差异）——
    pmin = pts.min(dim=0).values
    pmax = pts.max(dim=0).values
    rng = (pmax - pmin).clamp_min(1e-6)
    pts_norm = (pts - pmin) / rng

    # 组装特征（F=4）
    feats = torch.cat([pts_norm, imp[:, None]], dim=1)  # [m,4]

    # —— 计算空间稀疏性分数（这里用 KNN=8 的平均距离近似，成本低）
    # 为保证速度，对超大簇先子采样作为参考集合
    m = feats.shape[0]
    if m > 4096:
        ref_idx = torch.randint(0, m, (4096,), device=device)
        ref = pts_norm[ref_idx]
        d = torch.cdist(pts_norm, ref)  # [m,4096]
        knn = 8
        spatial_score, _ = d.topk(knn, largest=False)   # 距离越大越稀疏
        spatial_score = spatial_score.mean(dim=1)       # [m]
    elif m > 2:
        d = torch.cdist(pts_norm, pts_norm)             # [m,m]
        knn = min(8, max(1, m-1))
        spatial_score, _ = d.topk(knn+1, largest=False) # 含自身0距离
        spatial_score = spatial_score[:, 1:].mean(dim=1)

    # 归一化空间分数 & 重要度
    s_sp = (spatial_score - spatial_score.min()) / (spatial_score.max()-spatial_score.min()+1e-8)
    s_imp = (imp - imp.min()) / (imp.max()-imp.min()+1e-8)

    # —— Tucker：对该体素随机采样最多 MAX_POINTS_FOR_TUCKER 个点做分解 ——
    if m > MAX_POINTS_FOR_TUCKER:  #这里是为了避免显存过大，进行抽样，可以删掉
        samp_idx = torch.randint(0, m, (MAX_POINTS_FOR_TUCKER,), device=device)
        feats_samp = feats[samp_idx]
    else:
        feats_samp = feats

    # (1, M, F) 形式
    X = feats_samp[None, :, :]  # shape: [1, Ms, 4]

    # 兼容不同 tensorly 版本（ranks / rank）
    try:
        core, factors = tucker(
            X, ranks=[1, min(RANK_POINT, X.shape[1]), min(RANK_FEATURE, X.shape[2])]
        )
    except TypeError:
        core, factors = tucker(
            X, rank=[1, min(RANK_POINT, X.shape[1]), min(RANK_FEATURE, X.shape[2])]
        )

    # feature-mode 子空间 U_feature: 形状 [F, rF]
    U_feature = factors[2]  # [4, rF]

    # 所有点的低秩分数 = || feats @ U_feature ||_2
    low_rank_vec = feats @ U_feature   # [m, rF]
    s_lr = torch.norm(low_rank_vec, dim=1)  # [m]
    s_lr = (s_lr - s_lr.min()) / (s_lr.max()-s_lr.min()+1e-8)

    # —— 融合三类分数 ——
    score = ALPHA_SPATIAL * s_sp + BETA_IMPORT * s_imp + GAMMA_LOW_RANK * s_lr  # [m]

    # —— 在该体素内剔除少量低分点 ——
    keep_m = max(1, int(m * KEEP_RATIO_PER_CLUSTER))
    _, keep_local = torch.topk(score, k=keep_m, largest=True)
    # 写入总分（仅对保留点记录分数，剔除点保持0）
    final_scores[idx[keep_local]] = score[keep_local]

# 仅保留 final_scores > 0 的点（被选中）
dense_keep_mask = final_scores > 0
kept_coords = dense_coords[dense_keep_mask]
kept_importance = dense_importance[dense_keep_mask]

# 非密集体素中的点（如果你也想保留一部分，可按策略拼回）
# 这里示例为：仅使用密集体素内保留的点
print(f"[Result] 密集体素内点: {dense_coords.shape[0]} → 保留: {kept_coords.shape[0]}")

# =============== 4) Demo：返回结果 ===============
# 你可以将 kept_coords / kept_importance 作为“重要点集”继续下游任务
# 例如：用于初始化锚点/高斯中心，或作为后续训练/渲染的子集。

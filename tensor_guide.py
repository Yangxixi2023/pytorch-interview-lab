TENSOR_GUIDE = '''import torch
from einops import rearrange, repeat, reduce

# 1. reshape / view：不改变元素顺序；view 需要兼容的 stride。
x = torch.arange(24.0).reshape(2, 3, 4)  # [B=2,T=3,D=4]
flat = x.view(2, -1)                    # [2,12]
assert flat.shape == (2, 12)

# 2. transpose / permute：交换维度，不会自动把内存变连续。
xt = x.transpose(1, 2)                  # [2,4,3]
xp = x.permute(0, 2, 1)                 # 同上
assert torch.equal(xt, xp)
contiguous_flat = xt.contiguous().view(2, -1)
reshape_flat = xt.reshape(2, -1)         # 必要时会复制
assert torch.equal(contiguous_flat, reshape_flat)

# 3. unsqueeze / squeeze：添加/删除长度为1的维度。
expanded_axis = x.unsqueeze(1)          # [2,1,3,4]
assert torch.equal(expanded_axis.squeeze(1), x)
# 指定 squeeze 的维度，避免 batch=1 时误删 batch 维。

# 4. expand 是广播视图；repeat（torch）会复制数据。
row = torch.arange(4.0).reshape(1, 4)
broadcast = row.expand(3, 4)            # 零 stride，不独立存储每一行
copied = row.repeat(3, 1)               # 真实复制
assert torch.equal(broadcast, copied)
# 不要对共享存储的 expand 结果随意原地写入；需要时 clone。

# 5. cat 在已有维度拼接；stack 新增一个维度。
assert torch.cat([x, x], dim=1).shape == (2, 6, 4)
assert torch.stack([x, x], dim=1).shape == (2, 2, 3, 4)

# 6. split 指定各段大小；chunk 按目标块数划分，实际块数可能较少。
parts = torch.split(x, [1, 2], dim=1)
assert [p.shape[1] for p in parts] == [1, 2]
chunks = torch.chunk(torch.arange(5), 4)
assert len(chunks) == 3

# 7. where 和广播：condition、两分支要可广播。
mask = torch.tensor([[True, False, True], [False, True, True]])
masked = torch.where(mask[..., None], x, torch.zeros_like(x))
assert masked.shape == x.shape
# where 的两侧表达式会先求值，不用于回避先发生的除零。

# 8. einops：把拆头、交换和合并意图写进模式字符串。
features = torch.arange(48.0).reshape(2, 2, 12)
heads = rearrange(features, 'b t (h d) -> b h t d', h=3)
merged = rearrange(heads, 'b h t d -> b t (h d)')
assert heads.shape == (2, 3, 2, 4)
assert torch.equal(features, merged)

# 9. einops.repeat / reduce：明确新增维度及归约方向。
tokens = repeat(torch.arange(4.0), 'd -> b t d', b=2, t=3)
pooled = reduce(tokens, 'b t d -> b d', 'mean')
assert tokens.shape == (2, 3, 4)
assert pooled.shape == (2, 4)

print('所有张量变换示例通过')
print('拆头 shape:', tuple(heads.shape))
print('聚合 shape:', tuple(pooled.shape))
'''

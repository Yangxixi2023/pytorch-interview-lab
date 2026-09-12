# 全题库：输入、返回与参考实现

89 道题；每个场景使用三个固定种子执行。

| # | 题目 | 分类 | 难度 | 场景数 | 参考实现 |
| --- | --- | --- | --- | --- | --- |
| 1 | PPO 损失 | 对齐与强化学习 | 中等 | 3 | [ppo_loss.py](../references/ppo_loss.py) |
| 2 | GSPO 损失 | 对齐与强化学习 | 困难 | 3 | [gspo_loss.py](../references/gspo_loss.py) |
| 3 | DAPO 损失 | 对齐与强化学习 | 困难 | 3 | [dapo_loss.py](../references/dapo_loss.py) |
| 4 | GRPO 损失 | 对齐与强化学习 | 困难 | 3 | [grpo_loss.py](../references/grpo_loss.py) |
| 5 | DPO 损失 | 对齐与强化学习 | 中等 | 2 | [dpo_loss.py](../references/dpo_loss.py) |
| 6 | GAE 广义优势估计 | 对齐与强化学习 | 中等 | 2 | [gae_advantage.py](../references/gae_advantage.py) |
| 7 | Bradley–Terry 奖励损失 | 对齐与强化学习 | 中等 | 2 | [reward_model.py](../references/reward_model.py) |
| 8 | 实现 Softmax | Transformer 内部机制 | 简单 | 2 | [softmax.py](../references/softmax.py) |
| 9 | 交叉熵损失 | 从零训练 GPT | 简单 | 2 | [cross_entropy.py](../references/cross_entropy.py) |
| 10 | 实现 ReLU | Transformer 内部机制 | 简单 | 2 | [relu.py](../references/relu.py) |
| 11 | GELU 激活函数 | Transformer 内部机制 | 简单 | 2 | [gelu.py](../references/gelu.py) |
| 12 | SwiGLU 激活函数 | Transformer 内部机制 | 简单 | 2 | [swiglu.py](../references/swiglu.py) |
| 13 | 实现 RMSNorm | Transformer 内部机制 | 中等 | 2 | [rmsnorm.py](../references/rmsnorm.py) |
| 14 | 实现 LayerNorm | Transformer 内部机制 | 中等 | 2 | [layernorm.py](../references/layernorm.py) |
| 15 | 简单线性层 | Transformer 内部机制 | 中等 | 2 | [linear.py](../references/linear.py) |
| 16 | Embedding 层 | 从零训练 GPT | 简单 | 2 | [embedding.py](../references/embedding.py) |
| 17 | Softmax 注意力 | Transformer 内部机制 | 简单 | 2 | [attention.py](../references/attention.py) |
| 18 | 因果自注意力 | 注意力与位置编码 | 中等 | 2 | [causal_attention.py](../references/causal_attention.py) |
| 19 | 滑动窗口注意力 | 注意力与位置编码 | 中等 | 2 | [sliding_window.py](../references/sliding_window.py) |
| 20 | ALiBi 注意力 | 注意力与位置编码 | 中等 | 2 | [alibi.py](../references/alibi.py) |
| 21 | 多头注意力 | Transformer 内部机制 | 困难 | 2 | [mha.py](../references/mha.py) |
| 22 | 分组查询注意力 GQA | 注意力与位置编码 | 困难 | 2 | [gqa.py](../references/gqa.py) |
| 23 | 旋转位置编码 RoPE | 注意力与位置编码 | 中等 | 2 | [rope.py](../references/rope.py) |
| 24 | LoRA 低秩适配 | 参数高效训练 | 中等 | 2 | [lora.py](../references/lora.py) |
| 25 | 实现 Dropout | 从零训练 GPT | 简单 | 2 | [dropout.py](../references/dropout.py) |
| 26 | 流匹配损失 | 扩散模型与 DiT | 简单 | 2 | [flow_matching.py](../references/flow_matching.py) |
| 27 | 标签平滑损失 | 从零训练 GPT | 简单 | 2 | [label_smoothing.py](../references/label_smoothing.py) |
| 28 | Focal Loss | 损失函数 | 中等 | 2 | [focal_loss.py](../references/focal_loss.py) |
| 29 | 对比损失 InfoNCE | 损失函数 | 中等 | 2 | [contrastive_loss.py](../references/contrastive_loss.py) |
| 30 | 梯度累积 | 从零训练 GPT | 简单 | 2 | [gradient_accumulation.py](../references/gradient_accumulation.py) |
| 31 | 梯度范数裁剪 | 从零训练 GPT | 简单 | 2 | [gradient_clipping.py](../references/gradient_clipping.py) |
| 32 | Adam 优化器 | 从零训练 GPT | 中等 | 2 | [adam.py](../references/adam.py) |
| 33 | 余弦学习率（含预热） | 从零训练 GPT | 中等 | 4 | [cosine_lr.py](../references/cosine_lr.py) |
| 34 | Kaiming 初始化 | 从零训练 GPT | 简单 | 2 | [weight_init.py](../references/weight_init.py) |
| 35 | 正弦位置编码 | 注意力与位置编码 | 简单 | 2 | [sinusoidal_pe.py](../references/sinusoidal_pe.py) |
| 36 | NTK-aware RoPE 缩放 | 注意力与位置编码 | 简单 | 2 | [ntk_rope.py](../references/ntk_rope.py) |
| 37 | 扩散噪声调度 | 扩散模型与 DiT | 简单 | 2 | [noise_schedule.py](../references/noise_schedule.py) |
| 38 | DDIM 采样步骤 | 扩散模型与 DiT | 中等 | 2 | [ddim_step.py](../references/ddim_step.py) |
| 39 | 自适应 LayerNorm Zero | 扩散模型与 DiT | 中等 | 2 | [adaln_zero.py](../references/adaln_zero.py) |
| 40 | 实现 BatchNorm | Transformer 内部机制 | 中等 | 2 | [batchnorm.py](../references/batchnorm.py) |
| 41 | 二维最大池化 | Vision Transformer | 简单 | 2 | [max_pool2d.py](../references/max_pool2d.py) |
| 42 | ViT Patch Embedding | Vision Transformer | 简单 | 2 | [vit_patch.py](../references/vit_patch.py) |
| 43 | 二维卷积 | Vision Transformer | 中等 | 2 | [conv2d.py](../references/conv2d.py) |
| 44 | 深度可分离卷积 | Vision Transformer | 中等 | 2 | [depthwise_conv.py](../references/depthwise_conv.py) |
| 45 | 多头交叉注意力 | 注意力与位置编码 | 中等 | 2 | [cross_attention.py](../references/cross_attention.py) |
| 46 | SwiGLU MLP | Transformer 内部机制 | 中等 | 2 | [mlp.py](../references/mlp.py) |
| 47 | 多元线性回归与反向传播 | 经典手撕 | 中等 | 2 | [linear_regression.py](../references/linear_regression.py) |
| 48 | GCN 层（图卷积） | 图神经网络 | 中等 | 2 | [gcn_layer.py](../references/gcn_layer.py) |
| 49 | GIN 层（图同构网络） | 图神经网络 | 中等 | 2 | [gin_layer.py](../references/gin_layer.py) |
| 50 | GraphSAGE 层 | 图神经网络 | 中等 | 2 | [graphsage_layer.py](../references/graphsage_layer.py) |
| 51 | MPNN 消息传递 | 图神经网络 | 中等 | 2 | [mpnn_layer.py](../references/mpnn_layer.py) |
| 52 | 图读出（图级池化） | 图神经网络 | 简单 | 2 | [graph_readout.py](../references/graph_readout.py) |
| 53 | GAT 层（图注意力） | 图神经网络 | 中等 | 2 | [gat_layer.py](../references/gat_layer.py) |
| 54 | 图自编码器 GAE | 图神经网络 | 困难 | 2 | [gae.py](../references/gae.py) |
| 55 | 链接预测 | 图神经网络 | 困难 | 2 | [link_prediction.py](../references/link_prediction.py) |
| 56 | MoE 负载均衡损失 | LLM 前沿架构 | 中等 | 2 | [moe_load_balance.py](../references/moe_load_balance.py) |
| 57 | 多 Token 预测 | LLM 前沿架构 | 中等 | 2 | [multi_token_prediction.py](../references/multi_token_prediction.py) |
| 58 | 差分注意力 | LLM 前沿架构 | 困难 | 2 | [diff_attention.py](../references/diff_attention.py) |
| 59 | 多头潜在注意力 MLA | LLM 前沿架构 | 困难 | 2 | [mla.py](../references/mla.py) |
| 60 | 混合专家 MoE | LLM 前沿架构 | 困难 | 2 | [moe.py](../references/moe.py) |
| 61 | Flash Attention 分块 | 推理与分布式训练 | 困难 | 2 | [flash_attention.py](../references/flash_attention.py) |
| 62 | 环形注意力（单机模拟） | 推理与分布式训练 | 困难 | 2 | [ring_attention.py](../references/ring_attention.py) |
| 63 | 线性自注意力 | 注意力与位置编码 | 困难 | 2 | [linear_attention.py](../references/linear_attention.py) |
| 64 | KV Cache 注意力 | 推理与分布式训练 | 困难 | 2 | [kv_cache.py](../references/kv_cache.py) |
| 65 | 分页注意力 | 推理与分布式训练 | 困难 | 2 | [paged_attention.py](../references/paged_attention.py) |
| 66 | INT8 量化线性层 | 推理与分布式训练 | 困难 | 2 | [int8_quantization.py](../references/int8_quantization.py) |
| 67 | QLoRA 量化基座与低秩更新 | 参数高效训练 | 困难 | 2 | [qlora.py](../references/qlora.py) |
| 68 | Mamba SSM 步骤 | LLM 前沿架构 | 困难 | 2 | [mamba_ssm.py](../references/mamba_ssm.py) |
| 69 | 张量并行 MLP（单机模拟） | 推理与分布式训练 | 困难 | 2 | [tensor_parallel.py](../references/tensor_parallel.py) |
| 70 | FSDP 训练步骤（单机模拟） | 推理与分布式训练 | 困难 | 2 | [fsdp_step.py](../references/fsdp_step.py) |
| 71 | 混合精度训练步骤 | 从零训练 GPT | 中等 | 2 | [mixed_precision.py](../references/mixed_precision.py) |
| 72 | 激活检查点 | 从零训练 GPT | 中等 | 2 | [activation_checkpointing.py](../references/activation_checkpointing.py) |
| 73 | GPT-2 Transformer Block | Transformer 内部机制 | 困难 | 2 | [gpt2_block.py](../references/gpt2_block.py) |
| 74 | ViT Transformer Block | Vision Transformer | 困难 | 2 | [vit_block.py](../references/vit_block.py) |
| 75 | Top-k / Top-p 采样分布 | 推理与分布式训练 | 中等 | 3 | [topk_sampling.py](../references/topk_sampling.py) |
| 76 | 束搜索解码 | 推理与分布式训练 | 中等 | 2 | [beam_search.py](../references/beam_search.py) |
| 77 | 推测解码接受步骤 | 推理与分布式训练 | 困难 | 2 | [speculative_decoding.py](../references/speculative_decoding.py) |
| 78 | 蒙特卡洛树搜索 PUCT 选择 | 对齐与强化学习 | 困难 | 2 | [mcts_search.py](../references/mcts_search.py) |
| 79 | 字节对编码 BPE | 经典手撕 | 困难 | 3 | [bpe.py](../references/bpe.py) |
| 80 | K-means 聚类 | 经典手撕 | 中等 | 2 | [kmeans.py](../references/kmeans.py) |
| 81 | 快速排序（递归） | 经典手撕 | 中等 | 3 | [quicksort_recursive.py](../references/quicksort_recursive.py) |
| 82 | 快速排序（非递归） | 经典手撕 | 中等 | 3 | [quicksort_iterative.py](../references/quicksort_iterative.py) |
| 83 | 拓扑排序 | 经典手撕 | 中等 | 3 | [topo.py](../references/topo.py) |
| 84 | Top-k 最大元素 | 经典手撕 | 中等 | 3 | [topk.py](../references/topk.py) |
| 85 | 模拟马走日 | 经典手撕 | 中等 | 4 | [knight.py](../references/knight.py) |
| 86 | 张量变换与多头重排 | 张量基础 | 简单 | 2 | [tensor_heads.py](../references/tensor_heads.py) |
| 87 | FFN 前馈网络 | Transformer 内部机制 | 中等 | 2 | [ffn.py](../references/ffn.py) |
| 88 | Entropy Loss 策略熵 | 对齐与强化学习 | 简单 | 2 | [entropy_loss.py](../references/entropy_loss.py) |
| 89 | SFT Loss 因果语言模型 | 从零训练 GPT | 中等 | 2 | [sft_loss.py](../references/sft_loss.py) |

## PPO 损失

```python
def solve(new_logp, old_logp, advantages, mask, clip_eps=0.2):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `new_logp` | 浮点 Tensor | `[B, T]` | 当前策略对已采样 token 的自然对数概率；需要保留梯度。 |
| `old_logp` | 浮点 Tensor | `[B, T]` | 采样时旧策略的 log probability；计算时 detach。 |
| `advantages` | 浮点 Tensor | `[B, T]` | 逐 token 优势，可正可负；计算时 detach。 |
| `mask` | bool Tensor | `[B, T]` | True 为有效 token，False 为 padding；整个 batch 至少一个 True。 |
| `clip_eps` | float | `—` | 对称裁剪半径；ratio 裁剪到 [1-clip_eps, 1+clip_eps]。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

实现 PPO 的 clipped policy loss（本题只考 actor，不含 value loss 和 entropy bonus）。
输入均为 [B,T]，mask 为 bool，有效 token 至少一个。old_logp 和 advantages 视为常量。
r=exp(new_logp-old_logp)，返回 -sum(mask*min(r*A, clip(r,1-eps,1+eps)*A))/sum(mask)。
返回可反向传播的标量；padding 不参与计算。禁止调用现成 PPO loss。

## GSPO 损失

```python
def solve(new_logp, old_logp, advantages, mask, clip_low=0.2, clip_high=0.28):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `new_logp` | 浮点 Tensor | `[B, T]` | 当前策略的 token log probability；保留梯度。 |
| `old_logp` | 浮点 Tensor | `[B, T]` | 旧策略 token log probability；detach 后使用。 |
| `advantages` | 浮点 Tensor | `[B]` | 每条序列一个已标准化优势；不是 [B,T]；detach 后使用。 |
| `mask` | bool Tensor | `[B, T]` | True 为有效 token；每条序列至少一个 True。 |
| `clip_low` | float | `—` | ratio 的下界为 1-clip_low。 |
| `clip_high` | float | `—` | ratio 的上界为 1+clip_high。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

new_logp/old_logp/mask: [B,T]；advantages: [B]，是已算好的组内标准化优势。每条序列至少一个有效 token，mask 为 bool。old_logp 和 advantages 视为常量。返回可求导标量。
对每条序列计算 s=exp(sum(mask*(new-old))/length)，再对序列做 min(s*A,clip(s,1-low,1+high)*A)，最终取负的序列平均。不添加 KL。

## DAPO 损失

```python
def solve(new_logp, old_logp, advantages, mask, clip_low=0.2, clip_high=0.28):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `new_logp` | 浮点 Tensor | `[B, T]` | 当前策略的 token log probability；保留梯度。 |
| `old_logp` | 浮点 Tensor | `[B, T]` | 旧策略 token log probability；detach 后使用。 |
| `advantages` | 浮点 Tensor | `[B]` | 每条序列一个已标准化优势；不是 [B,T]；detach 后使用。 |
| `mask` | bool Tensor | `[B, T]` | True 为有效 token；每条序列至少一个 True。 |
| `clip_low` | float | `—` | ratio 的下界为 1-clip_low。 |
| `clip_high` | float | `—` | ratio 的上界为 1+clip_high。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

new_logp/old_logp/mask: [B,T]；advantages: [B]，是已算好的组内标准化优势。每条序列至少一个有效 token，mask 为 bool。old_logp 和 advantages 视为常量。返回可求导标量。
对每个 token 计算 ratio 与 clipped surrogate，负的有效 token 总和除以整个 batch 的有效 token 数。输入是已经完成动态采样和奖励处理的 batch，本题不实现采样循环，不加 KL。

## GRPO 损失

```python
def solve(new_logp, old_logp, advantages, mask, clip_low=0.2, clip_high=0.28):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `new_logp` | 浮点 Tensor | `[B, T]` | 当前策略的 token log probability；保留梯度。 |
| `old_logp` | 浮点 Tensor | `[B, T]` | 旧策略 token log probability；detach 后使用。 |
| `advantages` | 浮点 Tensor | `[B]` | 每条序列一个已标准化优势；不是 [B,T]；detach 后使用。 |
| `mask` | bool Tensor | `[B, T]` | True 为有效 token；每条序列至少一个 True。 |
| `clip_low` | float | `—` | ratio 的下界为 1-clip_low。 |
| `clip_high` | float | `—` | ratio 的上界为 1+clip_high。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

new_logp/old_logp/mask: [B,T]；advantages: [B]，是已算好的组内标准化优势。每条序列至少一个有效 token，mask 为 bool。old_logp 和 advantages 视为常量。返回可求导标量。
对每个 token 计算 ratio 与 clipped surrogate，先在每条序列的有效 token 内平均，再对序列平均并取负。此题考 beta=0 的 GRPO policy loss，不含 KL。

## DPO 损失

```python
def solve(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `chosen` | 浮点 Tensor | `[B]` | 当前策略对偏好回答的整序列 log probability 之和；保留梯度。 |
| `rejected` | 浮点 Tensor | `[B]` | 当前策略对非偏好回答的整序列 log probability 之和；保留梯度。 |
| `ref_chosen` | 浮点 Tensor | `[B]` | 参考策略偏好回答 log probability；detach。 |
| `ref_rejected` | 浮点 Tensor | `[B]` | 参考策略非偏好回答 log probability；detach。 |
| `beta` | float | `—` | 偏好 margin 的缩放系数。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

四个 [B] 张量是整条回答的 log probability 之和。返回 mean(-logsigmoid(beta*((chosen-rejected)-(ref_chosen-ref_rejected))))；reference 不求梯度。需对极大负 margin 稳定。

## GAE 广义优势估计

```python
def solve(rewards, values, terminated, gamma=0.99, lam=0.95):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `rewards` | 浮点 Tensor | `[T, B]` | 每个时间步的奖励；本题时间维在前。 |
| `values` | 浮点 Tensor | `[T+1, B]` | 状态价值，最后一行用于末端 bootstrap。 |
| `terminated` | bool Tensor | `[T, B]` | True 表示该步后真正终止，阻断 bootstrap 与优势递推；不表示时间截断。 |
| `gamma` | float | `—` | 奖励折扣系数。 |
| `lam` | float | `—` | GAE 衰减系数。 |

返回：

- `advantages`：浮点 Tensor，`[T, B]`。优势张量；必须 detach。
- `returns`：浮点 Tensor，`[T, B]`。advantages + values[:-1]；必须 detach。

多项返回时按上面顺序组成元组。

rewards/terminated 为 [T,B]，values 为 [T+1,B]。terminated[t] 表示该步后真正终止（不是时间截断）。delta=r+gamma*(1-done)*V_next-V；A_t=delta+gamma*lam*(1-done)*A_next。返回 (advantages, returns)，returns=A+values[:-1]，两者均 detach。

## Bradley–Terry 奖励损失

```python
def solve(chosen_rewards, rejected_rewards):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `chosen_rewards` | 浮点 Tensor | `[B]` | 偏好回答的奖励。 |
| `rejected_rewards` | 浮点 Tensor | `[B]` | 非偏好回答的奖励。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

输入 [B] 奖励，返回 mean(softplus(rejected-chosen))，需要数值稳定和可导。

## 实现 Softmax

```python
def solve(x, dim=-1):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `任意形状` | 输入 logits，可能包含很大的正数。 |
| `dim` | int | `—` | 进行归一化的维度，支持负索引。 |

返回：

- `probabilities`：浮点 Tensor，`与 x 相同`。沿 dim 的和为 1，保留输入梯度；不能调用现成 softmax。

单项直接返回，不包装成元组。

沿 dim 实现数值稳定 softmax，保持形状。不能调用 torch.softmax / F.softmax。

## 交叉熵损失

```python
def solve(logits, targets):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `logits` | 浮点 Tensor | `[N, C]` | 未经 softmax 的分类分数。 |
| `targets` | int64 Tensor | `[N]` | 每个样本的正确类别，取值 0..C-1。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

logits [N,C]，targets [N] 为 int64 类别。返回平均交叉熵。不能调用 cross_entropy 或 log_softmax。

## 实现 ReLU

```python
def solve(x):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `任意形状` | 输入激活值，保留梯度。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。逐元素激活，保持 dtype、device 和形状。

单项直接返回，不包装成元组。

max(x,0)，不能调用 relu。

## GELU 激活函数

```python
def solve(x):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `任意形状` | 输入激活值，保留梯度。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。逐元素激活，保持 dtype、device 和形状。

单项直接返回，不包装成元组。

使用精确 erf 版本 x*Phi(x)，不是 tanh 近似。

## SwiGLU 激活函数

```python
def solve(x):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., 2D]` | 最后一维为偶数；前半为 a，后半为 b。 |

返回：

- `output`：浮点 Tensor，`[..., D]`。SiLU(a)*b，最后一维减半，保留梯度。

单项直接返回，不包装成元组。

最后一维长度为偶数，分成 a,b，返回 SiLU(a)*b。

## 实现 RMSNorm

```python
def solve(x, weight, eps=1e-6):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | 沿最后一维计算均方，不减均值。 |
| `weight` | 浮点 Tensor | `[D]` | 逐特征缩放权重。 |
| `eps` | float | `—` | 加在均方内部的稳定项。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。归一化后乘 weight；x 和 weight 均可求导。

单项直接返回，不包装成元组。

x [...,D]，weight [D]。沿最后一维返回 x/sqrt(mean(x²)+eps)*weight，不减均值。

## 实现 LayerNorm

```python
def solve(x, weight, bias, eps=1e-5):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | 只归一化最后一维，使用总体方差。 |
| `weight` | 浮点 Tensor | `[D]` | 缩放参数。 |
| `bias` | 浮点 Tensor | `[D]` | 偏移参数。 |
| `eps` | float | `—` | 加在方差内部的稳定项。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。LayerNorm 的仿射输出；保留输入和参数梯度。

单项直接返回，不包装成元组。

沿最后一维归一化，方差为总体方差 correction=0，返回归一化值*weight+bias。禁止调用 layer_norm。

## 简单线性层

```python
def solve(x, weight, bias):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., Din]` | 任意前导 batch 维度。 |
| `weight` | 浮点 Tensor | `[Dout, Din]` | 注意权重以输出维度在前。 |
| `bias` | 浮点 Tensor | `[Dout]` | 输出偏移。 |

返回：

- `output`：浮点 Tensor，`[..., Dout]`。x @ weight.T + bias，保留梯度。

单项直接返回，不包装成元组。

x [...,Din]，weight [Dout,Din]，bias [Dout]，返回 x@weight.T+bias，不能调用 F.linear。

## Embedding 层

```python
def solve(ids, weight):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `ids` | int64 Tensor | `任意形状` | 词表索引，取值 0..V-1；可以重复。 |
| `weight` | 浮点 Tensor | `[V, D]` | 可训练的 embedding 表。 |

返回：

- `output`：浮点 Tensor，`[*ids.shape, D]`。查表结果；重复索引对应 weight 梯度需累加。

单项直接返回，不包装成元组。

ids 是任意形状 int64 张量，weight [V,D]，返回 [...,D]。重复索引的梯度需累加，不能调用 embedding。

## Softmax 注意力

```python
def solve(q, k, v):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, Q, D]` | 查询。 |
| `k` | 浮点 Tensor | `[B, K, D]` | 键；K 可以不等于 Q。 |
| `v` | 浮点 Tensor | `[B, K, Dv]` | 值；Dv 可以不等于 D。 |

返回：

- `output`：浮点 Tensor，`[B, Q, Dv]`。非因果 scaled dot-product attention，保留 q/k/v 梯度。

单项直接返回，不包装成元组。

q [B,Q,D]，k [B,K,D]，v [B,K,Dv]；返回 softmax(qkᵀ/sqrt(D))v。不能调用 scaled_dot_product_attention。

## 因果自注意力

```python
def solve(q, k, v):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 query。 |
| `k` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 key。 |
| `v` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 value。 |

返回：

- `output`：浮点 Tensor，`[B, H, T, D]`。因果注意力输出；不允许读取未来位置，保留梯度。

单项直接返回，不包装成元组。

q,k,v [B,H,T,D]，返回同形输出。只允许看到当前和过去位置。

## 滑动窗口注意力

```python
def solve(q, k, v, window=3):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 query。 |
| `k` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 key。 |
| `v` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 value。 |
| `window` | int | `—` | 可见窗口长度，包含当前位置；为正整数。 |

返回：

- `output`：浮点 Tensor，`[B, H, T, D]`。因果注意力输出；不允许读取未来位置，保留梯度。

单项直接返回，不包装成元组。

q,k,v [B,H,T,D]，返回同形输出。只允许看到当前和过去位置。 window 包含当前位置，额外屏蔽距离 >=window 的历史 token。

## ALiBi 注意力

```python
def solve(q, k, v, slopes):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 query。 |
| `k` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 key。 |
| `v` | 浮点 Tensor | `[B, H, T, D]` | 已拆头的 value。 |
| `slopes` | 浮点 Tensor | `[H]` | 每个 head 的位置惩罚斜率。 |

返回：

- `output`：浮点 Tensor，`[B, H, T, D]`。因果注意力输出；不允许读取未来位置，保留梯度。

单项直接返回，不包装成元组。

q,k,v [B,H,T,D]，返回同形输出。只允许看到当前和过去位置。 slopes [H]，scores 额外减去 slopes[h]*(i-j)。

## 多头注意力

```python
def solve(x, wq, wk, wv, wo, heads):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[B, T, D]` | 未投影输入。 |
| `wq` | 浮点 Tensor | `[D, D]` | 按 x @ wq 使用；不转置。 |
| `wk` | 浮点 Tensor | `[D, D]` | 按 x @ wk 使用；不转置。 |
| `wv` | 浮点 Tensor | `[D, D]` | 按 x @ wv 使用；不转置。 |
| `wo` | 浮点 Tensor | `[D, D]` | 按 x @ wo 使用；不转置。 |
| `heads` | int | `—` | head 数 H，D 必须可被 H 整除。 |

返回：

- `output`：浮点 Tensor，`[B, T, D]`。非因果多头注意力合并后经过 wo 投影。

单项直接返回，不包装成元组。

x [B,T,D]，四个权重均 [D,D]，按 x@w 投影。拆 H 个头做非因果 attention，合并后 @wo，无 bias/dropout，D 可被 heads 整除。

## 分组查询注意力 GQA

```python
def solve(q, k, v):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, Hq, T, D]` | Hq 必须可被 Hkv 整除。 |
| `k` | 浮点 Tensor | `[B, Hkv, S, D]` | 连续 Hq/Hkv 个 query head 共用一个 KV head。 |
| `v` | 浮点 Tensor | `[B, Hkv, S, D]` | 与 key 的头数和序列长度一致。 |

返回：

- `output`：浮点 Tensor，`[B, Hq, T, D]`。非因果 GQA 输出，不合并 head 维。

单项直接返回，不包装成元组。

q [B,Hq,T,D]，k,v [B,Hkv,S,D]，Hq 是 Hkv 整数倍。连续 Hq/Hkv 个 query head 共享一个 KV head。返回非因果 attention [B,Hq,T,D]。

## 旋转位置编码 RoPE

```python
def solve(x, positions, base=10000.0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[B, H, T, D]` | D 为偶数，采用相邻偶奇维配对。 |
| `positions` | 整数 Tensor | `[T]` | 各 token 的绝对位置，可从非零位置开始。 |
| `base` | float | `—` | 旋转频率的底数。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。旋转后的张量；保留 x 的梯度。

单项直接返回，不包装成元组。

x [B,H,T,D]，D 偶数；positions [T]。采用相邻偶奇配对 (x0,x1),(x2,x3)，theta=position*base^(-2j/D)，输出每对 (a*cos-b*sin,a*sin+b*cos)。

## LoRA 低秩适配

```python
def solve(x, weight, a, b, alpha):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., Din]` | 输入特征。 |
| `weight` | 浮点 Tensor | `[Dout, Din]` | 冻结的基座权重，必须 detach。 |
| `a` | 浮点 Tensor | `[r, Din]` | 低秩降维权重。 |
| `b` | 浮点 Tensor | `[Dout, r]` | 低秩升维权重。 |
| `alpha` | float | `—` | 低秩分支缩放分子，实际系数 alpha/r。 |

返回：

- `output`：浮点 Tensor，`[..., Dout]`。基座输出加低秩更新；x/a/b 保留梯度，weight 不求导。

单项直接返回，不包装成元组。

x [...,Din]，weight [Dout,Din]，a [r,Din]，b [Dout,r]。返回 x@weight.T + alpha/r*(x@a.T@b.T)。weight 冻结，x/a/b 可求导。

## 实现 Dropout

```python
def solve(x, p, training, keep_mask):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `任意形状` | 待处理激活。 |
| `p` | float | `—` | 丢弃概率，0<=p<1。 |
| `training` | bool | `—` | True 训练，False 推理。 |
| `keep_mask` | bool Tensor | `与 x 相同` | 预先采样的保留掩码；本题不要求重新随机采样。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。训练：x*keep_mask/(1-p)；推理：x。保留 x 梯度。

单项直接返回，不包装成元组。

实现 inverted dropout。为方便确定性判题，提供已采样的 bool keep_mask，与 x 同形。training=False 返回 x；训练时返回 x*keep_mask/(1-p)。0<=p<1。无需自己采样。

## 流匹配损失

```python
def solve(pred_velocity, x0, x1):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `pred_velocity` | 浮点 Tensor | `任意形状` | 可训练的预测速度。 |
| `x0` | 浮点 Tensor | `与 pred_velocity 相同` | 路径起点，视为常量。 |
| `x1` | 浮点 Tensor | `与 pred_velocity 相同` | 路径终点，视为常量。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

线性路径 x_t=(1-t)x0+t*x1，目标速度为 x1-x0；返回所有元素平均 MSE，x0/x1 视为常量。

## 标签平滑损失

```python
def solve(logits, targets, smoothing=0.1):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `logits` | 浮点 Tensor | `[N, C]` | 未经 softmax 的分类分数。 |
| `targets` | int64 Tensor | `[N]` | 每个样本的正确类别，取值 0..C-1。 |
| `smoothing` | float | `—` | 平滑比例，标签分布为 (1-smoothing)*one_hot+smoothing/C。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

logits [N,C]，平滑标签为 (1-eps)*one_hot+eps/C，返回平均交叉熵。

## Focal Loss

```python
def solve(logits, targets, gamma=2.0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `logits` | 浮点 Tensor | `[N, C]` | 未经 softmax 的分类分数。 |
| `targets` | int64 Tensor | `[N]` | 每个样本的正确类别，取值 0..C-1。 |
| `gamma` | float | `—` | Focal 调制指数；gamma=0 退化为交叉熵。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

多分类无 alpha 加权版本；pt=softmax(logits)[target]，返回 mean(-(1-pt)^gamma*log(pt))。

## 对比损失 InfoNCE

```python
def solve(queries, keys, temperature=0.1):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `queries` | 浮点 Tensor | `[N, D]` | 需要在最后一维 L2 normalize。 |
| `keys` | 浮点 Tensor | `[N, D]` | 第 i 个 key 是第 i 个 query 的正样本。 |
| `temperature` | float | `—` | 正的温度系数。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

queries/keys [N,D]，先 L2 normalize；第 i 个 query 的正例为第 i 个 key，其他为负例。返回单向平均交叉熵。

## 梯度累积

```python
def solve(x, y, weight, microbatch_size):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, D]` | 输入特征。 |
| `y` | 浮点 Tensor | `[N, O]` | 回归目标。 |
| `weight` | 浮点 Tensor | `[D, O]` | 线性映射参数。 |
| `microbatch_size` | int | `—` | 正整数；最后一个 microbatch 可以不足此大小。 |

返回：

- `grad_weight`：浮点 Tensor，`[D, O]`。全 batch、所有输出元素的 mean MSE 对 weight 的手动梯度；不要用 autograd。

单项直接返回，不包装成元组。

x [N,D]，y [N,O]，weight [D,O]。手动计算全 batch mean MSE 对 weight 的梯度，按 microbatch 累加；最后一个不足 batch 也要按元素数正确加权。返回梯度，不修改输入，不用 autograd。

## 梯度范数裁剪

```python
def solve(grads, max_norm, eps=1e-6):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `grads` | list[Tensor] | `各元素形状可不同` | 所有需要一起计算全局 L2 范数的梯度。 |
| `max_norm` | float | `—` | 允许的最大范数。 |
| `eps` | float | `—` | 加在分母中的稳定项。 |

返回：

- `clipped_grads`：list[Tensor]，`与 grads 一一对应`。裁剪后的新梯度列表，不原地修改输入。
- `total_norm`：浮点 Tensor，`[]`。裁剪之前的全局 L2 范数。

多项返回时按上面顺序组成元组。

grads 为张量列表，计算所有元素的全局 L2 范数 norm；返回 (新梯度列表, norm)，新梯度=g*min(1,max_norm/(norm+eps))，不原地修改。

## Adam 优化器

```python
def solve(param, grad, m, v, step, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `param` | 浮点 Tensor | `与 param 相同` | 当前参数。 |
| `grad` | 浮点 Tensor | `与 param 相同` | 当前梯度。 |
| `m` | 浮点 Tensor | `与 param 相同` | 上一时刻一阶动量。 |
| `v` | 浮点 Tensor | `与 param 相同` | 上一时刻二阶动量。 |
| `step` | int | `—` | 本次更新步数，从 1 开始。 |
| `lr` | float | `—` | 学习率。 |
| `beta1` | float | `—` | 一阶动量衰减系数。 |
| `beta2` | float | `—` | 二阶动量衰减系数。 |
| `eps` | float | `—` | 稳定项，放在 sqrt 外。 |

返回：

- `new_param`：浮点 Tensor，`与 param 相同`。更新后的参数，不修改 param。
- `new_m`：浮点 Tensor，`与 m 相同`。更新后的一阶动量。
- `new_v`：浮点 Tensor，`与 v 相同`。更新后的二阶动量。

多项返回时按上面顺序组成元组。

实现一次 Adam 更新，step 从 1 开始；偏置修正 m/(1-beta1^step)、v/(1-beta2^step)，epsilon 在 sqrt 外。返回 (new_param,new_m,new_v)，不修改输入。

## 余弦学习率（含预热）

```python
def solve(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `step` | int | `—` | 当前步，0<=step<=total_steps。 |
| `warmup_steps` | int | `—` | 预热步数，0<=warmup_steps<total_steps。 |
| `total_steps` | int | `—` | 总步数。 |
| `max_lr` | float | `—` | 预热结束时的学习率。 |
| `min_lr` | float | `—` | 余弦阶段终点学习率。 |

返回：

- `learning_rate`：float，`—`。当前步的 Python 浮点学习率，不返回 Tensor。

单项直接返回，不包装成元组。

0<=step<=total_steps 且 0<=warmup<total。warmup>0 且 step<warmup 时 lr=max_lr*step/warmup；此后从 max_lr 余弦降到 min_lr。返回 float。

## Kaiming 初始化

```python
def solve(standard_normal, fan_in):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `standard_normal` | 浮点 Tensor | `任意形状` | 已采样的标准正态噪声；不要再次采样。 |
| `fan_in` | int | `—` | 输入连接数，正整数。 |

返回：

- `weight`：浮点 Tensor，`与 standard_normal 相同`。按 sqrt(2/fan_in) 缩放。

单项直接返回，不包装成元组。

输入预采样标准正态张量，按 ReLU fan_in Kaiming normal 缩放为 sqrt(2/fan_in)*z；不重新采样。

## 正弦位置编码

```python
def solve(length, dim, base=10000.0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `length` | int | `—` | 序列长度。 |
| `dim` | int | `—` | 偶数特征维度。 |
| `base` | float | `—` | 频率底数。 |

返回：

- `encoding`：浮点 Tensor，`[length, dim]`。float32 位置编码，偶数列 sin、奇数列 cos。

单项直接返回，不包装成元组。

dim 为偶数，返回 float32 [length,dim]；PE[p,2i]=sin(p/base^(2i/dim))，奇数位用 cos。

## NTK-aware RoPE 缩放

```python
def solve(dim, scale, base=10000.0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `dim` | int | `—` | 偶数且大于 2。 |
| `scale` | float | `—` | 上下文缩放倍数，>=1。 |
| `base` | float | `—` | 原始 RoPE 底数。 |

返回：

- `inverse_frequencies`：浮点 Tensor，`[dim/2]`。float32 的缩放后逆频率。

单项直接返回，不包装成元组。

考固定 NTK 缩放的频率计算，不是动态长度策略。dim 为偶数且 >2，scale>=1；新 base=base*scale^(dim/(dim-2))，返回 float32 [dim/2] 逆频率 new_base^(-2i/dim)。

## 扩散噪声调度

```python
def solve(steps, beta_start, beta_end):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `steps` | int | `—` | 正的总步数。 |
| `beta_start` | float | `—` | 第一个 beta。 |
| `beta_end` | float | `—` | 最后一个 beta（steps=1 时只用起点）。 |

返回：

- `betas`：浮点 Tensor，`[steps]`。float32 线性 beta 序列。
- `alphas`：浮点 Tensor，`[steps]`。1-betas。
- `alpha_bars`：浮点 Tensor，`[steps]`。alphas 的前缀连乘。

多项返回时按上面顺序组成元组。

线性 beta 调度，包含两个端点。返回 (betas,alphas,alpha_bars)，alpha=1-beta，alpha_bar=cumprod(alpha)，float32。

## DDIM 采样步骤

```python
def solve(xt, eps_pred, alpha_bar_t, alpha_bar_prev):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `xt` | 浮点 Tensor | `任意形状` | 时刻 t 的噪声样本。 |
| `eps_pred` | 浮点 Tensor | `与 xt 相同` | 模型预测的噪声。 |
| `alpha_bar_t` | float | `—` | 当前累计 alpha，>0。 |
| `alpha_bar_prev` | float | `—` | 前一时刻累计 alpha，<=1。 |

返回：

- `x_prev`：浮点 Tensor，`与 xt 相同`。eta=0 的确定性 DDIM 更新结果，不截断预测 x0。

单项直接返回，不包装成元组。

确定性 eta=0 DDIM。先预测 x0=(xt-sqrt(1-at)*eps)/sqrt(at)，再返回 sqrt(ap)*x0+sqrt(1-ap)*eps；不 clip x0。

## 自适应 LayerNorm Zero

```python
def solve(x, residual, shift, scale, gate, eps=1e-6):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[B, T, D]` | 待归一化激活。 |
| `residual` | 浮点 Tensor | `[B, T, D]` | 残差分支输入。 |
| `shift` | 浮点 Tensor | `[B, D]` | 每样本特征偏移。 |
| `scale` | 浮点 Tensor | `[B, D]` | 每样本特征缩放增量。 |
| `gate` | 浮点 Tensor | `[B, D]` | 残差更新的门控。 |
| `eps` | float | `—` | LayerNorm 稳定项。 |

返回：

- `output`：浮点 Tensor，`[B, T, D]`。门控调制后的残差输出；gate=0 时等于 residual。

单项直接返回，不包装成元组。

x/residual [B,T,D]，shift/scale/gate [B,D]。返回 residual+gate[:,None]*((1+scale[:,None])*LN(x)+shift[:,None])，LN 无仿射，方差 correction=0。这是已给调制参数的 AdaLN-Zero 残差核心。

## 实现 BatchNorm

```python
def solve(x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, C]` | N>1；沿 N 维统计。 |
| `weight` | 浮点 Tensor | `[C]` | 缩放。 |
| `bias` | 浮点 Tensor | `[C]` | 偏移。 |
| `running_mean` | 浮点 Tensor | `[C]` | 历史均值。 |
| `running_var` | 浮点 Tensor | `[C]` | 历史方差。 |
| `training` | bool | `—` | 是否使用 batch 统计。 |
| `momentum` | float | `—` | 新统计在 running 更新中的权重。 |
| `eps` | float | `—` | 方差稳定项。 |

返回：

- `output`：浮点 Tensor，`[N, C]`。训练归一化使用总体方差，保留梯度。
- `new_running_mean`：浮点 Tensor，`[C]`。更新后的均值；训练统计 detach，不修改输入。
- `new_running_var`：浮点 Tensor，`[C]`。更新使用无偏方差；训练统计 detach，不修改输入。

多项返回时按上面顺序组成元组。

x [N,C]，N>1。训练输出用总体方差；更新 running_var 使用无偏方差，running=(1-m)*running+m*batch。推理使用 running 统计。返回 (output,new_running_mean,new_running_var)，统计需 detach，不原地修改。

## 二维最大池化

```python
def solve(x, kernel_size, stride):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, C, H, W]` | 图像输入，无 padding。 |
| `kernel_size` | int | `—` | 正方形池化窗口边长。 |
| `stride` | int | `—` | 滑动步长。 |

返回：

- `output`：浮点 Tensor，`[N, C, Hout, Wout]`。Hout=floor((H-kernel_size)/stride)+1，Wout 同理。

单项直接返回，不包装成元组。

x [N,C,H,W]，无 padding，窗口 kernel_size、步长 stride，向下取整；禁止调用 max_pool2d。

## ViT Patch Embedding

```python
def solve(x, kernel_size, stride):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, C, H, W]` | H、W 都可被 patch 边长 p 整除。 |
| `kernel_size` | int | `—` | patch 边长 p。 |
| `stride` | int | `—` | 本题等于 kernel_size。 |

返回：

- `patches`：浮点 Tensor，`[N, (H/p)*(W/p), C*p*p]`。patch 先行后列；每个 patch 内按 C、ph、pw 展平，不含投影。

单项直接返回，不包装成元组。

本题考 patchify，不含线性投影。x [N,C,H,W]，kernel_size=stride=p，H/W 均可整除 p。输出 [N,num_patches,C*p*p]，patch 顺序先行后列，内部按 C,ph,pw 展平。

## 二维卷积

```python
def solve(x, weight, bias, stride=1, padding=0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, Cin, H, W]` | 输入图像。 |
| `weight` | 浮点 Tensor | `[Cout, Cin, Kh, Kw]` | 互相关卷积核，不翻转。 |
| `bias` | 浮点 Tensor | `[Cout]` | 输出偏移。 |
| `stride` | int | `—` | 两个空间维度使用相同步长。 |
| `padding` | int | `—` | 四边相同的零 padding。 |

返回：

- `output`：浮点 Tensor，`[N, Cout, Hout, Wout]`。Hout=floor((H+2*padding-Kh)/stride)+1，Wout 同理；groups=1。

单项直接返回，不包装成元组。

x [N,Cin,H,W]，weight [Cout,Cin,Kh,Kw]；实现互相关式 conv2d，无 dilation，groups=1。禁止调用 conv2d，可用 unfold。

## 深度可分离卷积

```python
def solve(x, depthwise, pointwise):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, C, H, W]` | 输入图像。 |
| `depthwise` | 浮点 Tensor | `[C, 1, K, K]` | K 为奇数，逐通道卷积，padding=K//2。 |
| `pointwise` | 浮点 Tensor | `[O, C, 1, 1]` | 逐点 1×1 通道混合核。 |

返回：

- `output`：浮点 Tensor，`[N, O, H, W]`。深度卷积后做逐点卷积，空间大小不变，无 bias。

单项直接返回，不包装成元组。

x [N,C,H,W]，depthwise [C,1,K,K]，pointwise [O,C,1,1]，K 为奇数。先逐通道卷积（padding=K//2），再 1x1 混合，无 bias，stride=1。不能调用 conv2d。

## 多头交叉注意力

```python
def solve(q, k, v):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, H, T, D]` | 已投影的 query。 |
| `k` | 浮点 Tensor | `[B, H, S, D]` | 已投影的 key，S 可以不等于 T。 |
| `v` | 浮点 Tensor | `[B, H, S, D]` | 已投影的 value。 |

返回：

- `output`：浮点 Tensor，`[B, H, T, D]`。非因果注意力；不合并 head 维。

单项直接返回，不包装成元组。

输入已经投影并拆头：q [B,H,T,D]，k/v [B,H,S,D]，允许 T!=S。返回 scaled dot-product attention，不使用因果 mask。

## SwiGLU MLP

```python
def solve(x, gate_weight, up_weight, down_weight):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | 输入特征。 |
| `gate_weight` | 浮点 Tensor | `[D, F]` | 门控投影。 |
| `up_weight` | 浮点 Tensor | `[D, F]` | 上投影。 |
| `down_weight` | 浮点 Tensor | `[F, D]` | 下投影。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。(SiLU(x@gate_weight)*(x@up_weight))@down_weight。

单项直接返回，不包装成元组。

x [...,D]，gate/up [D,F]，down [F,D]；返回 (SiLU(x@gate)*(x@up))@down，无 bias。

## 多元线性回归与反向传播

```python
def solve(x, y, weight, bias):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, D]` | 输入特征。 |
| `y` | 浮点 Tensor | `[N, O]` | 目标输出。 |
| `weight` | 浮点 Tensor | `[D, O]` | 线性回归权重。 |
| `bias` | 浮点 Tensor | `[O]` | 偏移。 |

返回：

- `loss`：浮点 Tensor，`[]`。所有 N*O 元素的 mean MSE。
- `dweight`：浮点 Tensor，`[D, O]`。手动推导的 weight 梯度，不用 autograd。
- `dbias`：浮点 Tensor，`[O]`。手动推导的 bias 梯度。

多项返回时按上面顺序组成元组。

x [N,D]，y [N,O]，weight [D,O]，bias [O]。返回 (mean MSE, dweight, dbias)，手动推导梯度，不能用 autograd。

## GCN 层（图卷积）

```python
def solve(x, adj, weight):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, F]` | 每个节点的输入特征。 |
| `adj` | 浮点 Tensor | `[N, N]` | adj[i,j] 表示 j 向 i 传递信息的边；输入无自环，GCN 要求对称邻接。 |
| `weight` | 浮点 Tensor | `[F, O]` | 特征变换矩阵。 |

返回：

- `output`：浮点 Tensor，`[N, O]`。按题目指定的消息与归约规则得到的节点特征；保留梯度。

单项直接返回，不包装成元组。

x [N,F]，adj [N,N]，weight [F,O]。A 是非负对称邻接矩阵，无自环。加 I，做 D^-1/2*(A+I)*D^-1/2*X*W，不加激活。

## GIN 层（图同构网络）

```python
def solve(x, adj, weight):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, F]` | 每个节点的输入特征。 |
| `adj` | 浮点 Tensor | `[N, N]` | adj[i,j] 表示 j 向 i 传递信息的边；输入无自环，GCN 要求对称邻接。 |
| `weight` | 浮点 Tensor | `[F, O]` | 特征变换矩阵。 |

返回：

- `output`：浮点 Tensor，`[N, O]`。按题目指定的消息与归约规则得到的节点特征；保留梯度。

单项直接返回，不包装成元组。

x [N,F]，adj [N,N]，weight [F,O]。A 无自环；固定 epsilon=0，线性 MLP，无激活，返回 (X+A@X)@W。

## GraphSAGE 层

```python
def solve(x, adj, weight):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, F]` | 每个节点的输入特征。 |
| `adj` | 浮点 Tensor | `[N, N]` | adj[i,j] 表示 j 向 i 传递信息的边；输入无自环，GCN 要求对称邻接。 |
| `weight` | 浮点 Tensor | `[F, O]` | 特征变换矩阵。 |

返回：

- `output`：浮点 Tensor，`[N, O]`。按题目指定的消息与归约规则得到的节点特征；保留梯度。

单项直接返回，不包装成元组。

x [N,F]，adj [N,N]，weight [F,O]。A 无自环；先对邻居取均值，孤立节点的邻居均值为零，再返回 (X+neighbor_mean)@W。此题固定使用加法融合、无激活。

## MPNN 消息传递

```python
def solve(x, adj, weight):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, F]` | 每个节点的输入特征。 |
| `adj` | 浮点 Tensor | `[N, N]` | adj[i,j] 表示 j 向 i 传递信息的边；输入无自环，GCN 要求对称邻接。 |
| `weight` | 浮点 Tensor | `[F, O]` | 特征变换矩阵。 |

返回：

- `output`：浮点 Tensor，`[N, O]`。按题目指定的消息与归约规则得到的节点特征；保留梯度。

单项直接返回，不包装成元组。

x [N,F]，adj [N,N]，weight [F,O]。每条 j→i 边的消息为 (Xj-Xi)@W；用 A[i,j] 加权求和，无自环，无更新 MLP。

## 图读出（图级池化）

```python
def solve(x, batch, num_graphs):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, F]` | 节点特征。 |
| `batch` | int64 Tensor | `[N]` | 每节点所属图的编号，范围 0..num_graphs-1。 |
| `num_graphs` | int | `—` | 图的总数量，可包含没有节点的图。 |

返回：

- `output`：浮点 Tensor，`[num_graphs, F]`。每图节点特征均值；空图输出全零。

单项直接返回，不包装成元组。

x [N,F]，batch [N] 表示每节点归属；返回每图 mean pooling [num_graphs,F]，没有节点的图输出零。

## GAT 层（图注意力）

```python
def solve(x, adj, weight, attn_src, attn_dst):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, F]` | 节点输入特征。 |
| `adj` | bool Tensor | `[N, N]` | 可连接的邻接掩码，实现时再添加自环。 |
| `weight` | 浮点 Tensor | `[F, O]` | 特征投影。 |
| `attn_src` | 浮点 Tensor | `[O]` | 接收节点 i 的注意力向量。 |
| `attn_dst` | 浮点 Tensor | `[O]` | 发送节点 j 的注意力向量。 |

返回：

- `output`：浮点 Tensor，`[N, O]`。单头 GAT 聚合输出，不含末端激活。

单项直接返回，不包装成元组。

单头 GAT：h=XW；e[i,j]=LeakyReLU(h[i]·attn_src+h[j]·attn_dst,0.2)。为 A 添加自环后按行 masked softmax，再乘 h，无输出激活。A 为 bool [N,N]。

## 图自编码器 GAE

```python
def solve(z, adjacency):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `z` | 浮点 Tensor | `[N, D]` | 给定的节点编码，不需要实现编码器。 |
| `adjacency` | 浮点 Tensor | `[N, N]` | 0/1 重构目标，包含对角线。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

本题是 Graph AutoEncoder 的重构损失，不是优势估计。已给编码 z [N,D]；内积解码 logits=z@z.T，所有 N² 元素（包含对角线）与 0/1 adjacency 做平均 BCE-with-logits。返回标量。

## 链接预测

```python
def solve(z, edges):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `z` | 浮点 Tensor | `[N, D]` | 节点编码。 |
| `edges` | int64 Tensor | `[2, E]` | 第一行是源节点，第二行是目标节点。 |

返回：

- `probabilities`：浮点 Tensor，`[E]`。每条边的 sigmoid 内积概率，按输入边的顺序返回。

单项直接返回，不包装成元组。

z [N,D]，edges [2,E] 是源/目标索引，返回 E 个 sigmoid(z[u]·z[v]) 概率，保持边顺序。

## MoE 负载均衡损失

```python
def solve(router_logits):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `router_logits` | 浮点 Tensor | `[T, E]` | 每 token 对每 expert 的未归一化分数。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

router_logits [T,E]。p=softmax(logits)，f 是 argmax top-1 路由各专家频率（不求导），P 是平均路由概率。返回 E*sum(f*P)。平分时 argmax 选最小索引。

## 多 Token 预测

```python
def solve(logits, tokens):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `logits` | 浮点 Tensor | `[K, B, T, V]` | 第 k 个 head 在 t 位置预测 t+k+1；k 从0开始。 |
| `tokens` | int64 Tensor | `[B, T]` | 目标 token ID；保证 T>K。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

logits [K,B,T,V]，tokens [B,T]。第 k（从0开始）个 head 在位置 t 预测 tokens[t+k+1]，只取仍在序列内的位置；返回所有 head 有效位置的 CE 总和/有效位置总数。保证 T>K。

## 差分注意力

```python
def solve(q1, k1, q2, k2, v, lam):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q1` | 浮点 Tensor | `[B, H, T, D]` | 第一组 query。 |
| `k1` | 浮点 Tensor | `[B, H, T, D]` | 第一组 key。 |
| `q2` | 浮点 Tensor | `[B, H, T, D]` | 第二组 query。 |
| `k2` | 浮点 Tensor | `[B, H, T, D]` | 第二组 key。 |
| `v` | 浮点 Tensor | `[B, H, T, D]` | 两组注意力共享的 value。 |
| `lam` | float | `—` | 第二组 softmax 权重的减法系数。 |

返回：

- `output`：浮点 Tensor，`[B, H, T, D]`。两组注意力之差作用到 v 的输出；没有因果 mask 或 head norm。

单项直接返回，不包装成元组。

各输入 [B,H,T,D]；返回 (softmax(q1k1ᵀ/sqrt(D))-lam*softmax(q2k2ᵀ/sqrt(D)))@v，无因果 mask。本题是差分注意力核心，不含 head norm。

## 多头潜在注意力 MLA

```python
def solve(q, latent, wk, wv):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, H, T, D]` | 已投影的 query。 |
| `latent` | 浮点 Tensor | `[B, S, R]` | 共享的低秩 KV 表示。 |
| `wk` | 浮点 Tensor | `[H, R, D]` | 每头 key 重构矩阵。 |
| `wv` | 浮点 Tensor | `[H, R, D]` | 每头 value 重构矩阵。 |

返回：

- `output`：浮点 Tensor，`[B, H, T, D]`。重构 KV 后的非因果注意力输出，不含 RoPE。

单项直接返回，不包装成元组。

考低秩 KV 重构核心，不含 decoupled RoPE。q [B,H,T,D]，latent [B,S,R]，wk/wv [H,R,D]。先各头 latent@wk/wv，再做非因果 attention，返回 [B,H,T,D]。

## 混合专家 MoE

```python
def solve(x, router_logits, weights, top_k=2):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[T, D]` | token 特征。 |
| `router_logits` | 浮点 Tensor | `[T, E]` | 专家路由分数，测试不含并列。 |
| `weights` | 浮点 Tensor | `[E, D, O]` | 每个专家的线性层权重。 |
| `top_k` | int | `—` | 每 token 选择的专家数，1<=top_k<=E。 |

返回：

- `output`：浮点 Tensor，`[T, O]`。只在选中专家 logits 内归一化，并加权聚合输出。

单项直接返回，不包装成元组。

x [T,D]，router [T,E]，weights [E,D,O]。选 logits 的 top-k 专家，仅在选中 logits 内 softmax，输出各专家线性输出的加权和 [T,O]。不含容量限制。测试不包含路由分数并列。

## Flash Attention 分块

```python
def solve(q, k, v, block_size=2):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[Q, D]` | 单头 query，无 batch 维。 |
| `k` | 浮点 Tensor | `[K, D]` | 按序列维分块处理的 key。 |
| `v` | 浮点 Tensor | `[K, V]` | 按同样块划分的 value。 |
| `block_size` | int | `—` | 正的 KV 分块大小，K 不一定可整除。 |

返回：

- `output`：浮点 Tensor，`[Q, V]`。精确非因果注意力，不构造完整 Q×K 分数矩阵。

单项直接返回，不包装成元组。

q [Q,D]，k [K,D]，v [K,V]；按 KV 块维护 running max、exp 和、加权 value 和，实现精确非因果 attention。禁止构造完整 Q×K 分数矩阵。本题考 online softmax，不要求写 CUDA kernel。

## 环形注意力（单机模拟）

```python
def solve(q, k, v, block_size=2):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[Q, D]` | 单头 query，无 batch 维。 |
| `k` | 浮点 Tensor | `[K, D]` | 按序列维分块处理的 key。 |
| `v` | 浮点 Tensor | `[K, V]` | 按同样块划分的 value。 |
| `block_size` | int | `—` | 正的 KV 分块大小，K 不一定可整除。 |

返回：

- `output`：浮点 Tensor，`[Q, V]`。精确非因果注意力，不构造完整 Q×K 分数矩阵。

单项直接返回，不包装成元组。

q [Q,D]，k [K,D]，v [K,V]；按 KV 块维护 running max、exp 和、加权 value 和，实现精确非因果 attention。禁止构造完整 Q×K 分数矩阵。本题模拟环上依次接收 KV 块的数学过程，不执行跨机通信。

## 线性自注意力

```python
def solve(q, k, v, eps=1e-6):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, T, D]` | 应用 ELU+1 特征映射的 query。 |
| `k` | 浮点 Tensor | `[B, T, D]` | 应用 ELU+1 特征映射的 key。 |
| `v` | 浮点 Tensor | `[B, T, D]` | value 不做特征映射。 |
| `eps` | float | `—` | 归一化分母的稳定项。 |

返回：

- `output`：浮点 Tensor，`[B, T, D]`。非因果线性注意力，不构造 T×T 矩阵。

单项直接返回，不包装成元组。

q,k,v [B,T,D]，特征映射 phi(x)=ELU(x)+1。非因果，返回 phi(Q)*(phi(K)ᵀV)/(phi(Q)*sum(phi(K))+eps)。不要构造 T×T 矩阵。

## KV Cache 注意力

```python
def solve(q, new_k, new_v, cached_k, cached_v):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[B, H, 1, D]` | 当前解码步的 query。 |
| `new_k` | 浮点 Tensor | `[B, H, 1, D]` | 当前新增 key。 |
| `new_v` | 浮点 Tensor | `[B, H, 1, D]` | 当前新增 value。 |
| `cached_k` | 浮点 Tensor | `[B, H, S, D]` | 历史 key，可为空 S=0。 |
| `cached_v` | 浮点 Tensor | `[B, H, S, D]` | 历史 value，与 cached_k 长度相同。 |

返回：

- `output`：浮点 Tensor，`[B, H, 1, D]`。当前 query 对所有已缓存和新增 KV 的注意力。
- `updated_k`：浮点 Tensor，`[B, H, S+1, D]`。追加后的 key，不修改原输入。
- `updated_v`：浮点 Tensor，`[B, H, S+1, D]`。追加后的 value，不修改原输入。

多项返回时按上面顺序组成元组。

单 token 解码：q/new_k/new_v [B,H,1,D]，cache [B,H,S,D] 可为空。沿序列维追加，返回 (attention_output,updated_k,updated_v)，不修改原 cache。

## 分页注意力

```python
def solve(q, key_pages, value_pages, block_table, length):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `q` | 浮点 Tensor | `[D]` | 单序列单步 query。 |
| `key_pages` | 浮点 Tensor | `[P, page_size, D]` | 物理 key 页。 |
| `value_pages` | 浮点 Tensor | `[P, page_size, D]` | 物理 value 页。 |
| `block_table` | int64 Tensor | `[num_blocks]` | 按逻辑序列顺序列出的物理页索引。 |
| `length` | int | `—` | 有效 KV token 数，>=1；末页 padding 不计入。 |

返回：

- `output`：浮点 Tensor，`[D]`。按逻辑顺序收集前 length 个 KV 后的注意力输出。

单项直接返回，不包装成元组。

单序列 q [D]，pages [P,page_size,D]，block_table [num_blocks] 是逻辑→物理页索引。收集逻辑顺序前 length 个 KV，计算 attention [D]；末页 padding 不参与。

## INT8 量化线性层

```python
def solve(x, weight):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | 输入特征。 |
| `weight` | 浮点 Tensor | `[O, D]` | 待量化权重，每输出行一个 scale。 |

返回：

- `output`：浮点 Tensor，`[..., O]`。使用反量化权重计算的线性输出。
- `q`：int8 Tensor，`[O, D]`。round 后限制到 [-127,127]，必须返回 int8。
- `scale`：浮点 Tensor，`[O, 1]`。逐行量化比例；全零行固定用 1。

多项返回时按上面顺序组成元组。

weight [O,D]，每输出通道对称量化：scale=max(abs(row))/127；全零行 scale=1。q=round(w/scale).clamp(-127,127) int8；用反量化权重算 x@w_hat.T。返回 (output,q,scale)，scale [O,1]，不考梯度。

## QLoRA 量化基座与低秩更新

```python
def solve(x, codes, codebook, scales, a, b, alpha):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | 输入特征，保留梯度。 |
| `codes` | int64 Tensor | `[O, D]` | 0..15 的给定量化码。 |
| `codebook` | 浮点 Tensor | `[16]` | 输入提供的码本，视为常量。 |
| `scales` | 浮点 Tensor | `[O, 1]` | 基座逐行缩放，视为常量。 |
| `a` | 浮点 Tensor | `[r, D]` | 可训练的低秩降维矩阵。 |
| `b` | 浮点 Tensor | `[O, r]` | 可训练的低秩升维矩阵。 |
| `alpha` | float | `—` | 低秩分支的缩放分子，实际系数 alpha/r。 |

返回：

- `output`：浮点 Tensor，`[..., O]`。量化基座输出加 LoRA 更新；只对 x/a/b 保留梯度。

单项直接返回，不包装成元组。

给定已经量化的 4-bit 基座：codes [O,D] 为 0..15 索引，codebook [16] 为输入提供的码本，scales [O,1]；W=codebook[codes]*scales。a [r,D], b [O,r]。返回 x@W.T+alpha/r*x@a.T@b.T，基座与码本冻结。此题不考 NF4 码本生成、双重量化或 CUDA。

## Mamba SSM 步骤

```python
def solve(u, delta, a, b, c, d, initial):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `u` | 浮点 Tensor | `[B, T, D]` | 序列输入。 |
| `delta` | 浮点 Tensor | `[B, T, D]` | 每步离散化步长。 |
| `a` | 浮点 Tensor | `[D, N]` | 状态转移参数。 |
| `b` | 浮点 Tensor | `[B, T, N]` | 选择性输入参数。 |
| `c` | 浮点 Tensor | `[B, T, N]` | 选择性读出参数。 |
| `d` | 浮点 Tensor | `[D]` | 输入跳连系数。 |
| `initial` | 浮点 Tensor | `[B, D, N]` | 初始状态，可以非零。 |

返回：

- `y`：浮点 Tensor，`[B, T, D]`。每步状态读出加输入跳连。
- `final_h`：浮点 Tensor，`[B, D, N]`。最后一个时间步之后的状态。

多项返回时按上面顺序组成元组。

u/delta [B,T,D]，a [D,N]，b/c [B,T,N]，d [D]，initial [B,D,N]。递推 h=exp(delta*a)*h+delta*b*u；y=sum(c*h,N)+d*u。返回 (y [B,T,D],final_h)。考 selective scan 核心，采用 delta*B 输入离散化。

## 张量并行 MLP（单机模拟）

```python
def solve(x, up_shards, down_shards):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | 输入特征。 |
| `up_shards` | list[Tensor] | `第 i 项 [D, Fi]` | 列并行上投影，分片宽度可以不同。 |
| `down_shards` | list[Tensor] | `第 i 项 [Fi, O]` | 与 up_shards 一一配对的下投影。 |

返回：

- `output`：浮点 Tensor，`[..., O]`。各分片 ReLU(x@up_i)@down_i 的和。

单项直接返回，不包装成元组。

列并行 up shard [D,Fi]，行并行 down shard [Fi,O]。返回 sum(ReLU(x@up_i)@down_i)。在单机模拟通信求和，不启动进程组，无 bias。

## FSDP 训练步骤（单机模拟）

```python
def solve(param_shards, rank_grads, lr):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `param_shards` | list[Tensor] | `第 i 项 [Si]` | 一维参数分片，完整长度 S=sum(Si)。 |
| `rank_grads` | list[Tensor] | `每项 [S]` | 每个 rank 对完整参数的梯度。 |
| `lr` | float | `—` | SGD 学习率。 |

返回：

- `updated_shards`：list[Tensor]，`与 param_shards 一一对应`。先跨 rank 平均完整梯度，再按原长度切片更新；不修改输入。

单项直接返回，不包装成元组。

模拟 all-gather 参数与 reduce-scatter 平均梯度后 SGD。param_shards 为一维分片列表；rank_grads 为每 rank 的完整一维梯度（均为全参数长度）。平均梯度后按 shard 长度拆分，返回更新后的分片列表。不执行真实分布式通信。

## 混合精度训练步骤

```python
def solve(param, scaled_grad, loss_scale, lr):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `param` | 浮点 Tensor | `任意形状` | float32 参数，不原地修改。 |
| `scaled_grad` | 浮点 Tensor | `与 param 相同` | 经过 loss scaling 的梯度，可能包含 NaN/Inf。 |
| `loss_scale` | float | `—` | 正的 loss scaling 系数。 |
| `lr` | float | `—` | 学习率。 |

返回：

- `new_param`：浮点 Tensor，`与 param 相同`。有限梯度时更新，否则复制原参数。
- `updated`：bool，`—`。更新成功返回 True，出现非有限梯度并跳过返回 False。

多项返回时按上面顺序组成元组。

考 loss scaling 的更新核心，输入 float32。不启动 GPU。先 unscale；若梯度存在 NaN/Inf 则跳过并返回 (param.clone(),False)，否则返回 (param-lr*grad/loss_scale,True)。不原地修改。

## 激活检查点

```python
def solve(x, w1, w2):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | MLP 输入；可能不需要梯度。 |
| `w1` | 浮点 Tensor | `[D, F]` | 第一层可训练权重。 |
| `w2` | 浮点 Tensor | `[F, O]` | 第二层可训练权重。 |

返回：

- `output`：浮点 Tensor，`[..., O]`。checkpoint 包裹的 ReLU(x@w1)@w2，保留参数梯度。

单项直接返回，不包装成元组。

实现 checkpointed 两层 MLP，返回 ReLU(x@w1)@w2。必须调用 torch.utils.checkpoint.checkpoint，use_reentrant=False；保留反向梯度。本题允许使用该机制 API，重点理解重算。

## GPT-2 Transformer Block

```python
def solve(x, wqkv, wo, w1, w2, heads, eps=1e-5):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[B, T, D]` | block 输入。 |
| `wqkv` | 浮点 Tensor | `[D, 3D]` | 联合 QKV 投影，按最后一维分成 q/k/v。 |
| `wo` | 浮点 Tensor | `[D, D]` | 注意力输出投影。 |
| `w1` | 浮点 Tensor | `[D, F]` | MLP 上投影。 |
| `w2` | 浮点 Tensor | `[F, D]` | MLP 下投影。 |
| `heads` | int | `—` | head 数，必须整除 D。 |
| `eps` | float | `—` | 两处无仿射 LayerNorm 的稳定项。 |

返回：

- `output`：浮点 Tensor，`[B, T, D]`。pre-LN 双残差 block 输出；GPT-2 因果，ViT 双向。

单项直接返回，不包装成元组。

实现 pre-LN block，无 bias/dropout，LN 无仿射：a=x+Attention(LN(x))；返回 a+GELU(LN(a)@w1)@w2。wqkv [D,3D]，wo [D,D]，w1 [D,F]，w2 [F,D]，GELU 用精确 erf。attention 使用因果 mask。

## ViT Transformer Block

```python
def solve(x, wqkv, wo, w1, w2, heads, eps=1e-5):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[B, T, D]` | block 输入。 |
| `wqkv` | 浮点 Tensor | `[D, 3D]` | 联合 QKV 投影，按最后一维分成 q/k/v。 |
| `wo` | 浮点 Tensor | `[D, D]` | 注意力输出投影。 |
| `w1` | 浮点 Tensor | `[D, F]` | MLP 上投影。 |
| `w2` | 浮点 Tensor | `[F, D]` | MLP 下投影。 |
| `heads` | int | `—` | head 数，必须整除 D。 |
| `eps` | float | `—` | 两处无仿射 LayerNorm 的稳定项。 |

返回：

- `output`：浮点 Tensor，`[B, T, D]`。pre-LN 双残差 block 输出；GPT-2 因果，ViT 双向。

单项直接返回，不包装成元组。

实现 pre-LN block，无 bias/dropout，LN 无仿射：a=x+Attention(LN(x))；返回 a+GELU(LN(a)@w1)@w2。wqkv [D,3D]，wo [D,D]，w1 [D,F]，w2 [F,D]，GELU 用精确 erf。attention 双向，无因果 mask。

## Top-k / Top-p 采样分布

```python
def solve(logits, top_k, top_p, temperature=1.0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `logits` | 浮点 Tensor | `[V]` | 一维词表分数，测试不含并列。 |
| `top_k` | int | `—` | 0 表示不做 top-k；否则 1..V。 |
| `top_p` | float | `—` | 累计概率阈值，0<top_p<=1。 |
| `temperature` | float | `—` | 严格正的温度。 |

返回：

- `probabilities`：浮点 Tensor，`[V]`。原词表顺序的归一化概率，被过滤项为0；返回分布，不采样 token。

单项直接返回，不包装成元组。

输入一维 logits。先除温度，再保留 top_k（0 表示不裁剪）；在剩余归一化分布上做 nucleus，保留首次使累计概率达到/超过 top_p 的 token；返回原词表顺序的归一化概率。本题不随机采样。分数不并列，0<top_p<=1。

## 束搜索解码

```python
def solve(log_probs, beam_size):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `log_probs` | 浮点 Tensor | `[T, V]` | 给定的每步 log probability；本题分布与历史无关。 |
| `beam_size` | int | `—` | 每步保留的最大候选数，正整数。 |

返回：

- `beams`：list[tuple[list[int], float]]，`最多 beam_size 项`。每项为 (token 序列, 累计 log 分数)；分数降序，并列时序列字典序升序。

单项直接返回，不包装成元组。

log_probs [T,V] 为给定的逐步 log 概率（与历史无关的简化模型）。从空序列开始，每步展开并保留累计分数最高 beam_size 个，返回 [(token_list, score)]。并列按 token 序列字典序升序。本题无 EOS/长度惩罚。

## 推测解码接受步骤

```python
def solve(draft_tokens, draft_probs, target_probs, uniforms):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `draft_tokens` | int64 Tensor | `[K]` | 草稿 token ID。 |
| `draft_probs` | 浮点 Tensor | `[K, V]` | 草稿模型的已归一化概率，候选 token 概率>0。 |
| `target_probs` | 浮点 Tensor | `[K, V]` | 目标模型的已归一化概率。 |
| `uniforms` | 浮点 Tensor | `[K]` | 提供的 [0,1) 随机数，不要重新采样。 |

返回：

- `accepted`：list[int]，`长度 0..K`。从前向后已接受的草稿 token。
- `residual`：Tensor 或 None，`[V] 或 —`。首次拒绝时返回 normalize(max(p-q,0))；全部接受返回 None。

多项返回时按上面顺序组成元组。

实现单次 speculative acceptance 核心。draft_tokens [K]；draft/target_probs [K,V]，uniforms [K]。逐步接受条件 u<min(1,p[token]/q[token])。首次拒绝返回 (已接受 token 列表, normalize(max(p-q,0)))；全部接受返回 (全部 token,None)。本题不采样替换 token/额外 token。q 对候选 token 严格正；拒绝时剩余分布和>0。

## 蒙特卡洛树搜索 PUCT 选择

```python
def solve(priors, value_sums, visits, parent_visits, c_puct=1.0):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `priors` | 浮点 Tensor | `[A]` | 各候选动作的先验概率。 |
| `value_sums` | 浮点 Tensor | `[A]` | 各动作累计价值 W。 |
| `visits` | 整数 Tensor | `[A]` | 各动作访问次数 N，允许0。 |
| `parent_visits` | int | `—` | 父节点访问次数。 |
| `c_puct` | float | `—` | 探索项系数。 |

返回：

- `action_index`：int，`—`。PUCT 最大动作的 Python 整数下标，并列取最小下标。

单项直接返回，不包装成元组。

考一次 MCTS selection：Q=W/N，未访问 Q=0；U=c*P*sqrt(parent_visits)/(1+N)。返回 argmax(Q+U) 的 Python int，并列取最小下标。不模拟环境 rollout。

## 字节对编码 BPE

```python
def solve(words, num_merges):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `words` | dict[str, int] | `—` | 词到正整数频次的映射；可为空。 |
| `num_merges` | int | `—` | 最多合并次数，非负整数。 |

返回：

- `merges`：list[tuple[str, str]]，`最多 num_merges 项`。按执行顺序记录每次合并的 token pair。
- `tokenized`：dict[str, list[str]]，`与 words 的键相同`。每个原始词经过合并后的 token 列表。

多项返回时按上面顺序组成元组。

words 是 {字符串:正整数词频}。初始化为字符列表（本题不添加词尾标记），迭代统计每个词内部相邻 token pair 的加权频次，合并最多 pair；并列按 pair 的 Python 字典序。从左到右非重叠合并，直到无 pair 或达到次数。返回 (merges, tokenized_dict)，merges 是 pair 元组列表。

## K-means 聚类

```python
def solve(x, initial_centers, iterations):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[N, D]` | 待聚类数据。 |
| `initial_centers` | 浮点 Tensor | `[K, D]` | 指定的初始中心，不要随机初始化。 |
| `iterations` | int | `—` | 恰好执行的 Lloyd 更新次数，允许0。 |

返回：

- `centers`：浮点 Tensor，`[K, D]`。更新后的中心；空簇保留原中心。
- `labels`：int64 Tensor，`[N]`。根据最终中心重新分配；并列最小索引。

多项返回时按上面顺序组成元组。

x [N,D]，初始中心 [K,D]。恰好执行 iterations 次 Lloyd 更新：最近中心（平方欧氏距，并列最小索引），再求每簇均值。空簇保留原中心。最后用更新完的中心重算 labels。返回 (centers,labels)，不考 autograd。

## 快速排序（递归）

```python
def solve(nums):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `nums` | list[int] | `任意长度` | 可为空，包含负数或重复元素；不修改输入。 |

返回：

- `sorted_nums`：list[int]，`与 nums 等长`。升序新列表，保留重复值。

单项直接返回，不包装成元组。

返回升序新列表，保留重复值，不修改输入。禁止 sorted/list.sort。练习递归分治。

## 快速排序（非递归）

```python
def solve(nums):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `nums` | list[int] | `任意长度` | 可为空，包含负数或重复元素；不修改输入。 |

返回：

- `sorted_nums`：list[int]，`与 nums 等长`。升序新列表，保留重复值。

单项直接返回，不包装成元组。

返回升序新列表，保留重复值，不修改输入。禁止 sorted/list.sort。必须使用显式栈，不递归。

## 拓扑排序

```python
def solve(num_nodes, edges):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `num_nodes` | int | `—` | 节点编号 0..num_nodes-1，允许0。 |
| `edges` | list[tuple[int, int]] | `任意长度` | (u,v) 表示 u→v，无重复边。 |

返回：

- `order`：list[int]，`num_nodes 项或 []`。字典序最小的拓扑序，含孤立节点；存在环返回 []。

单项直接返回，不包装成元组。

节点 0..n-1，edges 为 (u,v) 有向边列表，无重复边。返回字典序最小的合法拓扑序；存在环返回 []。需包含孤立节点。

## Top-k 最大元素

```python
def solve(nums, k):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `nums` | list[int] | `任意长度` | 允许负数和重复元素。 |
| `k` | int | `—` | 0<=k<=len(nums)。 |

返回：

- `largest`：list[int]，`k 项`。最大的 k 个元素，降序排列，保留重复。

单项直接返回，不包装成元组。

0<=k<=len(nums)，返回降序的 k 个最大元素，保留重复；禁止对全数组排序、torch.topk。可用大小 k 的堆。

## 模拟马走日

```python
def solve(rows, cols, start, target, blocked):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `rows` | int | `—` | 棋盘行数，正整数。 |
| `cols` | int | `—` | 棋盘列数，正整数。 |
| `start` | tuple[int, int] | `(row, col)` | 0-based 起点，位于棋盘内。 |
| `target` | tuple[int, int] | `(row, col)` | 0-based 终点，位于棋盘内。 |
| `blocked` | list[tuple[int, int]] | `任意长度` | 不可落点坐标，允许为空。 |

返回：

- `min_steps`：int，`—`。最少步数；不可达或起终点被阻挡返回 -1；合法同点返回0。

单项直接返回，不包装成元组。

将“马走日”具体设为国际象棋骑士最短路（无蹩马腿规则）：棋盘 rows×cols，坐标从0开始。每步 (±1,±2)/(±2,±1)，blocked 是不可落点坐标列表。返回最少步数，不可达或起终点 blocked 返回 -1。

## 张量变换与多头重排

```python
def solve(x, heads):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[B, T, D]` | 可能不是连续张量，D 可被 heads 整除。 |
| `heads` | int | `—` | head 数 H，正整数。 |

返回：

- `split_heads`：浮点 Tensor，`[B, H, T, D/H]`。拆头并交换时间与头维度，保留梯度。
- `merged`：浮点 Tensor，`[B, T, D]`。合并还原的张量，值与输入一致。

多项返回时按上面顺序组成元组。

x [B,T,D]，D 可被 heads 整除。将最后一维拆成 H 个 head，并交换为 [B,H,T,D/H]；再合并还原。返回 (split_heads,merged)。允许使用 view/reshape/transpose 或 einops，不修改输入。

## FFN 前馈网络

```python
def solve(x, w1, b1, w2, b2):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `x` | 浮点 Tensor | `[..., D]` | 输入。 |
| `w1` | 浮点 Tensor | `[D, F]` | 上投影。 |
| `b1` | 浮点 Tensor | `[F]` | 上投影偏移。 |
| `w2` | 浮点 Tensor | `[F, D]` | 下投影。 |
| `b2` | 浮点 Tensor | `[D]` | 输出偏移。 |

返回：

- `output`：浮点 Tensor，`与 x 相同`。精确 GELU 的双线性层输出，保留梯度，不包含残差。

单项直接返回，不包装成元组。

实现普通 Transformer FFN：GELU(x@w1+b1)@w2+b2。x [...,D]，w1 [D,F]，b1 [F]，w2 [F,D]，b2 [D]。GELU 使用精确 erf 形式，不含残差/归一化/dropout。

## Entropy Loss 策略熵

```python
def solve(logits, mask):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `logits` | 浮点 Tensor | `[B, T, V]` | 未归一化的词表分数。 |
| `mask` | bool Tensor | `[B, T]` | 至少一个 True；仅在有效位置平均。 |

返回：

- `entropy`：浮点 Tensor，`[]`。正号平均熵，零维可导张量；不是负熵正则 loss。

单项直接返回，不包装成元组。

logits [B,T,V]，mask [B,T] 为 bool，至少一个有效位置。对每个有效位置计算 H=-sum_v p_v*log(p_v)，返回有效位置的平均熵（正号）。如果训练目标要鼓励探索，最小化 loss 时应减去该值；此函数不取负号。使用稳定 log_softmax，可调用 torch.log_softmax。

## SFT Loss 因果语言模型

```python
def solve(logits, labels, ignore_index=-100):
    ...
```

| 参数 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `logits` | 浮点 Tensor | `[B, T, V]` | t 位置预测 t+1 标签，T>=2。 |
| `labels` | int64 Tensor | `[B, T]` | 目标 token ID，或 ignore_index；shift 后至少一个有效位置。 |
| `ignore_index` | int | `—` | 屏蔽 prompt/padding 的特殊标签值。 |

返回：

- `loss`：浮点 Tensor，`[]`。零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。

单项直接返回，不包装成元组。

logits [B,T,V]，labels [B,T] int64。logits[:,t] 预测 labels[:,t+1]；丢掉最后一个预测与第一个标签。labels 等于 ignore_index 的位置不计 loss，常用于 prompt 和 padding。返回所有有效目标 token 的平均交叉熵，不是每条序列平均。T>=2，shift 后至少一个有效目标。

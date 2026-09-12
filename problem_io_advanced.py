def extend(describe,tensor,scalar,loss):
    describe('gcn_layer gin_layer graphsage_layer mpnn_layer',[
        tensor('x','[N, F]','每个节点的输入特征。'),tensor('adj','[N, N]','adj[i,j] 表示 j 向 i 传递信息的边；输入无自环，GCN 要求对称邻接。'),tensor('weight','[F, O]','特征变换矩阵。')],
        [tensor('output','[N, O]','按题目指定的消息与归约规则得到的节点特征；保留梯度。')])
    describe('graph_readout',[tensor('x','[N, F]','节点特征。'),tensor('batch','[N]','每节点所属图的编号，范围 0..num_graphs-1。','int64 Tensor'),scalar('num_graphs','图的总数量，可包含没有节点的图。','int')],
        [tensor('output','[num_graphs, F]','每图节点特征均值；空图输出全零。')])
    describe('gat_layer',[tensor('x','[N, F]','节点输入特征。'),tensor('adj','[N, N]','可连接的邻接掩码，实现时再添加自环。','bool Tensor'),tensor('weight','[F, O]','特征投影。'),tensor('attn_src','[O]','接收节点 i 的注意力向量。'),tensor('attn_dst','[O]','发送节点 j 的注意力向量。')],
        [tensor('output','[N, O]','单头 GAT 聚合输出，不含末端激活。')])
    describe('gae',[tensor('z','[N, D]','给定的节点编码，不需要实现编码器。'),tensor('adjacency','[N, N]','0/1 重构目标，包含对角线。')],loss)
    describe('link_prediction',[tensor('z','[N, D]','节点编码。'),tensor('edges','[2, E]','第一行是源节点，第二行是目标节点。','int64 Tensor')],[tensor('probabilities','[E]','每条边的 sigmoid 内积概率，按输入边的顺序返回。')])
    describe('moe_load_balance',[tensor('router_logits','[T, E]','每 token 对每 expert 的未归一化分数。')],loss)
    describe('multi_token_prediction',[tensor('logits','[K, B, T, V]','第 k 个 head 在 t 位置预测 t+k+1；k 从0开始。'),tensor('tokens','[B, T]','目标 token ID；保证 T>K。','int64 Tensor')],loss)
    describe('diff_attention',[tensor(n,'[B, H, T, D]',d) for n,d in [('q1','第一组 query。'),('k1','第一组 key。'),('q2','第二组 query。'),('k2','第二组 key。'),('v','两组注意力共享的 value。')]]+[scalar('lam','第二组 softmax 权重的减法系数。')],
        [tensor('output','[B, H, T, D]','两组注意力之差作用到 v 的输出；没有因果 mask 或 head norm。')])
    describe('mla',[tensor('q','[B, H, T, D]','已投影的 query。'),tensor('latent','[B, S, R]','共享的低秩 KV 表示。'),tensor('wk','[H, R, D]','每头 key 重构矩阵。'),tensor('wv','[H, R, D]','每头 value 重构矩阵。')],
        [tensor('output','[B, H, T, D]','重构 KV 后的非因果注意力输出，不含 RoPE。')])
    describe('moe',[tensor('x','[T, D]','token 特征。'),tensor('router_logits','[T, E]','专家路由分数，测试不含并列。'),tensor('weights','[E, D, O]','每个专家的线性层权重。'),scalar('top_k','每 token 选择的专家数，1<=top_k<=E。','int')],
        [tensor('output','[T, O]','只在选中专家 logits 内归一化，并加权聚合输出。')])
    describe('flash_attention ring_attention',[tensor('q','[Q, D]','单头 query，无 batch 维。'),tensor('k','[K, D]','按序列维分块处理的 key。'),tensor('v','[K, V]','按同样块划分的 value。'),scalar('block_size','正的 KV 分块大小，K 不一定可整除。','int')],
        [tensor('output','[Q, V]','精确非因果注意力，不构造完整 Q×K 分数矩阵。')])
    describe('linear_attention',[tensor('q','[B, T, D]','应用 ELU+1 特征映射的 query。'),tensor('k','[B, T, D]','应用 ELU+1 特征映射的 key。'),tensor('v','[B, T, D]','value 不做特征映射。'),scalar('eps','归一化分母的稳定项。')],
        [tensor('output','[B, T, D]','非因果线性注意力，不构造 T×T 矩阵。')])
    describe('kv_cache',[tensor(n,'[B, H, 1, D]',d) for n,d in [('q','当前解码步的 query。'),('new_k','当前新增 key。'),('new_v','当前新增 value。')]]+[tensor('cached_k','[B, H, S, D]','历史 key，可为空 S=0。'),tensor('cached_v','[B, H, S, D]','历史 value，与 cached_k 长度相同。')],
        [tensor('output','[B, H, 1, D]','当前 query 对所有已缓存和新增 KV 的注意力。'),tensor('updated_k','[B, H, S+1, D]','追加后的 key，不修改原输入。'),tensor('updated_v','[B, H, S+1, D]','追加后的 value，不修改原输入。')])
    describe('paged_attention',[tensor('q','[D]','单序列单步 query。'),tensor('key_pages','[P, page_size, D]','物理 key 页。'),tensor('value_pages','[P, page_size, D]','物理 value 页。'),tensor('block_table','[num_blocks]','按逻辑序列顺序列出的物理页索引。','int64 Tensor'),scalar('length','有效 KV token 数，>=1；末页 padding 不计入。','int')],
        [tensor('output','[D]','按逻辑顺序收集前 length 个 KV 后的注意力输出。')])
    describe('int8_quantization',[tensor('x','[..., D]','输入特征。'),tensor('weight','[O, D]','待量化权重，每输出行一个 scale。')],
        [tensor('output','[..., O]','使用反量化权重计算的线性输出。'),tensor('q','[O, D]','round 后限制到 [-127,127]，必须返回 int8。','int8 Tensor'),tensor('scale','[O, 1]','逐行量化比例；全零行固定用 1。')])
    describe('qlora',[tensor('x','[..., D]','输入特征，保留梯度。'),tensor('codes','[O, D]','0..15 的给定量化码。','int64 Tensor'),tensor('codebook','[16]','输入提供的码本，视为常量。'),tensor('scales','[O, 1]','基座逐行缩放，视为常量。'),tensor('a','[r, D]','可训练的低秩降维矩阵。'),tensor('b','[O, r]','可训练的低秩升维矩阵。'),scalar('alpha','低秩分支的缩放分子，实际系数 alpha/r。')],
        [tensor('output','[..., O]','量化基座输出加 LoRA 更新；只对 x/a/b 保留梯度。')])
    describe('mamba_ssm',[tensor('u','[B, T, D]','序列输入。'),tensor('delta','[B, T, D]','每步离散化步长。'),tensor('a','[D, N]','状态转移参数。'),tensor('b','[B, T, N]','选择性输入参数。'),tensor('c','[B, T, N]','选择性读出参数。'),tensor('d','[D]','输入跳连系数。'),tensor('initial','[B, D, N]','初始状态，可以非零。')],
        [tensor('y','[B, T, D]','每步状态读出加输入跳连。'),tensor('final_h','[B, D, N]','最后一个时间步之后的状态。')])
    describe('tensor_parallel',[tensor('x','[..., D]','输入特征。'),('up_shards','list[Tensor]','第 i 项 [D, Fi]','列并行上投影，分片宽度可以不同。'),('down_shards','list[Tensor]','第 i 项 [Fi, O]','与 up_shards 一一配对的下投影。')],
        [tensor('output','[..., O]','各分片 ReLU(x@up_i)@down_i 的和。')])
    describe('fsdp_step',[('param_shards','list[Tensor]','第 i 项 [Si]','一维参数分片，完整长度 S=sum(Si)。'),('rank_grads','list[Tensor]','每项 [S]','每个 rank 对完整参数的梯度。'),scalar('lr','SGD 学习率。')],
        [('updated_shards','list[Tensor]','与 param_shards 一一对应','先跨 rank 平均完整梯度，再按原长度切片更新；不修改输入。')])
    describe('mixed_precision',[tensor('param','任意形状','float32 参数，不原地修改。'),tensor('scaled_grad','与 param 相同','经过 loss scaling 的梯度，可能包含 NaN/Inf。'),scalar('loss_scale','正的 loss scaling 系数。'),scalar('lr','学习率。')],
        [tensor('new_param','与 param 相同','有限梯度时更新，否则复制原参数。'),('updated','bool','—','更新成功返回 True，出现非有限梯度并跳过返回 False。')])
    describe('activation_checkpointing',[tensor('x','[..., D]','MLP 输入；可能不需要梯度。'),tensor('w1','[D, F]','第一层可训练权重。'),tensor('w2','[F, O]','第二层可训练权重。')],
        [tensor('output','[..., O]','checkpoint 包裹的 ReLU(x@w1)@w2，保留参数梯度。')])
    describe('gpt2_block vit_block',[tensor('x','[B, T, D]','block 输入。'),tensor('wqkv','[D, 3D]','联合 QKV 投影，按最后一维分成 q/k/v。'),tensor('wo','[D, D]','注意力输出投影。'),tensor('w1','[D, F]','MLP 上投影。'),tensor('w2','[F, D]','MLP 下投影。'),scalar('heads','head 数，必须整除 D。','int'),scalar('eps','两处无仿射 LayerNorm 的稳定项。')],
        [tensor('output','[B, T, D]','pre-LN 双残差 block 输出；GPT-2 因果，ViT 双向。')])
    describe('topk_sampling',[tensor('logits','[V]','一维词表分数，测试不含并列。'),scalar('top_k','0 表示不做 top-k；否则 1..V。','int'),scalar('top_p','累计概率阈值，0<top_p<=1。'),scalar('temperature','严格正的温度。')],
        [tensor('probabilities','[V]','原词表顺序的归一化概率，被过滤项为0；返回分布，不采样 token。')])
    describe('beam_search',[tensor('log_probs','[T, V]','给定的每步 log probability；本题分布与历史无关。'),scalar('beam_size','每步保留的最大候选数，正整数。','int')],
        [('beams','list[tuple[list[int], float]]','最多 beam_size 项','每项为 (token 序列, 累计 log 分数)；分数降序，并列时序列字典序升序。')])
    describe('speculative_decoding',[tensor('draft_tokens','[K]','草稿 token ID。','int64 Tensor'),tensor('draft_probs','[K, V]','草稿模型的已归一化概率，候选 token 概率>0。'),tensor('target_probs','[K, V]','目标模型的已归一化概率。'),tensor('uniforms','[K]','提供的 [0,1) 随机数，不要重新采样。')],
        [('accepted','list[int]','长度 0..K','从前向后已接受的草稿 token。'),('residual','Tensor 或 None','[V] 或 —','首次拒绝时返回 normalize(max(p-q,0))；全部接受返回 None。')])
    describe('mcts_search',[tensor('priors','[A]','各候选动作的先验概率。'),tensor('value_sums','[A]','各动作累计价值 W。'),tensor('visits','[A]','各动作访问次数 N，允许0。','整数 Tensor'),scalar('parent_visits','父节点访问次数。','int'),scalar('c_puct','探索项系数。')],
        [('action_index','int','—','PUCT 最大动作的 Python 整数下标，并列取最小下标。')])
    describe('bpe',[('words','dict[str, int]','—','词到正整数频次的映射；可为空。'),scalar('num_merges','最多合并次数，非负整数。','int')],
        [('merges','list[tuple[str, str]]','最多 num_merges 项','按执行顺序记录每次合并的 token pair。'),('tokenized','dict[str, list[str]]','与 words 的键相同','每个原始词经过合并后的 token 列表。')])
    describe('kmeans',[tensor('x','[N, D]','待聚类数据。'),tensor('initial_centers','[K, D]','指定的初始中心，不要随机初始化。'),scalar('iterations','恰好执行的 Lloyd 更新次数，允许0。','int')],
        [tensor('centers','[K, D]','更新后的中心；空簇保留原中心。'),tensor('labels','[N]','根据最终中心重新分配；并列最小索引。','int64 Tensor')])
    describe('quicksort_recursive quicksort_iterative',[('nums','list[int]','任意长度','可为空，包含负数或重复元素；不修改输入。')],
        [('sorted_nums','list[int]','与 nums 等长','升序新列表，保留重复值。')])
    describe('topo',[scalar('num_nodes','节点编号 0..num_nodes-1，允许0。','int'),('edges','list[tuple[int, int]]','任意长度','(u,v) 表示 u→v，无重复边。')],
        [('order','list[int]','num_nodes 项或 []','字典序最小的拓扑序，含孤立节点；存在环返回 []。')])
    describe('topk',[('nums','list[int]','任意长度','允许负数和重复元素。'),scalar('k','0<=k<=len(nums)。','int')],
        [('largest','list[int]','k 项','最大的 k 个元素，降序排列，保留重复。')])
    describe('knight',[scalar('rows','棋盘行数，正整数。','int'),scalar('cols','棋盘列数，正整数。','int'),('start','tuple[int, int]','(row, col)','0-based 起点，位于棋盘内。'),('target','tuple[int, int]','(row, col)','0-based 终点，位于棋盘内。'),('blocked','list[tuple[int, int]]','任意长度','不可落点坐标，允许为空。')],
        [('min_steps','int','—','最少步数；不可达或起终点被阻挡返回 -1；合法同点返回0。')])

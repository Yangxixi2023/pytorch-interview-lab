def populate(add):
    graphcases=[('含孤立节点',"a=torch.tensor([[0.,1.,0.,0.],[1.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]]); args=(torch.randn(4,3,requires_grad=True),a,torch.randn(3,2,requires_grad=True))"),('稠密图',"args=(torch.randn(3,2,requires_grad=True),torch.ones(3,3)-torch.eye(3),torch.randn(2,4,requires_grad=True))")]
    for slug,title,desc,body in [
        ('gcn_layer','GCN 层（图卷积）','A 是非负对称邻接矩阵，无自环。加 I，做 D^-1/2*(A+I)*D^-1/2*X*W，不加激活。','a=adj+torch.eye(len(adj),device=adj.device)\nr=a.sum(-1).rsqrt()\nreturn (r[:,None]*a*r[None,:])@x@weight'),
        ('gin_layer','GIN 层（图同构网络）','A 无自环；固定 epsilon=0，线性 MLP，无激活，返回 (X+A@X)@W。','return (x+adj@x)@weight'),
        ('graphsage_layer','GraphSAGE 层','A 无自环；先对邻居取均值，孤立节点的邻居均值为零，再返回 (X+neighbor_mean)@W。此题固定使用加法融合、无激活。','degree=adj.sum(-1,keepdim=True)\nneighbor=(adj@x)/degree.clamp_min(1)\nreturn (x+neighbor)@weight'),
        ('mpnn_layer','MPNN 消息传递','每条 j→i 边的消息为 (Xj-Xi)@W；用 A[i,j] 加权求和，无自环，无更新 MLP。','messages=x@weight\nreturn adj@messages-adj.sum(-1,keepdim=True)*messages')]:
        add(slug,title,'图神经网络','中等','x, adj, weight','x [N,F]，adj [N,N]，weight [F,O]。'+desc,body,graphcases,gradient=True)
    add('graph_readout','图读出（图级池化）','图神经网络','简单','x, batch, num_graphs',
        'x [N,F]，batch [N] 表示每节点归属；返回每图 mean pooling [num_graphs,F]，没有节点的图输出零。',
        'out=x.new_zeros(num_graphs,x.shape[-1]).index_add(0,batch,x)\ncount=torch.bincount(batch,minlength=num_graphs).clamp_min(1)\nreturn out/count[:,None]',
        [('含空图',"args=(torch.randn(5,3,requires_grad=True),torch.tensor([0,2,0,2,2]),4)"),('单图',"args=(torch.randn(3,2,requires_grad=True),torch.zeros(3,dtype=torch.long),1)")],gradient=True)
    add('gat_layer','GAT 层（图注意力）','图神经网络','中等','x, adj, weight, attn_src, attn_dst',
        '单头 GAT：h=XW；e[i,j]=LeakyReLU(h[i]·attn_src+h[j]·attn_dst,0.2)。为 A 添加自环后按行 masked softmax，再乘 h，无输出激活。A 为 bool [N,N]。',
        'h=x@weight\ne=F.leaky_relu((h@attn_src)[:,None]+(h@attn_dst)[None,:],0.2)\nmask=adj|torch.eye(len(adj),device=adj.device,dtype=torch.bool)\nreturn e.masked_fill(~mask,float("-inf")).softmax(-1)@h',
        [('含孤立点',"args=(torch.randn(4,3,requires_grad=True),torch.zeros(4,4,dtype=torch.bool),torch.randn(3,2,requires_grad=True),torch.randn(2,requires_grad=True),torch.randn(2,requires_grad=True))"),('有向图',"args=(torch.randn(3,2,requires_grad=True),torch.tensor([[0,1,0],[0,0,1],[1,0,0]],dtype=torch.bool),torch.randn(2,3,requires_grad=True),torch.randn(3,requires_grad=True),torch.randn(3,requires_grad=True))")],gradient=True)
    add('gae','图自编码器 GAE','图神经网络','困难','z, adjacency',
        '本题是 Graph AutoEncoder 的重构损失，不是优势估计。已给编码 z [N,D]；内积解码 logits=z@z.T，所有 N² 元素（包含对角线）与 0/1 adjacency 做平均 BCE-with-logits。返回标量。',
        'logits=z@z.T\nreturn (F.softplus(logits)-adjacency*logits).mean()',
        [('稠密重构',"args=(torch.randn(4,3,requires_grad=True),torch.eye(4))"),('大幅嵌入',"args=(torch.randn(3,2,requires_grad=True)*20,torch.ones(3,3))")],gradient=True)
    add('link_prediction','链接预测','图神经网络','困难','z, edges',
        'z [N,D]，edges [2,E] 是源/目标索引，返回 E 个 sigmoid(z[u]·z[v]) 概率，保持边顺序。',
        'return (z[edges[0]]*z[edges[1]]).sum(-1).sigmoid()',
        [('重复边',"args=(torch.randn(4,3,requires_grad=True),torch.tensor([[0,1,0],[1,2,1]]))"),('自环',"args=(torch.randn(2,2,requires_grad=True),torch.tensor([[0,1],[0,1]]))")],gradient=True)
    add('moe_load_balance','MoE 负载均衡损失','LLM 前沿架构','中等','router_logits',
        'router_logits [T,E]。p=softmax(logits)，f 是 argmax top-1 路由各专家频率（不求导），P 是平均路由概率。返回 E*sum(f*P)。平分时 argmax 选最小索引。',
        'p=router_logits.softmax(-1)\ne=p.shape[-1]\nf=F.one_hot(p.argmax(-1),e).to(p.dtype).mean(0)\nreturn e*(f*p.mean(0)).sum()',
        [('路由偏斜',"args=(torch.randn(8,3,requires_grad=True),)"),('均匀概率',"args=(torch.zeros(4,2,requires_grad=True),)")],gradient=True)
    add('multi_token_prediction','多 Token 预测','LLM 前沿架构','中等','logits, tokens',
        'logits [K,B,T,V]，tokens [B,T]。第 k（从0开始）个 head 在位置 t 预测 tokens[t+k+1]，只取仍在序列内的位置；返回所有 head 有效位置的 CE 总和/有效位置总数。保证 T>K。',
        'loss=logits.new_zeros(()); count=0\nfor k in range(logits.shape[0]):\n    target=tokens[:,k+1:]\n    loss=loss+F.cross_entropy(logits[k,:,:-k-1].reshape(-1,logits.shape[-1]),target.reshape(-1),reduction="sum")\n    count+=target.numel()\nreturn loss/count',
        [('多个未来 head',"args=(torch.randn(3,2,5,7,requires_grad=True),torch.randint(7,(2,5)))"),('单 head',"args=(torch.randn(1,1,3,4,requires_grad=True),torch.tensor([[0,1,2]]))")],gradient=True)
    add('diff_attention','差分注意力','LLM 前沿架构','困难','q1, k1, q2, k2, v, lam',
        '各输入 [B,H,T,D]；返回 (softmax(q1k1ᵀ/sqrt(D))-lam*softmax(q2k2ᵀ/sqrt(D)))@v，无因果 mask。本题是差分注意力核心，不含 head norm。',
        's=math.sqrt(q1.shape[-1])\nreturn ((q1@k1.transpose(-2,-1)/s).softmax(-1)-lam*(q2@k2.transpose(-2,-1)/s).softmax(-1))@v',
        [('差分',"args=tuple(torch.randn(1,2,3,4,requires_grad=True) for _ in range(5))+(0.8,)"),('lambda 零',"args=tuple(torch.randn(2,1,2,3,requires_grad=True) for _ in range(5))+(0.,)")],gradient=True)
    add('mla','多头潜在注意力 MLA','LLM 前沿架构','困难','q, latent, wk, wv',
        '考低秩 KV 重构核心，不含 decoupled RoPE。q [B,H,T,D]，latent [B,S,R]，wk/wv [H,R,D]。先各头 latent@wk/wv，再做非因果 attention，返回 [B,H,T,D]。',
        'k=torch.einsum("bsr,hrd->bhsd",latent,wk)\nv=torch.einsum("bsr,hrd->bhsd",latent,wv)\nreturn (q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])).softmax(-1)@v',
        [('低秩 KV',"args=(torch.randn(2,3,4,5,requires_grad=True),torch.randn(2,6,2,requires_grad=True),torch.randn(3,2,5,requires_grad=True),torch.randn(3,2,5,requires_grad=True))"),('单步',"args=(torch.randn(1,1,1,2,requires_grad=True),torch.randn(1,1,1,requires_grad=True),torch.randn(1,1,2,requires_grad=True),torch.randn(1,1,2,requires_grad=True))")],gradient=True)
    add('moe','混合专家 MoE','LLM 前沿架构','困难','x, router_logits, weights, top_k=2',
        'x [T,D]，router [T,E]，weights [E,D,O]。选 logits 的 top-k 专家，仅在选中 logits 内 softmax，输出各专家线性输出的加权和 [T,O]。不含容量限制。测试不包含路由分数并列。',
        'scores,ids=router_logits.topk(top_k,dim=-1)\np=scores.softmax(-1)\ny=torch.einsum("td,tkdo->tko",x,weights[ids])\nreturn (y*p[...,None]).sum(1)',
        [('top-2 路由',"args=(torch.randn(5,3,requires_grad=True),torch.randn(5,4,requires_grad=True),torch.randn(4,3,2,requires_grad=True),2)"),('top-1',"args=(torch.randn(3,2,requires_grad=True),torch.randn(3,3,requires_grad=True),torch.randn(3,2,4,requires_grad=True),1)")],gradient=True)
    for slug,title in [('flash_attention','Flash Attention 分块'),('ring_attention','环形注意力（单机模拟）')]:
        add(slug,title,'推理与分布式训练','困难','q, k, v, block_size=2',
            'q [Q,D]，k [K,D]，v [K,V]；按 KV 块维护 running max、exp 和、加权 value 和，实现精确非因果 attention。禁止构造完整 Q×K 分数矩阵。'+('本题模拟环上依次接收 KV 块的数学过程，不执行跨机通信。' if slug=='ring_attention' else '本题考 online softmax，不要求写 CUDA kernel。'),
            '''m=q.new_full((len(q),1),float('-inf')); l=q.new_zeros(len(q),1); acc=q.new_zeros(len(q),v.shape[-1])
for start in range(0,len(k),block_size):
    s=q@k[start:start+block_size].T/math.sqrt(q.shape[-1])
    new_m=torch.maximum(m,s.amax(-1,keepdim=True)); scale=(m-new_m).exp(); p=(s-new_m).exp()
    acc=acc*scale+p@v[start:start+block_size]; l=l*scale+p.sum(-1,keepdim=True); m=new_m
return acc/l''',
            [('不整除分块',"args=(torch.randn(4,3,requires_grad=True),torch.randn(7,3,requires_grad=True),torch.randn(7,5,requires_grad=True),3)"),('大分数稳定性',"args=(torch.randn(2,3,requires_grad=True)*30,torch.randn(5,3,requires_grad=True)*30,torch.randn(5,2,requires_grad=True),1)")],gradient=True)
    add('linear_attention','线性自注意力','注意力与位置编码','困难','q, k, v, eps=1e-6',
        'q,k,v [B,T,D]，特征映射 phi(x)=ELU(x)+1。非因果，返回 phi(Q)*(phi(K)ᵀV)/(phi(Q)*sum(phi(K))+eps)。不要构造 T×T 矩阵。',
        'q=F.elu(q)+1; k=F.elu(k)+1\nkv=k.transpose(-2,-1)@v\nreturn (q@kv)/((q*k.sum(-2,keepdim=True)).sum(-1,keepdim=True)+eps)',
        [('随机序列',"args=tuple(torch.randn(2,5,3,requires_grad=True) for _ in range(3))"),('负输入',"args=(torch.full((1,2,3),-4.,requires_grad=True),torch.full((1,2,3),-3.,requires_grad=True),torch.randn(1,2,3,requires_grad=True))")],gradient=True)
    add('kv_cache','KV Cache 注意力','推理与分布式训练','困难','q, new_k, new_v, cached_k, cached_v',
        '单 token 解码：q/new_k/new_v [B,H,1,D]，cache [B,H,S,D] 可为空。沿序列维追加，返回 (attention_output,updated_k,updated_v)，不修改原 cache。',
        'k=torch.cat((cached_k,new_k),dim=-2); v=torch.cat((cached_v,new_v),dim=-2)\nreturn (q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])).softmax(-1)@v,k,v',
        [('追加解码',"args=tuple(torch.randn(1,2,1,3,requires_grad=True) for _ in range(3))+tuple(torch.randn(1,2,4,3) for _ in range(2))"),('空缓存',"args=tuple(torch.randn(1,1,1,2,requires_grad=True) for _ in range(3))+(torch.empty(1,1,0,2),torch.empty(1,1,0,2))")],gradient=True)
    add('paged_attention','分页注意力','推理与分布式训练','困难','q, key_pages, value_pages, block_table, length',
        '单序列 q [D]，pages [P,page_size,D]，block_table [num_blocks] 是逻辑→物理页索引。收集逻辑顺序前 length 个 KV，计算 attention [D]；末页 padding 不参与。',
        'k=key_pages[block_table].flatten(0,1)[:length]; v=value_pages[block_table].flatten(0,1)[:length]\nreturn (q@k.T/math.sqrt(q.shape[-1])).softmax(-1)@v',
        [('乱序物理页与尾部 padding',"args=(torch.randn(4,requires_grad=True),torch.randn(4,3,4,requires_grad=True),torch.randn(4,3,4,requires_grad=True),torch.tensor([2,0,3]),7)"),('单页',"args=(torch.randn(2,requires_grad=True),torch.randn(2,2,2),torch.randn(2,2,2),torch.tensor([1]),1)")],gradient=True)
    add('int8_quantization','INT8 量化线性层','推理与分布式训练','困难','x, weight',
        'weight [O,D]，每输出通道对称量化：scale=max(abs(row))/127；全零行 scale=1。q=round(w/scale).clamp(-127,127) int8；用反量化权重算 x@w_hat.T。返回 (output,q,scale)，scale [O,1]，不考梯度。',
        'scale=weight.abs().amax(-1,keepdim=True)/127\nscale=torch.where(scale==0,torch.ones_like(scale),scale)\nq=(weight/scale).round().clamp(-127,127).to(torch.int8)\nreturn x@(q.to(x.dtype)*scale).T,q,scale',
        [('随机权重',"args=(torch.randn(2,5),torch.randn(3,5))"),('零通道',"args=(torch.ones(2,3),torch.tensor([[0.,0.,0.],[1.,-2.,3.]]))")])
    add('qlora','QLoRA 量化基座与低秩更新','参数高效训练','困难','x, codes, codebook, scales, a, b, alpha',
        '给定已经量化的 4-bit 基座：codes [O,D] 为 0..15 索引，codebook [16] 为输入提供的码本，scales [O,1]；W=codebook[codes]*scales。a [r,D], b [O,r]。返回 x@W.T+alpha/r*x@a.T@b.T，基座与码本冻结。此题不考 NF4 码本生成、双重量化或 CUDA。',
        'w=(codebook[codes]*scales).detach()\nreturn x@w.T+alpha/a.shape[0]*(x@a.T@b.T)',
        [('量化基座',"args=(torch.randn(2,4,requires_grad=True),torch.randint(16,(3,4)),torch.linspace(-1,1,16),torch.rand(3,1),torch.randn(2,4,requires_grad=True),torch.randn(3,2,requires_grad=True),4.)"),('零低秩更新',"args=(torch.ones(1,2,requires_grad=True),torch.tensor([[0,15]]),torch.linspace(-1,1,16),torch.ones(1,1),torch.ones(1,2,requires_grad=True),torch.zeros(1,1,requires_grad=True),1.)")],gradient=True)
    add('mamba_ssm','Mamba SSM 步骤','LLM 前沿架构','困难','u, delta, a, b, c, d, initial',
        'u/delta [B,T,D]，a [D,N]，b/c [B,T,N]，d [D]，initial [B,D,N]。递推 h=exp(delta*a)*h+delta*b*u；y=sum(c*h,N)+d*u。返回 (y [B,T,D],final_h)。考 selective scan 核心，采用 delta*B 输入离散化。',
        'h=initial; ys=[]\nfor t in range(u.shape[1]):\n    dt=delta[:,t,:,None]\n    h=(dt*a).exp()*h+dt*b[:,t,None,:]*u[:,t,:,None]\n    ys.append((h*c[:,t,None,:]).sum(-1)+d*u[:,t])\nreturn torch.stack(ys,1),h',
        [('状态递推',"args=(torch.randn(2,4,3,requires_grad=True),torch.rand(2,4,3),-torch.rand(3,2),torch.randn(2,4,2),torch.randn(2,4,2),torch.randn(3),torch.zeros(2,3,2))"),('非零初态',"args=(torch.randn(1,1,2,requires_grad=True),torch.rand(1,1,2),-torch.rand(2,3),torch.randn(1,1,3),torch.randn(1,1,3),torch.ones(2),torch.ones(1,2,3))")],gradient=True)
    add('tensor_parallel','张量并行 MLP（单机模拟）','推理与分布式训练','困难','x, up_shards, down_shards',
        '列并行 up shard [D,Fi]，行并行 down shard [Fi,O]。返回 sum(ReLU(x@up_i)@down_i)。在单机模拟通信求和，不启动进程组，无 bias。',
        'return sum(torch.relu(x@u)@d for u,d in zip(up_shards,down_shards))',
        [('不等宽分片',"args=(torch.randn(2,4,requires_grad=True),[torch.randn(4,2,requires_grad=True),torch.randn(4,3,requires_grad=True)],[torch.randn(2,5,requires_grad=True),torch.randn(3,5,requires_grad=True)])"),('单分片',"args=(torch.randn(3,2),[torch.randn(2,4)],[torch.randn(4,2)])")],gradient=True)
    add('fsdp_step','FSDP 训练步骤（单机模拟）','推理与分布式训练','困难','param_shards, rank_grads, lr',
        '模拟 all-gather 参数与 reduce-scatter 平均梯度后 SGD。param_shards 为一维分片列表；rank_grads 为每 rank 的完整一维梯度（均为全参数长度）。平均梯度后按 shard 长度拆分，返回更新后的分片列表。不执行真实分布式通信。',
        'g=torch.stack(rank_grads).mean(0); out=[]; start=0\nfor p in param_shards:\n    out.append(p-lr*g[start:start+p.numel()]); start+=p.numel()\nreturn out',
        [('不等分片',"args=([torch.ones(2),torch.ones(3)],[torch.arange(5.),torch.arange(5.)*3],0.1)"),('单 rank',"args=([torch.randn(3)],[torch.randn(3)],0.01)")])
    add('mixed_precision','混合精度训练步骤','从零训练 GPT','中等','param, scaled_grad, loss_scale, lr',
        '考 loss scaling 的更新核心，输入 float32。不启动 GPU。先 unscale；若梯度存在 NaN/Inf 则跳过并返回 (param.clone(),False)，否则返回 (param-lr*grad/loss_scale,True)。不原地修改。',
        'g=scaled_grad/loss_scale\nif not torch.isfinite(g).all():\n    return param.clone(),False\nreturn param-lr*g,True',
        [('正常更新',"args=(torch.ones(3),torch.tensor([128.,-256.,0.]),128.,0.1)"),('溢出跳过',"args=(torch.ones(2),torch.tensor([float('inf'),1.]),128.,0.1)")])
    add('activation_checkpointing','激活检查点','从零训练 GPT','中等','x, w1, w2',
        '实现 checkpointed 两层 MLP，返回 ReLU(x@w1)@w2。必须调用 torch.utils.checkpoint.checkpoint，use_reentrant=False；保留反向梯度。本题允许使用该机制 API，重点理解重算。',
        'from torch.utils.checkpoint import checkpoint\nreturn checkpoint(lambda a,b,c: torch.relu(a@b)@c,x,w1,w2,use_reentrant=False)',
        [('前向与 backward',"args=(torch.randn(2,4,requires_grad=True),torch.randn(4,6,requires_grad=True),torch.randn(6,3,requires_grad=True))"),('输入无梯度但参数有',"args=(torch.randn(3,2),torch.randn(2,4,requires_grad=True),torch.randn(4,2,requires_grad=True))")],gradient=True)
    for slug,title,causal in [('gpt2_block','GPT-2 Transformer Block',True),('vit_block','ViT Transformer Block',False)]:
        add(slug,title,'Transformer 内部机制' if causal else 'Vision Transformer','困难','x, wqkv, wo, w1, w2, heads, eps=1e-5',
            '实现 pre-LN block，无 bias/dropout，LN 无仿射：a=x+Attention(LN(x))；返回 a+GELU(LN(a)@w1)@w2。wqkv [D,3D]，wo [D,D]，w1 [D,F]，w2 [F,D]，GELU 用精确 erf。'+('attention 使用因果 mask。' if causal else 'attention 双向，无因果 mask。'),
            '''b,t,d=x.shape; z=F.layer_norm(x,(d,),eps=eps)
q,k,v=[part.reshape(b,t,heads,d//heads).transpose(1,2) for part in (z@wqkv).chunk(3,-1)]
s=q@k.transpose(-2,-1)/math.sqrt(d//heads)
'''+('s=s.masked_fill(torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1),float("-inf"))\n' if causal else '')+'''a=x+(s.softmax(-1)@v).transpose(1,2).reshape(b,t,d)@wo
return a+F.gelu(F.layer_norm(a,(d,),eps=eps)@w1)@w2''',
            [('双头 block',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(4,12,requires_grad=True)*0.2,torch.randn(4,4,requires_grad=True)*0.2,torch.randn(4,6,requires_grad=True)*0.2,torch.randn(6,4,requires_grad=True)*0.2,2)"),('单 token',"args=(torch.randn(1,1,4,requires_grad=True),torch.randn(4,12)*0.1,torch.eye(4),torch.randn(4,5)*0.1,torch.randn(5,4)*0.1,1)")],gradient=True)
    from catalog_algorithms import populate as algorithms
    algorithms(add)

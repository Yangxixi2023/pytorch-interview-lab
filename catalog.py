"""Executable, explicit interview exercises. References are disclosed only on request."""
import textwrap

PROBLEMS = {}


def add(slug, title, category, difficulty, signature, statement, body, cases, *, hint='', sources=(), gradient=False):
    source = 'import math\nimport torch\nimport torch.nn.functional as F\n\n'
    source += f'def solve({signature}):\n' + textwrap.indent(textwrap.dedent(body).strip(), '    ') + '\n'
    PROBLEMS[slug] = dict(id=slug, title=title, category=category, difficulty=difficulty,
        signature=signature, statement=statement, reference=source,
        starter='import math\nimport torch\n\n' + f'def solve({signature}):\n    # 在这里实现；可以使用 PyTorch 基础张量运算\n    raise NotImplementedError\n',
        cases=[dict(name=c[0], setup=c[1]) for c in cases], hint=hint,
        sources=list(sources), gradient=gradient)


add('ppo_loss','PPO 损失','对齐与强化学习','中等',
    'new_logp, old_logp, advantages, mask, clip_eps=0.2',
    '''实现 PPO 的 clipped policy loss（本题只考 actor，不含 value loss 和 entropy bonus）。
输入均为 [B,T]，mask 为 bool，有效 token 至少一个。old_logp 和 advantages 视为常量。
r=exp(new_logp-old_logp)，返回 -sum(mask*min(r*A, clip(r,1-eps,1+eps)*A))/sum(mask)。
返回可反向传播的标量；padding 不参与计算。禁止调用现成 PPO loss。''',
    '''r = (new_logp - old_logp.detach()).exp()
a = advantages.detach()
return -torch.minimum(r*a, r.clamp(1-clip_eps,1+clip_eps)*a)[mask].mean()''',
    [('正负优势与双侧裁剪', "new=torch.tensor([[0.,0.7,-0.7,0.1]],requires_grad=True); args=(new,torch.zeros_like(new),torch.tensor([[1.,1.,-1.,-2.]]),torch.ones_like(new,dtype=torch.bool))"),
     ('变长与 padding', "new=torch.randn(3,5,requires_grad=True)*0.3; args=(new,torch.randn(3,5)*0.2,torch.randn(3,5),torch.tensor([[1,0,0,0,0],[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool),0.1)"),
     ('旧策略与优势必须 detach', "args=(torch.randn(2,3,requires_grad=True),torch.randn(2,3,requires_grad=True),torch.randn(2,3,requires_grad=True),torch.ones(2,3,dtype=torch.bool))")],
    hint='负优势下也取 minimum；不能把 ratio 先裁剪再直接乘 advantage。',sources=['https://arxiv.org/abs/1707.06347'],gradient=True)

for slug,title in [('gspo_loss','GSPO 损失'),('dapo_loss','DAPO 损失'),('grpo_loss','GRPO 损失')]:
    if slug=='gspo_loss':
        formula='对每条序列计算 s=exp(sum(mask*(new-old))/length)，再对序列做 min(s*A,clip(s,1-low,1+high)*A)，最终取负的序列平均。不添加 KL。'
        body='''length = mask.sum(-1)
a = advantages.detach()
s = ((new_logp-old_logp.detach()).masked_fill(~mask,0).sum(-1)/length).exp()
return -torch.minimum(s*a,s.clamp(1-clip_low,1+clip_high)*a).mean()'''
        src='https://arxiv.org/abs/2507.18071'
    else:
        formula=('对每个 token 计算 ratio 与 clipped surrogate，负的有效 token 总和除以整个 batch 的有效 token 数。输入是已经完成动态采样和奖励处理的 batch，本题不实现采样循环，不加 KL。' if slug=='dapo_loss' else '对每个 token 计算 ratio 与 clipped surrogate，先在每条序列的有效 token 内平均，再对序列平均并取负。此题考 beta=0 的 GRPO policy loss，不含 KL。')
        reduction='surrogate[mask].mean()' if slug=='dapo_loss' else '(surrogate.masked_fill(~mask,0).sum(-1)/mask.sum(-1)).mean()'
        body=f'''r=(new_logp-old_logp.detach()).exp()
a=advantages.detach().unsqueeze(-1)
surrogate=torch.minimum(r*a,r.clamp(1-clip_low,1+clip_high)*a)
return -{reduction}'''
        src='https://arxiv.org/abs/2503.14476' if slug=='dapo_loss' else 'https://arxiv.org/abs/2402.03300'
    add(slug,title,'对齐与强化学习','困难','new_logp, old_logp, advantages, mask, clip_low=0.2, clip_high=0.28',
        'new_logp/old_logp/mask: [B,T]；advantages: [B]，是已算好的组内标准化优势。每条序列至少一个有效 token，mask 为 bool。old_logp 和 advantages 视为常量。返回可求导标量。\n'+formula,
        body,[('不等长、正负优势',"args=(torch.tensor([[0.1,0.4,-0.4],[0.5,-0.5,0.1]],requires_grad=True),torch.zeros(2,3),torch.tensor([1.,-1.]),torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool))"),
        ('非对称裁剪',"args=(torch.randn(4,6,requires_grad=True),torch.randn(4,6)*0.2,torch.tensor([1.,-2.,0.,0.5]),torch.rand(4,6)>-1,0.1,0.4)"),
        ('梯度隔离',"args=(torch.randn(2,3,requires_grad=True)*0.1,torch.zeros(2,3,requires_grad=True),torch.tensor([1.,-1.],requires_grad=True),torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool))")],
        hint='先确定优化的单位是 token 还是 sequence；变长样本可以区分三种归约。',sources=[src],gradient=True)

add('dpo_loss','DPO 损失','对齐与强化学习','中等','chosen, rejected, ref_chosen, ref_rejected, beta=0.1',
    '四个 [B] 张量是整条回答的 log probability 之和。返回 mean(-logsigmoid(beta*((chosen-rejected)-(ref_chosen-ref_rejected))))；reference 不求梯度。需对极大负 margin 稳定。',
    'return -F.logsigmoid(beta*((chosen-rejected)-(ref_chosen.detach()-ref_rejected.detach()))).mean()',
    [('普通偏好',"args=tuple(torch.randn(5,requires_grad=True) for _ in range(4))"),('极端 margin',"args=(torch.tensor([-10000.,10000.],requires_grad=True),torch.zeros(2),torch.zeros(2),torch.zeros(2))")],gradient=True)

add('gae_advantage','GAE 广义优势估计','对齐与强化学习','中等','rewards, values, terminated, gamma=0.99, lam=0.95',
    'rewards/terminated 为 [T,B]，values 为 [T+1,B]。terminated[t] 表示该步后真正终止（不是时间截断）。delta=r+gamma*(1-done)*V_next-V；A_t=delta+gamma*lam*(1-done)*A_next。返回 (advantages, returns)，returns=A+values[:-1]，两者均 detach。',
    '''with torch.no_grad():
    a=torch.zeros_like(rewards); running=torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0]-1,-1,-1):
        live=(~terminated[t]).to(rewards.dtype)
        delta=rewards[t]+gamma*live*values[t+1]-values[t]
        running=delta+gamma*lam*live*running
        a[t]=running
    return a,a+values[:-1]''',
    [('中途终止与 bootstrap',"args=(torch.ones(4,2),torch.arange(10.).reshape(5,2).requires_grad_(),torch.tensor([[0,0],[1,0],[0,0],[1,0]],dtype=torch.bool))"),('lambda 为零',"args=(torch.randn(3,1),torch.randn(4,1),torch.zeros(3,1,dtype=torch.bool),0.9,0.)")])

add('reward_model','Bradley–Terry 奖励损失','对齐与强化学习','中等','chosen_rewards, rejected_rewards',
    '输入 [B] 奖励，返回 mean(softplus(rejected-chosen))，需要数值稳定和可导。',
    'return F.softplus(rejected_rewards-chosen_rewards).mean()', [('偏好',"args=(torch.randn(7,requires_grad=True),torch.randn(7,requires_grad=True))"),('极端',"args=(torch.tensor([-1000.,1000.],requires_grad=True),torch.zeros(2))")],gradient=True)

add('softmax','实现 Softmax','Transformer 内部机制','简单','x, dim=-1',
    '沿 dim 实现数值稳定 softmax，保持形状。不能调用 torch.softmax / F.softmax。',
    'z=x-x.amax(dim=dim,keepdim=True)\ne=z.exp()\nreturn e/e.sum(dim=dim,keepdim=True)',
    [('大数',"args=(torch.tensor([[1000.,1001.,999.]],requires_grad=True),)"),('指定维度',"args=(torch.randn(2,3,4,requires_grad=True),1)")],gradient=True)
add('cross_entropy','交叉熵损失','从零训练 GPT','简单','logits, targets',
    'logits [N,C]，targets [N] 为 int64 类别。返回平均交叉熵。不能调用 cross_entropy 或 log_softmax。',
    'return (torch.logsumexp(logits,dim=-1)-logits.gather(1,targets[:,None]).squeeze(1)).mean()',
    [('普通多分类',"args=(torch.randn(5,4,requires_grad=True),torch.tensor([0,1,3,2,1]))"),('大 logits',"args=(torch.tensor([[1000.,-1000.],[1001.,1000.]],requires_grad=True),torch.tensor([1,0]))")],gradient=True)
for slug,title,body,desc in [
    ('relu','实现 ReLU','return x.clamp_min(0)','max(x,0)，不能调用 relu。'),
    ('gelu','GELU 激活函数','return 0.5*x*(1+torch.erf(x/math.sqrt(2)))','使用精确 erf 版本 x*Phi(x)，不是 tanh 近似。'),
    ('swiglu','SwiGLU 激活函数','a,b=x.chunk(2,dim=-1)\nreturn (a*torch.sigmoid(a))*b','最后一维长度为偶数，分成 a,b，返回 SiLU(a)*b。')]:
    add(slug,title,'Transformer 内部机制','简单','x',desc,body,[('正负数',"args=(torch.randn(2,3,6,requires_grad=True),)"),('大幅输入',"args=(torch.tensor([[-20.,-1.,1.,20.]],requires_grad=True),)")],gradient=True)
add('rmsnorm','实现 RMSNorm','Transformer 内部机制','中等','x, weight, eps=1e-6',
    'x [...,D]，weight [D]。沿最后一维返回 x/sqrt(mean(x²)+eps)*weight，不减均值。',
    'return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*weight',
    [('随机输入',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(4,requires_grad=True))"),('零输入',"args=(torch.zeros(2,4,requires_grad=True),torch.ones(4))")],gradient=True)
add('layernorm','实现 LayerNorm','Transformer 内部机制','中等','x, weight, bias, eps=1e-5',
    '沿最后一维归一化，方差为总体方差 correction=0，返回归一化值*weight+bias。禁止调用 layer_norm。',
    'm=x.mean(-1,keepdim=True)\nreturn (x-m)*torch.rsqrt((x-m).square().mean(-1,keepdim=True)+eps)*weight+bias',
    [('多维',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(4,requires_grad=True),torch.randn(4,requires_grad=True))"),('常量',"args=(torch.ones(2,3,requires_grad=True),torch.ones(3),torch.zeros(3))")],gradient=True)
add('linear','简单线性层','Transformer 内部机制','中等','x, weight, bias',
    'x [...,Din]，weight [Dout,Din]，bias [Dout]，返回 x@weight.T+bias，不能调用 F.linear。',
    'return x@weight.T+bias', [('多维 batch',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(5,4,requires_grad=True),torch.randn(5,requires_grad=True))"),('单样本',"args=(torch.randn(4,requires_grad=True),torch.randn(1,4,requires_grad=True),torch.randn(1,requires_grad=True))")],gradient=True)
add('embedding','Embedding 层','从零训练 GPT','简单','ids, weight',
    'ids 是任意形状 int64 张量，weight [V,D]，返回 [...,D]。重复索引的梯度需累加，不能调用 embedding。',
    'return weight[ids]', [('重复索引',"args=(torch.tensor([[1,1,3],[0,2,1]]),torch.randn(5,4,requires_grad=True))"),('一维索引',"args=(torch.tensor([0]),torch.randn(2,3,requires_grad=True))")],gradient=True)

add('attention','Softmax 注意力','Transformer 内部机制','简单','q, k, v',
    'q [B,Q,D]，k [B,K,D]，v [B,K,Dv]；返回 softmax(qkᵀ/sqrt(D))v。不能调用 scaled_dot_product_attention。',
    'return (q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])).softmax(-1)@v',
    [('交叉长度',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(2,5,4,requires_grad=True),torch.randn(2,5,6,requires_grad=True))"),('单 token',"args=tuple(torch.randn(1,1,3,requires_grad=True) for _ in range(3))")],gradient=True)

for slug,title,extra,maskexpr in [('causal_attention','因果自注意力','', 'j>i'),('sliding_window','滑动窗口注意力',', window=3','(j>i)|(j<i-window+1)'),('alibi','ALiBi 注意力',', slopes','j>i')]:
    desc='q,k,v [B,H,T,D]，返回同形输出。只允许看到当前和过去位置。'
    if slug=='sliding_window': desc+=' window 包含当前位置，额外屏蔽距离 >=window 的历史 token。'
    if slug=='alibi': desc+=' slopes [H]，scores 额外减去 slopes[h]*(i-j)。'
    body=f'''t=q.shape[-2]; i=torch.arange(t,device=q.device)[:,None]; j=torch.arange(t,device=q.device)[None,:]
scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
'''+('scores=scores-slopes[None,:,None,None]*(i-j)\n' if slug=='alibi' else '')+f'return scores.masked_fill({maskexpr},float("-inf")).softmax(-1)@v'
    case="args=tuple(torch.randn(2,2,5,4,requires_grad=True) for _ in range(3))"+ ("+(torch.tensor([0.1,0.3]),)" if slug=='alibi' else '')
    add(slug,title,'注意力与位置编码','中等','q, k, v'+extra,desc,body,[('多头与因果 mask',case),('单步',"args=tuple(torch.randn(1,2,1,4,requires_grad=True) for _ in range(3))"+("+(torch.tensor([0.1,0.2]),)" if slug=='alibi' else ''))],gradient=True)

add('mha','多头注意力','Transformer 内部机制','困难','x, wq, wk, wv, wo, heads',
    'x [B,T,D]，四个权重均 [D,D]，按 x@w 投影。拆 H 个头做非因果 attention，合并后 @wo，无 bias/dropout，D 可被 heads 整除。',
    '''b,t,d=x.shape
q,k,v=[(x@w).reshape(b,t,heads,d//heads).transpose(1,2) for w in (wq,wk,wv)]
y=(q@k.transpose(-2,-1)/math.sqrt(d//heads)).softmax(-1)@v
return y.transpose(1,2).reshape(b,t,d)@wo''',
    [('多头',"args=(torch.randn(2,3,8,requires_grad=True),)+tuple(torch.randn(8,8,requires_grad=True)*0.2 for _ in range(4))+(2,)"),('单头',"args=(torch.randn(1,4,4,requires_grad=True),)+tuple(torch.eye(4) for _ in range(4))+(1,)")],gradient=True)
add('gqa','分组查询注意力 GQA','注意力与位置编码','困难','q, k, v',
    'q [B,Hq,T,D]，k,v [B,Hkv,S,D]，Hq 是 Hkv 整数倍。连续 Hq/Hkv 个 query head 共享一个 KV head。返回非因果 attention [B,Hq,T,D]。',
    'n=q.shape[1]//k.shape[1]\nk=k.repeat_interleave(n,dim=1); v=v.repeat_interleave(n,dim=1)\nreturn (q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])).softmax(-1)@v',
    [('4Q/2KV',"args=(torch.randn(2,4,3,4,requires_grad=True),torch.randn(2,2,5,4,requires_grad=True),torch.randn(2,2,5,4,requires_grad=True))"),('MQA',"args=(torch.randn(1,4,2,3,requires_grad=True),torch.randn(1,1,3,3,requires_grad=True),torch.randn(1,1,3,3,requires_grad=True))")],gradient=True)

add('rope','旋转位置编码 RoPE','注意力与位置编码','中等','x, positions, base=10000.0',
    'x [B,H,T,D]，D 偶数；positions [T]。采用相邻偶奇配对 (x0,x1),(x2,x3)，theta=position*base^(-2j/D)，输出每对 (a*cos-b*sin,a*sin+b*cos)。',
    '''d=x.shape[-1]
freq=base**(-torch.arange(0,d,2,device=x.device,dtype=x.dtype)/d)
angle=positions[:,None]*freq; c=angle.cos(); s=angle.sin()
a=x[...,0::2]; b=x[...,1::2]
return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(-2)''',
    [('偏移位置',"args=(torch.randn(2,2,3,8,requires_grad=True),torch.tensor([3,4,5]))"),('位置零',"args=(torch.randn(1,1,1,4,requires_grad=True),torch.tensor([0]))")],gradient=True)

add('lora','LoRA 低秩适配','参数高效训练','中等','x, weight, a, b, alpha',
    'x [...,Din]，weight [Dout,Din]，a [r,Din]，b [Dout,r]。返回 x@weight.T + alpha/r*(x@a.T@b.T)。weight 冻结，x/a/b 可求导。',
    'return x@weight.detach().T+(alpha/a.shape[0])*(x@a.T@b.T)',
    [('低秩更新',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(5,4,requires_grad=True),torch.randn(2,4,requires_grad=True),torch.randn(5,2,requires_grad=True),4.)"),('零初始化 B',"args=(torch.randn(2,4,requires_grad=True),torch.randn(3,4),torch.randn(1,4,requires_grad=True),torch.zeros(3,1,requires_grad=True),1.)")],gradient=True)

from catalog_extra import populate
populate(add)

from catalog_foundations import populate as populate_foundations
populate_foundations(add)

from problem_io import IO, FORMULAS
for slug, problem in PROBLEMS.items():
    problem.update(IO[slug])
    problem['formulas'] = FORMULAS.get(slug, [])

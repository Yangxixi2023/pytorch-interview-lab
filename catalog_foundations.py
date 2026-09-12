def populate(add):
    add('tensor_heads','张量变换与多头重排','张量基础','简单','x, heads',
        'x [B,T,D]，D 可被 heads 整除。将最后一维拆成 H 个 head，并交换为 [B,H,T,D/H]；再合并还原。返回 (split_heads,merged)。允许使用 view/reshape/transpose 或 einops，不修改输入。',
        'b,t,d=x.shape\nsplit=x.reshape(b,t,heads,d//heads).transpose(1,2)\nreturn split,split.transpose(1,2).reshape(b,t,d)',
        [('三头拆分',"args=(torch.arange(48.).reshape(2,2,12).requires_grad_(),3)"),('非连续输入',"args=(torch.randn(2,8,3).transpose(1,2).requires_grad_(),2)")],gradient=True)
    add('ffn','FFN 前馈网络','Transformer 内部机制','中等','x, w1, b1, w2, b2',
        '实现普通 Transformer FFN：GELU(x@w1+b1)@w2+b2。x [...,D]，w1 [D,F]，b1 [F]，w2 [F,D]，b2 [D]。GELU 使用精确 erf 形式，不含残差/归一化/dropout。',
        'z=x@w1+b1\ng=0.5*z*(1+torch.erf(z/math.sqrt(2)))\nreturn g@w2+b2',
        [('多维输入',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(4,7,requires_grad=True),torch.randn(7,requires_grad=True),torch.randn(7,4,requires_grad=True),torch.randn(4,requires_grad=True))"),('单特征',"args=(torch.randn(2,1,requires_grad=True),torch.randn(1,3),torch.randn(3),torch.randn(3,1),torch.randn(1))")],gradient=True)
    add('entropy_loss','Entropy Loss 策略熵','对齐与强化学习','简单','logits, mask',
        'logits [B,T,V]，mask [B,T] 为 bool，至少一个有效位置。对每个有效位置计算 H=-sum_v p_v*log(p_v)，返回有效位置的平均熵（正号）。如果训练目标要鼓励探索，最小化 loss 时应减去该值；此函数不取负号。使用稳定 log_softmax，可调用 torch.log_softmax。',
        'lp=logits.log_softmax(-1)\nentropy=-(lp.exp()*lp).sum(-1)\nreturn entropy[mask].mean()',
        [('均匀分布与 mask',"args=(torch.zeros(2,3,4,requires_grad=True),torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool))"),('极端 logits',"args=(torch.tensor([[[1000.,-1000.,0.],[-4.,2.,3.]]],requires_grad=True),torch.ones(1,2,dtype=torch.bool))")],gradient=True)
    add('sft_loss','SFT Loss 因果语言模型','从零训练 GPT','中等','logits, labels, ignore_index=-100',
        'logits [B,T,V]，labels [B,T] int64。logits[:,t] 预测 labels[:,t+1]；丢掉最后一个预测与第一个标签。labels 等于 ignore_index 的位置不计 loss，常用于 prompt 和 padding。返回所有有效目标 token 的平均交叉熵，不是每条序列平均。T>=2，shift 后至少一个有效目标。',
        'lp=logits[:,:-1].log_softmax(-1)\ny=labels[:,1:]\nvalid=y!=ignore_index\nreturn -lp[valid].gather(-1,y[valid,None]).mean()',
        [('shift 与 prompt 屏蔽',"args=(torch.randn(2,4,5,requires_grad=True),torch.tensor([[-100,-100,2,1],[-100,3,0,-100]]))"),('单个有效目标',"args=(torch.randn(1,3,4,requires_grad=True),torch.tensor([[-100,-100,2]]))")],gradient=True)

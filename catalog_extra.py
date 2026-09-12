def populate(add):
    add('dropout','实现 Dropout','从零训练 GPT','简单','x, p, training, keep_mask',
        '实现 inverted dropout。为方便确定性判题，提供已采样的 bool keep_mask，与 x 同形。training=False 返回 x；训练时返回 x*keep_mask/(1-p)。0<=p<1。无需自己采样。',
        'return x*keep_mask/(1-p) if training else x',
        [('训练缩放',"args=(torch.ones(2,4,requires_grad=True),0.5,True,torch.tensor([[1,0,1,0],[0,1,0,1]],dtype=torch.bool))"),('推理模式',"args=(torch.randn(3,requires_grad=True),0.8,False,torch.zeros(3,dtype=torch.bool))")],gradient=True)
    add('flow_matching','流匹配损失','扩散模型与 DiT','简单','pred_velocity, x0, x1',
        '线性路径 x_t=(1-t)x0+t*x1，目标速度为 x1-x0；返回所有元素平均 MSE，x0/x1 视为常量。',
        'return (pred_velocity-(x1-x0).detach()).square().mean()',
        [('随机速度',"args=tuple(torch.randn(2,3,4,requires_grad=True) for _ in range(3))"),('静止路径',"x=torch.ones(2,3); args=(torch.zeros(2,3,requires_grad=True),x,x)")],gradient=True)
    add('label_smoothing','标签平滑损失','从零训练 GPT','简单','logits, targets, smoothing=0.1',
        'logits [N,C]，平滑标签为 (1-eps)*one_hot+eps/C，返回平均交叉熵。',
        'lp=logits.log_softmax(-1)\nreturn (-(1-smoothing)*lp.gather(1,targets[:,None]).mean()-smoothing*lp.mean())',
        [('平滑',"args=(torch.randn(4,5,requires_grad=True),torch.tensor([0,2,3,1]),0.2)"),('退化交叉熵',"args=(torch.randn(2,3,requires_grad=True),torch.tensor([0,2]),0.)")],gradient=True)
    add('focal_loss','Focal Loss','损失函数','中等','logits, targets, gamma=2.0',
        '多分类无 alpha 加权版本；pt=softmax(logits)[target]，返回 mean(-(1-pt)^gamma*log(pt))。',
        'lp=logits.log_softmax(-1).gather(1,targets[:,None]).squeeze(1)\nreturn (-((1-lp.exp())**gamma)*lp).mean()',
        [('难易样本',"args=(torch.randn(6,4,requires_grad=True),torch.tensor([0,1,2,3,0,1]))"),('gamma 零',"args=(torch.randn(2,3,requires_grad=True),torch.tensor([1,2]),0.)")],gradient=True)
    add('contrastive_loss','对比损失 InfoNCE','损失函数','中等','queries, keys, temperature=0.1',
        'queries/keys [N,D]，先 L2 normalize；第 i 个 query 的正例为第 i 个 key，其他为负例。返回单向平均交叉熵。',
        'scores=F.normalize(queries,dim=-1)@F.normalize(keys,dim=-1).T/temperature\nreturn F.cross_entropy(scores,torch.arange(scores.shape[0],device=scores.device))',
        [('正负例',"args=(torch.randn(4,5,requires_grad=True),torch.randn(4,5,requires_grad=True))"),('单样本',"args=(torch.randn(1,3,requires_grad=True),torch.randn(1,3,requires_grad=True),0.5)")],gradient=True)
    add('gradient_accumulation','梯度累积','从零训练 GPT','简单','x, y, weight, microbatch_size',
        'x [N,D]，y [N,O]，weight [D,O]。手动计算全 batch mean MSE 对 weight 的梯度，按 microbatch 累加；最后一个不足 batch 也要按元素数正确加权。返回梯度，不修改输入，不用 autograd。',
        '''grad=torch.zeros_like(weight)
for start in range(0,len(x),microbatch_size):
    xb=x[start:start+microbatch_size]; yb=y[start:start+microbatch_size]
    grad+=2*xb.T@(xb@weight-yb)/y.numel()
return grad''',
        [('不整除 batch',"args=(torch.randn(7,3),torch.randn(7,2),torch.randn(3,2),3)"),('整 batch',"args=(torch.randn(4,2),torch.randn(4,1),torch.randn(2,1),4)")])
    add('gradient_clipping','梯度范数裁剪','从零训练 GPT','简单','grads, max_norm, eps=1e-6',
        'grads 为张量列表，计算所有元素的全局 L2 范数 norm；返回 (新梯度列表, norm)，新梯度=g*min(1,max_norm/(norm+eps))，不原地修改。',
        'norm=torch.sqrt(sum(g.square().sum() for g in grads))\nscale=(max_norm/(norm+eps)).clamp(max=1)\nreturn [g*scale for g in grads],norm',
        [('需要裁剪',"args=([torch.tensor([3.,4.]),torch.tensor([12.])],5.)"),('零梯度',"args=([torch.zeros(3),torch.zeros(2)],1.)")])
    add('adam','Adam 优化器','从零训练 GPT','中等','param, grad, m, v, step, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8',
        '实现一次 Adam 更新，step 从 1 开始；偏置修正 m/(1-beta1^step)、v/(1-beta2^step)，epsilon 在 sqrt 外。返回 (new_param,new_m,new_v)，不修改输入。',
        'm1=beta1*m+(1-beta1)*grad\nv1=beta2*v+(1-beta2)*grad.square()\np=param-lr*(m1/(1-beta1**step))/((v1/(1-beta2**step)).sqrt()+eps)\nreturn p,m1,v1',
        [('第一步',"args=(torch.ones(3),torch.tensor([1.,-2.,0.]),torch.zeros(3),torch.zeros(3),1)"),('已有动量',"args=(torch.randn(2,3),torch.randn(2,3),torch.randn(2,3),torch.rand(2,3),7,0.01)")])
    add('cosine_lr','余弦学习率（含预热）','从零训练 GPT','中等','step, warmup_steps, total_steps, max_lr, min_lr=0.0',
        '0<=step<=total_steps 且 0<=warmup<total。warmup>0 且 step<warmup 时 lr=max_lr*step/warmup；此后从 max_lr 余弦降到 min_lr。返回 float。',
        'if step<warmup_steps:\n    return max_lr*step/warmup_steps\np=(step-warmup_steps)/(total_steps-warmup_steps)\nreturn min_lr+0.5*(max_lr-min_lr)*(1+math.cos(math.pi*p))',
        [('预热',"args=(2,10,100,0.01,0.001)"),('起点',"args=(0,0,100,0.01)"),('中途',"args=(55,10,100,0.01,0.001)"),('终点',"args=(100,10,100,0.01,0.001)")])
    add('weight_init','Kaiming 初始化','从零训练 GPT','简单','standard_normal, fan_in',
        '输入预采样标准正态张量，按 ReLU fan_in Kaiming normal 缩放为 sqrt(2/fan_in)*z；不重新采样。',
        'return standard_normal*math.sqrt(2/fan_in)', [('一般形状',"args=(torch.randn(20,30),30)"),('fan_in=1',"args=(torch.tensor([-1.,0.,1.]),1)")])
    add('sinusoidal_pe','正弦位置编码','注意力与位置编码','简单','length, dim, base=10000.0',
        'dim 为偶数，返回 float32 [length,dim]；PE[p,2i]=sin(p/base^(2i/dim))，奇数位用 cos。',
        'freq=base**(-torch.arange(0,dim,2,dtype=torch.float32)/dim)\na=torch.arange(length)[:,None]*freq\nreturn torch.stack((a.sin(),a.cos()),-1).flatten(-2)',
        [('短序列',"args=(5,8)"),('单位置',"args=(1,2)")])
    add('ntk_rope','NTK-aware RoPE 缩放','注意力与位置编码','简单','dim, scale, base=10000.0',
        '考固定 NTK 缩放的频率计算，不是动态长度策略。dim 为偶数且 >2，scale>=1；新 base=base*scale^(dim/(dim-2))，返回 float32 [dim/2] 逆频率 new_base^(-2i/dim)。',
        'new_base=base*scale**(dim/(dim-2))\nreturn new_base**(-torch.arange(0,dim,2,dtype=torch.float32)/dim)',
        [('延长上下文',"args=(8,4.)"),('恒等缩放',"args=(6,1.)")])
    add('noise_schedule','扩散噪声调度','扩散模型与 DiT','简单','steps, beta_start, beta_end',
        '线性 beta 调度，包含两个端点。返回 (betas,alphas,alpha_bars)，alpha=1-beta，alpha_bar=cumprod(alpha)，float32。',
        'b=torch.linspace(beta_start,beta_end,steps)\na=1-b\nreturn b,a,a.cumprod(0)', [('五步',"args=(5,0.0001,0.02)"),('一步',"args=(1,0.01,0.1)")])
    add('ddim_step','DDIM 采样步骤','扩散模型与 DiT','中等','xt, eps_pred, alpha_bar_t, alpha_bar_prev',
        '确定性 eta=0 DDIM。先预测 x0=(xt-sqrt(1-at)*eps)/sqrt(at)，再返回 sqrt(ap)*x0+sqrt(1-ap)*eps；不 clip x0。',
        'x0=(xt-math.sqrt(1-alpha_bar_t)*eps_pred)/math.sqrt(alpha_bar_t)\nreturn math.sqrt(alpha_bar_prev)*x0+math.sqrt(1-alpha_bar_prev)*eps_pred',
        [('中间步',"args=(torch.randn(2,3,4),torch.randn(2,3,4),0.4,0.6)"),('恢复 x0',"args=(torch.randn(3),torch.randn(3),0.8,1.)")])
    add('adaln_zero','自适应 LayerNorm Zero','扩散模型与 DiT','中等','x, residual, shift, scale, gate, eps=1e-6',
        'x/residual [B,T,D]，shift/scale/gate [B,D]。返回 residual+gate[:,None]*((1+scale[:,None])*LN(x)+shift[:,None])，LN 无仿射，方差 correction=0。这是已给调制参数的 AdaLN-Zero 残差核心。',
        'z=F.layer_norm(x,(x.shape[-1],),eps=eps)\nreturn residual+gate[:,None]*((1+scale[:,None])*z+shift[:,None])',
        [('非零调制',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(2,3,4,requires_grad=True))+tuple(torch.randn(2,4,requires_grad=True) for _ in range(3))"),('零门恒等',"args=(torch.randn(1,2,3),torch.randn(1,2,3),torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1,3))")],gradient=True)
    add('batchnorm','实现 BatchNorm','Transformer 内部机制','中等','x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5',
        'x [N,C]，N>1。训练输出用总体方差；更新 running_var 使用无偏方差，running=(1-m)*running+m*batch。推理使用 running 统计。返回 (output,new_running_mean,new_running_var)，统计需 detach，不原地修改。',
        '''if training:
    mean=x.mean(0); var=x.var(0,unbiased=False)
    rm=(1-momentum)*running_mean+momentum*mean.detach()
    rv=(1-momentum)*running_var+momentum*x.detach().var(0,unbiased=True)
else:
    mean=running_mean; var=running_var; rm=running_mean; rv=running_var
return (x-mean)*torch.rsqrt(var+eps)*weight+bias,rm,rv''',
        [('训练统计',"args=(torch.randn(5,3,requires_grad=True),torch.ones(3),torch.zeros(3),torch.zeros(3),torch.ones(3),True)"),('推理统计',"args=(torch.randn(2,3,requires_grad=True),torch.randn(3),torch.randn(3),torch.randn(3),torch.rand(3)+0.1,False)")],gradient=True)
    for slug,title,body,statement in [
        ('max_pool2d','二维最大池化','windows=x.unfold(2,kernel_size,stride).unfold(3,kernel_size,stride)\nreturn windows.flatten(-2).max(-1).values','x [N,C,H,W]，无 padding，窗口 kernel_size、步长 stride，向下取整；禁止调用 max_pool2d。'),
        ('vit_patch','ViT Patch Embedding','n,c,h,w=x.shape\np=kernel_size\nreturn x.unfold(2,p,stride).unfold(3,p,stride).permute(0,2,3,1,4,5).reshape(n,-1,c*p*p)','本题考 patchify，不含线性投影。x [N,C,H,W]，kernel_size=stride=p，H/W 均可整除 p。输出 [N,num_patches,C*p*p]，patch 顺序先行后列，内部按 C,ph,pw 展平。')]:
        add(slug,title,'Vision Transformer','简单','x, kernel_size, stride',statement,body,
            [('多通道',"args=(torch.arange(96.).reshape(2,3,4,4).requires_grad_(),2,2)"),('小窗口',"args=(torch.randn(1,2,4,6,requires_grad=True),2,2)")],gradient=True)
    add('conv2d','二维卷积','Vision Transformer','中等','x, weight, bias, stride=1, padding=0',
        'x [N,Cin,H,W]，weight [Cout,Cin,Kh,Kw]；实现互相关式 conv2d，无 dilation，groups=1。禁止调用 conv2d，可用 unfold。',
        'kh,kw=weight.shape[-2:]\nz=F.pad(x,(padding,padding,padding,padding))\nwindows=z.unfold(2,kh,stride).unfold(3,kw,stride)\nreturn torch.einsum("nchwij,ocij->nohw",windows,weight)+bias[None,:,None,None]',
        [('矩形核与 padding',"args=(torch.randn(2,2,5,6,requires_grad=True),torch.randn(3,2,2,3,requires_grad=True),torch.randn(3,requires_grad=True),2,1)"),('1x1',"args=(torch.randn(1,2,3,3,requires_grad=True),torch.randn(4,2,1,1,requires_grad=True),torch.zeros(4))")],gradient=True)
    add('depthwise_conv','深度可分离卷积','Vision Transformer','中等','x, depthwise, pointwise',
        'x [N,C,H,W]，depthwise [C,1,K,K]，pointwise [O,C,1,1]，K 为奇数。先逐通道卷积（padding=K//2），再 1x1 混合，无 bias，stride=1。不能调用 conv2d。',
        'k=depthwise.shape[-1]; p=k//2\nwindows=F.pad(x,(p,p,p,p)).unfold(2,k,1).unfold(3,k,1)\nz=torch.einsum("nchwij,cij->nchw",windows,depthwise[:,0])\nreturn torch.einsum("nchw,oc->nohw",z,pointwise[:,:,0,0])',
        [('逐通道与混合',"args=(torch.randn(2,3,4,4,requires_grad=True),torch.randn(3,1,3,3,requires_grad=True),torch.randn(5,3,1,1,requires_grad=True))"),('单通道',"args=(torch.randn(1,1,3,3,requires_grad=True),torch.randn(1,1,1,1,requires_grad=True),torch.randn(2,1,1,1,requires_grad=True))")],gradient=True)
    add('cross_attention','多头交叉注意力','注意力与位置编码','中等','q, k, v',
        '输入已经投影并拆头：q [B,H,T,D]，k/v [B,H,S,D]，允许 T!=S。返回 scaled dot-product attention，不使用因果 mask。',
        'return (q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])).softmax(-1)@v',
        [('不同序列长度',"args=(torch.randn(2,3,4,5,requires_grad=True),torch.randn(2,3,7,5,requires_grad=True),torch.randn(2,3,7,5,requires_grad=True))"),('单个 key',"args=(torch.randn(1,2,3,4,requires_grad=True),torch.randn(1,2,1,4,requires_grad=True),torch.randn(1,2,1,4,requires_grad=True))")],gradient=True)
    add('mlp','SwiGLU MLP','Transformer 内部机制','中等','x, gate_weight, up_weight, down_weight',
        'x [...,D]，gate/up [D,F]，down [F,D]；返回 (SiLU(x@gate)*(x@up))@down，无 bias。',
        'g=x@gate_weight\nreturn ((g*g.sigmoid())*(x@up_weight))@down_weight',
        [('前向和梯度',"args=(torch.randn(2,3,4,requires_grad=True),torch.randn(4,6,requires_grad=True),torch.randn(4,6,requires_grad=True),torch.randn(6,4,requires_grad=True))"),('小维度',"args=(torch.randn(1,2,requires_grad=True),torch.randn(2,3),torch.randn(2,3),torch.randn(3,2))")],gradient=True)
    add('linear_regression','多元线性回归与反向传播','经典手撕','中等','x, y, weight, bias',
        'x [N,D]，y [N,O]，weight [D,O]，bias [O]。返回 (mean MSE, dweight, dbias)，手动推导梯度，不能用 autograd。',
        'err=x@weight+bias-y\nreturn err.square().mean(),2*x.T@err/err.numel(),2*err.sum(0)/err.numel()',
        [('多输出',"args=(torch.randn(5,3),torch.randn(5,2),torch.randn(3,2),torch.randn(2))"),('单样本',"args=(torch.tensor([[2.]]),torch.tensor([[1.]]),torch.tensor([[3.]]),torch.tensor([1.]))")])
    from catalog_advanced import populate as advanced
    advanced(add)

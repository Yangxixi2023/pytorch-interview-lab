"""题目输入和返回值说明；由题目定义使用，不改变判题规则。"""

IO = {}


def describe(slugs, inputs, returns):
    for slug in slugs.split():
        IO[slug] = {'inputs': [dict(name=n, type=t, shape=s, description=d) for n,t,s,d in inputs],
                    'returns': [dict(name=n, type=t, shape=s, description=d) for n,t,s,d in returns]}


def tensor(name, shape, meaning, dtype='浮点 Tensor'):
    return name,dtype,shape,meaning


def scalar(name, meaning, kind='float'):
    return name,kind,'—',meaning


loss=[tensor('loss','[]','零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。')]
describe('ppo_loss',[
    tensor('new_logp','[B, T]','当前策略对已采样 token 的自然对数概率；需要保留梯度。'),
    tensor('old_logp','[B, T]','采样时旧策略的 log probability；计算时 detach。'),
    tensor('advantages','[B, T]','逐 token 优势，可正可负；计算时 detach。'),
    tensor('mask','[B, T]','True 为有效 token，False 为 padding；整个 batch 至少一个 True。','bool Tensor'),
    scalar('clip_eps','对称裁剪半径；ratio 裁剪到 [1-clip_eps, 1+clip_eps]。')],loss)
describe('gspo_loss dapo_loss grpo_loss',[
    tensor('new_logp','[B, T]','当前策略的 token log probability；保留梯度。'),
    tensor('old_logp','[B, T]','旧策略 token log probability；detach 后使用。'),
    tensor('advantages','[B]','每条序列一个已标准化优势；不是 [B,T]；detach 后使用。'),
    tensor('mask','[B, T]','True 为有效 token；每条序列至少一个 True。','bool Tensor'),
    scalar('clip_low','ratio 的下界为 1-clip_low。'),scalar('clip_high','ratio 的上界为 1+clip_high。')],loss)
describe('dpo_loss',[tensor(n,'[B]',d) for n,d in [
    ('chosen','当前策略对偏好回答的整序列 log probability 之和；保留梯度。'),
    ('rejected','当前策略对非偏好回答的整序列 log probability 之和；保留梯度。'),
    ('ref_chosen','参考策略偏好回答 log probability；detach。'),
    ('ref_rejected','参考策略非偏好回答 log probability；detach。')]]+[scalar('beta','偏好 margin 的缩放系数。')],loss)
describe('gae_advantage',[
    tensor('rewards','[T, B]','每个时间步的奖励；本题时间维在前。'),tensor('values','[T+1, B]','状态价值，最后一行用于末端 bootstrap。'),
    tensor('terminated','[T, B]','True 表示该步后真正终止，阻断 bootstrap 与优势递推；不表示时间截断。','bool Tensor'),
    scalar('gamma','奖励折扣系数。'),scalar('lam','GAE 衰减系数。')],
    [tensor('advantages','[T, B]','优势张量；必须 detach。'),tensor('returns','[T, B]','advantages + values[:-1]；必须 detach。')])
describe('reward_model',[tensor('chosen_rewards','[B]','偏好回答的奖励。'),tensor('rejected_rewards','[B]','非偏好回答的奖励。')],loss)
describe('softmax',[tensor('x','任意形状','输入 logits，可能包含很大的正数。'),scalar('dim','进行归一化的维度，支持负索引。','int')],
    [tensor('probabilities','与 x 相同','沿 dim 的和为 1，保留输入梯度；不能调用现成 softmax。')])
for slug,extra in [('cross_entropy',[]),('label_smoothing',[scalar('smoothing','平滑比例，标签分布为 (1-smoothing)*one_hot+smoothing/C。')]),('focal_loss',[scalar('gamma','Focal 调制指数；gamma=0 退化为交叉熵。')])]:
    describe(slug,[tensor('logits','[N, C]','未经 softmax 的分类分数。'),tensor('targets','[N]','每个样本的正确类别，取值 0..C-1。','int64 Tensor')]+extra,loss)
describe('relu gelu',[tensor('x','任意形状','输入激活值，保留梯度。')],[tensor('output','与 x 相同','逐元素激活，保持 dtype、device 和形状。')])
describe('swiglu',[tensor('x','[..., 2D]','最后一维为偶数；前半为 a，后半为 b。')],[tensor('output','[..., D]','SiLU(a)*b，最后一维减半，保留梯度。')])
describe('rmsnorm',[tensor('x','[..., D]','沿最后一维计算均方，不减均值。'),tensor('weight','[D]','逐特征缩放权重。'),scalar('eps','加在均方内部的稳定项。')],
    [tensor('output','与 x 相同','归一化后乘 weight；x 和 weight 均可求导。')])
describe('layernorm',[tensor('x','[..., D]','只归一化最后一维，使用总体方差。'),tensor('weight','[D]','缩放参数。'),tensor('bias','[D]','偏移参数。'),scalar('eps','加在方差内部的稳定项。')],[tensor('output','与 x 相同','LayerNorm 的仿射输出；保留输入和参数梯度。')])
describe('linear',[tensor('x','[..., Din]','任意前导 batch 维度。'),tensor('weight','[Dout, Din]','注意权重以输出维度在前。'),tensor('bias','[Dout]','输出偏移。')],[tensor('output','[..., Dout]','x @ weight.T + bias，保留梯度。')])
describe('embedding',[tensor('ids','任意形状','词表索引，取值 0..V-1；可以重复。','int64 Tensor'),tensor('weight','[V, D]','可训练的 embedding 表。')],[tensor('output','[*ids.shape, D]','查表结果；重复索引对应 weight 梯度需累加。')])
describe('attention',[tensor('q','[B, Q, D]','查询。'),tensor('k','[B, K, D]','键；K 可以不等于 Q。'),tensor('v','[B, K, Dv]','值；Dv 可以不等于 D。')],[tensor('output','[B, Q, Dv]','非因果 scaled dot-product attention，保留 q/k/v 梯度。')])
for slug,extra in [('causal_attention',[]),('sliding_window',[scalar('window','可见窗口长度，包含当前位置；为正整数。','int')]),('alibi',[tensor('slopes','[H]','每个 head 的位置惩罚斜率。')])]:
    describe(slug,[tensor('q','[B, H, T, D]','已拆头的 query。'),tensor('k','[B, H, T, D]','已拆头的 key。'),tensor('v','[B, H, T, D]','已拆头的 value。')]+extra,
        [tensor('output','[B, H, T, D]','因果注意力输出；不允许读取未来位置，保留梯度。')])
describe('mha',[tensor('x','[B, T, D]','未投影输入。')]+[tensor(n,'[D, D]','按 x @ '+n+' 使用；不转置。') for n in ('wq','wk','wv','wo')]+[scalar('heads','head 数 H，D 必须可被 H 整除。','int')],
    [tensor('output','[B, T, D]','非因果多头注意力合并后经过 wo 投影。')])
describe('gqa',[tensor('q','[B, Hq, T, D]','Hq 必须可被 Hkv 整除。'),tensor('k','[B, Hkv, S, D]','连续 Hq/Hkv 个 query head 共用一个 KV head。'),tensor('v','[B, Hkv, S, D]','与 key 的头数和序列长度一致。')],
    [tensor('output','[B, Hq, T, D]','非因果 GQA 输出，不合并 head 维。')])
describe('rope',[tensor('x','[B, H, T, D]','D 为偶数，采用相邻偶奇维配对。'),tensor('positions','[T]','各 token 的绝对位置，可从非零位置开始。','整数 Tensor'),scalar('base','旋转频率的底数。')],
    [tensor('output','与 x 相同','旋转后的张量；保留 x 的梯度。')])
describe('lora',[tensor('x','[..., Din]','输入特征。'),tensor('weight','[Dout, Din]','冻结的基座权重，必须 detach。'),tensor('a','[r, Din]','低秩降维权重。'),tensor('b','[Dout, r]','低秩升维权重。'),scalar('alpha','低秩分支缩放分子，实际系数 alpha/r。')],[tensor('output','[..., Dout]','基座输出加低秩更新；x/a/b 保留梯度，weight 不求导。')])
describe('dropout',[tensor('x','任意形状','待处理激活。'),scalar('p','丢弃概率，0<=p<1。'),scalar('training','True 训练，False 推理。','bool'),tensor('keep_mask','与 x 相同','预先采样的保留掩码；本题不要求重新随机采样。','bool Tensor')],
    [tensor('output','与 x 相同','训练：x*keep_mask/(1-p)；推理：x。保留 x 梯度。')])
describe('flow_matching',[tensor('pred_velocity','任意形状','可训练的预测速度。'),tensor('x0','与 pred_velocity 相同','路径起点，视为常量。'),tensor('x1','与 pred_velocity 相同','路径终点，视为常量。')],loss)
describe('contrastive_loss',[tensor('queries','[N, D]','需要在最后一维 L2 normalize。'),tensor('keys','[N, D]','第 i 个 key 是第 i 个 query 的正样本。'),scalar('temperature','正的温度系数。')],loss)
describe('gradient_accumulation',[tensor('x','[N, D]','输入特征。'),tensor('y','[N, O]','回归目标。'),tensor('weight','[D, O]','线性映射参数。'),scalar('microbatch_size','正整数；最后一个 microbatch 可以不足此大小。','int')],
    [tensor('grad_weight','[D, O]','全 batch、所有输出元素的 mean MSE 对 weight 的手动梯度；不要用 autograd。')])
describe('gradient_clipping',[('grads','list[Tensor]','各元素形状可不同','所有需要一起计算全局 L2 范数的梯度。'),scalar('max_norm','允许的最大范数。'),scalar('eps','加在分母中的稳定项。')],
    [('clipped_grads','list[Tensor]','与 grads 一一对应','裁剪后的新梯度列表，不原地修改输入。'),tensor('total_norm','[]','裁剪之前的全局 L2 范数。')])
describe('adam',[tensor(n,'与 param 相同',d) for n,d in [('param','当前参数。'),('grad','当前梯度。'),('m','上一时刻一阶动量。'),('v','上一时刻二阶动量。')]]+
    [scalar('step','本次更新步数，从 1 开始。','int'),scalar('lr','学习率。'),scalar('beta1','一阶动量衰减系数。'),scalar('beta2','二阶动量衰减系数。'),scalar('eps','稳定项，放在 sqrt 外。')],
    [tensor('new_param','与 param 相同','更新后的参数，不修改 param。'),tensor('new_m','与 m 相同','更新后的一阶动量。'),tensor('new_v','与 v 相同','更新后的二阶动量。')])
describe('cosine_lr',[scalar('step','当前步，0<=step<=total_steps。','int'),scalar('warmup_steps','预热步数，0<=warmup_steps<total_steps。','int'),scalar('total_steps','总步数。','int'),scalar('max_lr','预热结束时的学习率。'),scalar('min_lr','余弦阶段终点学习率。')],
    [('learning_rate','float','—','当前步的 Python 浮点学习率，不返回 Tensor。')])
describe('weight_init',[tensor('standard_normal','任意形状','已采样的标准正态噪声；不要再次采样。'),scalar('fan_in','输入连接数，正整数。','int')],[tensor('weight','与 standard_normal 相同','按 sqrt(2/fan_in) 缩放。')])
describe('sinusoidal_pe',[scalar('length','序列长度。','int'),scalar('dim','偶数特征维度。','int'),scalar('base','频率底数。')],[tensor('encoding','[length, dim]','float32 位置编码，偶数列 sin、奇数列 cos。')])
describe('ntk_rope',[scalar('dim','偶数且大于 2。','int'),scalar('scale','上下文缩放倍数，>=1。'),scalar('base','原始 RoPE 底数。')],[tensor('inverse_frequencies','[dim/2]','float32 的缩放后逆频率。')])
describe('noise_schedule',[scalar('steps','正的总步数。','int'),scalar('beta_start','第一个 beta。'),scalar('beta_end','最后一个 beta（steps=1 时只用起点）。')],
    [tensor('betas','[steps]','float32 线性 beta 序列。'),tensor('alphas','[steps]','1-betas。'),tensor('alpha_bars','[steps]','alphas 的前缀连乘。')])
describe('ddim_step',[tensor('xt','任意形状','时刻 t 的噪声样本。'),tensor('eps_pred','与 xt 相同','模型预测的噪声。'),scalar('alpha_bar_t','当前累计 alpha，>0。'),scalar('alpha_bar_prev','前一时刻累计 alpha，<=1。')],[tensor('x_prev','与 xt 相同','eta=0 的确定性 DDIM 更新结果，不截断预测 x0。')])
describe('adaln_zero',[tensor('x','[B, T, D]','待归一化激活。'),tensor('residual','[B, T, D]','残差分支输入。')]+[tensor(n,'[B, D]',d) for n,d in [('shift','每样本特征偏移。'),('scale','每样本特征缩放增量。'),('gate','残差更新的门控。')]]+[scalar('eps','LayerNorm 稳定项。')],
    [tensor('output','[B, T, D]','门控调制后的残差输出；gate=0 时等于 residual。')])
describe('batchnorm',[tensor('x','[N, C]','N>1；沿 N 维统计。')]+[tensor(n,'[C]',d) for n,d in [('weight','缩放。'),('bias','偏移。'),('running_mean','历史均值。'),('running_var','历史方差。')]]+[scalar('training','是否使用 batch 统计。','bool'),scalar('momentum','新统计在 running 更新中的权重。'),scalar('eps','方差稳定项。')],
    [tensor('output','[N, C]','训练归一化使用总体方差，保留梯度。'),tensor('new_running_mean','[C]','更新后的均值；训练统计 detach，不修改输入。'),tensor('new_running_var','[C]','更新使用无偏方差；训练统计 detach，不修改输入。')])
describe('max_pool2d',[tensor('x','[N, C, H, W]','图像输入，无 padding。'),scalar('kernel_size','正方形池化窗口边长。','int'),scalar('stride','滑动步长。','int')],
    [tensor('output','[N, C, Hout, Wout]','Hout=floor((H-kernel_size)/stride)+1，Wout 同理。')])
describe('vit_patch',[tensor('x','[N, C, H, W]','H、W 都可被 patch 边长 p 整除。'),scalar('kernel_size','patch 边长 p。','int'),scalar('stride','本题等于 kernel_size。','int')],
    [tensor('patches','[N, (H/p)*(W/p), C*p*p]','patch 先行后列；每个 patch 内按 C、ph、pw 展平，不含投影。')])
describe('conv2d',[tensor('x','[N, Cin, H, W]','输入图像。'),tensor('weight','[Cout, Cin, Kh, Kw]','互相关卷积核，不翻转。'),tensor('bias','[Cout]','输出偏移。'),scalar('stride','两个空间维度使用相同步长。','int'),scalar('padding','四边相同的零 padding。','int')],
    [tensor('output','[N, Cout, Hout, Wout]','Hout=floor((H+2*padding-Kh)/stride)+1，Wout 同理；groups=1。')])
describe('depthwise_conv',[tensor('x','[N, C, H, W]','输入图像。'),tensor('depthwise','[C, 1, K, K]','K 为奇数，逐通道卷积，padding=K//2。'),tensor('pointwise','[O, C, 1, 1]','逐点 1×1 通道混合核。')],
    [tensor('output','[N, O, H, W]','深度卷积后做逐点卷积，空间大小不变，无 bias。')])
describe('cross_attention',[tensor('q','[B, H, T, D]','已投影的 query。'),tensor('k','[B, H, S, D]','已投影的 key，S 可以不等于 T。'),tensor('v','[B, H, S, D]','已投影的 value。')],[tensor('output','[B, H, T, D]','非因果注意力；不合并 head 维。')])
describe('mlp',[tensor('x','[..., D]','输入特征。'),tensor('gate_weight','[D, F]','门控投影。'),tensor('up_weight','[D, F]','上投影。'),tensor('down_weight','[F, D]','下投影。')],[tensor('output','与 x 相同','(SiLU(x@gate_weight)*(x@up_weight))@down_weight。')])
describe('linear_regression',[tensor('x','[N, D]','输入特征。'),tensor('y','[N, O]','目标输出。'),tensor('weight','[D, O]','线性回归权重。'),tensor('bias','[O]','偏移。')],
    [tensor('loss','[]','所有 N*O 元素的 mean MSE。'),tensor('dweight','[D, O]','手动推导的 weight 梯度，不用 autograd。'),tensor('dbias','[O]','手动推导的 bias 梯度。')])

from problem_io_advanced import extend
extend(describe,tensor,scalar,loss)

describe('tensor_heads',[tensor('x','[B, T, D]','可能不是连续张量，D 可被 heads 整除。'),scalar('heads','head 数 H，正整数。','int')],
    [tensor('split_heads','[B, H, T, D/H]','拆头并交换时间与头维度，保留梯度。'),tensor('merged','[B, T, D]','合并还原的张量，值与输入一致。')])
describe('ffn',[tensor('x','[..., D]','输入。'),tensor('w1','[D, F]','上投影。'),tensor('b1','[F]','上投影偏移。'),tensor('w2','[F, D]','下投影。'),tensor('b2','[D]','输出偏移。')],
    [tensor('output','与 x 相同','精确 GELU 的双线性层输出，保留梯度，不包含残差。')])
describe('entropy_loss',[tensor('logits','[B, T, V]','未归一化的词表分数。'),tensor('mask','[B, T]','至少一个 True；仅在有效位置平均。','bool Tensor')],
    [tensor('entropy','[]','正号平均熵，零维可导张量；不是负熵正则 loss。')])
describe('sft_loss',[tensor('logits','[B, T, V]','t 位置预测 t+1 标签，T>=2。'),tensor('labels','[B, T]','目标 token ID，或 ignore_index；shift 后至少一个有效位置。','int64 Tensor'),scalar('ignore_index','屏蔽 prompt/padding 的特殊标签值。','int')],loss)

FORMULAS = {
    'ppo_loss': [r'r_{bt}=\exp(\ell_{bt}-\ell^{\mathrm{old}}_{bt})', r'L=-\frac{\sum_{bt}m_{bt}\min(r_{bt}A_{bt},\operatorname{clip}(r_{bt},1-\epsilon,1+\epsilon)A_{bt})}{\sum_{bt}m_{bt}}'],
    'gspo_loss': [r'n_b=\sum_t m_{bt},\qquad s_b=\exp\left(\frac{\sum_t m_{bt}(\ell_{bt}-\ell^{\mathrm{old}}_{bt})}{n_b}\right)',r'L=-\frac{1}{B}\sum_b\min\left(s_b A_b,\operatorname{clip}(s_b,1-\epsilon_l,1+\epsilon_h)A_b\right)'],
    'dapo_loss': [r'r_{bt}=\exp(\ell_{bt}-\ell^{\mathrm{old}}_{bt})',r'L=-\frac{\sum_{bt}m_{bt}\min(r_{bt}A_b,\operatorname{clip}(r_{bt},1-\epsilon_l,1+\epsilon_h)A_b)}{\sum_{bt}m_{bt}}'],
    'grpo_loss': [r'r_{bt}=\exp(\ell_{bt}-\ell^{\mathrm{old}}_{bt})',r'L=-\frac{1}{B}\sum_b\frac{\sum_t m_{bt}\min(r_{bt}A_b,\operatorname{clip}(r_{bt},1-\epsilon_l,1+\epsilon_h)A_b)}{\sum_t m_{bt}}'],
    'dpo_loss':[r'L=-\frac{1}{B}\sum_b\log\sigma\left(\beta\left[(\ell^+_b-\ell^-_b)-(\ell^{+,\mathrm{ref}}_b-\ell^{-,\mathrm{ref}}_b)\right]\right)'],
    'gae_advantage':[r'\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t',r'A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1},\qquad R_t=A_t+V_t'],
    'rmsnorm':[r'y=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot w'],
    'layernorm':[r'y=\frac{x-\mu}{\sqrt{\operatorname{mean}((x-\mu)^2)+\epsilon}}\odot w+b'],
    'attention':[r'Y=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right)V'],
    'softmax':[r'p_i=\frac{\exp(x_i-\max_j x_j)}{\sum_j\exp(x_j-\max_k x_k)}'],
}

# 训练岗位面试问答

36 道问答，按机器学习、预训练、后训练分类。

## 如何判断欠拟合还是过拟合？

同时看训练误差、验证误差及它们随数据量和训练步数的变化。训练也差通常优先查优化与表达能力；训练好但验证差优先查泛化、数据分布和泄漏。

资料：[scikit-learn：学习曲线](https://scikit-learn.org/stable/modules/learning_curve.html)

## 训练集、验证集、测试集各做什么？

训练集拟合参数，验证集用于选择超参数与模型，测试集用于最后评估未见数据表现。反复看测试集并调整模型，会把它变成隐含验证集。

资料：[scikit-learn：交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)

## 数据泄漏通常发生在哪里？

在全量数据上拟合标准化、缺失值填充、特征选择或降维，都可能让验证信息进入训练。正确做法是在训练折上 fit，在验证折上仅 transform。

资料：[scikit-learn：数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)

## L1 和 L2 正则化有什么区别？

L1 惩罚绝对值，通常产生稀疏系数；L2 惩罚平方，倾向于连续缩小权重并缓解共线性。正则强度需要验证集选择。

资料：[scikit-learn：线性模型](https://scikit-learn.org/stable/modules/linear_model.html)

## 逻辑回归为什么用于分类？

它对特征的线性组合建模 log-odds，经 sigmoid 得到二分类概率，并通常通过对数似然/交叉熵训练。分类标签由概率与阈值共同决定。

资料：[scikit-learn：线性模型](https://scikit-learn.org/stable/modules/linear_model.html)

## 类别不平衡时为什么不能只看准确率？

多数类占比很高时，模型全预测多数类也能得到高准确率。应根据漏报与误报成本查看 precision、recall、F1、PR 曲线及混淆矩阵。

资料：[scikit-learn：评估指标](https://scikit-learn.org/stable/modules/model_evaluation.html)

## ROC-AUC、PR-AUC 和概率校准有何区别？

AUC 类指标衡量跨阈值的区分或排序表现；概率校准检查预测0.8的样本是否约80%为正。排序好不保证概率准确。

资料：[scikit-learn：评估指标](https://scikit-learn.org/stable/modules/model_evaluation.html) · [scikit-learn：概率校准](https://scikit-learn.org/stable/modules/calibration.html)

## 交叉验证为什么要和最终测试分开？

交叉验证可评估不同划分上的波动并选择模型，但调参本身会适配这些验证结果。独立测试或嵌套交叉验证用于减少选择偏差。

资料：[scikit-learn：交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)

## Bagging、随机森林与 Boosting 如何比较？

Bagging 对重采样训练的模型做集成；随机森林还随机选择候选特征。Boosting 逐轮构建模型，梯度提升针对当前损失的负梯度拟合增量。

资料：[scikit-learn：集成学习](https://scikit-learn.org/stable/modules/ensemble.html)

## PCA 保留高方差方向，是否一定保留预测信息？

不一定。PCA 是无监督线性降维，目标是解释输入方差；标签信息可能存在于低方差方向，因此维数仍需通过下游任务验证。

资料：[scikit-learn：PCA](https://scikit-learn.org/stable/modules/decomposition.html)

## 交叉熵、KL 和熵有什么关系？

对固定目标分布 p，交叉熵 H(p,q)=H(p)+KL(p||q)，所以最小化交叉熵等价于最小化该方向的 KL。模型自身熵 H(q) 则衡量输出分布的不确定性。

资料：[scikit-learn：评估指标](https://scikit-learn.org/stable/modules/model_evaluation.html)

## 固定随机种子能否保证完全复现？

不能。还受库版本、平台、设备和非确定性算子影响。应同时控制相关随机源，并记录运行环境；需要时使用确定性算法设置。

资料：[PyTorch：可复现性](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## Decoder-only 预训练目标是什么？

通过因果上下文预测下一个 token，最小化有效目标 token 的负对数似然。训练可以并行计算各位置 loss，但注意力不能访问目标位置之后的内容。

资料：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 模型大小、数据量和算力怎样分配？

固定训练计算量时，需要平衡参数量与训练 token 数。Chinchilla 的经验结论来自特定模型、数据与预算实验；实际方案还要考虑数据质量和推理成本。

资料：[Chinchilla：训练计算分配](https://arxiv.org/abs/2203.15556)

## 预训练数据清洗应关注什么？

关注来源质量、重复内容、文本可用性、语言和领域比例，以及评测污染。过滤策略会改变数据分布，应通过质量抽样和下游评测共同验证。

资料：[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

## Attention 为什么除以 sqrt(d)？

在分量近似独立且尺度适当时，点积方差随维度增长。除以 sqrt(d) 控制分数尺度，降低 softmax 过度饱和的倾向。

资料：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## LayerNorm 和 RMSNorm 有何区别？

LayerNorm 对特征减均值并按标准差缩放；RMSNorm 不做均值中心化，只按均方根归一化。两者统计都不依赖 batch 大小。

资料：[RMSNorm](https://arxiv.org/abs/1910.07467)

## Adam 与 AdamW 的权重衰减有何不同？

AdamW 将参数衰减与自适应梯度更新分开；把 L2 项直接加到 Adam 的梯度中，会经过动量和自适应缩放，通常不等价。

资料：[Decoupled Weight Decay](https://arxiv.org/abs/1711.05101)

## 混合精度中 autocast 与 loss scaling 各做什么？

autocast 为不同算子选择执行精度；loss scaling 放大 loss，减轻小梯度在低精度下下溢，并在更新前还原梯度。它们是不同机制。

资料：[PyTorch：混合精度训练](https://docs.pytorch.org/docs/main/notes/amp_examples.html)

## 梯度累积何时等价于大 batch？

在损失权重和更新频率一致等条件下，microbatch 梯度之和可复现大 batch 梯度。变长序列必须按有效 token 数正确加权。

资料：[PyTorch：混合精度训练](https://docs.pytorch.org/docs/main/notes/amp_examples.html)

## 训练显存主要被什么占用？

主要包括参数、梯度、优化器状态、激活及临时缓冲。参数量只能解释一部分；序列长度、batch 和实现方式也会显著影响峰值。

资料：[ZeRO](https://arxiv.org/abs/1910.02054)

## FlashAttention 为什么能减少显存和时间？

通过分块、重排计算和 online softmax，减少中间注意力矩阵在高带宽显存中的读写。它计算精确注意力，不是把注意力改成稀疏近似。

资料：[FlashAttention](https://arxiv.org/abs/2205.14135)

## 长上下文训练除了改位置编码还需要关注什么？

还要考虑长序列数据分布、计算与显存成本，以及长距离能力评估。位置编码的外推策略不能替代长上下文任务验证。

资料：[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

## 训练 loss 突然出现 NaN，如何定位？

先区分非有限输入、前向数值溢出和梯度问题，定位首次出现非有限值的张量。检查学习率、log/除法、mask、精度与梯度缩放。

资料：[PyTorch：混合精度训练](https://docs.pytorch.org/docs/main/notes/amp_examples.html)

## SFT、偏好优化和 RLVR 分别使用什么监督？

SFT 模仿示范回答；偏好优化比较候选回答；RLVR 使用可验证的结果信号，例如程序测试或数学答案。它们的信号粒度与误差来源不同。

资料：[InstructGPT](https://arxiv.org/abs/2203.02155) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## SFT 的 prompt mask 与 label shift 怎样实现？

如果只训练回答部分，可以把 prompt 和 padding 标签设为 ignore_index。因果模型用位置 t 的 logits 预测位置 t+1 标签；有效 token 平均的分母也要随 mask 改变。

资料：[InstructGPT](https://arxiv.org/abs/2203.02155)

## PPO 中 old policy 和 reference policy 是一回事吗？

不是。old policy 对应生成 rollout 的行为策略，用于 importance ratio；reference policy 常用于约束偏离初始或参考模型的程度。

资料：[PPO](https://arxiv.org/abs/1707.06347) · [InstructGPT](https://arxiv.org/abs/2203.02155)

## PPO clipped objective 为什么要取 minimum？

它对原 surrogate 和裁剪 surrogate 取较保守的一项。正负优势下激活的裁剪方向不同，不能简单地先 clip ratio 再乘 advantage。

资料：[PPO](https://arxiv.org/abs/1707.06347)

## GAE 中 gamma 与 lambda 有何作用？

gamma 折扣未来奖励；lambda 控制多步 TD 残差的混合长度，影响优势估计的偏差与方差。递推需要在真正终止处切断。

资料：[Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## GRPO 相比 PPO 省掉了什么？

GRPO 使用同一 prompt 多个回答的组内相对奖励构造优势，减少对独立价值模型的依赖。训练行为仍取决于具体 loss、归约和 KL 配置。

资料：[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

## DAPO 的关键改动是什么？

包括非对称裁剪、动态采样、token 级 loss 归约和超长奖励处理。它们分别影响探索、有效训练样本、长度权重和长回答反馈。

资料：[DAPO](https://arxiv.org/abs/2503.14476)

## GSPO 与 token-level ratio 的区别是什么？

GSPO 依据长度归一化序列似然构造 importance ratio，并在序列层裁剪和优化。核心变化是优化单位与奖励单位的对应关系。

资料：[GSPO](https://arxiv.org/abs/2507.18071)

## DPO 如何避免单独训练显式奖励模型？

DPO 利用带 KL 约束的奖励优化形式与最优策略之间的关系，把偏好学习写成策略与参考策略 log probability 差的分类目标。

资料：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

## 奖励模型升分但实际回答变差，可能是什么？

可能存在代理奖励与真实目标不一致，策略学会了利用奖励模型偏好或验证器漏洞。需要在独立评估上检查任务正确性和行为质量。

资料：[InstructGPT](https://arxiv.org/abs/2203.02155) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## LoRA 的 A/B 初始化和缩放如何解释？

LoRA 在冻结基座上训练低秩增量，通常以 alpha/r 缩放。常见初始化让一侧为零，使初始增量为零，同时后续梯度能够开启更新。

资料：[LoRA](https://arxiv.org/abs/2106.09685)

## QLoRA 相比 LoRA 额外做了什么？

QLoRA 在量化的冻结基座上训练低秩适配器，并提出 NF4、双重量化等降低存储开销的机制。计算时的 dtype 和权重存储格式要分开看。

资料：[QLoRA](https://arxiv.org/abs/2305.14314)

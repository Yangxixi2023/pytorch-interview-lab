# 训练岗位面试问答

36 道知识问答，按机器学习、预训练、后训练分类。根据原始论文与官方文档整理，不声称来自特定公司的真实面试。回答与追问由本项目重新组织；来源用于核对技术背景。

## q001 · 机器学习 · 如何判断欠拟合还是过拟合？

同时看训练误差、验证误差及它们随数据量和训练步数的变化。训练也差通常优先查优化与表达能力；训练好但验证差优先查泛化、数据分布和泄漏。

回答要点：

- 比较曲线前要确保训练/评估指标口径一致。
- 增加数据更可能缓解方差问题；并非所有训练问题都能靠加数据解决。

追问：为什么训练 loss 很低也可能是实现错误？

易错点：不能只凭一次验证分数给模型贴上“过拟合”标签。

资料：[scikit-learn：学习曲线](https://scikit-learn.org/stable/modules/learning_curve.html)

## q002 · 机器学习 · 训练集、验证集、测试集各做什么？

训练集拟合参数，验证集用于选择超参数与模型，测试集用于最后评估未见数据表现。反复看测试集并调整模型，会把它变成隐含验证集。

回答要点：

- 样本有用户、患者等分组时，拆分单位应与实际泛化目标一致。
- 时间序列预测通常按时间前后拆分，避免未来信息进入训练。

追问：同一用户多条样本随机划分会怎样？

易错点：分层抽样维持类别比例，不自动解决用户或时间泄漏。

资料：[scikit-learn：交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)

## q003 · 机器学习 · 数据泄漏通常发生在哪里？

在全量数据上拟合标准化、缺失值填充、特征选择或降维，都可能让验证信息进入训练。正确做法是在训练折上 fit，在验证折上仅 transform。

回答要点：

- 交叉验证时预处理也必须在每个训练折内拟合。
- Pipeline 可以把预处理与模型作为一个评估单元。

追问：目标编码怎样避免把标签泄漏进特征？

易错点：“没有直接使用测试标签”不代表没有泄漏。

资料：[scikit-learn：数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)

## q004 · 机器学习 · L1 和 L2 正则化有什么区别？

L1 惩罚绝对值，通常产生稀疏系数；L2 惩罚平方，倾向于连续缩小权重并缓解共线性。正则强度需要验证集选择。

回答要点：

- 特征尺度会影响惩罚的相对强度。
- 相关特征很多时，L1 的特征选择可能不稳定。

追问：为什么 ridge 能改善病态线性回归？

易错点：不要把正则化强度越大理解为泛化一定越好。

资料：[scikit-learn：线性模型](https://scikit-learn.org/stable/modules/linear_model.html)

## q005 · 机器学习 · 逻辑回归为什么用于分类？

它对特征的线性组合建模 log-odds，经 sigmoid 得到二分类概率，并通常通过对数似然/交叉熵训练。分类标签由概率与阈值共同决定。

回答要点：

- 非线性特征变换可以改变原输入空间中的决策边界。
- 阈值可以按误判成本调整，不必固定0.5。

追问：分类阈值改变会不会改变排序能力？

易错点：名字含“回归”不表示训练目标是普通最小二乘。

资料：[scikit-learn：线性模型](https://scikit-learn.org/stable/modules/linear_model.html)

## q006 · 机器学习 · 类别不平衡时为什么不能只看准确率？

多数类占比很高时，模型全预测多数类也能得到高准确率。应根据漏报与误报成本查看 precision、recall、F1、PR 曲线及混淆矩阵。

回答要点：

- 说明 positive class 的定义以及平均方式。
- 报告评估数据的类别比例和选阈值方式。

追问：宏平均和微平均分别重视什么？

易错点：F1 不考虑真负例，不能替代所有业务指标。

资料：[scikit-learn：评估指标](https://scikit-learn.org/stable/modules/model_evaluation.html)

## q007 · 机器学习 · ROC-AUC、PR-AUC 和概率校准有何区别？

AUC 类指标衡量跨阈值的区分或排序表现；概率校准检查预测0.8的样本是否约80%为正。排序好不保证概率准确。

回答要点：

- PR 的 precision 对正例比例敏感，跨数据集比较要小心。
- 概率用于风险定价或决策时，应另外检查校准。

追问：单调变换概率后，排序和校准会如何变化？

易错点：不能把 ROC-AUC 解释成预测概率正确的比例。

资料：[scikit-learn：评估指标](https://scikit-learn.org/stable/modules/model_evaluation.html) · [scikit-learn：概率校准](https://scikit-learn.org/stable/modules/calibration.html)

## q008 · 机器学习 · 交叉验证为什么要和最终测试分开？

交叉验证可评估不同划分上的波动并选择模型，但调参本身会适配这些验证结果。独立测试或嵌套交叉验证用于减少选择偏差。

回答要点：

- 保持分组、时间约束和预处理规则。
- 除了平均分，也看方差和各折失败模式。

追问：什么时候使用 GroupKFold 而不是 StratifiedKFold？

易错点：多试几百组参数之后，最好的 CV 分数仍可能乐观。

资料：[scikit-learn：交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)

## q009 · 机器学习 · Bagging、随机森林与 Boosting 如何比较？

Bagging 对重采样训练的模型做集成；随机森林还随机选择候选特征。Boosting 逐轮构建模型，梯度提升针对当前损失的负梯度拟合增量。

回答要点：

- Bagging 常用于降低不稳定学习器的方差。
- Boosting 的学习率、树深和轮数会共同影响拟合与泛化。

追问：GBDT 为什么既能回归也能分类？

易错点：不要把所有 boosting 都描述为同一种“重加权错分样本”算法。

资料：[scikit-learn：集成学习](https://scikit-learn.org/stable/modules/ensemble.html)

## q010 · 机器学习 · PCA 保留高方差方向，是否一定保留预测信息？

不一定。PCA 是无监督线性降维，目标是解释输入方差；标签信息可能存在于低方差方向，因此维数仍需通过下游任务验证。

回答要点：

- 通常先中心化，是否标准化取决于特征量纲和目标。
- 只能在训练数据上拟合 PCA。

追问：PCA 和监督式特征选择的目标有何不同？

易错点：不能把解释方差比例当作分类准确率保证。

资料：[scikit-learn：PCA](https://scikit-learn.org/stable/modules/decomposition.html)

## q011 · 机器学习 · 交叉熵、KL 和熵有什么关系？

对固定目标分布 p，交叉熵 H(p,q)=H(p)+KL(p||q)，所以最小化交叉熵等价于最小化该方向的 KL。模型自身熵 H(q) 则衡量输出分布的不确定性。

回答要点：

- one-hot 目标的交叉熵只取正确类别负 log probability。
- 鼓励探索通常在最小化目标中减去模型熵。

追问：标签平滑改变的是目标熵还是模型熵？

易错点：不要把 Entropy Loss 和监督交叉熵当成同一个函数。

资料：[scikit-learn：评估指标](https://scikit-learn.org/stable/modules/model_evaluation.html)

## q012 · 机器学习 · 固定随机种子能否保证完全复现？

不能。还受库版本、平台、设备和非确定性算子影响。应同时控制相关随机源，并记录运行环境；需要时使用确定性算法设置。

回答要点：

- Python、NumPy 与 PyTorch 可能有独立随机状态。
- 确定性实现有时会影响性能。

追问：为什么相同 seed 在 CPU 和 GPU 上仍可能不同？

易错点：复现一次运行不等于模型结论在多个种子上都稳定。

资料：[PyTorch：可复现性](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## q013 · 预训练 · Decoder-only 预训练目标是什么？

通过因果上下文预测下一个 token，最小化有效目标 token 的负对数似然。训练可以并行计算各位置 loss，但注意力不能访问目标位置之后的内容。

回答要点：

- 明确 logits 与 labels 的 shift，避免错一位。
- padding 和文档边界的处理决定哪些上下文和目标有效。

追问：为什么训练并行而自回归生成通常逐步进行？

易错点：并行训练不意味着允许未来 token 泄漏。

资料：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## q014 · 预训练 · 模型大小、数据量和算力怎样分配？

固定训练计算量时，需要平衡参数量与训练 token 数。Chinchilla 的经验结论来自特定模型、数据与预算实验；实际方案还要考虑数据质量和推理成本。

回答要点：

- 先明确优化的是训练预算还是全生命周期成本。
- 不要把某个 tokens/parameter 比例当成不变定律。

追问：为什么高推理量场景可能选择更小但训练更久的模型？

易错点：不同数据和架构下，经验缩放关系需要重新验证。

资料：[Chinchilla：训练计算分配](https://arxiv.org/abs/2203.15556)

## q015 · 预训练 · 预训练数据清洗应关注什么？

关注来源质量、重复内容、文本可用性、语言和领域比例，以及评测污染。过滤策略会改变数据分布，应通过质量抽样和下游评测共同验证。

回答要点：

- 文档级重复和局部重复可能需要不同处理。
- 保留训练/评测数据边界，避免用评测答案作为训练样本。

追问：强过滤可能误删哪些有价值的数据？

易错点：数据越干净不必然意味着覆盖越全面。

资料：[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

## q016 · 预训练 · Attention 为什么除以 sqrt(d)？

在分量近似独立且尺度适当时，点积方差随维度增长。除以 sqrt(d) 控制分数尺度，降低 softmax 过度饱和的倾向。

回答要点：

- 这里使用每个 head 的维度。
- 该推导依赖分量尺度假设，不是任意数据分布的精确恒等式。

追问：不缩放时对 softmax 梯度有何影响？

易错点：不能误用总 hidden size 代替 head dimension。

资料：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## q017 · 预训练 · LayerNorm 和 RMSNorm 有何区别？

LayerNorm 对特征减均值并按标准差缩放；RMSNorm 不做均值中心化，只按均方根归一化。两者统计都不依赖 batch 大小。

回答要点：

- 说明归一化维度、epsilon 位置和仿射参数。
- RMSNorm 省略中心化计算，但不能因此推断所有任务效果都更好。

追问：实现方差时 correction=0 和1会带来什么差异？

易错点：RMSNorm 不等于简单去掉 LayerNorm 的 bias。

资料：[RMSNorm](https://arxiv.org/abs/1910.07467)

## q018 · 预训练 · Adam 与 AdamW 的权重衰减有何不同？

AdamW 将参数衰减与自适应梯度更新分开；把 L2 项直接加到 Adam 的梯度中，会经过动量和自适应缩放，通常不等价。

回答要点：

- 普通 SGD 下的等价关系不能直接搬到 Adam。
- 讨论实现时明确 learning rate 与 weight decay 的乘法位置。

追问：为什么偏置和归一化参数常使用不同衰减设置？

易错点：不要把 Adam 的 L2 惩罚和 AdamW 视为同一更新式。

资料：[Decoupled Weight Decay](https://arxiv.org/abs/1711.05101)

## q019 · 预训练 · 混合精度中 autocast 与 loss scaling 各做什么？

autocast 为不同算子选择执行精度；loss scaling 放大 loss，减轻小梯度在低精度下下溢，并在更新前还原梯度。它们是不同机制。

回答要点：

- 梯度裁剪应作用于 unscale 后的梯度。
- 遇到非有限梯度时，scaler 可以跳过更新并调整 scale。

追问：BF16 和 FP16 对 loss scaling 的需求为何可能不同？

易错点：把整个模型直接转换成 half 不等于正确配置混合精度训练。

资料：[PyTorch：混合精度训练](https://docs.pytorch.org/docs/main/notes/amp_examples.html)

## q020 · 预训练 · 梯度累积何时等价于大 batch？

在损失权重和更新频率一致等条件下，microbatch 梯度之和可复现大 batch 梯度。变长序列必须按有效 token 数正确加权。

回答要点：

- 累计完再更新参数，而不是每个 microbatch 都 step。
- BatchNorm、随机层和数值舍入可能破坏严格等价。

追问：最后一个 microbatch 更小时应该怎样缩放？

易错点：直接平均各 microbatch 的 mean loss 可能给予小 batch 过高权重。

资料：[PyTorch：混合精度训练](https://docs.pytorch.org/docs/main/notes/amp_examples.html)

## q021 · 预训练 · 训练显存主要被什么占用？

主要包括参数、梯度、优化器状态、激活及临时缓冲。参数量只能解释一部分；序列长度、batch 和实现方式也会显著影响峰值。

回答要点：

- ZeRO 按阶段分片优化器状态、梯度、参数。
- 分片节省存储，但会引入通信与临时聚合开销。

追问：为什么减小 microbatch 后仍然可能 OOM？

易错点：不能只用参数量乘 dtype 字节数估算完整训练显存。

资料：[ZeRO](https://arxiv.org/abs/1910.02054)

## q022 · 预训练 · FlashAttention 为什么能减少显存和时间？

通过分块、重排计算和 online softmax，减少中间注意力矩阵在高带宽显存中的读写。它计算精确注意力，不是把注意力改成稀疏近似。

回答要点：

- 核心是 IO 与中间存储优化，不代表点积计算复杂度自动变为线性。
- 真实加速受硬件、shape 和 kernel 实现影响。

追问：跨 KV 分块怎样保持 softmax 的归一化正确？

易错点：单纯用 Python 分块并不等于获得高效 CUDA kernel 的速度。

资料：[FlashAttention](https://arxiv.org/abs/2205.14135)

## q023 · 预训练 · 长上下文训练除了改位置编码还需要关注什么？

还要考虑长序列数据分布、计算与显存成本，以及长距离能力评估。位置编码的外推策略不能替代长上下文任务验证。

回答要点：

- 评估不同长度和信息位置，不只测试单一检索设置。
- 检查长文本的目标 mask、截断与采样比例。

追问：为什么通过 needle retrieval 不等于具备长文推理能力？

易错点：上下文窗口标称长度不等于所有位置都同样可靠。

资料：[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

## q024 · 预训练 · 训练 loss 突然出现 NaN，如何定位？

先区分非有限输入、前向数值溢出和梯度问题，定位首次出现非有限值的张量。检查学习率、log/除法、mask、精度与梯度缩放。

回答要点：

- 用小批次复现，并对可疑子图临时用高精度核对。
- 如果发生在 AMP 更新，查看 unscale 后梯度和 scaler 状态。

追问：为什么先做 nan_to_num 可能掩盖训练错误？

易错点：跳过坏步是控制影响，不能替代查明错误来源。

资料：[PyTorch：混合精度训练](https://docs.pytorch.org/docs/main/notes/amp_examples.html)

## q025 · 后训练 · SFT、偏好优化和 RLVR 分别使用什么监督？

SFT 模仿示范回答；偏好优化比较候选回答；RLVR 使用可验证的结果信号，例如程序测试或数学答案。它们的信号粒度与误差来源不同。

回答要点：

- 示范质量、偏好一致性和验证器可靠性都要单独评估。
- 一种任务可以依次使用多种训练阶段。

追问：为什么通过格式检查不等于答案真的正确？

易错点：“能自动计算奖励”不意味着奖励一定准确覆盖目标。

资料：[InstructGPT](https://arxiv.org/abs/2203.02155) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## q026 · 后训练 · SFT 的 prompt mask 与 label shift 怎样实现？

如果只训练回答部分，可以把 prompt 和 padding 标签设为 ignore_index。因果模型用位置 t 的 logits 预测位置 t+1 标签；有效 token 平均的分母也要随 mask 改变。

回答要点：

- 检查 chat template 边界和结束符是否纳入目标。
- 当 shift 后没有有效标签时，应有明确定义，不能悄悄除零。

追问：为什么先求每条样本的 mean 再平均会改变长度权重？

易错点：attention mask 和 loss mask 的作用不同，不能互相替代。

资料：[InstructGPT](https://arxiv.org/abs/2203.02155)

## q027 · 后训练 · PPO 中 old policy 和 reference policy 是一回事吗？

不是。old policy 对应生成 rollout 的行为策略，用于 importance ratio；reference policy 常用于约束偏离初始或参考模型的程度。

回答要点：

- 两种 log probability 的更新时机不同。
- 计算 actor loss 时，旧策略和优势通常作为固定训练目标处理。

追问：多轮复用同一批 rollout 时，ratio 如何变化？

易错点：不能把 KL reference 的概率直接替换为 PPO ratio 的分母。

资料：[PPO](https://arxiv.org/abs/1707.06347) · [InstructGPT](https://arxiv.org/abs/2203.02155)

## q028 · 后训练 · PPO clipped objective 为什么要取 minimum？

它对原 surrogate 和裁剪 surrogate 取较保守的一项。正负优势下激活的裁剪方向不同，不能简单地先 clip ratio 再乘 advantage。

回答要点：

- 通常实现最小化负的 surrogate。
- clip 是目标函数的一部分，不是对参数更新量的硬约束。

追问：负优势、ratio 小于下界时梯度是什么？

易错点：不要把 minimum 写成 maximum，或忘记整体负号。

资料：[PPO](https://arxiv.org/abs/1707.06347)

## q029 · 后训练 · GAE 中 gamma 与 lambda 有何作用？

gamma 折扣未来奖励；lambda 控制多步 TD 残差的混合长度，影响优势估计的偏差与方差。递推需要在真正终止处切断。

回答要点：

- lambda=0 对应一步 TD 残差。
- 时间截断是否 bootstrap 要与环境语义一致。

追问：为什么 terminated 与 truncated 需要区分？

易错点：不要在 batch 维递推，也不要遗漏最后一个 bootstrap value。

资料：[Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## q030 · 后训练 · GRPO 相比 PPO 省掉了什么？

GRPO 使用同一 prompt 多个回答的组内相对奖励构造优势，减少对独立价值模型的依赖。训练行为仍取决于具体 loss、归约和 KL 配置。

回答要点：

- 组内奖励无差异时缺少相对学习信号。
- std 的约定与数值稳定项必须明确。

追问：组大小太小会怎样影响优势估计？

易错点：不要把所有命名为 GRPO 的代码都认为使用同一种 loss 归约。

资料：[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

## q031 · 后训练 · DAPO 的关键改动是什么？

包括非对称裁剪、动态采样、token 级 loss 归约和超长奖励处理。它们分别影响探索、有效训练样本、长度权重和长回答反馈。

回答要点：

- 动态采样与计算 loss 是训练流程中的不同部分。
- 全局 token 平均不同于先逐序列平均。

追问：为什么不等长样本可以检测错误归约？

易错点：只把 clip 上界调大，不能称为完整复现 DAPO。

资料：[DAPO](https://arxiv.org/abs/2503.14476)

## q032 · 后训练 · GSPO 与 token-level ratio 的区别是什么？

GSPO 依据长度归一化序列似然构造 importance ratio，并在序列层裁剪和优化。核心变化是优化单位与奖励单位的对应关系。

回答要点：

- 先平均有效 token 的 log ratio，再 exp。
- 不是对 token ratio 做普通算术平均。

追问：为什么先 exp 再 mean 与先 mean 再 exp 不相同？

易错点：不能把 GSPO 和 GSPO-token 等变体的梯度公式混用。

资料：[GSPO](https://arxiv.org/abs/2507.18071)

## q033 · 后训练 · DPO 如何避免单独训练显式奖励模型？

DPO 利用带 KL 约束的奖励优化形式与最优策略之间的关系，把偏好学习写成策略与参考策略 log probability 差的分类目标。

回答要点：

- 输入通常是 chosen/rejected 对及参考模型概率。
- 仍然依赖偏好数据质量和参考策略选择。

追问：beta 改变什么约束强度与训练行为？

易错点：DPO 不需要显式 RM，不代表它完全没有奖励建模假设。

资料：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

## q034 · 后训练 · 奖励模型升分但实际回答变差，可能是什么？

可能存在代理奖励与真实目标不一致，策略学会了利用奖励模型偏好或验证器漏洞。需要在独立评估上检查任务正确性和行为质量。

回答要点：

- 分开监控训练 reward、独立评测和长度等行为统计。
- 检查格式捷径、答案泄漏和重复模式。

追问：为什么只优化一个 judge 分数可能产生偏差？

易错点：奖励升高本身不能证明真实能力同步提升。

资料：[InstructGPT](https://arxiv.org/abs/2203.02155) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## q035 · 后训练 · LoRA 的 A/B 初始化和缩放如何解释？

LoRA 在冻结基座上训练低秩增量，通常以 alpha/r 缩放。常见初始化让一侧为零，使初始增量为零，同时后续梯度能够开启更新。

回答要点：

- 说明所用 A/B 的形状及乘法方向。
- 秩、目标层和数据一起决定适配能力。

追问：如果 A 和 B 都初始化成零会怎样？

易错点：低秩减少可训练参数，不代表激活显存也按同样比例减少。

资料：[LoRA](https://arxiv.org/abs/2106.09685)

## q036 · 后训练 · QLoRA 相比 LoRA 额外做了什么？

QLoRA 在量化的冻结基座上训练低秩适配器，并提出 NF4、双重量化等降低存储开销的机制。计算时的 dtype 和权重存储格式要分开看。

回答要点：

- 量化基座通常不作为普通可训练权重直接更新。
- 适配器、激活和优化器仍然占用显存。

追问：为什么“4-bit 模型”不等于所有计算都在4-bit执行？

易错点：不能只按4-bit权重大小估计总训练显存。

资料：[QLoRA](https://arxiv.org/abs/2305.14314)

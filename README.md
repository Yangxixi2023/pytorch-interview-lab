# 手撕实验室 · PyTorch Interview Lab

一个在本机运行的 **Python / PyTorch 面试手撕练习平台**。使用类似 LeetCode 的题库与分栏编辑器，练习模型组件、训练算法、推理机制和经典算法，直接执行代码并查看测试与调试结果。

**89 道题 · 192 种测试场景 · 576 次完整题库测试执行**

包含 **PPO、GRPO、GSPO、DAPO、DPO loss、GAE、MHA、GQA、RoPE、RMSNorm、LoRA** 等内容。每题提供中文题意、逐参数类型/shape/含义、返回值顺序与要求、待填写模板、测试用例和参考实现。核心 loss 公式使用数学排版。

> 前端构建产物已包含在仓库中。日常使用只需 Python 依赖，不需要安装 Node.js，也不需要 GPU。代码在本机 CPU PyTorch 中执行，不发送到远程判题服务。

## 快速开始

### 环境要求

- 建议使用 Python 3.11 或更新版本，并选择 PyTorch 有对应安装包的版本。
- 默认依赖：**PyTorch、NumPy、einops**，详见 [requirements.txt](requirements.txt)。
- 使用支持 ES modules 与 MathML 的现代浏览器。
- Git 用于克隆仓库；也可以使用 GitHub 的 **Code → Download ZIP**，解压后进入项目目录。
- Node.js 只用于修改前端后重新构建，运行现有页面不需要它。

已验证环境为 Windows、Python 3.14、CPU PyTorch 2.14。下方也提供 macOS/Linux 命令，但尚未在这些系统上完成同等验证。

### Windows（PowerShell）

```powershell
git clone https://github.com/Yangxixi2023/pytorch-interview-lab.git
cd pytorch-interview-lab

python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install numpy einops

.venv/Scripts/python.exe server.py
```

如果电脑使用 `py` 启动器，将创建环境的命令改为 `py -m venv .venv`。后续命令直接调用虚拟环境中的 Python，无需激活环境或修改 PowerShell 执行策略。

### macOS / Linux

```bash
git clone https://github.com/Yangxixi2023/pytorch-interview-lab.git
cd pytorch-interview-lab

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python server.py
```

Linux 若只需 CPU 版本，可将依赖安装一步换成：

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install numpy einops
```

### 打开与关闭

终端显示 `Interview Lab: http://127.0.0.1:8765` 后，打开：

- **[题库首页](http://127.0.0.1:8765/)**
- **[PPO 练习](http://127.0.0.1:8765/#ppo_loss)**
- **[GSPO 练习](http://127.0.0.1:8765/#gspo_loss)**
- **[DAPO 练习](http://127.0.0.1:8765/#dapo_loss)**

以上链接指向你自己的电脑，需先启动服务。保持终端开启；按 **Ctrl+C** 停止服务。下次使用只需进入项目目录，再次执行启动命令，不用重新安装。

端口占用时显式换一个端口：

```powershell
.venv/Scripts/python.exe server.py --port 8766
```

然后打开 [http://127.0.0.1:8766](http://127.0.0.1:8766)。macOS/Linux 对应使用 `.venv/bin/python`。

## 如何练习

1. 按关键词、分类、难度和状态筛选，点击题目进入工作台。
2. 先读 **输入参数** 表：确认每个参数的类型、shape、含义、默认值和约束。
3. 再读 **返回要求**：返回单个张量还是元组、元组的顺序、shape、dtype、梯度与归约方式均单独说明。
4. 在右侧实现 `solve(...)`。按题意返回结果，`print` 的内容仅用于调试，不作为判题返回值。
5. 点击 **运行示例** 执行首个场景；点击 **提交测试** 执行全部场景。
6. 展开结果查看输入、实际/期望输出和错误；在 **提交记录** 中可恢复某次代码，或下载当前 `.py`。

模板默认不包含答案，参考实现需主动展开。左侧章节导航可以跳到输入、返回和计算规则；测试页展示固定种子的命名输入、期望输出与可运行的构造代码。

| 操作 | Windows / Linux | macOS |
| --- | --- | --- |
| 运行示例 | Ctrl + Enter | Cmd + Enter |
| 提交全部测试 | Ctrl + Shift + Enter | Cmd + Shift + Enter |
| 保存代码 | Ctrl + S | Cmd + S |
| 缩进 / 减少缩进 | Tab / Shift + Tab | Tab / Shift + Tab |

快捷键在代码编辑器获得焦点时生效。界面支持分栏拖动、暂停计时、亮暗主题；窄屏时采用上下布局。

### 输入与返回值约定

- `[]` 表示零维张量，例如 `torch.tensor(0.5)`；与 shape 为 `[1]` 的单元素向量不同。
- `...` 表示任意前导维度。例如 `[..., D]` 可以是 `[B,T,D]`。
- loss 题若要求可求导标量，应返回 Tensor 本身；不要 `.item()` 或转换成 Python float。
- 题目要求返回元组时，按返回说明中的顺序返回；列表与元组不是同一种结构。
- dtype 和梯度行为都属于要求。例如 mask 通常是 bool，索引通常是 int64，旧策略概率在相关题目中必须 detach。
- `grad_fn` 的具体名称会因实现方式不同而变化，不需要和示例一致。

### 判题检查什么

- 张量形状、dtype 与数值，容差为 `rtol=2e-4`、`atol=2e-5`。
- Python 列表、元组、字典等返回值的结构与内容。
- 标注梯度检查的题目会比较输入和参数梯度，并检查题意中的梯度隔离要求。
- 每次运行独立进程，最长 **30 秒**，也可主动点击 **停止**。

完整提交使用随机种子 `17`、`41`、`103`。576 是 192 种场景乘三个种子的执行次数；确定性场景在不同种子下可能具有相同输入。

数值通过不能证明算法/API/空间复杂度要求全部满足，例如不能仅根据输出判断是否真的采用分块 Flash Attention。请同时遵守题面限制并进行源码自查。

## 导入 PyTorch、NumPy、einops 和其他库

默认环境允许在编辑器及自定义调试中直接导入：

```python
import math
import torch
import numpy as np
from einops import rearrange, repeat, reduce

x = torch.randn(2, 5, 12)
q = rearrange(x, 'b t (h d) -> b h t d', h=3)
print(q.shape)  # torch.Size([2, 3, 5, 4])
```

不做统一的 import 白名单拦截；题目只限制明确禁止的现成算法。例如允许 `einops.rearrange` 辅助拆头，但 Softmax 题仍不允许直接调用 `torch.softmax`。NumPy 适合不要求 autograd 的处理；需要 PyTorch 梯度的题目不要通过 NumPy 转换断开计算图。

需要额外的库时，用**启动服务的同一个虚拟环境**安装：

```powershell
# Windows，以 scipy 为例
.venv/Scripts/python.exe -m pip install scipy
```

```bash
# macOS / Linux
.venv/bin/python -m pip install scipy
```

每次判题都会新建 Python 进程，安装成功后下次运行即可导入。若要让仓库其他使用者也默认获得该库，应同时更新 `requirements.txt`。

## 自定义调试与变量快照

### 打印、断言和反向传播

打开 **自定义调试**，可直接调用当前代码中的 `solve`。也可在 **测试用例** 中点击 **载入自定义调试**。

例如完成 Softmax 后：

```python
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
y = solve(x, dim=-1)

print('概率：', y)
print('每行概率和：', y.sum(dim=-1))
assert y.shape == x.shape

y[0, 0].backward()
print('输入梯度：', x.grad)
```

点击 **运行调试**，在 **输出 / 错误** 中查看结果。自定义调试不计为正式提交，也不自动标记题目通过。

### 指定行快照

在 **快照行号** 输入英文逗号分隔的行号，例如 `6, 9`，再运行代码：

- 快照在对应行执行**之前**记录，展示在 **变量快照**。
- 行号对应题目编辑器，不是自定义调试片段。
- 循环反复经过同一行会产生多次快照，每次运行最多40条。
- 这是执行轨迹快照，不是暂停式断点或交互式单步调试器。

## 题库与算法范围

完整目录见 **[89 道题与输入返回说明](docs/PROBLEMS.md)**。

| 方向 | 示例 |
| --- | --- |
| 对齐与强化学习 | PPO、GRPO、GSPO、DAPO、DPO、GAE、Bradley–Terry、PUCT |
| Transformer 与位置编码 | Softmax、MHA、GQA、RoPE、ALiBi、RMSNorm、LayerNorm、SwiGLU |
| 训练与参数高效训练 | 交叉熵、梯度累积/裁剪、Adam、学习率调度、LoRA、QLoRA |
| 推理与分布式原理 | KV Cache、分页注意力、分块注意力、采样、束搜索、推测解码、张量并行 |
| 视觉与扩散 | 卷积、池化、patchify、ViT Block、Flow Matching、DDIM、AdaLN-Zero |
| 图神经网络 | GCN、GAT、GIN、GraphSAGE、MPNN、图池化、图自编码器、链接预测 |
| 前沿架构 | MoE、负载均衡、MLA、多 Token 预测、差分注意力、Mamba |
| 经典手撕 | K-means、两种快排、拓扑排序、Top-k、BPE、线性回归反向传播、马走日 |

部分题目专门考核心计算，题面已明确范围：

- **PPO**：clipped actor loss，不含 value loss、entropy bonus。
- **GRPO**：beta=0 的 policy loss，先序列内平均再跨序列平均，不含 KL。
- **DAPO**：对已完成动态采样与奖励处理的 batch 做非对称裁剪及全局有效 token 平均，不实现完整采样循环。
- **GSPO**：长度归一化序列 importance ratio 与序列平均，标准化优势作为输入提供。
- **GAE**：广义优势估计和图自编码器分成两题。
- **FSDP / 张量并行 / 环形注意力**：单机数学模拟，不启动多机通信。
- **Flash Attention**：PyTorch 分块 online softmax，不要求 CUDA kernel。
- **MLA / QLoRA / MCTS / ViT patch**：分别考低秩 KV、给定码本的量化基座、PUCT 选择、patchify 等核心，不代表完整生产实现。
- **马走日**：国际象棋骑士最短路，不含中国象棋蹩马腿规则。

原始资料：[PPO](https://arxiv.org/abs/1707.06347)、[GRPO](https://arxiv.org/abs/2402.03300)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)。

## 参考代码与背诵

点击顶栏 **参考代码**，可以搜索和筛选**全部89道题**的参考实现。每道题都有完整代码，以及可展开的输入/返回约定；不是仅为部分重点题提供答案。

- **遮住代码 / 显示参考代码**：自测是否记住实现步骤。
- **去练习 / 默写**：进入对应题目的编辑器，保留已有草稿；想从模板开始可点击重置。
- **复制代码 / 下载 .py**：便于保存和线下复习。
- **张量变换与 einops 完整教程**：包含 view、reshape、transpose、permute、contiguous、unsqueeze、squeeze、expand、repeat、cat、stack、split、chunk、where，以及 einops rearrange/repeat/reduce。

完整的独立 Python 参考文件在 **[references/](references/README.md)**。每个文件含参数说明、返回约定、完整实现和可运行示例。例如：

```powershell
.venv/Scripts/python.exe references/mha.py
.venv/Scripts/python.exe references/ppo_loss.py
.venv/Scripts/python.exe references/tensor_operations.py
```

macOS/Linux 将解释器路径替换为 `.venv/bin/python`。普通 FFN、Entropy Loss、SFT Loss 与张量拆头均是独立练习题，有测试和参考代码。

## 面试问答

点击顶栏 **面试问答**，按 **机器学习、预训练、后训练** 浏览36道知识问答。可以搜索关键词，先口述，再展开回答。每题都有：回答摘要、补充要点、面试追问、易错点、原始资料链接，部分题目还链接到相关代码练习。

这是根据原始论文和官方文档整理的知识题，不声称来自某家公司的真实面经。内容与追问由本项目重新组织，技术背景可通过链接核对。也可直接在 GitHub 阅读 **[面试问答文档](docs/INTERVIEW_QA.md)**。

## 进度保存

草稿、用时、通过状态、自定义调试片段和每题最近 **15 次提交代码**保存在当前浏览器 localStorage，键名为 `interview-lab`，没有远程账号同步。

- `localhost` 与 `127.0.0.1`、不同端口及不同浏览器的存储互相独立。建议始终使用同一个地址。
- 清理站点数据或结束隐私窗口可能丢失记录。
- **下载代码**只备份当前题的 `.py`，不是整个题库进度。
- “已通过”表示历史上曾通过全部测试；修改草稿后应重新提交验证当前代码。

## 测试与开发

```powershell
# Windows：全部回归测试
.venv/Scripts/python.exe -m unittest test_lab test_http -v
```

macOS/Linux 使用 `.venv/bin/python -m unittest test_lab test_http -v`。测试包含题库参考实现、参数文档、错误实现检测、独立数学/算子对照、einops 提交与调试、中文输出、快照、真实 HTTP、主动停止和超时。超时用例会真实等待约30秒，整套通常需要数十秒。

修改前端后：

```bash
npm ci
npm run build
```

PowerShell 若阻止执行 `npm.ps1`，改用 `npm.cmd ci`、`npm.cmd run build`。构建完成后刷新浏览器；修改 Python 后重启服务器。

修改题意、输入输出或测试用例时：

1. 在 `catalog*.py` 注册/修改题目，并在 `problem_io*.py` 写清每个输入和返回值。
2. 使用本机 Python 执行 `generate_examples.py` 更新 `examples.json`，所有显示的期望输出都由参考代码计算。然后执行 `generate_docs.py` 更新全部参考文件和文档。
3. 执行回归测试，更新 `docs/PROBLEMS.md` 和 README 中的数量。
4. 若修改前端，一并提交更新后的 `web/bundle.js`。

| 文件 | 用途 |
| --- | --- |
| `server.py` | 回环 HTTP 服务、异步任务、取消 |
| `runner.py` | 测试运行、输出/梯度比较、自定义调试、快照 |
| `catalog*.py` | 题意、模板、参考函数、测试场景 |
| `problem_io*.py` | 参数类型、shape、含义、返回顺序与公式 |
| `generate_examples.py` / `examples.json` | 生成并保存题面示例 |
| `test_lab.py` / `test_http.py` | 回归测试 |
| `generate_docs.py` / `references/` | 全题库独立参考文件与文档导出 |
| `interview_questions.py` | 带来源的面试问答 |
| `requirements.txt` | Python 依赖 |
| `package*.json` / `build.mjs` | 前端依赖及构建 |
| `web/` | 页面、样式、编辑器和数学公式渲染 |
| `THIRD_PARTY_NOTICES.md` | 随前端分发的第三方许可声明 |

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `No module named torch / einops` | 用运行服务器的同一个虚拟环境安装依赖。 |
| PyTorch 没有匹配安装包 | 检查 Python 版本与系统架构，参考 [PyTorch 安装页](https://pytorch.org/get-started/locally/)。 |
| 浏览器打不开 | 确认服务器终端仍在运行，端口与访问地址一致。 |
| “请刷新页面后重试” | 服务重启会更换会话 token，刷新页面再运行。 |
| `NotImplementedError` | 模板尚未实现，需要先填写 `solve`。 |
| 输出正确但提交失败 | 检查返回结构、dtype、梯度和 detach 要求。 |
| 修改页面后没有变化 | 执行 `npm run build`，然后刷新。 |
| 进度看起来丢失 | 检查浏览器、域名、端口是否变化，是否清理了站点数据。 |
| 想直接用 GitHub Pages 运行 | 该应用需要 Python 后端；仅静态托管不能运行 PyTorch 判题，请克隆后本地启动。 |

## 执行边界

**这是个人练习工具，不是多用户不可信代码沙箱。** 提交代码拥有启动服务器用户的本机权限；独立子进程和超时机制不提供操作系统级隔离。

服务仅监听 `127.0.0.1`，并检查 Host、Origin 和会话 token。仅运行自己的代码，不通过公网代理或端口转发公开这个执行服务。公开 GitHub 仓库只公开源码，不会公开你电脑上的判题进程。

普通打印最多记录30000字符，变量快照有数量和展示长度限制。测试辅助练习，不保证发现任意错误实现。

## 第三方组件

界面使用 CodeMirror、Lezer 和 KaTeX，使用 esbuild 构建。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。PyTorch、NumPy 和 einops 安装时分别附带其自身许可。

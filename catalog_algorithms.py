def populate(add):
    add('topk_sampling','Top-k / Top-p 采样分布','推理与分布式训练','中等','logits, top_k, top_p, temperature=1.0',
        '输入一维 logits。先除温度，再保留 top_k（0 表示不裁剪）；在剩余归一化分布上做 nucleus，保留首次使累计概率达到/超过 top_p 的 token；返回原词表顺序的归一化概率。本题不随机采样。分数不并列，0<top_p<=1。',
        '''z=logits/temperature
if top_k>0:
    vals,ids=z.topk(top_k); filtered=torch.full_like(z,float('-inf')); filtered[ids]=vals; z=filtered
vals,ids=z.sort(descending=True); p=vals.softmax(-1)
remove=(p.cumsum(-1)-p)>=top_p
p=p.masked_fill(remove,0); p=p/p.sum()
out=torch.zeros_like(p); out[ids]=p
return out''',
        [('top-k 后 nucleus',"args=(torch.tensor([0.1,2.,1.,-1.,0.5]),3,0.8,0.7)"),('保留最高一项',"args=(torch.tensor([1.,2.,3.]),0,0.1)"),('不裁剪',"args=(torch.tensor([1.,2.,3.]),0,1.)")])
    add('beam_search','束搜索解码','推理与分布式训练','中等','log_probs, beam_size',
        'log_probs [T,V] 为给定的逐步 log 概率（与历史无关的简化模型）。从空序列开始，每步展开并保留累计分数最高 beam_size 个，返回 [(token_list, score)]。并列按 token 序列字典序升序。本题无 EOS/长度惩罚。',
        'beams=[([],0.)]\nfor row in log_probs:\n    candidates=[(seq+[i],score+float(p)) for seq,score in beams for i,p in enumerate(row)]\n    beams=sorted(candidates,key=lambda x:(-x[1],x[0]))[:beam_size]\nreturn beams',
        [('多步搜索',"args=(torch.tensor([[-0.2,-1.],[-1.2,-0.3],[-0.5,-0.8]]),3)"),('并列规则',"args=(torch.zeros(2,3),2)")])
    add('speculative_decoding','推测解码接受步骤','推理与分布式训练','困难','draft_tokens, draft_probs, target_probs, uniforms',
        '实现单次 speculative acceptance 核心。draft_tokens [K]；draft/target_probs [K,V]，uniforms [K]。逐步接受条件 u<min(1,p[token]/q[token])。首次拒绝返回 (已接受 token 列表, normalize(max(p-q,0)))；全部接受返回 (全部 token,None)。本题不采样替换 token/额外 token。q 对候选 token 严格正；拒绝时剩余分布和>0。',
        'accepted=[]\nfor i,token in enumerate(draft_tokens.tolist()):\n    ratio=min(1.,float(target_probs[i,token]/draft_probs[i,token]))\n    if float(uniforms[i])>=ratio:\n        residual=(target_probs[i]-draft_probs[i]).clamp_min(0)\n        return accepted,residual/residual.sum()\n    accepted.append(token)\nreturn accepted,None',
        [('第二步拒绝',"args=(torch.tensor([0,1]),torch.tensor([[0.5,0.5],[0.1,0.9]]),torch.tensor([[0.7,0.3],[0.8,0.2]]),torch.tensor([0.2,0.8]))"),('全部接受',"args=(torch.tensor([0,1]),torch.full((2,2),0.5),torch.full((2,2),0.5),torch.tensor([0.3,0.8]))")])
    add('mcts_search','蒙特卡洛树搜索 PUCT 选择','对齐与强化学习','困难','priors, value_sums, visits, parent_visits, c_puct=1.0',
        '考一次 MCTS selection：Q=W/N，未访问 Q=0；U=c*P*sqrt(parent_visits)/(1+N)。返回 argmax(Q+U) 的 Python int，并列取最小下标。不模拟环境 rollout。',
        'q=torch.where(visits>0,value_sums/visits.clamp_min(1),torch.zeros_like(value_sums))\nu=c_puct*priors*math.sqrt(parent_visits)/(1+visits)\nreturn int((q+u).argmax())',
        [('探索与利用',"args=(torch.tensor([0.2,0.6,0.2]),torch.tensor([5.,0.,1.]),torch.tensor([10,0,2]),12,2.)"),('并列',"args=(torch.ones(3)/3,torch.zeros(3),torch.zeros(3,dtype=torch.long),0)")])
    add('bpe','字节对编码 BPE','经典手撕','困难','words, num_merges',
        'words 是 {字符串:正整数词频}。初始化为字符列表（本题不添加词尾标记），迭代统计每个词内部相邻 token pair 的加权频次，合并最多 pair；并列按 pair 的 Python 字典序。从左到右非重叠合并，直到无 pair 或达到次数。返回 (merges, tokenized_dict)，merges 是 pair 元组列表。',
        '''tokens={w:list(w) for w in words}; merges=[]
for _ in range(num_merges):
    counts={}
    for w,seq in tokens.items():
        for pair in zip(seq,seq[1:]): counts[pair]=counts.get(pair,0)+words[w]
    if not counts: break
    pair=min(counts,key=lambda p:(-counts[p],p)); merges.append(pair)
    for w,seq in tokens.items():
        result=[]; i=0
        while i<len(seq):
            if i+1<len(seq) and (seq[i],seq[i+1])==pair:
                result.append(seq[i]+seq[i+1]); i+=2
            else: result.append(seq[i]); i+=1
        tokens[w]=result
return merges,tokens''',
        [('词频与重叠',"args=({'low':5,'lower':2,'newest':3,'aaaa':2},4)"),('并列与短词',"args=({'ab':1,'ac':1,'x':3},5)"),('空词表',"args=({},3)")])
    add('kmeans','K-means 聚类','经典手撕','中等','x, initial_centers, iterations',
        'x [N,D]，初始中心 [K,D]。恰好执行 iterations 次 Lloyd 更新：最近中心（平方欧氏距，并列最小索引），再求每簇均值。空簇保留原中心。最后用更新完的中心重算 labels。返回 (centers,labels)，不考 autograd。',
        '''centers=initial_centers.clone()
for _ in range(iterations):
    labels=(x[:,None]-centers[None,:]).square().sum(-1).argmin(-1)
    centers=torch.stack([x[labels==k].mean(0) if (labels==k).any() else centers[k] for k in range(len(centers))])
labels=(x[:,None]-centers[None,:]).square().sum(-1).argmin(-1)
return centers,labels''',
        [('空簇与两团',"args=(torch.tensor([[0.,0.],[1.,0.],[9.,9.],[10.,9.]]),torch.tensor([[0.,0.],[9.,9.],[100.,100.]]),3)"),('不迭代',"args=(torch.randn(5,2),torch.randn(2,2),0)")])
    for slug,title,body in [
        ('quicksort_recursive','快速排序（递归）','def sort(a):\n    if len(a)<2: return a\n    p=a[len(a)//2]\n    return sort([v for v in a if v<p])+[v for v in a if v==p]+sort([v for v in a if v>p])\nreturn sort(list(nums))'),
        ('quicksort_iterative','快速排序（非递归）','a=list(nums); stack=[(0,len(a)-1)]\nwhile stack:\n    lo,hi=stack.pop()\n    if lo>=hi: continue\n    pivot=a[hi]; i=lo\n    for j in range(lo,hi):\n        if a[j]<pivot: a[i],a[j]=a[j],a[i]; i+=1\n    a[i],a[hi]=a[hi],a[i]\n    stack.extend([(lo,i-1),(i+1,hi)])\nreturn a')]:
        add(slug,title,'经典手撕','中等','nums','返回升序新列表，保留重复值，不修改输入。禁止 sorted/list.sort。'+('必须使用显式栈，不递归。' if slug.endswith('iterative') else '练习递归分治。'),body,
            [('重复与负数',"args=([3,-1,2,3,0,-1],)"),('空列表',"args=([],)"),('随机数组',"args=(torch.randint(-20,20,(50,)).tolist(),)")])
    add('topo','拓扑排序','经典手撕','中等','num_nodes, edges',
        '节点 0..n-1，edges 为 (u,v) 有向边列表，无重复边。返回字典序最小的合法拓扑序；存在环返回 []。需包含孤立节点。',
        'import heapq\nadj=[[] for _ in range(num_nodes)]; degree=[0]*num_nodes\nfor u,v in edges: adj[u].append(v); degree[v]+=1\nheap=[i for i in range(num_nodes) if degree[i]==0]; heapq.heapify(heap); out=[]\nwhile heap:\n    u=heapq.heappop(heap); out.append(u)\n    for v in adj[u]:\n        degree[v]-=1\n        if degree[v]==0: heapq.heappush(heap,v)\nreturn out if len(out)==num_nodes else []',
        [('DAG 与孤点',"args=(6,[(0,2),(1,2),(2,3),(1,4)])"),('有环',"args=(3,[(0,1),(1,0)])"),('空图',"args=(0,[])")])
    add('topk','Top-k 最大元素','经典手撕','中等','nums, k',
        '0<=k<=len(nums)，返回降序的 k 个最大元素，保留重复；禁止对全数组排序、torch.topk。可用大小 k 的堆。',
        'import heapq\nif k==0: return []\nheap=[]\nfor n in nums:\n    if len(heap)<k: heapq.heappush(heap,n)\n    elif n>heap[0]: heapq.heapreplace(heap,n)\nreturn [heapq.heappop(heap) for _ in range(k)][::-1]',
        [('重复最大值',"args=([2,7,3,7,-1],3)"),('k 零',"args=([1,2],0)"),('随机',"args=(torch.randint(-100,100,(60,)).tolist(),7)")])
    add('knight','模拟马走日','经典手撕','中等','rows, cols, start, target, blocked',
        '将“马走日”具体设为国际象棋骑士最短路（无蹩马腿规则）：棋盘 rows×cols，坐标从0开始。每步 (±1,±2)/(±2,±1)，blocked 是不可落点坐标列表。返回最少步数，不可达或起终点 blocked 返回 -1。',
        'from collections import deque\nblocked=set(map(tuple,blocked)); start=tuple(start); target=tuple(target)\nif start in blocked or target in blocked: return -1\nqueue=deque([(start,0)]); seen={start}\nwhile queue:\n    (r,c),dist=queue.popleft()\n    if (r,c)==target: return dist\n    for dr,dc in [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)]:\n        nxt=(r+dr,c+dc)\n        if 0<=nxt[0]<rows and 0<=nxt[1]<cols and nxt not in seen and nxt not in blocked:\n            seen.add(nxt); queue.append((nxt,dist+1))\nreturn -1',
        [('标准棋盘',"args=(8,8,(0,0),(7,7),[])"),('不可达',"args=(2,2,(0,0),(1,1),[])"),('同位置',"args=(3,3,(1,1),(1,1),[])"),('阻挡',"args=(8,8,(0,0),(7,7),[(1,2),(2,1)])")])

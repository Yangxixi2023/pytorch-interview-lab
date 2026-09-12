"""字节对编码 BPE

输入：
words: dict[str, int] — — 词到正整数频次的映射；可为空。
num_merges: int — — 最多合并次数，非负整数。

返回（多项按元组顺序）：
merges: list[tuple[str, str]] 最多 num_merges 项 — 按执行顺序记录每次合并的 token pair。
tokenized: dict[str, list[str]] 与 words 的键相同 — 每个原始词经过合并后的 token 列表。
"""

import torch

def solve(words, num_merges):
    tokens = {w: list(w) for w in words}
    merges = []
    for _ in range(num_merges):
        counts = {}
        for w, seq in tokens.items():
            for pair in zip(seq, seq[1:]):
                counts[pair] = counts.get(pair, 0) + words[w]
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        merges.append(pair)
        for w, seq in tokens.items():
            result = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                    result.append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    result.append(seq[i])
                    i += 1
            tokens[w] = result
    return (merges, tokens)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = ({'low': 5, 'lower': 2, 'newest': 3, 'aaaa': 2}, 4)
    print(solve(*args))

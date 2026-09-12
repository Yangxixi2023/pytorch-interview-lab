"""字节对编码 BPE

输入：
words: dict[str, int] —
    词到正整数频次的映射；可为空。
num_merges: int —
    最多合并次数，非负整数。

返回：
merges: list[tuple[str, str]] 最多 num_merges 项
    按执行顺序记录每次合并的 token pair。
tokenized: dict[str, list[str]] 与 words 的键相同
    每个原始词经过合并后的 token 列表。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(words, num_merges):
    tokenized = {word: list(word) for word in words}
    merges = []

    for merge_step in range(num_merges):
        pair_counts = {}
        for word, tokens in tokenized.items():
            frequency = words[word]
            for index in range(len(tokens) - 1):
                pair = (tokens[index], tokens[index + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency
        if not pair_counts:
            break

        # 优先最高频次；相同频次时按 pair 字典序选择。
        best_pair = min(
            pair_counts, key=lambda pair: (-pair_counts[pair], pair)
        )
        merges.append(best_pair)

        for word, tokens in tokenized.items():
            merged_tokens = []
            index = 0
            while index < len(tokens):
                has_next = index + 1 < len(tokens)
                if (
                    has_next
                    and (tokens[index], tokens[index + 1]) == best_pair
                ):
                    merged_tokens.append(tokens[index] + tokens[index + 1])
                    index += 2  # 从左到右合并，配对不能重叠。
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            tokenized[word] = merged_tokens

    return merges, tokenized


if __name__ == "__main__":
    torch.manual_seed(17)
    args = ({"low": 5, "lower": 2, "newest": 3, "aaaa": 2}, 4)
    print(solve(*args))

import pandas as pd


def get_token_alignment_df(prompt, tokenizer):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    input_ids = inputs["input_ids"][0]
    offset_mapping = inputs["offset_mapping"][0]

    data = []
    for i, (token_id, (start, end)) in enumerate(zip(input_ids.tolist(), offset_mapping.tolist())):
        token_str = tokenizer.decode([token_id])
        char_str = prompt[start:end] if start != end else ""
        data.append({
            "Index": i,
            "Token ID": token_id,
            "Token": token_str,
            # "char": char_str
        })

    return pd.DataFrame(data)


def print_token_alignment(prompt, tokenizer):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    input_ids = inputs["input_ids"][0]
    offset_mapping = inputs["offset_mapping"][0]

    # print(f"\n🧾 原始 Prompt: {prompt}\n")
    # print(f"{'Idx':<5}{'Token ID':<10}{'Token':<10}{'对应字符':<10}")
    # print("=" * 40)

    for i, (token_id, (start, end)) in enumerate(zip(input_ids.tolist(), offset_mapping.tolist())):
        token_str = tokenizer.decode([token_id])
        char_str = prompt[start:end] if start != end else ""  # 忽略空位
        print(f"{i:<5}{token_id:<10}{token_str:<10}{repr(char_str):<10}")

    print("=" * 40)

# # ✅ 使用方法（你已有 tokenizer）：
# print_token_alignment("请你详细介绍一下西湖。", tokenizer)

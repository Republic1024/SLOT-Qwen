# delta_trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm


def get_empty_delta(model, tokenizer, prompt, H_state):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    hidden_size = H_state.size(-1)
    delta = nn.Parameter(torch.zeros((1, 1, hidden_size), device=H_state.device, requires_grad=True))

    return delta


def train_delta_from_H(model, tokenizer, prompt, H_state, step=3, lr=1e-2):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    current_ids = input_ids[:, :-1]
    target_ids = input_ids[:, 1:]

    hidden_size = H_state.size(-1)
    delta = nn.Parameter(torch.zeros((1, 1, hidden_size), device=H_state.device, requires_grad=True))
    if step == 0:
        return delta

    optimizer = torch.optim.Adam([delta], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_log = []

    for i in range(step):
        optimizer.zero_grad()
        delta_broadcast = delta.expand(H_state[:, :-1, :].shape)
        adjusted_H = H_state[:, :-1, :] + delta_broadcast
        logits = torch.matmul(adjusted_H, model.lm_head.weight.T)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    return delta


def generate_by_H(model, prompt, tokenizer, delta, answer_len=100):
    """
    基于隐藏状态 H 添加扰动 delta 的方式进行文本生成。

    参数：
    - model: LLM 模型
    - prompt: 输入提示词
    - tokenizer: 分词器
    - delta: 扰动张量，shape=[1, 1, hidden_size]
    - answer_len: 生成 token 数量

    返回：
    - record: Tensor，只包含新增的 token ids（不含 prompt 部分）
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]  # shape: [1, L_prompt]
    generated_ids = input_ids.clone()
    record = torch.empty((1, 0), dtype=torch.long, device=generated_ids.device)

    for step in tqdm(range(answer_len)):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, output_hidden_states=True, return_dict=True)
            H_cur = outputs.hidden_states[-1]  # shape: [1, cur_len, hidden_size]

        last_H = H_cur[:, -1, :] + delta.squeeze(1)  # 加扰动
        logits = torch.matmul(last_H, model.lm_head.weight.T)  # shape: [1, vocab_size]
        next_token_id = torch.argmax(logits, dim=-1)  # shape: [1]

        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
        record = torch.cat([record, next_token_id.unsqueeze(0)], dim=-1)

    record_txt = tokenizer.decode(record[0], skip_special_tokens=True)
    return record_txt


import torch
from tqdm import tqdm


def generate_by_H_fast(model, prompt, tokenizer, delta, answer_len=100):
    """
    使用 past_key_values 加速生成，基于隐藏状态 H 添加扰动 delta。

    参数：
    - model: 支持 use_cache 的语言模型（如 GPT）
    - prompt: 输入提示文本
    - tokenizer: 分词器
    - delta: shape=[1, 1, hidden_size] 的扰动张量
    - answer_len: 生成 token 数量

    返回：
    - record_txt: 解码后的生成文本（不含 prompt）
    """
    # 编码 prompt，获取首个 token
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]  # [1, L]

    # 第一步：获取 initial past_key_values
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True, output_hidden_states=True, use_cache=True)
        past_key_values = outputs.past_key_values

        # 取最后一个 token 的 hidden state 加 delta
        H_last = outputs.hidden_states[-1][:, -1, :] + delta.squeeze(1)  # [1, hidden_size]
        logits = torch.matmul(H_last, model.lm_head.weight.T)  # [1, vocab_size]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]

    # 初始化结果
    generated = [next_token_id]

    # 后续 token 逐步生成
    for _ in tqdm(range(answer_len - 1), desc="Generating"):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                return_dict=True,
                output_hidden_states=True,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            H_last = outputs.hidden_states[-1][:, -1, :] + delta.squeeze(1)  # [1, hidden_size]
            logits = torch.matmul(H_last, model.lm_head.weight.T)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]

        generated.append(next_token_id)

    # 拼接所有生成 token（不含 prompt）
    gen_ids = torch.cat(generated, dim=-1)  # [1, T]
    record_txt = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return record_txt


def evaluate_slot_ceval(model, tokenizer, delta, example, max_len=20, verbose=True):
    """
    用于基于 SLOT+Delta 方法评估选择题问题（符合 C-Eval 格式）。

    参数：
    - model, tokenizer: 加载好的模型与分词器
    - delta: 经训练后的扰动张量
    - example: 单个 C-Eval 格式题目字典，如：
        {
            'id': 0,
            'question': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____',
            'A': '1',
            'B': '2',
            'C': '3',
            'D': '4',
            'answer': 'C',
            'explanation': ''
        }
    - max_len: 生成的最大 token 数（默认 100）
    - verbose: 是否打印中间推理文本

    返回：
    - predict_option: 模型预测的选项（如 'A', 'B', 'C'）
    - is_correct: 是否预测正确
    """
    # === 构建 Prompt ===
    prompt = f"""以下是一道单项选择题，请你阅读题目并选择最合适的选项。

题目：{example['question']}

选项：
A. {example['A']}
B. {example['B']}
C. {example['C']}
D. {example['D']}

答案是："""

    # === 生成答案文本 ===
    output_text = generate_by_H(model, prompt, tokenizer, delta, answer_len=max_len)

    if verbose:
        print("🔍 模型生成结果:\n", output_text)

    # === 判断哪个选项被提及 ===
    predict_option = None
    for option in ['A', 'B', 'C', 'D']:
        if option in output_text:
            predict_option = option
            break

    is_correct = (predict_option == example['answer'])
    return predict_option, is_correct


def generate_by_H_eos(model, prompt, tokenizer, delta, answer_len=100):
    """
    基于隐藏状态 H 添加扰动 delta 的方式进行文本生成，支持 <eos> 提前终止。

    参数：
    - model: LLM 模型
    - prompt: 输入提示词
    - tokenizer: 分词器
    - delta: 扰动张量，shape=[1, 1, hidden_size]
    - answer_len: 最多生成 token 数量

    返回：
    - record_txt: 解码后的生成文本（不含 prompt 部分）
    """
    eos_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]  # shape: [1, L_prompt]
    generated_ids = input_ids.clone()
    record = torch.empty((1, 0), dtype=torch.long, device=generated_ids.device)

    for _ in range(answer_len):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, output_hidden_states=True, return_dict=True)
            H_cur = outputs.hidden_states[-1]  # shape: [1, cur_len, hidden_size]

        last_H = H_cur[:, -1, :] + delta.squeeze(1)
        logits = torch.matmul(last_H, model.lm_head.weight.T)
        next_token_id = torch.argmax(logits, dim=-1)

        if next_token_id.item() == eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
        record = torch.cat([record, next_token_id.unsqueeze(0)], dim=-1)

    record_txt = tokenizer.decode(record[0], skip_special_tokens=True)
    return record_txt


def evaluate_slot_ceval_eos(model, tokenizer, delta, example, max_len=20, verbose=True):
    """
    基于 generate_by_H_eos 的评估函数，用于 C-Eval 单选题目。

    返回：
    - predict_option: 预测选项，如 'A'
    - is_correct: 是否预测正确
    """
    prompt = f"""以下是一道单项选择题，请你阅读题目并选择最合适的选项。

题目：{example['question']}

选项：
A. {example['A']}
B. {example['B']}
C. {example['C']}
D. {example['D']}

答案是："""

    output_text = generate_by_H_eos(model, prompt, tokenizer, delta, answer_len=max_len)

    if verbose:
        print("🔍 模型生成结果:\n", output_text)

    predict_option = None
    for option in ['A', 'B', 'C', 'D']:
        if option in output_text:
            predict_option = option
            break

    is_correct = (predict_option == example['answer'])
    return predict_option, is_correct


def evaluate_slot_ceval_eos_2(model, tokenizer, delta, example, max_len=20, verbose=True):
    """
    基于 generate_by_H_eos 的评估函数，用于 C-Eval 单选题目。

    返回：
    - predict_option: 预测选项，如 'A'
    - is_correct: 是否预测正确
    """
    prompt = f"""以下是一道单项选择题，请你阅读题目并选择最合适的选项。

题目：{example['question']}

选项：
A. {example['A']}
B. {example['B']}
C. {example['C']}
D. {example['D']}

答案是："""

    output_text = generate_by_H_eos(model, prompt, tokenizer, delta, answer_len=max_len)

    if verbose:
        print("🔍 模型生成结果:\n", output_text)

    predict_option = None
    for option in ['A', 'B', 'C', 'D']:
        if option in output_text:
            predict_option = option
            break

    print(predict_option, example['answer'])
    is_correct = (predict_option == example['answer'])
    return predict_option, predict_option, example['answer']

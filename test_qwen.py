from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-llama-8b", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-r1-distill-llama-8b",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).eval()
response, history = model.chat(tokenizer, "从以下微博内容中提取实体关系，并返回 (实体1, 关系, 实体2) 的格式列表：实际上，现在在韩国各地，尹锡悦粉丝开始针对中国人， 包括但不限于围攻中国记者，高喊“我们韩国人不喜欢中国人”， 围攻中国游客， 穿着美国队长制服闯中国大使馆。。。。  这也能说明尹锡悦的底色。。#尹锡悦将返回总统官邸##尹锡悦被捕52天后获释# #韩国人开始大量买入中国股票# 请仅返回列表，例如：[('实体1', '关系', '实体2'), ...]", history=None)
print(response)
# 你好！很高兴为你提供帮助。

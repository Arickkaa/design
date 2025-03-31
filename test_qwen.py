from transformers import AutoTokenizer, AutoModelForCausalLM

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好,给我讲一个笑话", history=None)
print(response)
# 你好！很高兴为你提供帮助。

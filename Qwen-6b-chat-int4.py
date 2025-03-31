import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import unicodedata
from tqdm import tqdm
import ast

def fetch_data(csv_file="weibo_data.csv"):
    """从 CSV 文件中读取数据"""
    return pd.read_csv(csv_file, encoding="utf-8")

class RelationExtractor:
    def __init__(self, model_name="Qwen/Qwen-7B-Chat-Int4"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto"
        )
        self.model.eval()

    def extract_relation(self, text):
        prompt = (
            f"从以下文本提取实体关系，返回 (实体1, 关系, 实体2) 的格式列表：\n\n"
            f"{text}\n\n仅返回列表如：[('实体1', '关系', '实体2'), ...]，禁止添加其他任何内容"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128, use_cache=False)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result

def main():
    df = fetch_data()
    extractor = RelationExtractor()
    with open("relations.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["note_id", "实体1", "关系", "实体2"])
        for _, row in tqdm(df.iterrows(), total=len(df), desc="抽取进度"):
            note_id, text = row["note_id"], row["content"]
            try:
                output = extractor.extract_relation(text)
                # 归一化并替换智能引号和全角逗号
                normalized = unicodedata.normalize('NFKC', output)
                normalized = normalized.replace("“", "\"").replace("”", "\"").replace("，", ",")
                # 安全解析字符串，要求模型返回合法的 Python 列表表达式
                relations = ast.literal_eval(normalized)
                if isinstance(relations, list):
                    for rel in relations:
                        if isinstance(rel, (list, tuple)) and len(rel) == 3:
                            writer.writerow([note_id, *rel])
                            print(f"抽取到关系: {note_id}\t{rel[0]}\t{rel[1]}\t{rel[2]}")
                else:
                    print(f"返回结果格式不符 (note_id={note_id}): {normalized}")
            except Exception as e:
                print(f"解析错误 (note_id={note_id}): {e}")

if __name__ == "__main__":
    main()

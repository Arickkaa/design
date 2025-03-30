import os
import csv
import pymysql
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import logging
import unicodedata
from tqdm import tqdm

os.environ["USE_PARALLEL_KERNEL"] = "0"

# 设置日志，用于记录错误信息
logging.basicConfig(filename="relation_extraction.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# 1. MySQL 配置信息
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "123456",
    "database": "media_crawler",
    "charset": "utf8mb4"
}

# 从 MySQL 中读取微博数据
def fetch_data():
    conn = pymysql.connect(**MYSQL_CONFIG)
    query = "SELECT note_id, content FROM weibo_note WHERE content IS NOT NULL LIMIT 100;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 2. 使用 ChatGLM3-6B 进行关系抽取
class RelationExtractor:
    def __init__(self, model_name="THUDM/chatglm2-6b", token=None):
        """
        加载 ChatGLM3-6B 模型，若模型为私有仓库请传入 token，否则 token 设置为 None。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
        # 使用 AutoModel 加载模型，使用 half() 降低显存占用；若显存不足可改为 .to("cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model.eval()

    def extract_relation(self, text):
        prompt = (
            f"从以下微博内容中提取实体关系，并返回 (实体1, 关系, 实体2) 的 JSON 格式列表：\n\n"
            f"{text}\n\n请仅返回列表，例如：[('实体1', '关系', '实体2'), ...]"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        # 将输入数据转移到模型所在设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            # 这里增加 use_cache=False 避免缓存相关问题
            output = self.model.generate(**inputs, max_new_tokens=128, use_cache=False)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result

# 3. 主流程：抽取关系并逐行写入 CSV 文件，同时显示进度
def main():
    df = fetch_data()
    # 请根据需要传入 token；若模型公开，则 token 设置为 None
    extractor = RelationExtractor(token=None)
    
    with open("relations.csv", "w", encoding="utf-8", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入 CSV 表头
        csv_writer.writerow(["note_id", "实体1", "关系", "实体2"])
        csvfile.flush()  # 立即写入表头
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="抽取进度"):
            note_id, text = row["note_id"], row["content"]
            result = None
            try:
                result = extractor.extract_relation(text)
                # 对输出进行 Unicode 规范化，将全角字符转换为半角
                result_normalized = unicodedata.normalize('NFKC', result)
                # 替换智能引号和全角逗号为标准符号
                result_normalized = result_normalized.replace("“", "\"").replace("”", "\"")
                result_normalized = result_normalized.replace("，", ",")
                # 尝试解析返回结果，要求模型返回合法的 Python 表达式（如列表）
                extracted_relations = eval(result_normalized)
                if isinstance(extracted_relations, list):
                    for relation_tuple in extracted_relations:
                        if len(relation_tuple) == 3:
                            head, relation, tail = relation_tuple
                            csv_writer.writerow([note_id, head, relation, tail])
                            csvfile.flush()  # 每写入一行立即写入磁盘
                            print(f"抽取到关系: {note_id}\t{head}\t{relation}\t{tail}")
                        else:
                            logging.info(f"返回结果格式错误 (note_id={note_id}): {relation_tuple}")
                else:
                    logging.info(f"返回结果非列表格式 (note_id={note_id}): {result_normalized}")
            except Exception as e:
                logging.error(f"无法解析模型输出 (note_id={note_id}): {result}")
                logging.error(f"错误信息： {e}")

if __name__ == "__main__":
    main()

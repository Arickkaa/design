import pymysql
import torch
from transformers import AutoTokenizer, AutoModel

# ---------- 1. 从 MySQL 中读取数据 ----------
def fetch_content():
    # 修改为你的数据库连接参数
    connection = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='123456',
        database='media_crawler',
        charset='utf8mb4'
    )
    try:
        with connection.cursor() as cursor:
            # 这里假设你的表名为 weibo_data，且字段 content 存储微博文本
            sql = "SELECT content FROM weibo_note LIMIT 1;"
            cursor.execute(sql)
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return None
    finally:
        connection.close()

# ---------- 2. 加载 ChatGLM-6B 模型 ----------
def load_chatglm_model():
    model_name = "THUDM/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 指定设备为 CPU
    device = torch.device("cpu")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    # 如果设备是 CPU，则确保模型为 float32
    if device.type == "cpu":
        model = model.float()
    else:
        model = model.half()
    model.to(device)
    return tokenizer, model, device

# ---------- 3. 利用大模型进行知识抽取 ----------
def extract_knowledge(content, tokenizer, model):
    # 构造抽取 prompt，要求输出格式为 (实体,关系,实体) 的三元组
    prompt = (
        "请从以下微博内容中抽取知识，输出实体和关系的三元组，格式为 (实体,关系,实体)：\n"
        f"内容：{content}\n"
        "输出："
    )
    # 调用 ChatGLM-6B 模型的 chat 方法进行对话式生成
    # history 参数用于记录对话历史，这里置空即可
    response, _ = model.chat(tokenizer, prompt, history=[])
    return response

# ---------- 主函数 ----------
def main():
    content = fetch_content()
    if not content:
        print("未能获取到数据，请检查数据库连接或数据表内容。")
        return

    print("原始微博内容：")
    print(content)
    print("-" * 50)

    tokenizer, model, device = load_chatglm_model()
    result = extract_knowledge(content, tokenizer, model)
    
    print("知识抽取结果：")
    print(result)

if __name__ == "__main__":
    main()

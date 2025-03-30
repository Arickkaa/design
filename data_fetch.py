import pymysql
import pandas as pd

# MySQL 配置信息
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "123456",
    "database": "media_crawler",
    "charset": "utf8mb4"
}

# 从 MySQL 中读取微博数据（500条）
def fetch_data():
    conn = pymysql.connect(**MYSQL_CONFIG)
    query = "SELECT note_id, content FROM weibo_note WHERE content IS NOT NULL;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def main():
    df = fetch_data()
    # 将数据保存到 CSV 文件中，设置utf-8-sig编码以防中文乱码
    df.to_csv('weibo_data.csv', index=False, encoding='utf-8-sig')
    print("成功将500条数据写入文件：weibo_data.csv")

if __name__ == "__main__":
    main()

import os
import re
import pymysql

# 1. 数据库连接配置
db_config = {
    "host": "127.0.0.1",  # Docker 映射到本地的地址
    "port": 3306,
    "user": "admin",  # 推荐使用项目专用用户
    "password": "123456",
    "database": "cp_rag_db",
    "charset": "utf8mb4"
}


def insert_to_mysql():
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    dataset_path = "../data/leetcode_data/note_data"

    # 遍历数据集
    for folder_name in os.listdir(dataset_path):
        # 正则提取题号和题目名
        match = re.match(r'^(\d+)-(.*)', folder_name)
        if not match:
            continue

        original_id = match.group(1)
        title = match.group(2).replace("-", " ").title()
        formatted_id = f"LC{int(original_id):04d}"

        folder_full_path = os.path.join(dataset_path, folder_name)

        # 读取已完成繁简转换的 Note.md
        note_path = os.path.join(folder_full_path, "Note.md")
        content = ""
        if os.path.exists(note_path):
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read()

        # 读取 AC 代码 answer.ts
        code_path = os.path.join(folder_full_path, "answer.ts")
        ac_code = ""
        if os.path.exists(code_path):
            with open(code_path, 'r', encoding='utf-8') as f:
                ac_code = f.read()

        # 执行插入
        sql = "INSERT INTO solutions (id, title, content, ac_code) VALUES (%s, %s, %s, %s)"
        try:
            cursor.execute(sql, (formatted_id, title, content, ac_code))
        except Exception as e:
            print(f"Error inserting {formatted_id}: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    print("数据入库任务完成！")


if __name__ == "__main__":
    insert_to_mysql()
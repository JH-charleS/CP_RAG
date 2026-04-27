import os
import opencc

# 初始化转换器：'t2s' 代表 Traditional to Simplified (繁转简)
converter = opencc.OpenCC('t2s')

# 假设你的数据集放在 'leetcode_dataset' 文件夹下
dataset_path = '../data/leetcode_data/note_data'

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file == 'Note.md':
            file_path = os.path.join(root, file)

            # 读取繁体内容
            with open(file_path, 'r', encoding='utf-8') as f:
                trad_content = f.read()

            # 执行转换
            simp_content = converter.convert(trad_content)

            # 覆盖原文件（或另存为 Note_simp.md）
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(simp_content)

print("全部 Note.md 繁简转换完成！")
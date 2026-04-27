import os
import shutil
from pathlib import Path


def extract_specific_files(source_dir, target_dir):
    """
    从源文件夹及其子文件夹中提取特定类型的文档到目标文件夹。
    支持格式：txt, doc, docx, ppt, pptx, pdf
    """
    # 定义允许的后缀名（统一转为小写处理）
    ALLOWED_EXTENSIONS = {'.txt', '.doc', '.docx', '.ppt', '.pptx', '.pdf'}

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 创建目标文件夹
    if not target_path.exists():
        target_path.mkdir(parents=True)

    count = 0

    # 使用 rglob('*') 递归遍历所有文件
    for file_path in source_path.rglob('*'):
        # 只处理文件，且后缀名在允许列表中
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:

            dest_file_path = target_path / file_path.name

            # 处理同名文件冲突
            counter = 1
            while dest_file_path.exists():
                new_name = f"{file_path.stem}_{counter}{file_path.suffix}"
                dest_file_path = target_path / new_name
                counter += 1

            try:
                shutil.copy2(file_path, dest_file_path)
                count += 1
                # print(f"已复制: {file_path.name}") # 如果文件太多建议关闭打印
            except Exception as e:
                print(f"处理文件 {file_path.name} 时出错: {e}")

    return count


# 如果你想单独运行测试这个脚本，保留下面的代码：
if __name__ == "__main__":
    src = r"/source_data\Competitive-Programming-Docs\国家集训队历年论文集"
    dst = r"/source_data\temp_data"
    total = extract_specific_files(src, dst)
    print(f"任务完成，共提取文档：{total} 个")
import os
import shutil
from pathlib import Path

# 主目录路径
main_dir = "/home/vgg16_aml_classification/aml_data/RUNX1_RUNX1T1"  #换成每个类别提取

# 初始化计数器
total_files_before = 0
total_files_after = 0
files_by_folder = {}

# 获取所有子文件夹
subdirs = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]

# 统计提取前的文件数量
print("提取前统计：")
for subdir in subdirs:
    subdir_path = os.path.join(main_dir, subdir)

    # 获取子文件夹中的所有tif文件
    tif_files = [f for f in os.listdir(subdir_path) if f.endswith('.tif')]
    folder_file_count = len(tif_files)

    # 更新统计信息
    files_by_folder[subdir] = folder_file_count
    total_files_before += folder_file_count

    print(f"  子文件夹 {subdir}: {folder_file_count} 个tif文件")

print(f"提取前总计: {total_files_before} 个tif文件\n")

# 遍历每个子文件夹并提取文件
print("开始提取文件...")
for subdir in subdirs:
    subdir_path = os.path.join(main_dir, subdir)

    # 获取子文件夹中的所有tif文件
    tif_files = [f for f in os.listdir(subdir_path) if f.endswith('.tif')]

    # 遍历每个tif文件
    for tif_file in tif_files:
        # 原始文件路径
        original_path = os.path.join(subdir_path, tif_file)

        # 新文件名: 原始文件名_子文件夹名称.tif
        new_filename = f"{os.path.splitext(tif_file)[0]}_{subdir}.tif"

        # 新文件路径
        new_path = os.path.join(main_dir, new_filename)

        # 复制文件到主目录并重命名
        shutil.copy2(original_path, new_path)
        print(f"  已复制: {original_path} -> {new_path}")

# 验证提取后的文件数量
extracted_files = [f for f in os.listdir(main_dir) if f.endswith('.tif') and os.path.isfile(os.path.join(main_dir, f))]
total_files_after = len(extracted_files)

print(f"\n提取后总计: {total_files_after} 个tif文件")

# 如果计数不一致，发出警告
if total_files_after != total_files_before:
    print(f"警告: 提取前后文件数量不一致! 差异: {total_files_after - total_files_before}")

# 删除原子文件夹及其内容
print("\n开始删除原子文件夹...")
for subdir in subdirs:
    subdir_path = os.path.join(main_dir, subdir)
    shutil.rmtree(subdir_path)
    print(f"  已删除子文件夹: {subdir}")

print("\n处理完成！所有文件已提取并重命名，原子文件夹已删除。")

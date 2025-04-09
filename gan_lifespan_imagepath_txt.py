# 生成lifespan需要的txt目录图片文件

import os

def save_file_paths_to_txt(directory, output_txt):
    with open(output_txt, 'w') as file:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                # 构建完整的相对路径
                relative_path = os.path.join(root, file_name)
                # 调整路径为所需的格式，这里假设所有文件都应归类于"datasets/males/"
                # 实际应用中可能需要根据文件的实际路径调整逻辑
                formatted_path = relative_path.replace(directory, "").lstrip('/')
                formatted_path = "datasets/females/" + formatted_path
                file.write(formatted_path + "\n")

# 示例用法
directory_to_scan = "/data/Lifespan_Age_Transformation_Synthesis-master/datasets/females"  # 替换为你的目录路径
output_file = "file_paths.txt"  # 输出的TXT文件名
save_file_paths_to_txt(directory_to_scan, output_file)

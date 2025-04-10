import os
import re
from openai import OpenAI


# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="DASHSCOPE_API_KEY",  # 替换为您的 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def model_compare(image_path1, image_path2):
    """
    使用大模型判断两张图片是否是同一个人。
    
    参数:
    - image_path1: 第一张图片的路径。
    - image_path2: 第二张图片的路径。
    
    返回:
    - True: 如果大模型判断是同一个人。
    - False: 如果大模型判断不是同一个人或发生错误。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请判断这两张图片中的人物是否是同一个人，只允许返回“是”或“不是”"},
                {"type": "image_url", "image_url": {"url": f"file://{image_path1}"}},
                {"type": "image_url", "image_url": {"url": f"file://{image_path2}"}}
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 使用适合的视觉语言模型
            messages=messages
        )

        # 解析返回结果
        response_content = completion.choices[0].message.content.lower()
        if "是" in response_content:
            return True  # 判断为同一个人
        else:
            return False  # 判断为不同人
    except Exception as e:
        print(f"模型判断失败: {e}")
        return False

def find_and_compare_images(dir1, dir2, out_file):
    """
    遍历两个目录，找到相同文件名（忽略后缀）的图片，并使用大模型判断是否是同一个人。
    
    参数:
    - dir1: 第一个目录路径。
    - dir2: 第二个目录路径。
    - out_file: 输出结果文件路径。
    """
    with open(out_file, "a") as f:
        # 获取两个目录下的文件列表
        files_dir1 = {os.path.splitext(file)[0]: file for file in os.listdir(dir1)}
        files_dir2 = {os.path.splitext(file)[0]: file for file in os.listdir(dir2)}

        # 找到两个目录中共有的文件名（忽略后缀）
        common_files = set(files_dir1.keys()).intersection(set(files_dir2.keys()))
        results = []

        for filename in sorted(common_files):
            # 构建完整的文件路径
            path1 = os.path.join(dir1, files_dir1[filename])
            path2 = os.path.join(dir2, files_dir2[filename])

            # 调用大模型进行判断
            is_same_person = model_compare(path1, path2)
            result = f"文件 {filename}: {'是同一个人' if is_same_person else '不是同一个人'}"
            print(result)
            f.write(result + "\n")
            results.append(is_same_person)

        # 统计结果
        total_files = len(results)
        same_count = sum(results)
        similarity_rate = (same_count / total_files) * 100 if total_files > 0 else 0
        summary = f"总文件数: {total_files}, 同一个人的比例: {similarity_rate:.2f}%"
        print(summary)
        f.write(summary + "\n")

# 使用示例
dir_path1 = "path/to/your/directory1"  # 原图路径
dir_path2 = "path/to/your/directory2"  # 目标年龄目录路径
out_file = "XX_face_compare.txt"
find_and_compare_images(dir_path1, dir_path2, out_file)

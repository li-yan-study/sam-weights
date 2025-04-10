import os
import requests
from json import JSONDecoder

def compareIm(faceId1, faceId2, similarity_threshold=75):
    """
    使用Face++ API比较两张图片的相似度。
    
    参数:
    - faceId1: 第一张图片的路径。
    - faceId2: 第二张图片的路径。
    - similarity_threshold: 相似度阈值，默认为75。
    
    返回:
    - confidence: 图片的相似度值，如果请求成功且相似度超过阈值。
    - None: 如果请求失败或相似度未达到阈值。
    """
    compare_url = "https://api-cn.faceplusplus.com/facepp/v3/compare"
    key = "你自己申请的key"  # 替换为您的 Face++ API Key
    secret = "你自己申请的secret"  # 替换为您的 Face++ API Secret
    
    try:
        with open(faceId1, "rb") as f1, open(faceId2, "rb") as f2:
            files = {"image_file1": f1, "image_file2": f2}
            data = {"api_key": key, "api_secret": secret}
            response = requests.post(compare_url, data=data, files=files)
            req_con = response.content.decode('utf-8')
            req_dict = JSONDecoder().decode(req_con)
            
            confidence = req_dict.get('confidence', None)
            if confidence and confidence > similarity_threshold:
                print(f"图片相似度：{confidence}")
                return 100
            else:
                return 0
            
    except Exception as e:
        print("发生错误：", e)
    
    return None  # 表示请求失败或不满足条件

def find_and_compare_images(dir1, dir2, out_file, similarity_threshold=75):
    """
    遍历两个目录，找到相同文件名（忽略后缀）的图片，并使用Face++ API比较它们的相似度。
    
    参数:
    - dir1, dir2: 两个需要比较的目录路径。
    - out_file: 输出结果文件路径。
    - similarity_threshold: 相似度阈值，默认为75。
    """
    with open(out_file, "a") as f:
        # 获取两个目录下的文件列表，并提取文件名（去掉后缀）
        files_dir1 = {os.path.splitext(file)[0]: file for file in os.listdir(dir1)}
        files_dir2 = {os.path.splitext(file)[0]: file for file in os.listdir(dir2)}

        # 找到两个目录中共有的文件名（忽略后缀）
        common_files = set(files_dir1.keys()).intersection(set(files_dir2.keys()))
        confidences = []

        target_age = os.path.basename(dir2)  # 获取目标年龄目录名称
        for filename in sorted(common_files):
            # 构建完整的文件路径
            path1 = os.path.join(dir1, files_dir1[filename])
            path2 = os.path.join(dir2, files_dir2[filename])
            
            # 调用compareIm函数进行比较
            confidence = compareIm(path1, path2, similarity_threshold)
            if confidence is not None:
                print(f"文件 {filename} 的相似度为：{confidence}")
                confidences.append(confidence)
            else:
                print(f"文件 {filename} 无法比较或未达到阈值。")

        # 计算平均相似度
        if confidences:
            average_confidence = sum(confidences) / len(confidences)
            avg_str = f"目标年龄：{target_age}, 相似概率：{average_confidence:.2f}"
            f.write(avg_str + "\n")
            print(avg_str)
        else:
            print("没有有效的相似度数据可供计算。")

# 使用示例
dir_path1 = "path/to/your/directory1"  # 原图路径
dir_path2 = "path/to/your/directory2"  # 目标年龄目录路径
out_file = "XX_face_compare.txt"
find_and_compare_images(dir_path1, dir_path2, out_file)

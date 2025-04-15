import os
import requests
from json import JSONDecoder
import time

def compareIm(faceId1, faceId2):
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
    key = "UoP6baxo5c9lmcW4DcRpK9aaPsv6dGva"  # 替换为您的 Face++ API Key
    secret = "HIPPhrYrLAis0qqBij4Km1iWpjQFmieO"  # 替换为您的 Face++ API Secret
    
    try:
        with open(faceId1, "rb") as f1, open(faceId2, "rb") as f2:
            files = {"image_file1": f1, "image_file2": f2}
            data = {"api_key": key, "api_secret": secret}
            response = requests.post(compare_url, data=data, files=files)
            req_con = response.content.decode('utf-8')
            req_dict = JSONDecoder().decode(req_con)
            
            confidence = req_dict.get('confidence', None)
            if confidence is not None:
                return confidence
            else:
                return None
            
    except Exception as e:
        print("发生错误：", e)
    
    return None  # 表示请求失败或不满足条件

def find_and_compare_images(dir1, dir2_root, out_file):
    """
    遍历两个目录，找到相同文件名（忽略后缀）的图片，并使用Face++ API比较它们的相似度。
    
    参数:
    - dir1: 第一个需要比较的目录路径。
    - dir2_root: 第二个根目录路径，包含多个子目录。
    - out_file: 输出结果文件路径。
    # - similarity_threshold: 相似度阈值，默认为75。
    """
    with open(out_file, "a",encoding="utf-8") as f:
        # 获取第一个目录下的文件列表，并提取文件名（去掉后缀）
        files_dir1 = {os.path.splitext(file)[0]: file for file in os.listdir(dir1)}

        # 遍历第二个根目录下的所有子目录
        for subdir_name in os.listdir(dir2_root):
            subdir_path = os.path.join(dir2_root, subdir_name)
            if not os.path.isdir(subdir_path):
                continue  # 跳过非目录项

            # 获取子目录下的文件列表，并提取文件名（去掉后缀）
            files_dir2 = {os.path.splitext(file)[0]: file for file in os.listdir(subdir_path)}

            # 找到两个目录中共有的文件名（忽略后缀）
            common_files = set(files_dir1.keys()).intersection(set(files_dir2.keys()))
            confidences = []

            for filename in sorted(common_files):
                # 构建完整的文件路径
                path1 = os.path.join(dir1, files_dir1[filename])
                path2 = os.path.join(subdir_path, files_dir2[filename])
                
                # 调用compareIm函数进行比较
                confidence = compareIm(path1, path2)
                if confidence is not None:
                    sim_str = f"文件 {filename} 的相似度为：{confidence}"
                    print(sim_str)
                    confidences.append(confidence)
                    f.write(sim_str + "\n")
                else:
                    print(f"文件 {filename} 无法比较。")
                
                time.sleep(1.1) # 延时1.1秒，避免频繁请求API
            # 计算平均相似度
            if confidences:
                average_confidence = sum(confidences) / len(confidences)
                avg_str = f"目标年龄目录：{subdir_name}, 平均相似概率：{average_confidence:.2f}"
                f.write(avg_str + "\n")
                print(avg_str)
            else:
                print(f"子目录 {subdir_name} 没有有效的相似度数据可供计算。")

# 使用示例
dir_path1 = "data_200_hebing"  # 原图路径
dir_path2_root = "cusp_200"  # 包含多个子目录的目标年龄根目录
out_file = "cusp_face_compare.txt"
find_and_compare_images(dir_path1, dir_path2_root, out_file)

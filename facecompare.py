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
    key = "你自己申请的key"
    secret = "你自己申请的secret"
    
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
                return confidence
            
    except Exception as e:
        print("发生错误：", e)
    
    return None  # 表示请求失败或不满足条件

def find_and_compare_images(dir1, dir2, similarity_threshold=75):
    """
    遍历两个目录，找到相同文件名的图片，并使用Face++ API比较它们的相似度。
    
    参数:

    - dir1, dir2: 两个需要比较的目录路径。

    - similarity_threshold: 相似度阈值，默认为75。
    """
    # 获取两个目录下的文件列表
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))
    
    # 找到两个目录中共有的文件名
    common_files = files_dir1.intersection(files_dir2)
    
    for filename in common_files:
        # 构建完整的文件路径
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)
        
        # 调用compareIm函数进行比较
        confidence = compareIm(path1, path2, similarity_threshold)
        
        if confidence is not None:
            print(f"文件 {filename} 的相似度为：{confidence}")
        else:
            print(f"文件 {filename} 无法比较或未达到阈值。")
        
    print("所有相同文件名的图片已处理完成。")

# 使用示例
dir_path1 = "path/to/your/directory1"
dir_path2 = "path/to/your/directory2"
find_and_compare_images(dir_path1, dir_path2)

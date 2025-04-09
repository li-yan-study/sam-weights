import os
import requests
from json import JSONDecoder

http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
key = "你的API Key"
secret = "你的API Secret"
# filepath1 = "D:\\py\\image\\WIN_20180412_21_52_13_Pro.jpg"

data = {
"api_key": key,
"api_secret": secret,
"return_attributes": "age"
}

# files = {"image_file": open(filepath1, "rb")}

# response = requests.post(http_url, data=data, files=files)
# req_con = response.content.decode('utf-8')
# req_dict = JSONDecoder().decode(req_con)
# print(req_dict)


# 人脸年龄检测
def face_detect(http_url, data,image):
    response = requests.post(http_url, data=data, files=image)
    req_con = response.content.decode('utf-8')
    result = JSONDecoder().decode(req_con)
    print(result)
    return result


def process_images(base_dir,http_url,data, out_file):
    with open(out_file, "w") as f:
        total_differences = []
        total_blurs = []
        total_facequlitys = []
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                print(f"正在处理子目录：{subdir}")
                differences = []
                blurs = []
                facequalitys =[]
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    if filename.lower().endswith(('.png','.jpg','.jpeg')):
                        try:
                            files = {"image_file": open(file_path, "rb")}
                            result = face_detect(http_url=http_url, data=data, image=files)
                            age = result['faces'][0]['attributes']['age']['value']
                            blur = result['faces'][0]['attributes']['blur']['value']
                            facequality = result['faces'][0]['attributes']['facequlity']['value']
                            difference = abs(int(subdir)- age)
                            differences.append(difference) # 记录差异值
                            total_differences.append(difference)
                            blurs.append(blur)
                            total_blurs.append(blur)
                            facequalitys.append(facequality)
                            total_facequlitys.append(facequality)
                            # 打印和保存每张图片的处理结果
                            result_str = f"图片：{file_path}, 目标年龄：{subdir}, 检测年龄：{age},差异:{difference},模糊度blur:{blur},人脸质量：{facequality}"
                            print(result_str)
                            f.write(result_str + "\n")
                        except Exception as e:
                            print(f"处理图片{file_path}时出错")


                average_difference = sum(differences) / len(differences)
                aveblur = sum(blurs)/len(blurs)
                avefacequality = sum(facequalitys)/len(facequalitys)
                avg_str= f"目标年龄：{subdir},平均差异：{average_difference:.4f},平均模糊度:{aveblur:.4f},平均人脸质量:{avefacequality:.4f}"
                print(avg_str)
                f.write(avg_str + "\n")

        total_average_difference = sum(total_differences) / len(total_differences)
        total_aveblur = sum(total_blurs)/len(total_blurs)
        total_avefacequality = sum(total_facequlitys)/len(total_facequlitys)
        avg_str= f"年龄总平均差异：{total_average_difference:.4f},总平均模糊度:{total_aveblur:.4f},总平均人脸质量:{total_avefacequality:.4f}"
        print(avg_str)
        f.write(avg_str + "\n")
image_directory = ""  #图片目录
out_file = "ageDif-SAM.txt"  # 保存 各图片、子目录年龄差异的文本结果

process_images(image_directory,http_url, data,out_file=out_file)

# imagepath = ""
# face_detect(imageToBase64(imagepath), imageType, options)

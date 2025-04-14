import os
import argparse
import requests
from json import JSONDecoder
import time
# from tenacity import retry, stop_after_attempt, wait_exponential


#   python script.py --restart   全新开始
#   python script.py --resume    断点续传
# 参数解析设置
parser = argparse.ArgumentParser(description='人脸年龄检测批处理工具')
parser.add_argument('--resume', action='store_true', help='从上次中断处继续运行')
parser.add_argument('--restart', action='store_true', help='清除检查点重新开始')
args = parser.parse_args()

# 检查点管理
CHECKPOINT_FILE = "processing.checkpoint"
if args.restart and os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("已清除历史检查点")

# API配置
HTTP_URL = "https://api-cn.faceplusplus.com/facepp/v3/detect"
API_KEY = "UoP6baxo5c9lmcW4DcRpK9aaPsv6dGva"    # 替换实际值
API_SECRET = "HIPPhrYrLAis0qqBij4Km1iWpjQFmieO"  # 替换实际值

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def face_detect(http_url, data, files):
    response = requests.post(http_url, data=data, files=files)
    response.raise_for_status()
    return JSONDecoder().decode(response.content.decode('utf-8'))

def load_processed_files():
    processed = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as cf:
            processed = {os.path.abspath(line.strip()) for line in cf}
    return processed

def process_images(base_dir, out_file="ageDif-SAM.txt"):
    processed = load_processed_files()
    
    with open(out_file, 'a' if args.resume else 'w',encoding="utf-8") as f:  # 结果文件模式控制
        total_metrics = {'diff': [], 'blur': [], 'quality': []}
        
        for subdir in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # current_metrics = {'diff': [], 'blur': [], 'quality': []}
            current_metrics = {'diff': [],  'quality': []}
            print(f"\n处理目录: {subdir_path}")
            
            for filename in sorted(os.listdir(subdir_path)):
                file_path = os.path.abspath(os.path.join(subdir_path, filename))
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                if file_path in processed:
                    print(f"跳过已处理: {filename}")
                    continue
                
                try:
                    with open(file_path, 'rb') as img_file:
                        result = face_detect(
                            http_url=HTTP_URL,
                            data={"api_key": API_KEY, "api_secret": API_SECRET, "return_attributes": "age,facequality"},
                            files={"image_file": img_file}
                        )
                    
                    # 数据提取
                    face = result['faces'][0]
                    age = face['attributes']['age']['value']
                    # blur = face['attributes']['blur']['value']
                    quality = face['attributes']['facequality']['value']
                    
                    # 记录结果
                    diff = abs(int(subdir) - age)
                    log_line = f"{file_path} | 目标:{subdir} | 检测:{age} | 差异:{diff} | 质量:{quality:.2f}"
                    print(log_line)
                    f.write(log_line + '\n')
                    
                    # 更新指标
                    current_metrics['diff'].append(diff)
                    # current_metrics['blur'].append(blur)
                    current_metrics['quality'].append(quality)
                    
                    # 更新检查点
                    with open(CHECKPOINT_FILE, 'a') as cf:
                        cf.write(file_path + '\n')
                    processed.add(file_path)
                    
                    # 延时2s
                    time.sleep(1.1)
                except Exception as e:
                    print(f"处理失败: {filename} - {str(e)}"+'\n')
                    print(result)
                    continue

            # 目录级统计
            if current_metrics['diff']:
                avg_line = f"[{subdir}] 平均差异:{sum(current_metrics['diff'])/len(current_metrics['diff']):.2f} | " \
                          f"平均模糊:{sum(current_metrics['blur'])/len(current_metrics['blur']):.2f} | " \
                          f"平均质量:{sum(current_metrics['quality'])/len(current_metrics['quality']):.2f}"
                print(avg_line)
                f.write(avg_line + '\n')
                total_metrics['diff'].extend(current_metrics['diff'])
                total_metrics['blur'].extend(current_metrics['blur'])
                total_metrics['quality'].extend(current_metrics['quality'])

        # 全局统计
        if total_metrics['diff']:
            final_line = f"[总计] 平均差异:{sum(total_metrics['diff'])/len(total_metrics['diff']):.2f} | " \
                        f"平均模糊:{sum(total_metrics['blur'])/len(total_metrics['blur']):.2f} | " \
                        f"平均质量:{sum(total_metrics['quality'])/len(total_metrics['quality']):.2f}"
            print('\n' + final_line)
            f.write(final_line + '\n')

if __name__ == "__main__":
    process_images(
        base_dir="cusp_out",  # 替换实际路径
        out_file="cusp_analysis_result.txt"
    )
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)  # 完成后自动清理检查点

import os
import re
import argparse
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


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

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="DASHSCOPE_API_KEY",  # 替换为您的 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def model_detect(image_path):
    """
    使用大模型检测图片中人物的年龄。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请检测图片中人物的年龄,并且仅仅返回检测到年龄的数字，不要有任何其他输出"},
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 使用适合的视觉语言模型
            messages=messages
        )

        # 解析返回结果，提取年龄数字
        response_content = completion.choices[0].message.content
        age_match = re.search(r'\d+', response_content)  # 匹配连续数字
        if age_match:
            return int(age_match.group(0))  # 返回年龄数字
        else:
            raise ValueError("未检测到年龄")
    except Exception as e:
        raise RuntimeError(f"模型检测失败: {e}")

def load_processed_files():
    processed = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as cf:
            processed = {os.path.abspath(line.strip()) for line in cf}
    return processed

def process_images(base_dir, out_file="ageDif-SAM.txt"):
    processed = load_processed_files()
    
    with open(out_file, 'a' if args.resume else 'w') as f:  # 结果文件模式控制
        total_metrics = {'diff': []}
        
        for subdir in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            current_metrics = {'diff': []}
            print(f"\n处理目录: {subdir_path}")
            
            for filename in sorted(os.listdir(subdir_path)):
                file_path = os.path.abspath(os.path.join(subdir_path, filename))
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                if file_path in processed:
                    print(f"跳过已处理: {filename}")
                    continue
                
                try:
                    # 调用大模型进行年龄检测
                    detected_age = model_detect(file_path)
                    
                    # 数据记录
                    target_age = int(subdir)  # 假设子目录名称为目标年龄
                    diff = abs(target_age - detected_age)
                    log_line = f"{file_path} | 目标:{target_age} | 检测:{detected_age} | 差异:{diff}"
                    print(log_line)
                    f.write(log_line + '\n')
                    
                    # 更新指标
                    current_metrics['diff'].append(diff)
                    
                    # 更新检查点
                    with open(CHECKPOINT_FILE, 'a') as cf:
                        cf.write(file_path + '\n')
                    processed.add(file_path)
                        
                except Exception as e:
                    print(f"处理失败: {filename} - {str(e)}")
                    continue

            # 目录级统计
            if current_metrics['diff']:
                avg_line = f"[{subdir}] 平均差异:{sum(current_metrics['diff']) / len(current_metrics['diff']):.2f}"
                print(avg_line)
                f.write(avg_line + '\n')
                total_metrics['diff'].extend(current_metrics['diff'])

        # 全局统计
        if total_metrics['diff']:
            final_line = f"[总计] 平均差异:{sum(total_metrics['diff']) / len(total_metrics['diff']):.2f}"
            print('\n' + final_line)
            f.write(final_line + '\n')

if __name__ == "__main__":
    process_images(
        base_dir="/data/SAM-master/test-out/inference_results",  # 替换实际路径
        out_file="analysis_result.txt"
    )
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)  # 完成后自动清理检查点

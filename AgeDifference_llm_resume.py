import os
import re
import json
import argparse
import base64
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
    api_key="sk-1b8326be718d4a5698dc1f9d86e0052c",  # 替换为您的 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def model_detect(image_path):
    """
    使用大模型检测图片中人物的年龄和质量评分。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请检测图片中人物的年龄和质量评分(0-100)，并以以下格式返回: age:{年龄}, quality:{质量评分}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_path}"}}
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 使用适合的视觉语言模型
            messages=messages
        )

        # 解析返回结果，提取年龄和质量评分
        response_content = completion.choices[0].message.content.strip()
        age_match = re.search(r'age:(\d+)', response_content)  # 匹配年龄
        quality_match = re.search(r'quality:(\d+(\.\d+)?)', response_content)  # 匹配质量评分

        if age_match and quality_match:
            age = int(age_match.group(1))  # 提取年龄
            quality = float(quality_match.group(1))  # 提取质量评分
            return {'age': age, 'quality': quality}
        else:
            raise ValueError("未检测到年龄或质量评分")
    except Exception as e:
        raise RuntimeError(f"模型检测失败: {e}")

def load_checkpoint():
    """
    加载检查点文件，恢复已处理的文件和统计信息。
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return {
            "processed_files": set(),
            "subdir_metrics": {},
            "total_metrics": {"diff": [], "quality": []}
        }

    with open(CHECKPOINT_FILE, 'r') as cf:
        checkpoint_data = json.load(cf)
    
    # 将 processed_files 转换为 set
    checkpoint_data["processed_files"] = set(checkpoint_data.get("processed_files", []))
    return checkpoint_data

def save_checkpoint(checkpoint_data):
    """
    保存检查点文件，记录当前状态。
    """
    checkpoint_data["processed_files"] = list(checkpoint_data["processed_files"])  # 转换为列表以便序列化
    with open(CHECKPOINT_FILE, 'w') as cf:
        json.dump(checkpoint_data, cf, indent=4)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_images(base_dir, out_file="age_quality_analysis.txt"):
    checkpoint_data = load_checkpoint()
    processed_files = checkpoint_data["processed_files"]
    subdir_metrics = checkpoint_data["subdir_metrics"]
    total_metrics = checkpoint_data["total_metrics"]

    with open(out_file, 'a' if args.resume else 'w') as f:  # 结果文件模式控制
        for subdir in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            if subdir in subdir_metrics:  # 如果子目录已处理过，跳过
                print(f"跳过已处理目录: {subdir}")
                continue

            current_metrics = {'diff': [], 'quality': []}
            print(f"\n处理目录: {subdir_path}")
            
            for filename in sorted(os.listdir(subdir_path)):
                file_path = os.path.abspath(os.path.join(subdir_path, filename))
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                if file_path in processed_files:
                    print(f"跳过已处理: {filename}")
                    continue
                
                try:
                    # 编码图像并调用大模型进行检测
                    base64_image = encode_image(file_path)
                    detection_result = model_detect(base64_image)
                    
                    # 数据记录
                    target_age = int(subdir)  # 假设子目录名称为目标年龄
                    detected_age = detection_result['age']
                    quality = detection_result['quality']
                    diff = abs(target_age - detected_age)
                    log_line = f"{file_path} | 目标:{target_age} | 检测:{detected_age} | 差异:{diff} | 质量评分:{quality:.2f}"
                    print(log_line)
                    f.write(log_line + '\n')
                    
                    # 更新指标
                    current_metrics['diff'].append(diff)
                    current_metrics['quality'].append(quality)
                    
                    # 更新检查点
                    processed_files.add(file_path)
                    save_checkpoint({
                        "processed_files": processed_files,
                        "subdir_metrics": subdir_metrics,
                        "total_metrics": total_metrics
                    })
                
                except Exception as e:
                    print(f"处理失败: {filename} - {str(e)}")
                    continue

            # 更新子目录统计
            if current_metrics['diff'] and current_metrics['quality']:
                avg_diff = sum(current_metrics['diff']) / len(current_metrics['diff'])
                avg_quality = sum(current_metrics['quality']) / len(current_metrics['quality'])
                avg_line = f"[{subdir}] 平均差异:{avg_diff:.2f} | 平均质量评分:{avg_quality:.2f}"
                print(avg_line)
                f.write(avg_line + '\n')

                subdir_metrics[subdir] = {
                    "avg_diff": avg_diff,
                    "avg_quality": avg_quality
                }

                # 更新全局统计
                total_metrics['diff'].extend(current_metrics['diff'])
                total_metrics['quality'].extend(current_metrics['quality'])

        # 全局统计
        if total_metrics['diff'] and total_metrics['quality']:
            final_avg_diff = sum(total_metrics['diff']) / len(total_metrics['diff'])
            final_avg_quality = sum(total_metrics['quality']) / len(total_metrics['quality'])
            final_line = f"[总计] 平均差异:{final_avg_diff:.2f} | 平均质量评分:{final_avg_quality:.2f}"
            print('\n' + final_line)
            f.write(final_line + '\n')

        # 保存最终检查点
        save_checkpoint({
            "processed_files": processed_files,
            "subdir_metrics": subdir_metrics,
            "total_metrics": total_metrics
        })

if __name__ == "__main__":
    process_images(
        base_dir="cusp_out",  # 替换实际路径
        out_file="cusp_result.txt"
    )
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)  # 完成后自动清理检查点

# 运行： CUDA_VISIBLE_DEVICES=0 python test.py --name males_model --which_epoch latest --display_id 0 --traverse --interp_step 0.05 --image_path_file males_image_list.txt  --in_the_wild --verbose 


import os
import shutil
import re
# 指定您要处理的目录路径
base_dir = 'Lifespan_Age_Transformation_Synthesis-master/results/males_model/test_latest/traversal/'
pattern = re.compile('_class_\d+\.png$')
# 遍历目录
for filename in os.listdir(base_dir):
    # 检查文件是否符合模式，这里假设文件名格式为：XXXXXX_..._class_数字.png
    if  pattern.search(filename):
        # 分割文件名，提取数字部分
        parts = filename.split('_')
        num_part = parts[-1].split('.')[0]  # 获取_class_后的数字
        base_num = filename[:6]  # 假设前6位是您想保留的部分

        # 创建或确认子目录
        sub_dir = os.path.join(base_dir, num_part)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        # 构建新文件名（仅保留前部数字）
        new_filename = f"{base_num}.png"

        # 移动文件到子目录，并重命名
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(sub_dir, new_filename)
        shutil.move(src_path, dst_path)

print("文件分类和重命名完成。")

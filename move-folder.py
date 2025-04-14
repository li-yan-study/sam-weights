import os
import shutil

def copy_folder_structure(src_folder, dst_folder):
    
    if not os.path.exists(src_folder):
        print(f"源文件夹{src_folder}不存在")
        return
    
    for root, dirs, files in os.walk(src_folder):
        relative_path = os.path.join(root, src_folder)

        # for dir_name in dirs:
        #     src_dir_path = os.path.relpath(root, src_folder)
        #     dst_dir_path = os.path.join(dst_folder, relative_path)
        for file_name in files:
            src_file_path = os.path.join(root, file_name)
            dst_file_path = os.path.join(dst_folder, file_name)
            shutil.copy2(src_file_path, dst_file_path)
            print(f"复制文件{src_file_path}到{dst_file_path}")


src_folder = "/data/datasets/celeba_hq/val/male"
dst_folder = "/data/SAM-master/test_data"
copy_folder_structure(src_folder,dst_folder)

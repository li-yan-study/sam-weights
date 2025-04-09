# 让lifespan能够正常推理，不检测关键点
# 使用gan_lifespan_imagepath_txt.py 生成图片目录文件
# 在数据集中新建目录：parsings，把数据集复制到里面。
# 找到data/multiclass_unaligned_dataset.py中的def get_item_from_path(self, path): 函数，修改 if self.in_the_wild:部分如下
# 分男女运行： CUDA_VISIBLE_DEVICES=0 python test.py --name males_model --which_epoch latest --display_id 0 --traverse --interp_step 0.05 --image_path_file males_image_list.txt  --in_the_wild --verbose 
    def get_item_from_path(self, path):
        path_dir, im_name = os.path.split(path)
        img = Image.open(path).convert('RGB')
        img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)

        if self.in_the_wild:
            # img, parsing = self.preprocessor.forward(img)
            parsing_path = os.path.join(path_dir, 'parsings', im_name[:-4] + '.jpg')
            parsing = Image.open(parsing_path).convert('RGB')
            parsing = np.array(parsing.getdata(), dtype=np.uint8).reshape(parsing.size[1], parsing.size[0], 3)
        
        else:
            parsing_path = os.path.join(path_dir, 'parsings', im_name[:-4] + '.png')
            parsing = Image.open(parsing_path).convert('RGB')
            parsing = np.array(parsing.getdata(), dtype=np.uint8).reshape(parsing.size[1], parsing.size[0], 3)

        img = Image.fromarray(self.mask_image(img, parsing))
        img = self.transform(img).unsqueeze(0)

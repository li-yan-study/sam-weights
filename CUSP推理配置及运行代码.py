"""为了获得年龄信息，我们使用在 IMDB-WIKI 数据集上预训练的年龄分类器。我们使用了 Rothe 等人从没有面部标志的单张图像中发布的对真实和表观年龄的深度期望中发布的模型。

要准备模型，您需要下载原始 caffe 模型并将其转换为 PyTorch 格式。我们使用 Vadim Kantorov 发布的转换器 caffemodel2pytorch。然后将 PyTorch 模型命名为 as 并将其放入文件夹 .dex_imdb_wiki.caffemodel.pt/models"""

## 1 首先需要下载预训练权重 分为LS和RR版本。  
## 2 然后下载HRFAE中DEX分类器权重（https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_imdb_wiki.caffemodel），并使用caffemodel2pytorch（https://github.com/vadimkantorov/caffemodel2pytorch）转换成.pt格式
## 3 使用以下代码进行推理

# vgg16-aml-classification
vgg16 aml classification
"# vgg16_aml_classification" 


一：下载数据
https://www.kaggle.com/datasets/gchan357/human-aml-cytomorphology-dataset
数据描述：
人类白血病细胞形态学数据集
正常人和白血病患者的血细胞图像数据集
关于数据集
该数据集包括四个流行的急性髓系白血病（AML）亚型的血细胞图像，这些亚型具有定义性的遗传异常和典型的形态特征，根据世卫组织2022年的分类:（i） APL与PML::RARA融合，（ii） AML与NPM1突变，（iii） AML和CBFB::MYH11融合（无NPM1变异），以及（iv） AML的RUNX1::RUNX1T1融合，以及对照组的健康干细胞捐献者。

根目录下的每个文件夹代表一个类。这些是：

（1）正常患者（对照）
（2） APL与PML::RARA融合（PML_RARA）
（3）带有 NPM1 突变（ NPM1 ）的AML
（4）具有CBFB::MYH11融合且无NPM1突变的AML（CBFB_MYH11）
（5） AML与RUNX1::RUNX1T1融合（RUNX1_RUNX1T1 ）。

每个类文件夹下的每个子文件夹代表来自单个白血病患者的图像。每个患者的首字母是该文件夹的名称。

请参阅本笔记本，了解每类图像的图像编号、图像尺寸和颜色信息。所有图像的尺寸为144x144。
https://www.kaggle.com/code/gchan357/aml-image-number-dimension-and-color-information/

所有彩色图像。所有图像文件均为.tif格式。

背景资料:

2009年至2020年慕尼黑白血病实验室（MLL）数据库中共有189份外周血涂片被数字化。首先，对所有血液涂片进行10倍放大扫描，并创建一个概览图像。使用Metasystems Metafer平台，使用分割阈值和对数颜色变换自动执行细胞检测。自动对血液涂抹区域的质量进行进一步分析。每个患者的99-500个白细胞，然后通过油浸显微镜以40倍放大的TIF格式扫描，对应于24.9μm x 24.9μm（144x144像素）。为此，使用了Meta Systems的CMOS彩色相机，分辨率为4096x3000px，像素大小为3,45μm x 3,45μm。四个像素被分成一个像素，其大小为6.9μm x 6.9μm，分辨率为6.9μm/40（1 px=01725μm）。


二、数据提取与预处理
python data_extraction.py

三、vgg16模型构建、训练、验证、测试、评估
python train.py

四、模型对比
python model_comparison.py

五、是否添加BN层模型效果对比
python model comparison_bn_nobn.py

# 第四次作业

廖沩健

自动化65

2160504124

提交日期:  2019年3月17日

摘要:

这次作业聚焦于空域高低通滤波处理，低通滤波器选用了中值滤波器和高斯滤波器(固定方差为1.5)，高通滤波选择了unsharp masking的方法，以及sobel算子和laplace算子。其中所有的算法除Canny外都是用python从头实现的，使用到的库主要有numpy,cv2,matplotlib等。大部分参考的都是课本与ppt的内容。对于所有处理结果的讨论，都展示在下面的对应的部分。


## 空域低通滤波
### 中值滤波
#### test1

3x3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test1.pgm_median3x3.bmp)

5x5

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test1.pgm_median5x5.bmp)

7x7

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test1.pgm_median7x7.bmp)

#### test2

3x3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test2.tif_median3x3.bmp)

5x5

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test2.tif_median5x5.bmp)

7x7

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test2.tif_median7x7.bmp)

### 高斯滤波
我使用了“gaussian.pdf”文件中给出的公式计算高斯滤波器，然后对所有的值做了归一化处理，保证权值和为1。

#### test1

3x3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test1.pgm_gaussian3x3.bmp)

5x5

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test1.pgm_gaussian5x5.bmp)

7x7

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test1.pgm_gaussian7x7.bmp)

#### test2

3x3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test2.tif_gaussian3x3.bmp)

5x5

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test2.tif_gaussian5x5.bmp)

7x7

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test2.tif_gaussian7x7.bmp)

由以上所有图可以看出一个趋势，那就是随着过滤器的尺寸变大，图片会越来越模糊。对比中值和高斯滤波可以发现，中值滤波器可以有效去出图上的“白点”，这一点在test1中尤其明显，并且随着滤波器尺寸增大，白点越来少，到7的时候，几乎没有白点了；而高斯滤波器对这一点却有些无能为力，白点同整个图像一起模糊，但并未消除。但是相比于中值滤波，高斯滤波器可以很好的保持图片内容的细节，即使在大尺寸的滤波器处理下，细节的轮廓仍然是存在甚至清新的，而中值滤波在大尺寸的情况下就是直接混淆甚至去除了细节，使图片失真严重。


## 空域高通滤波（边缘提取）
### unsharp masking

#### test3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test3_corrupt.pgm_unshapen.bmp)

#### test4

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test4.tif_unshapen.bmp)

### sobel edge detector
#### test3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test3_corrupt.pgm_sobel.bmp)

#### test4

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test4.tif_sobel.bmp)

### laplace edge detector
#### test3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test3_corrupt.pgm_laplacian.bmp)

#### test4

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test4.tif_laplacian.bmp)

### canny
#### test3

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test3_corrupt.pgm_canny.bmp)

#### test4

![](https://raw.githubusercontent.com/mutewall/homework_img/master/test4.tif_canny.bmp)

以上算法均是用python从头实现的，除了cannny以外。其中,在laplace算法中，由于其对噪声很敏感，所以先使用之前实现的高斯滤波器进行了一下低通滤波。从最后的效果来看，显然canny算法的效果是最好的，毕竟也是业界认可的最好边缘提取算法，但它有两个阈值需要人为调节，不同的值带来效果确实有不小区别。在此，我选择的是30和200。相比与laplace算法，sobel有更明显的边缘。在laplace提取的test4的结果图中，乍一看似乎没有提取出来，但向屏幕凑近一些，仔细看，还是有比较明显的边缘的。但是sobel算法带来的边缘比较粗糙，含有更多的噪声。
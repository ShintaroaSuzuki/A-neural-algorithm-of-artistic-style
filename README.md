# 概要

TensorFlow を使って、[Leon A. Gatys el at. "Image Style Transfer Using Convolutional Neural Networks". The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2414-2423.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)のアルゴリズムを実装しました。

動かす際は、[こちら](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)から、VGG19 または VGG16 のモデルをダウンロードし、models フォルダに格納してください。

style_transfer.py は VGG19 を使った通常の画風変換のアルゴリズムで、new_style_transfer.py は VGG16 を使ってクオリティを犠牲に多少高速化したアルゴリズムです。
style_transfer_square.py は出力を正方形にしただけです。

![result_cat1](https://user-images.githubusercontent.com/50007328/137363419-e527b199-74d4-4e37-a33b-044b1f21abaa.png)

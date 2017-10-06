# CaptchaPredictor
Using MLP to predicte Tsinghua course selecting system's captcha

> Author: 	Tinyalpha
>
> Email:	2366759958@qq.com

### Dependency:

* python 3.5+

* tensorflow 1.3.0+

* keras 2.0.8+

* scikit-learn 0.19.0+

* numpy 1.11.1+

* pillow 4.2.1+

### How to Use:

* Run `Predictor/predict.py` to predict a captcha
* Note that loading model and initializing predictor need a lot of time, if you want to predict more than one captchas, you'd better rewrite the main function in `predict.py`

### How to Train:

* Paste your labeled captchas to `Database\labeled` like the original
* Run `crop.py` to separate it in characters 
* Run `pro.py` to process the captchas
* Open `Train\Train.ipynb` to train the two models
* Replace `Predictor\char.h5 and length.h5` with new one

### Improve:

* Now predictor's  accuracy is about 70% on test set and 89% on train set.
* Since the Author is a bit lazy, I only have 504 labeled captchas to train the two model.
* If you have labeled more captchas, welcome to share with me by email or pull requests. ^_^
* To separate captchas into characters, I have used kmeans algorithm. But, as you see, it did't work very well. If you want to improve the accuracy, you'd better improve it first.
* And you can use CNN in place of MLP too.
* Here are two simple tool on `Tools` to get and filter captchas.
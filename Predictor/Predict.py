# coding: utf-8
# author: H.Chen (Tinyalpha)

import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from keras.models import Sequential, load_model
from keras.layers import Dense,Activation,Dropout

class Predictor():
    def __init__(self):
        self.model_l = load_model("length.h5")
        self.model_c = load_model("char.h5")
    
    def turn(self, _arr):
        return _arr.reshape(1,_arr.size)
    
    def hotone(self, _arr):
        u = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        idx = np.where(_arr==_arr.max())[0][0]
        return u[idx]
    
    def argmax(self, _mat):
        ans = np.zeros(_mat.shape[0], dtype="int")
        for i in range(_mat.shape[0]):
            ans[i] = np.where(_mat[i] == _mat[i].max())[0][0]        
        return ans
    
    def getchar(self, charimg):
        arr_ = np.array(charimg.getdata())
        arr_ = self.turn(arr_)
        p = self.hotone(self.model_c.predict(arr_)[0])
        return p
    
    def getleng(self, captcha):
        captarr = np.array(captcha.getdata())
        captarr = self.turn(captarr)
        out = self.model_l.predict(captarr)
        p = self.argmax(out)[0]+4
        return p

class Captcha():
    def __init__(self, img):
        self.n = 0
        self.subimg = None
        self.submat = None
        self.img = img.convert("L")
        self.mat = np.array(self.img.getdata(), dtype="int").reshape(self.img.size[::-1])
        self.chars = list() 

    def mat2img(self, _mat):
        # 将矩阵转换为图
        _mat *= 255
        _l = _mat.shape[0] * _mat.shape[1]
        _img = Image.new(mode = 'L', size = _mat.T.shape, color = 0)
        _img.putdata(_mat.reshape(_l))
        return _img
    
    def img2mat(self, _img):
        # 图转化为矩阵
        arr = np.array(_img.getdata())
        return arr.reshape(_img.size[::-1])

    def bin_or(self, _mat, _thr=128):
        # 二值化并取反
        for i in range(_mat.shape[0]):
            for j in range(_mat.shape[1]):
                _mat[i,j] = 0 if _mat[i,j] > _thr else 1
        return _mat
    
    def antinoise(self, _mat):
        # 去除干扰线 ;规则：纵向长度不超过3像素
        for i in range(2,_mat.shape[0]-2):
            for j in range(2,_mat.shape[1]-2):
                if _mat[i,j] == 0:
                    continue
                elif (_mat[i-2,j] + _mat[i+2,j]) == 0:
                    _mat[i-1:i+2,j] = np.zeros(3)
        # 去除噪点 ;规则：孤立
        for i in range(1,_mat.shape[0]-1):
            for j in range(1,_mat.shape[1]-1):
                if _mat[i,j] == 0:
                    continue
                elif (_mat[i,j-1]+_mat[i,j+1]+_mat[i-1,j]+_mat[i+1,j]) == 0:
                    _mat[i,j] = 0
        return _mat
    
    def proimg(self):
        # 处理图像
        self.mat = self.bin_or(self.mat)
        self.mat = self.antinoise(self.mat)
        self.img = self.mat2img(self.mat)

    def getpoints(self, _mat, yu = 0.5):
        # 返回mat中值大于yu的点的坐标列表组成的n*2矩阵
        w, h = _mat.shape
        lis = list()
        for i in range(w):
            for j in range(h):
                if _mat[i,j] > yu:
                    lis.append((i,j))
        return np.array(lis).reshape(len(lis),2)

    def separ(self, _centers, _halfwidth=12):
        # 根据传入的中心位置切割图像
        for center in _centers:
            left = center[0] - _halfwidth
            rigt = center[0] + _halfwidth
            char = self.subimg.crop((left, 0, rigt, self.subimg.size[1]))
            self.chars.append(char)
        return self.chars
    
    def getsub(self):
        # 获得subimg,submat
        bounder = (51, 12, 150, 48) if self.n==4 else (40, 12, 160, 48)
        self.subimg = self.img.crop(bounder)
        self.submat = self.img2mat(self.subimg)
            
    def cluster(self):
        if not self.subimg:
            self.getsub()
        points = self.getpoints(self.submat)
        kmeans = KMeans(n_clusters = self.n)
        kmeans.fit(points)
        labels = kmeans.predict(points)
        centers = kmeans.cluster_centers_
        # sort centers
        centers = np.array(sorted(centers[:,::-1], key=lambda x: x[0]))
        return self.separ(centers)

if __name__ == "__main__":
    print("initializing predictor ...")
    pr = Predictor()
    print("predictor ready")

    def predict(file):
        ans = ""
        img = Image.open(file)
        capt = Captcha(img)
        capt.proimg()
        capt.n = pr.getleng(capt.img)
        chars = capt.cluster()
        for each in chars:
            ans += pr.getchar(each)
        return ans
        
    def test(path, _predict):
          lis = os.listdir(path)
          correct = 0
          for each in lis:
              r = each.split(".")[0]
              p = _predict(path+"\\"+each)
              if p == r:
                  correct += 1
          print("correct rate:%.3f" %(correct/len(lis)))

    def named(path):
    	lis = os.listdir(path)
    	for each in lis:
    		p = predict(path+"\\"+each)
    		print(p)
    		os.rename(path+"\\"+each, path+"\\"+p+".jpg")
    #named(r"D:\tensorflow\验证码\test")
    
    #test(r"D:\tensorflow\验证码\Predictor\food\labeled", predict)
    while 1:
        file = input("filename:")
        print(predict(file))
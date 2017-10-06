import os
import numpy as np
import os.path as op
from PIL import Image,ImageDraw
from sklearn.cluster import KMeans


def bin_or(_mat, _thr):

    # 二值化并取反
    for i in range(_mat.shape[0]):
        for j in range(_mat.shape[1]):
            _mat[i,j] = 0 if _mat[i,j] > _thr else 1
    return _mat     


def antinoise(_mat):

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


def mat2img(_mat):

    _mat *= 255
    _l = _mat.shape[0] * _mat.shape[1]
    _img = Image.new(mode = 'L', size = _mat.T.shape, color = 0)
    _img.putdata(_mat.reshape(_l))

    return _img


def getpoints(my_mat, yu = 0.5):

    # 返回mat中大于yu的点的坐标列表组成的n*2矩阵
    w, h = my_mat.shape
    lis = list()
    for i in range(w):
        for j in range(h):
            if my_mat[i,j] > yu:
                lis.append((i,j))

    return np.array(lis).reshape(len(lis),2)


def autopro(_img):

    mat_a = np.array(_img.getdata(), dtype="int").reshape(_img.size[::-1])
    mat_a = bin_or(mat_a, 128)
    mat_a = antinoise(mat_a)

    return mat_a


def cropit(_img, _centers, _halfwidth=12):

    # 根据传入的中心位置切割图像
    char_list = list()
    for center in _centers:
        left = int(center[0]) - _halfwidth
        rigt = int(center[0]) + _halfwidth
        char = _img.crop((left, 0, rigt, _img.size[1]))
        char_list.append(char)

    return char_list


def docrop(path, to="chars"):

    j = 0
    flabels = list()
    bounders = [(51, 12, 150, 48), 
                (40, 12, 160, 48)]

    for each in os.listdir(path):
        # 预处理
        eachlabel = each.split(".")[0]
        n = len(eachlabel)
        bounder = bounders[n-4]

        # 处理图像
        _img = Image.open(path+op.sep+each).convert("L").crop(bounder)
        _mat = autopro(_img)
        newimg = mat2img(_mat)
     
        # 定centers
        points = getpoints(_mat)
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(points)
        labels = kmeans.predict(points)
        centers = kmeans.cluster_centers_
        centers = np.array(sorted(centers[:,::-1], key=lambda x: x[0]))
        
        # 切图并保存
        chars = cropit(newimg, centers)
        for i,charimg in enumerate(chars):
            charimg = charimg.convert("RGB")
            charimg.save(r"%s\%d.png" %(to,j))
            flabels.append(eachlabel[i])
            j += 1

        if j/50 == 0:
        	print(j)

    # 存储label
    with open(r"charslabel.txt","w") as file:
        file.write(str(flabels))

if __name__ == "__main__":
    docrop("labeled")
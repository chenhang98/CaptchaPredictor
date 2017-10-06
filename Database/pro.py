from PIL import Image
import numpy as np
import os

def bin_or(_mat, _thr):
    # 二值化,取反
    for i in range(_mat.shape[0]):
        for j in range(_mat.shape[1]):
            _mat[i,j] = 0 if _mat[i,j] > _thr else 1
    return _mat
            
def mat2img(_mat0):
    # 矩阵转换为图像
    _mat = _mat0.copy()
    _mat *= 255
    _l = _mat.shape[0] * _mat.shape[1]
    _img = Image.new(mode = 'L', size = _mat.T.shape, color = 0)
    _img.putdata(_mat.reshape(_l))
    return _img

def antinoise(_mat0):
    _mat = _mat0.copy()
    # 尝试去除干扰线
    for i in range(2,_mat.shape[0]-2):
        for j in range(2,_mat.shape[1]-2):
            if _mat[i,j] == 0:
                continue
            elif (_mat[i-2,j] + _mat[i+2,j]) == 0:
                _mat[i,j] = 0
                _mat[i-1,j] = 0
                _mat[i+1,j] = 0     

    # 去除噪点  
    for i in range(1,_mat.shape[0]-1):
        for j in range(1,_mat.shape[1]-1):
            if _mat[i,j] == 0:
                continue
            if (_mat[i,j-1] + _mat[i,j+1] + _mat[i-1,j] + _mat[i+1,j]) == 0:
                _mat[i,j] = 0
    return _mat

def autopro(_img):
    # 处理图像
    mat_a = np.array(_img.getdata(), dtype="int").reshape(_img.size[::-1])
    mat_a = bin_or(mat_a, 128)
    mat_a = antinoise(mat_a)
    return mat2img(mat_a)

if __name__ == "__main__":
    path = "labeled"
    for each in os.listdir(path):
        img = Image.open(path+"\\"+each).convert("L")
        img = autopro(img).convert("RGB")
        img.save("proed\\%s" %each)
import hashlib
from collections import defaultdict

n = 1000

def getmd5(filename):
	with open(filename, "rb") as file:
		fmd5 = hashlib.md5(file.read())
	return fmd5.hexdigest()

def copy(_from, _to):
	with open(_from, "rb") as file:
		with open(_to, "wb") as copyto:
			copyto.write(file.read())

uni = defaultdict(lambda: [])

for i in range(n):
	filename = r"data\%d.jpg" %i
	md5 = getmd5(filename)
	uni[md5].append(i)


sample = []
for key in uni.keys():
	if len(uni[key]) != 0:
		sample.append(uni[key][0])


for i in sample:
	copy(r"data\%d.jpg" %i, r"unidata_1\%d.jpg" %i)
	print(i)

print("all: ", len(sample))


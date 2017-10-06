import time, requests

startnum = 0
url = r'http://zhjwxk.cic.tsinghua.edu.cn/login-jcaptcah.jpg?captchaflag=login1'

# get captchas

for i in range(1000):
	if i%10 == 0:
		print(i)
	fig = requests.get(url).content
	with open(r'data\%d.jpg' %(i+startnum), 'wb') as file:
		file.write(fig)
	time.sleep(0.1)
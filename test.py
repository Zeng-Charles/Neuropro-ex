import numpy as np

import numpy as np

# 假设 txt 文件名为 'data.txt'
data = np.loadtxt('label_maintenance.txt', delimiter=',')
print(data)
print(data.shape)
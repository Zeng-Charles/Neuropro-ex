import numpy as np

data = np.array

correlationmatrix  = np.corrcoef(data,data.T)

print(correlationmatrix)
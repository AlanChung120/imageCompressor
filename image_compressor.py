import numpy as np
import matplotlib.pyplot as plt

img = 'operahall.png'
f = np.array(plt.imread(img), dtype=float)
plt.imshow(f,cmap='gray')
subblock = 32

# Compress Image Based on given Tolerance
def Compress(X, tol):
  numRows = X.shape[0] // subblock
  numCols = X.shape[1]  // subblock
  Y = np.zeros(X.shape)
  drop = 0
  nonzero = 0
  for l in range(numRows):
    for m in range(numCols):
      subX = X[l * subblock: (l + 1) * subblock, m * subblock: (m + 1) * subblock]
      subXF = np.fft.fft2(subX)
      maxF = np.amax(abs(subXF))
      for i in range(subXF.shape[0]):
        for j in range(subXF.shape[1]):
          if (subXF[i][j] != 0):
            nonzero += 1
          if (abs(subXF[i][j]) <= maxF * tol):
            if (subXF[i][j] != 0):
              drop += 1
            subXF[i][j] = 0
      compressedSubX = np.real(np.fft.ifft2(subXF))
      for p in range(l * subblock, (l + 1) * subblock):
        for q in range(m * subblock, (m + 1) * subblock):
          Y[p][q] = compressedSubX[p - l * subblock][q - m * subblock]
  return (Y, drop / nonzero)

# Compress by 50%
compressed5 = Compress(f, 0.001)
plt.imshow(compressed5[0],cmap='gray')
plt.title('Compressed Image drop ratio = 0.5 tol = 0.001')

# Compress by 70%
compressed7 = Compress(f, 0.0025)
plt.imshow(compressed7[0],cmap='gray')
plt.title('Compressed Image drop ratio = 0.7 tol = 0.0025')

# Compress by 90%
compressed9 = Compress(f, 0.0075)
plt.imshow(compressed9[0],cmap='gray')
plt.title('Compressed Image drop ratio = 0.9 tol = 0.0075')
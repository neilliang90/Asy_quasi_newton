from sklearn import datasets

def readlibsvm(name):
  x, y = datasets.load_svmlight_file(name)
  #x = normalize(x, norm='l2', axis=1)
  #print np.linalg.norm(x[1,].todense())
  return x, y



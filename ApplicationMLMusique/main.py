import ctypes

mydll = ctypes.CDLL('libmydll.dll')

print("resultat", mydll.myfunc())
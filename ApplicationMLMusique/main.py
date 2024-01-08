import array
import ctypes
import numpy as np
from PIL import Image

class MultiLayerPerceptron(ctypes.Structure):
    _fields_ = [
        ("n_entree", ctypes.c_size_t),
        ("n_sortie", ctypes.c_size_t),
        ("n_hidden", ctypes.c_size_t),
        ("n_poids", ctypes.c_size_t),
        ("poids", ctypes.POINTER(ctypes.c_double)),
        ("entrees", ctypes.POINTER(ctypes.c_double)),
        ("sorties", ctypes.POINTER(ctypes.c_double)),
    ]

# Chargement d'une image
image_path = r'C:\Users\HP\Desktop\Projets ESGI\ML_RVJV5_Dug_Zhe_You\image.png'
image = Image.open(image_path)

largeur, hauteur = image.size
pixels = list(image.getdata())
print(pixels[:10])
image.close()

# Chargement de la dll
my_dll = ctypes.CDLL(r'C:\Users\HP\Desktop\Projets ESGI\ML_RVJV5_Dug_Zhe_You\MLlib\target\debug\MLlib.dll')
create_mlp = my_dll.create_mlp
create_mlp.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
create_mlp.restype = ctypes.POINTER(MultiLayerPerceptron)

mlp_ptr = create_mlp(2, 2, 1)

mlp = mlp_ptr.contents

# Exemple du xor sous la forme (x0, x1, y)
data = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0]
float_array = np.array(data, dtype=np.float64)

my_dll.mlpLearning(mlp_ptr, float_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(float_array))
free_mlp = my_dll.free_mlp
free_mlp.argtypes = [ctypes.POINTER(MultiLayerPerceptron)]
free_mlp(mlp_ptr)

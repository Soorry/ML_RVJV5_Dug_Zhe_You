import array
import ctypes

# Définir la structure équivalente en Python
import numpy as np

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

# Charger la DLL Rust
my_dll = ctypes.CDLL(r'C:\Users\user\Desktop\ML_RVJV5_Dug_Zhe_You\MLlib\target\debug\MLlib.dll')  # Remplacez le chemin par votre DLL

# Appeler la fonction create_mlp pour initialiser le MultiLayerPerceptron
create_mlp = my_dll.create_mlp
create_mlp.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
create_mlp.restype = ctypes.POINTER(MultiLayerPerceptron)

mlp_ptr = create_mlp(2, 2, 1)

# Accéder à la structure MultiLayerPerceptron en Python
mlp = mlp_ptr.contents

# Créer un tableau de f64 en Python
data = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0]
float_array = np.array(data, dtype=np.float64)

# Appeler la fonction Rust avec les paramètres convertis
# my_dll.ask_lin_mod(mlp_ptr, float_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(float_array))

my_dll.mlpLearning(mlp_ptr, float_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(float_array))
# Libérer la mémoire lorsque vous avez fini d'utiliser le MultiLayerPerceptron
free_mlp = my_dll.free_mlp
free_mlp.argtypes = [ctypes.POINTER(MultiLayerPerceptron)]
free_mlp(mlp_ptr)

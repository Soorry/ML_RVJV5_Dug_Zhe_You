import array
import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Chargement d'une image
# image_path = r'C:\Users\HP\Desktop\Projets ESGI\ML_RVJV5_Dug_Zhe_You\image.png'
# image = Image.open(image_path)

# largeur, hauteur = image.size
# pixels = list(image.getdata())
# print(pixels[:10])
# image.close()


# Chargement de la dll
my_dll = ctypes.CDLL(r'C:\Users\HP\Desktop\Projets ESGI\ML_RVJV5_Dug_Zhe_You\MLlib\target\release\MLlib.dll')

isLm = False

if isLm:
    # config lm
    my_dll.create_lm.restype = ctypes.POINTER(ctypes.c_void_p)
    my_dll.train_lm.restype = ctypes.POINTER(ctypes.c_double)
    my_dll.predict_lm.restype = ctypes.c_double

    model = my_dll.create_lm()

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    nb_errors = 1000
    errs = my_dll.train_lm(model, ctypes.c_double(0.001), 100000, nb_errors, train_input_data_pointer, len(X), len(X[0]),
                        train_output_data_pointer, len(Y))
    array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_errors)).contents, dtype=np.float64)
    plt.plot(range(0, nb_errors), array_from_ptr)
    plt.show()

    outputs = []
    for x, y in X:
        data = np.array([x, y], dtype=np.float64)
        predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result = my_dll.predict_lm(model, predict_data_pointer, len(data))
        outputs.append(result)

    inputs = np.array(X)
    outputs = np.array(outputs)

    plt.scatter(inputs[outputs >= 0][:, 0], inputs[outputs >= 0][:, 1], color='red')
    plt.scatter(inputs[outputs < 0][:, 0], inputs[outputs < 0][:, 1], color='blue')
    plt.show()
    plt.clf()
else:
    # config mlp
    my_dll.create_mlp.restype = ctypes.POINTER(ctypes.c_void_p)
    my_dll.train_mlp.restype = ctypes.POINTER(ctypes.c_double)
    my_dll.predict_mlp.restype = ctypes.POINTER(ctypes.c_double)

    model = my_dll.create_mlp(5, 6, 1)

    #XOR
    #model = my_dll.create_mlp(2, 2, 1)
    #xor_input = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    #xor_output = np.array([[-1.0], [1.0], [1.0], [-1.0]], dtype=np.float64)

    #Cross
    #model = my_dll.create_mlp(3, 4, 1) #alpha 0.001, iter 10000000
    #X = np.random.random((400, 2)) * 2.0 - 1.0
    #Y = np.array([[1.0] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1.0] for p in X], dtype=np.float64)

    #MultiLinear
    #model = my_dll.create_mlp(2, 2, 1) #alpha 0.01, iter 50000
    #X = np.random.random((500, 2)) * 2.0 - 1.0
    #Y = np.array([[1.0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  #[0.0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  #[-1.0] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  #[2] for p in X])
    #X = X[[not np.all(arr == [2]) for arr in Y]]
    #Y = Y[[not np.all(arr == [2]) for arr in Y]]

    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[-1.0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0.0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [1.0] for p in X])

    train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    nb_err = 1000
    errs = my_dll.train_mlp(model, ctypes.c_double(0.001), 50000, nb_err, train_input_data_pointer, len(X), len(X[0]), train_output_data_pointer, len(Y), len(Y[0]))

    array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_err)).contents, dtype=np.float64)

    plt.plot(range(0, nb_err), array_from_ptr)
    plt.show()

    # Créer un tableau d'entrées
    inputs = []
    outputs = []
    """
    for x in np.arange(-1, 1, 0.1):
        for y in np.arange(-1, 1, 0.1):
            data = np.array([x, y], dtype=np.float64)
            predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            result = my_dll.predict_mlp(model, predict_data_pointer, len(data))
            inputs.append([x, y])
            outputs.append(result[0])
    # Convertir les tableaux en numpy arrays pour une utilisation facile avec Matplotlib
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    """
    """
    for x, y in X:
        data = np.array([x, y], dtype=np.float64)
        predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result = my_dll.predict_mlp(model, predict_data_pointer, len(data))
        outputs.append(result[0])
    """
    inputs = np.array(X)
    outputs = np.array(Y)

    plt.scatter(inputs[outputs >= 0.5][:, 0], inputs[outputs >= 0.5][:, 1], color='red')
    plt.scatter(inputs[outputs < -0.5][:, 0], inputs[outputs < -0.5][:, 1], color='blue')
    plt.scatter(inputs[(outputs > -0.5) & (outputs < 0.5)][:, 0], inputs[(outputs > -0.5) & (outputs < 0.5)][:, 1], color='green')
    plt.show()
    plt.clf()
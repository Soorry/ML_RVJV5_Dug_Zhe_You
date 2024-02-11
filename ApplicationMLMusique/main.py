import array
import ctypes
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Chargement de la dll
my_dll = ctypes.CDLL(r'C:\Users\HP\Desktop\Projets ESGI\ML_RVJV5_Dug_Zhe_You\MLlib\target\release\MLlib.dll')

#Modèle disponible
# 0 : Modèle Linéaire, 1 : MLP, 2 : RBF

#Exemples disponibles Modèle Linéaire :
# 0 : Cas simple, 1 : Cas multiple, 2 : Régression simple

#Exemples disponibles MLP :
# 0 : XOR, 1 : Cross, 2 : Linéaire multiple, 3: Multicross

modele_utilise = 3

exemple_utilise = 2

if modele_utilise == 0:
    # config lm
    my_dll.create_lm.restype = ctypes.POINTER(ctypes.c_void_p)
    my_dll.train_lm.restype = ctypes.POINTER(ctypes.c_double)
    my_dll.predict_lm.restype = ctypes.c_double

    if exemple_utilise == 0:
        #Cas linéaire simple
        model = my_dll.create_lm(2)

        X = np.array([
            [1.0, 1.0],
            [2.0, 3.0],
            [3.0, 3.0]
        ])
        Y = np.array([
            1.0,
            -1.0,
            -1.0
        ])
        train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        nb_errors = 1000
        errs = my_dll.train_lm(model, ctypes.c_double(0.001), 50000, nb_errors, train_input_data_pointer, len(X),
                               len(X[0]), train_output_data_pointer, len(Y))
        array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_errors)).contents,
                                       dtype=np.float64)
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

        backX = []
        backY = []
        for x in np.arange(1, 3, 0.01):
            for y in np.arange(1, 3, 0.01):
                data = np.array([x, y], dtype=np.float64)
                predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                result = my_dll.predict_lm(model, predict_data_pointer, len(data))
                backX.append([x, y])
                backY.append(result)

        backX = np.array(backX)
        backY = np.array(backY)

        plt.scatter(backX[backY >= 0][:, 0], backX[backY >= 0][:, 1], color='lightcoral')
        plt.scatter(backX[backY < 0][:, 0], backX[backY < 0][:, 1], color='lightblue')
        plt.scatter(inputs[outputs >= 0][:, 0], inputs[outputs >= 0][:, 1], color='red')
        plt.scatter(inputs[outputs < 0][:, 0], inputs[outputs < 0][:, 1], color='blue')
        plt.show()
        plt.clf()

    elif exemple_utilise == 1:
        model = my_dll.create_lm(2)
        X = np.concatenate(
            [np.random.random((100, 2)) * 0.9 + np.array([1, 1]), np.random.random((100, 2)) * 0.9 + np.array([2, 2])])
        Y = np.concatenate([np.ones((100, 1)), np.ones((100, 1)) * -1.0])

        train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        nb_errors = 1000
        errs = my_dll.train_lm(model, ctypes.c_double(0.001), 100000, nb_errors, train_input_data_pointer, len(X),
                               len(X[0]), train_output_data_pointer, len(Y))
        array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_errors)).contents,
                                       dtype=np.float64)
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

        backX = []
        backY = []
        for x in np.arange(1, 3, 0.01):
            for y in np.arange(1, 3, 0.01):
                data = np.array([x, y], dtype=np.float64)
                predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                result = my_dll.predict_lm(model, predict_data_pointer, len(data))
                backX.append([x, y])
                backY.append(result)

        backX = np.array(backX)
        backY = np.array(backY)

        plt.scatter(backX[backY >= 0][:, 0], backX[backY >= 0][:, 1], color='lightcoral')
        plt.scatter(backX[backY < 0][:, 0], backX[backY < 0][:, 1], color='lightblue')
        plt.scatter(inputs[outputs >= 0][:, 0], inputs[outputs >= 0][:, 1], color='red')
        plt.scatter(inputs[outputs < 0][:, 0], inputs[outputs < 0][:, 1], color='blue')
        plt.show()
        plt.clf()

    elif exemple_utilise > 1:
        X = []
        Y = []
        if exemple_utilise == 2:
            model = my_dll.create_lm(2)
            X = np.array([
                [1.0],
                [2.0]
            ])
            Y = np.array([
                2.0,
                3.0
            ])
        elif exemple_utilise == 3:
            model = my_dll.create_lm(3)
            X = np.array([
                [1.0],
                [2.0],
                [3.0]
            ])
            Y = np.array([
                2.0,
                3.0,
                2.5
            ])

        train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        nb_errors = 1000
        errs = my_dll.train_lm(model, ctypes.c_double(0.001), 100000, nb_errors, train_input_data_pointer, len(X),
                               len(X[0]), train_output_data_pointer, len(Y))
        array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_errors)).contents,
                                       dtype=np.float64)
        plt.plot(range(0, nb_errors), array_from_ptr)
        plt.show()

        inputs = np.array(X)
        outputs = np.array(Y)

        backX = []
        backY = []
        for x in np.arange(0, 3, 0.01):
            data = np.array([x], dtype=np.float64)
            predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            result = my_dll.predict_lm(model, predict_data_pointer, len(data))
            backX.append(x)
            backY.append(result)

        backX = np.array(backX)
        backY = np.array(backY)

        plt.scatter(backX, backY, color='black', s=1)
        plt.scatter(inputs, outputs, color='blue')

        plt.show()
        plt.clf()

elif modele_utilise == 1:
    # config mlp
    my_dll.create_mlp.restype = ctypes.POINTER(ctypes.c_void_p)
    my_dll.train_mlp.restype = ctypes.POINTER(ctypes.c_double)
    my_dll.predict_mlp.restype = ctypes.POINTER(ctypes.c_double)

    if exemple_utilise == 0:  # XOR
        # Creation du modèle
        model = my_dll.create_mlp(2, 2, 1)
        # Creation du dataset d'entrainement
        xor_input = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        xor_output = np.array([[-1.0], [1.0], [1.0], [-1.0]], dtype=np.float64)
        train_input_data_pointer = xor_input.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        train_output_data_pointer = xor_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # entrainement du modèle
        nb_err = 1000
        alpha = 0.01
        iter = 10000
        errs = my_dll.train_mlp(model, ctypes.c_double(alpha), iter, nb_err, train_input_data_pointer,
                                len(xor_input), len(xor_input[0]), train_output_data_pointer, len(xor_output),
                                len(xor_output[0]))

        # Affichage des erreurs
        array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_err)).contents,
                                       dtype=np.float64)

        plt.plot(range(0, nb_err), array_from_ptr)
        plt.show()

        # Affichage des resultats
        outputs = []
        inputs = np.array(xor_input)
        for y in xor_output:
            outputs.append(y[0])
        outputs = np.array(outputs)

        backX = []
        backY = []
        for x in np.arange(0, 1, 0.01):
            for y in np.arange(0, 1, 0.01):
                data = np.array([x, y], dtype=np.float64)
                predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                result = my_dll.predict_mlp(model, predict_data_pointer, len(data))
                backX.append([x, y])
                backY.append(result[0])

        backX = np.array(backX)
        backY = np.array(backY)

        plt.scatter(backX[backY >= 0][:, 0], backX[backY >= 0][:, 1], color='lightcoral')
        plt.scatter(backX[backY < 0][:, 0], backX[backY < 0][:, 1], color='lightblue')
        plt.scatter(inputs[outputs >= 0][:, 0], inputs[outputs >= 0][:, 1], color='red')
        plt.scatter(inputs[outputs < 0][:, 0], inputs[outputs < 0][:, 1], color='blue')
        plt.show()
        plt.clf()

    elif exemple_utilise == 1:  # Cross
        # Creation du dataset d'entrainement
        X = np.array(np.random.random((250, 2)) * 2.0 - 1.0)
        Y = np.array([[1.0] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1.0] for p in X], dtype=np.float64)

        train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # entrainement du modèle
        nb_err = 100
        alpha = 0.001
        iter = 2500000

        model = my_dll.create_mlp(4, 2, 1)
        errs = my_dll.train_mlp(model, ctypes.c_double(alpha), iter, nb_err, train_input_data_pointer, len(X),
                                len(X[0]),
                                train_output_data_pointer, len(Y), len(Y[0]))

        array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_err)).contents,
                                       dtype=np.float64)

        plt.plot(range(0, nb_err), array_from_ptr)
        plt.show()

        # Affichage des resultats
        inputs = []
        outputs = []

        inputs = np.array(X)
        for y in Y:
            outputs.append(y[0])
        outputs = np.array(outputs)

        backX = []
        backY = []
        for x in np.arange(-1, 1, 0.01):
            for y in np.arange(-1, 1, 0.01):
                data = np.array([x, y], dtype=np.float64)
                predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                result = my_dll.predict_mlp(model, predict_data_pointer, len(data))
                backX.append([x, y])
                backY.append(result[0])
        backX = np.array(backX)
        backY = np.array(backY)

        plt.scatter(backX[backY >= 0.0][:, 0], backX[backY >= 0.0][:, 1], color='lightcoral')
        plt.scatter(backX[backY < 0.0][:, 0], backX[backY < 0.0][:, 1], color='lightblue')
        plt.scatter(inputs[outputs >= 0.0][:, 0], inputs[outputs >= 0.0][:, 1], color='red')
        plt.scatter(inputs[outputs < 0.0][:, 0], inputs[outputs < 0.0][:, 1], color='blue')
        plt.show()
        plt.clf()

    elif exemple_utilise == 2:
        # MultiLinear
        model = my_dll.create_mlp(2, 2, 1)
        X = np.random.random((500, 2)) * 2.0 - 1.0
        Y = np.array([[1.0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                      [0.0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                      [-1.0] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                      [2] for p in X])
        X = X[[not np.all(arr == [2]) for arr in Y]]
        Y = Y[[not np.all(arr == [2]) for arr in Y]]

        train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        nb_err = 1000
        alpha = 0.001
        iter = 1000000
        errs = my_dll.train_mlp(model, ctypes.c_double(alpha), iter, nb_err, train_input_data_pointer, len(X),
                                len(X[0]), train_output_data_pointer, len(Y), len(Y[0]))

        # Créer un tableau d'entrées
        inputs = []
        outputs = []
        for y in Y:
            outputs.append(y[0])
        outputs = np.array(outputs)
        inputs = np.array(X)

        backX = []
        backY = []
        for x in np.arange(-1, 1, 0.01):
            for y in np.arange(-1, 1, 0.01):
                data = np.array([x, y], dtype=np.float64)
                predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                result = my_dll.predict_mlp(model, predict_data_pointer, len(data))
                backX.append([x, y])
                backY.append(result[0])
        # Convertir les tableaux en numpy arrays pour une utilisation facile avec Matplotlib
        backX = np.array(backX)
        backY = np.array(backY)

        #filtered_indices = np.where()

        plt.scatter(backX[backY >= 0.5][:, 0], backX[backY >= 0.5][:, 1], color='lightcoral')
        plt.scatter(backX[backY < -0.5][:, 0], backX[backY < -0.5][:, 1], color='lightblue')
        plt.scatter(backX[(backY * backY) < 0.5][:, 0], backX[(backY * backY) < 0.5][:, 1], color='lime')

        plt.scatter(inputs[outputs >= 0.5][:, 0], inputs[outputs >= 0.5][:, 1], color='red')
        plt.scatter(inputs[outputs < -0.5][:, 0], inputs[outputs < -0.5][:, 1], color='blue')
        plt.scatter(inputs[(outputs * outputs) < 0.5][:, 0], inputs[(outputs * outputs) < 0.5][:, 1], color='green')
        plt.show()
        plt.clf()

        array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_err)).contents,
                                       dtype=np.float64)

        plt.plot(range(0, nb_err), array_from_ptr)
        plt.show()

    elif exemple_utilise == 3:
        # MultiLinear
        model = my_dll.create_mlp(6, 3, 1)
        X = np.random.random((300, 2)) * 2.0 - 1.0
        Y = np.array([[-1.0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0.0] if abs(
                p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [1.0] for p in X])

        train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        nb_err = 1000
        alpha = 0.01
        iter = 10000000
        errs = my_dll.train_mlp(model, ctypes.c_double(alpha), iter, nb_err, train_input_data_pointer, len(X),
                                len(X[0]), train_output_data_pointer, len(Y), len(Y[0]))

        # Créer un tableau d'entrées
        inputs = []
        outputs = []
        for y in Y:
            outputs.append(y[0])
        outputs = np.array(outputs)
        inputs = np.array(X)

        backX = []
        backY = []
        for x in np.arange(-1, 1, 0.01):
            for y in np.arange(-1, 1, 0.01):
                data = np.array([x, y], dtype=np.float64)
                predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                result = my_dll.predict_mlp(model, predict_data_pointer, len(data))
                backX.append([x, y])
                backY.append(result[0])
        # Convertir les tableaux en numpy arrays pour une utilisation facile avec Matplotlib
        backX = np.array(backX)
        backY = np.array(backY)

        plt.scatter(backX[backY >= 0.5][:, 0], backX[backY >= 0.5][:, 1], color='lightcoral')
        plt.scatter(backX[backY < -0.5][:, 0], backX[backY < -0.5][:, 1], color='lightblue')
        plt.scatter(backX[(backY * backY) < 0.5][:, 0], backX[(backY * backY) < 0.5][:, 1], color='lime')

        plt.scatter(inputs[outputs >= 0.5][:, 0], inputs[outputs >= 0.5][:, 1], color='red')
        plt.scatter(inputs[outputs < -0.5][:, 0], inputs[outputs < -0.5][:, 1], color='blue')
        plt.scatter(inputs[(outputs * outputs) < 0.5][:, 0], inputs[(outputs * outputs) < 0.5][:, 1], color='green')
        plt.show()
        plt.clf()

        array_from_ptr = np.frombuffer(ctypes.cast(errs, ctypes.POINTER(ctypes.c_double * nb_err)).contents,
                                       dtype=np.float64)

        plt.plot(range(0, nb_err), array_from_ptr)
        plt.show()
else:
    #RBF
    my_dll.create_rbf.restype = ctypes.POINTER(ctypes.c_void_p)
    my_dll.predict_rbf.restype = ctypes.c_double

    X = np.random.random((200, 2)) * 2.0 - 1.0
    Y = np.array([-1.0 if p[0] > random.random()/10.0 else 1.0 for p in X])
    train_input_data_pointer = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    train_output_data_pointer = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    model = my_dll.create_rbf(200, ctypes.c_double(100), train_input_data_pointer, len(X),
                           len(X[0]), train_output_data_pointer, len(Y))

    backX = []
    backY = []
    for x in np.arange(-1, 1, 0.01):
        for y in np.arange(-1, 1, 0.01):
            data = np.array([x, y], dtype=np.float64)
            predict_data_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            result = my_dll.predict_rbf(model, predict_data_pointer, 2)
            backX.append([x, y])
            backY.append(result)
    # Convertir les tableaux en numpy arrays pour une utilisation facile avec Matplotlib
    backX = np.array(backX)
    backY = np.array(backY)

    plt.scatter(backX[backY >= 0.0][:, 0], backX[backY >= 0.0][:, 1], color='lightcoral')
    plt.scatter(backX[backY < 0.0][:, 0], backX[backY < 0.0][:, 1], color='lightblue')
    plt.scatter(X[Y >= 0.0][:, 0], X[Y >= 0.0][:, 1], color='red')
    plt.scatter(X[Y < 0.0][:, 0], X[Y < 0.0][:, 1], color='blue')
    plt.show()
    plt.clf()
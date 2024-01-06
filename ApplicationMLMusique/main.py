import ctypes

# Utilisation de r devant le chemin pour éviter l'échappement
mydll = ctypes.CDLL(r'C:\Users\zheng\OneDrive\Documents\GitHub\ML_RVJV5_Dug_Zhe_You\MLlib\target\debug\MLlib.dll')

# Appel de la fonction main si elle ne prend pas d'arguments
result = mydll.main(2,2,1)

print("Resultat :", result)

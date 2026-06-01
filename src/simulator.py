import numpy as np

# 1. Simulación de trayectorias mediante el método de Euler
def simulate_path(ti, tf, dt, x0, a, b, c, d):
    # Cantidad n de pasos de tiempo en la partición
    steps_number = int((tf - ti) / dt) 
    
    # Inicialización del vector de la EDE
    X = np.zeros(steps_number + 1)
    X[0] = x0 
    
    # Cálculo iterativo de cada punto
    for i in range(1, steps_number + 1):
        epsilon = np.random.normal(0, 1) # Variable normal estándar N(0,1)
        dB = np.sqrt(dt) * epsilon       # Incremento Browniano Delta B_i
        
        # Método de Euler para la EDE Autónoma
        X[i] = X[i-1] + ((a * X[i-1] + b) * dt) + ((c * X[i-1] + d) * dB)
        
    return X

# 2. Cálculo de la media exacta en cada instante de tiempo
def determine_mean(vector_tiempos, x0, a, b):
    mean = np.zeros(len(vector_tiempos))
    
    for i, t in enumerate(vector_tiempos):
        if a != 0:
            mean[i] = (x0 * np.exp(a * t)) + ((b / a) * (np.exp(a * t) - 1))
        else:
            mean[i] = x0 + b * t
            
    return mean


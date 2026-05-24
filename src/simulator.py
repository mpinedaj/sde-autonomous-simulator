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

# 3. Cálculo de la varianza exacta en cada instante de tiempo
def determine_variance(vector_tiempos, lista_mean, x0, a, b, c, d):
    variance = np.zeros(len(vector_tiempos))
    
    alpha = 2.0 * a + c ** 2
    beta = 2.0 * b + 2.0 * c * d
    
    for i, t in enumerate(vector_tiempos):
        # Para evitar divisiones por cero o comportamientos inestables si a = 0
        if abs(a) > 1e-12:
            P = x0 + b / a
            A = beta * P
            C_cte = d ** 2 - beta * (b / a)
            diff = a - alpha
            
            # Integral 1 (I1)
            if abs(diff) > 1e-12:
                I1 = (np.exp(diff * t) - 1.0) / diff
            else:
                I1 = t
                
            # Integral 2 (I2)
            if abs(alpha) > 1e-12:
                I2 = (1.0 - np.exp(-alpha * t)) / alpha
            else:
                I2 = t
                
            # Segundo momento E[X_t^2]
            M2 = np.exp(alpha * t) * (x0 ** 2 + A * I1 + C_cte * I2)
            
        else: # Caso especial cuando la deriva lineal es cero (a = 0)
            R = beta * x0 + d ** 2
            S = beta * b
            if abs(alpha) > 1e-12:
                e_neg = np.exp(-alpha * t)
                J1 = R * (1.0 - e_neg) / alpha
                J2 = S * (-t / alpha * e_neg + (1.0 - e_neg) / alpha ** 2)
                M2 = np.exp(alpha * t) * (x0 ** 2 + J1 + J2)
            else:
                M2 = x0 ** 2 + R * t + S * t ** 2 / 2.0
                
        # Varianza = E[X_t^2] - (E[X_t])^2
        var = M2 - lista_mean[i] ** 2
        # Evitar varianzas negativas por redondeo numérico flotante
        variance[i] = max(var, 0.0)
        
    return variance
"""
Implementación de Red Neuronal de Función de Base Radial (RBF)
Siguiendo la especificación del examen
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

class RBFNeuralNetwork:
    def __init__(self, num_centros, error_optimo=0.1):
        """
        Inicializa la red RBF
        
        Args:
            num_centros: Número de centros radiales (neuronas ocultas)
            error_optimo: Error de aproximación óptimo para convergencia
        """
        self.num_centros = num_centros
        self.error_optimo = error_optimo
        self.centros = None
        self.pesos = None
        self.phi_train = None
        self.historia_entrenamiento = {}
        
    def funcion_activacion(self, distancia):
        """
        Función de activación radial: FA(d) = d^2 * ln(d)
        Maneja el caso especial cuando d ≈ 0
        
        Args:
            distancia: Distancia euclidiana
            
        Returns:
            Valor de activación
        """
        # Evitar log(0) agregando un epsilon pequeño
        epsilon = 1e-10
        d = np.where(distancia < epsilon, epsilon, distancia)
        return (d ** 2) * np.log(d)
    
    def calcular_distancias(self, X, centros):
        """
        Calcula distancias euclidianas entre patrones y centros
        
        Args:
            X: Matriz de patrones (n_patrones, n_caracteristicas)
            centros: Matriz de centros (n_centros, n_caracteristicas)
            
        Returns:
            Matriz de distancias (n_patrones, n_centros)
        """
        n_patrones = X.shape[0]
        n_centros = centros.shape[0]
        distancias = np.zeros((n_patrones, n_centros))
        
        for i in range(n_patrones):
            for j in range(n_centros):
                # Distancia euclidiana: sqrt(sum((X_p - R_j)^2))
                distancias[i, j] = np.sqrt(np.sum((X[i] - centros[j]) ** 2))
        
        return distancias
    
    def calcular_activaciones(self, distancias):
        """
        Calcula las activaciones usando la función radial
        
        Args:
            distancias: Matriz de distancias
            
        Returns:
            Matriz Φ (Phi) de activaciones
        """
        return self.funcion_activacion(distancias)
    
    def construir_matriz_interpolacion(self, phi):
        """
        Construye la matriz de interpolación A = [1 | Φ]
        Agrega columna de unos (umbrales)
        
        Args:
            phi: Matriz de activaciones
            
        Returns:
            Matriz A de interpolación
        """
        unos = np.ones((phi.shape[0], 1))
        A = np.hstack([unos, phi])
        return A
    
    def entrenar(self, X_train, y_train):
        """
        Entrena la red RBF usando el método de mínimos cuadrados
        
        Args:
            X_train: Datos de entrenamiento (n_patrones, n_caracteristicas)
            y_train: Etiquetas de entrenamiento (n_patrones, n_salidas)
            
        Returns:
            dict con información del entrenamiento
        """
        print("=" * 60)
        print("INICIANDO ENTRENAMIENTO DE RED RBF")
        print("=" * 60)
        
        # Paso 1: Inicializar centros radiales aleatoriamente
        print("\n[Paso 1] Inicializando centros radiales...")
        n_caracteristicas = X_train.shape[0]
        
        # Seleccionar centros aleatorios del conjunto de entrenamiento
        indices_centros = np.random.choice(n_caracteristicas, 
                                          self.num_centros, 
                                          replace=False)
        self.centros = X_train[indices_centros].copy()
        
        print(f"  ✓ {self.num_centros} centros inicializados")
        print(f"  ✓ Forma de centros: {self.centros.shape}")
        
        # Paso 2: Calcular distancias
        print("\n[Paso 2] Calculando distancias entre patrones y centros...")
        distancias = self.calcular_distancias(X_train, self.centros)
        print(f"  ✓ Matriz de distancias: {distancias.shape}")
        print(f"  ✓ Rango de distancias: [{distancias.min():.4f}, {distancias.max():.4f}]")
        
        # Paso 3: Calcular activaciones (Φ)
        print("\n[Paso 3] Aplicando función de activación radial...")
        self.phi_train = self.calcular_activaciones(distancias)
        print(f"  ✓ Matriz Φ (Phi): {self.phi_train.shape}")
        print(f"  ✓ Rango de activaciones: [{self.phi_train.min():.4f}, {self.phi_train.max():.4f}]")
        
        # Paso 4: Construir matriz de interpolación
        print("\n[Paso 4] Construyendo matriz de interpolación A = [1 | Φ]...")
        A = self.construir_matriz_interpolacion(self.phi_train)
        print(f"  ✓ Matriz A: {A.shape}")
        
        # Paso 5: Calcular pesos usando mínimos cuadrados
        # W = (A^T * A)^-1 * A^T * y
        print("\n[Paso 5] Calculando pesos mediante mínimos cuadrados...")
        print("  Fórmula: W = (A^T * A)^-1 * A^T * y")
        
        try:
            # Calcular (A^T * A)
            ATA = np.dot(A.T, A)
            print(f"  ✓ A^T * A calculado: {ATA.shape}")
            
            # Invertir (A^T * A)
            ATA_inv = np.linalg.inv(ATA)
            print(f"  ✓ (A^T * A)^-1 calculado")
            
            # Calcular W
            self.pesos = np.dot(np.dot(ATA_inv, A.T), y_train)
            print(f"  ✓ Pesos W calculados: {self.pesos.shape}")
            print(f"  ✓ W0 (umbral): {self.pesos[0]}")
            print(f"  ✓ W1...Wn (pesos): {self.pesos[1:5]}..." if len(self.pesos) > 5 else f"  ✓ W1...Wn: {self.pesos[1:]}")
            
        except np.linalg.LinAlgError:
            print("  ✗ Error: Matriz singular. Usando pseudoinversa...")
            self.pesos = np.dot(np.linalg.pinv(A), y_train)
        
        # Paso 6: Predicción y cálculo de métricas
        print("\n[Paso 6] Evaluando modelo en conjunto de entrenamiento...")
        y_pred = self.predecir(X_train)
        metricas = self.calcular_metricas(y_train, y_pred)
        
        print(f"\n  MÉTRICAS DE ENTRENAMIENTO:")
        print(f"  ├─ Error General (EG): {metricas['EG']:.6f}")
        print(f"  ├─ MAE: {metricas['MAE']:.6f}")
        print(f"  ├─ RMSE: {metricas['RMSE']:.6f}")
        print(f"  └─ Convergencia: {'✓ SÍ' if metricas['Converge'] else '✗ NO'}")
        
        if metricas['Converge']:
            print(f"\n  ¡Éxito! EG ({metricas['EG']:.6f}) ≤ Error Óptimo ({self.error_optimo})")
        else:
            porcentaje = (metricas['EG'] / self.error_optimo) * 100
            print(f"\n  ⚠️ No converge: EG ({metricas['EG']:.6f}) > Error Óptimo ({self.error_optimo})")
            print(f"  📊 Alcanzado: {porcentaje:.1f}% del objetivo")
            
            if porcentaje < 150:
                print(f"\n  💡 SUGERENCIA: Está muy cerca!")
                print(f"     → Aumentar centros a {self.num_centros + 3}")
            else:
                centros_sugeridos = min(self.num_centros + 5, 30)
                error_sugerido = round(metricas['EG'] * 1.1, 3)
                print(f"\n  💡 SUGERENCIAS:")
                print(f"     Opción 1: Aumentar centros a {centros_sugeridos}")
                print(f"     Opción 2: Ajustar error óptimo a {error_sugerido}")
                print(f"     Opción 3: Usar 'Configuración Automática' en la app")
        
        # Guardar historia
        self.historia_entrenamiento = {
            'num_patrones': X_train.shape[0],
            'num_caracteristicas': X_train.shape[1],
            'num_centros': self.num_centros,
            'metricas': metricas
        }
        
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 60 + "\n")
        
        return metricas
    
    def predecir(self, X):
        """
        Realiza predicciones con la red entrenada
        
        Args:
            X: Datos de entrada (n_patrones, n_caracteristicas)
            
        Returns:
            Predicciones (n_patrones, n_salidas)
        """
        if self.centros is None or self.pesos is None:
            raise ValueError("La red debe ser entrenada primero")
        
        # Calcular distancias y activaciones
        distancias = self.calcular_distancias(X, self.centros)
        phi = self.calcular_activaciones(distancias)
        
        # Construir matriz A
        A = self.construir_matriz_interpolacion(phi)
        
        # Calcular predicciones: y_pred = A * W
        y_pred = np.dot(A, self.pesos)
        
        return y_pred
    
    def calcular_metricas(self, y_real, y_pred):
        """
        Calcula las métricas de evaluación: EG, MAE, RMSE
        
        Args:
            y_real: Valores reales
            y_pred: Valores predichos
            
        Returns:
            dict con métricas
        """
        N = len(y_real)
        
        # Error General (EG)
        eg = np.sum(np.abs(y_real - y_pred)) / N
        
        # Error Absoluto Medio (MAE)
        mae = np.mean(np.abs(y_real - y_pred))
        
        # Raíz del Error Cuadrático Medio (RMSE)
        rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
        
        # Verificar convergencia
        converge = eg <= self.error_optimo
        
        metricas = {
            'EG': float(eg),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'Converge': converge
        }
        
        return metricas
    
    def evaluar(self, X_test, y_test):
        """
        Evalúa el modelo en el conjunto de prueba
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas reales
            
        Returns:
            dict con métricas
        """
        print("\n" + "=" * 60)
        print("EVALUANDO EN CONJUNTO DE PRUEBA")
        print("=" * 60)
        
        y_pred = self.predecir(X_test)
        metricas = self.calcular_metricas(y_test, y_pred)
        
        print(f"\n  MÉTRICAS DE PRUEBA:")
        print(f"  ├─ Error General (EG): {metricas['EG']:.6f}")
        print(f"  ├─ MAE: {metricas['MAE']:.6f}")
        print(f"  └─ RMSE: {metricas['RMSE']:.6f}")
        
        print("\n" + "=" * 60 + "\n")
        
        return metricas
    
    def generar_graficos(self, X_train, y_train, X_test, y_test, 
                        metricas_train, metricas_test, ruta_salida='resultados/graficos'):
        """
        Genera visualizaciones del entrenamiento
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            metricas_train, metricas_test: Métricas calculadas
            ruta_salida: Directorio donde guardar los gráficos
        """
        os.makedirs(ruta_salida, exist_ok=True)
        
        # Predicciones
        y_pred_train = self.predecir(X_train)
        y_pred_test = self.predecir(X_test)
        
        # Gráfico 1: Yd vs Yr (Entrenamiento y Prueba)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(y_train, 'b-', label='Yd (Deseado)', linewidth=2)
        plt.plot(y_pred_train, 'r--', label='Yr (Obtenido)', linewidth=2)
        plt.xlabel('Patrón')
        plt.ylabel('Salida')
        plt.title('Entrenamiento: Salida Deseada vs Obtenida')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(y_test, 'b-', label='Yd (Deseado)', linewidth=2)
        plt.plot(y_pred_test, 'r--', label='Yr (Obtenido)', linewidth=2)
        plt.xlabel('Patrón')
        plt.ylabel('Salida')
        plt.title('Prueba: Salida Deseada vs Obtenida')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Comparación de métricas
        plt.subplot(1, 3, 3)
        conjuntos = ['Entrenamiento', 'Prueba']
        eg_values = [metricas_train['EG'], metricas_test['EG']]
        mae_values = [metricas_train['MAE'], metricas_test['MAE']]
        rmse_values = [metricas_train['RMSE'], metricas_test['RMSE']]
        
        x = np.arange(len(conjuntos))
        width = 0.25
        
        plt.bar(x - width, eg_values, width, label='EG', color='#ff6b6b')
        plt.bar(x, mae_values, width, label='MAE', color='#4ecdc4')
        plt.bar(x + width, rmse_values, width, label='RMSE', color='#45b7d1')
        
        plt.axhline(y=self.error_optimo, color='green', linestyle='--', 
                   label=f'Error Óptimo ({self.error_optimo})')
        
        plt.xlabel('Conjunto')
        plt.ylabel('Error')
        plt.title('Comparación de Métricas')
        plt.xticks(x, conjuntos)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{ruta_salida}/metricas_comparacion.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico 3: Dispersión de predicciones
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_pred_train, alpha=0.6, s=50)
        plt.plot([y_train.min(), y_train.max()], 
                [y_train.min(), y_train.max()], 
                'r--', linewidth=2, label='Ideal')
        plt.xlabel('Yd (Valor Real)')
        plt.ylabel('Yr (Predicción)')
        plt.title(f'Entrenamiento: Dispersión\nR²: {1 - metricas_train["RMSE"]**2 / np.var(y_train):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_test, alpha=0.6, s=50, color='orange')
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Ideal')
        plt.xlabel('Yd (Valor Real)')
        plt.ylabel('Yr (Predicción)')
        plt.title(f'Prueba: Dispersión\nR²: {1 - metricas_test["RMSE"]**2 / np.var(y_test):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{ruta_salida}/dispersion_predicciones.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráficos guardados en: {ruta_salida}/")
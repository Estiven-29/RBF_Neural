"""
Manejador de Datasets para Red Neuronal RBF
Carga, preprocesa y divide los datos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class DataHandler:
    def __init__(self):
        """Inicializa el manejador de datos"""
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.estadisticas = {}
        self.dataset_info = {}
        self.es_clasificacion = False
        
    def cargar_dataset(self, ruta_archivo):
        """
        Carga un dataset desde archivo CSV o JSON
        
        Args:
            ruta_archivo: Ruta al archivo del dataset
            
        Returns:
            dict con información del dataset
        """
        extension = os.path.splitext(ruta_archivo)[1].lower()
        
        try:
            if extension == '.csv':
                self.df = pd.read_csv(ruta_archivo)
            elif extension == '.json':
                self.df = pd.read_json(ruta_archivo)
            else:
                raise ValueError(f"Formato de archivo no soportado: {extension}")
            
            # Información básica del dataset
            self.dataset_info = {
                'nombre': os.path.basename(ruta_archivo),
                'num_patrones': len(self.df),
                'num_columnas': len(self.df.columns),
                'columnas': list(self.df.columns)
            }
            
            return self.dataset_info
            
        except Exception as e:
            raise Exception(f"Error al cargar dataset: {str(e)}")
    
    def verificar_dataset(self):
        """
        Verifica y muestra información del dataset cargado
        
        Returns:
            dict con estadísticas del dataset
        """
        if self.df is None:
            raise ValueError("No hay dataset cargado")
        
        info = {
            'patrones_totales': len(self.df),
            'columnas': list(self.df.columns),
            'tipos_datos': self.df.dtypes.to_dict(),
            'valores_faltantes': self.df.isnull().sum().to_dict(),
            'estadisticas_numericas': self.df.describe().to_dict()
        }
        
        return info
    
    def preprocesar_datos(self, columna_salida, normalizar=True):
        """
        Preprocesa los datos: identifica X e y, maneja valores faltantes, normaliza
        
        Args:
            columna_salida: Nombre de la columna objetivo/salida
            normalizar: Si normalizar las variables de entrada
            
        Returns:
            dict con información del preprocesamiento
        """
        if self.df is None:
            raise ValueError("No hay dataset cargado")
        
        if columna_salida not in self.df.columns:
            raise ValueError(f"La columna '{columna_salida}' no existe en el dataset")
        
        # Separar características (X) y objetivo (y)
        X = self.df.drop(columns=[columna_salida])
        y = self.df[columna_salida]
        
        # Detectar si es clasificación (salida categórica)
        self.es_clasificacion = y.dtype == 'object' or y.dtype.name == 'category'
        
        # Manejar variables categóricas en X
        columnas_categoricas = X.select_dtypes(include=['object']).columns
        if len(columnas_categoricas) > 0:
            # One-hot encoding para variables categóricas
            X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=False)
        
        # Codificar salida si es clasificación
        if self.es_clasificacion:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            self.dataset_info['clases'] = list(self.label_encoder.classes_)
        
        # Convertir a numpy arrays
        self.X = X.values.astype(float)
        self.y = y.values if isinstance(y, np.ndarray) else y.to_numpy()
        
        # Si es regresión, asegurar que y sea 2D
        if not self.es_clasificacion and self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        
        # Manejar valores faltantes (rellenar con la media)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)
        
        # Normalizar/Estandarizar variables de entrada
        if normalizar:
            self.X = self.scaler.fit_transform(self.X)
        
        # Calcular estadísticas
        self.estadisticas = {
            'num_entradas': self.X.shape[1],
            'num_salidas': self.y.shape[1] if self.y.ndim > 1 else 1,
            'es_clasificacion': self.es_clasificacion,
            'rango_X': {
                'min': float(self.X.min()),
                'max': float(self.X.max()),
                'media': float(self.X.mean()),
                'std': float(self.X.std())
            }
        }
        
        if self.es_clasificacion:
            self.estadisticas['num_clases'] = len(np.unique(self.y))
            self.estadisticas['distribucion_clases'] = {
                str(clase): int(count) 
                for clase, count in zip(*np.unique(self.y, return_counts=True))
            }
        else:
            self.estadisticas['rango_y'] = {
                'min': float(self.y.min()),
                'max': float(self.y.max()),
                'media': float(self.y.mean()),
                'std': float(self.y.std())
            }
        
        self.dataset_info['num_entradas'] = self.estadisticas['num_entradas']
        self.dataset_info['num_salidas'] = self.estadisticas['num_salidas']
        
        return self.estadisticas
    
    def dividir_datos(self, porcentaje_entrenamiento=0.7, semilla=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba
        
        Args:
            porcentaje_entrenamiento: Porcentaje de datos para entrenamiento (0-1)
            semilla: Semilla para reproducibilidad
            
        Returns:
            dict con información de la división
        """
        if self.X is None or self.y is None:
            raise ValueError("Debe preprocesar los datos primero")
        
        # Si es clasificación, usar y como está; si no, convertir a 1D para split
        y_for_split = self.y if self.es_clasificacion else self.y.ravel()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y,
            train_size=porcentaje_entrenamiento,
            random_state=semilla,
            stratify=y_for_split if self.es_clasificacion else None
        )
        
        info_division = {
            'total_patrones': len(self.X),
            'patrones_entrenamiento': len(self.X_train),
            'patrones_prueba': len(self.X_test),
            'porcentaje_entrenamiento': porcentaje_entrenamiento,
            'porcentaje_prueba': 1 - porcentaje_entrenamiento
        }
        
        return info_division
    
    def get_datos_entrenamiento(self):
        """Retorna los datos de entrenamiento"""
        if self.X_train is None:
            raise ValueError("Debe dividir los datos primero")
        return self.X_train, self.y_train
    
    def get_datos_prueba(self):
        """Retorna los datos de prueba"""
        if self.X_test is None:
            raise ValueError("Debe dividir los datos primero")
        return self.X_test, self.y_test
    
    def get_scaler(self):
        """Retorna el scaler entrenado"""
        return self.scaler
    
    def get_label_encoder(self):
        """Retorna el label encoder (si es clasificación)"""
        return self.label_encoder
    
    def get_estadisticas(self):
        """Retorna las estadísticas del dataset"""
        return self.estadisticas
    
    def get_dataset_info(self):
        """Retorna información del dataset"""
        return self.dataset_info
"""
Gestor de Persistencia para Red Neuronal RBF
Almacena entrenamientos, configuraciones y resultados en SQLite
"""

import sqlite3
import json
import pickle
import os
from datetime import datetime
from pathlib import Path

class StorageManager:
    def __init__(self, db_path='database/rbf_trainings.db'):
        """Inicializa el gestor de persistencia"""
        self.db_path = db_path
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Crear base de datos y tablas
        self._init_database()
    
    def _init_database(self):
        """Crea las tablas necesarias en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla principal de entrenamientos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entrenamientos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                dataset_nombre TEXT NOT NULL,
                fecha_creacion TEXT NOT NULL,
                num_patrones INTEGER,
                num_entradas INTEGER,
                num_salidas INTEGER,
                num_centros INTEGER,
                porcentaje_entrenamiento REAL,
                funcion_activacion TEXT,
                error_optimo REAL,
                descripcion TEXT
            )
        ''')
        
        # Tabla de configuración del modelo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS configuracion_modelo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entrenamiento_id INTEGER,
                centros_radiales BLOB,
                pesos BLOB,
                scaler_params BLOB,
                label_encoder BLOB,
                FOREIGN KEY (entrenamiento_id) REFERENCES entrenamientos(id)
            )
        ''')
        
        # Tabla de métricas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metricas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entrenamiento_id INTEGER,
                conjunto TEXT,
                eg REAL,
                mae REAL,
                rmse REAL,
                converge INTEGER,
                FOREIGN KEY (entrenamiento_id) REFERENCES entrenamientos(id)
            )
        ''')
        
        # Tabla de estadísticas del dataset
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS estadisticas_dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entrenamiento_id INTEGER,
                estadisticas_json TEXT,
                FOREIGN KEY (entrenamiento_id) REFERENCES entrenamientos(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def guardar_entrenamiento(self, nombre, dataset_info, config, modelo_data, 
                             metricas_train, metricas_test, estadisticas, descripcion=""):
        """
        Guarda un entrenamiento completo en la base de datos
        
        Args:
            nombre: Nombre del entrenamiento
            dataset_info: Información del dataset (dict)
            config: Configuración del modelo (dict)
            modelo_data: Datos del modelo entrenado (dict con centros, pesos, scaler, encoder)
            metricas_train: Métricas del conjunto de entrenamiento
            metricas_test: Métricas del conjunto de prueba
            estadisticas: Estadísticas del dataset
            descripcion: Descripción opcional
        
        Returns:
            id del entrenamiento guardado
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insertar información principal
            cursor.execute('''
                INSERT INTO entrenamientos 
                (nombre, dataset_nombre, fecha_creacion, num_patrones, num_entradas, 
                 num_salidas, num_centros, porcentaje_entrenamiento, funcion_activacion, 
                 error_optimo, descripcion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                nombre,
                dataset_info['nombre'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                dataset_info['num_patrones'],
                dataset_info['num_entradas'],
                dataset_info['num_salidas'],
                config['num_centros'],
                config['porcentaje_entrenamiento'],
                config['funcion_activacion'],
                config['error_optimo'],
                descripcion
            ))
            
            entrenamiento_id = cursor.lastrowid
            
            # Guardar configuración del modelo (serializado con pickle)
            cursor.execute('''
                INSERT INTO configuracion_modelo 
                (entrenamiento_id, centros_radiales, pesos, scaler_params, label_encoder)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                entrenamiento_id,
                pickle.dumps(modelo_data['centros']),
                pickle.dumps(modelo_data['pesos']),
                pickle.dumps(modelo_data['scaler']),
                pickle.dumps(modelo_data.get('label_encoder'))
            ))
            
            # Guardar métricas de entrenamiento
            cursor.execute('''
                INSERT INTO metricas (entrenamiento_id, conjunto, eg, mae, rmse, converge)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entrenamiento_id,
                'Entrenamiento',
                metricas_train['EG'],
                metricas_train['MAE'],
                metricas_train['RMSE'],
                1 if metricas_train['Converge'] else 0
            ))
            
            # Guardar métricas de prueba
            cursor.execute('''
                INSERT INTO metricas (entrenamiento_id, conjunto, eg, mae, rmse, converge)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entrenamiento_id,
                'Prueba',
                metricas_test['EG'],
                metricas_test['MAE'],
                metricas_test['RMSE'],
                0  # No aplica convergencia en prueba
            ))
            
            # Guardar estadísticas
            cursor.execute('''
                INSERT INTO estadisticas_dataset (entrenamiento_id, estadisticas_json)
                VALUES (?, ?)
            ''', (
                entrenamiento_id,
                json.dumps(estadisticas)
            ))
            
            conn.commit()
            return entrenamiento_id
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error al guardar entrenamiento: {str(e)}")
        finally:
            conn.close()
    
    def cargar_entrenamiento(self, entrenamiento_id):
        """
        Carga un entrenamiento completo desde la base de datos
        
        Returns:
            dict con toda la información del entrenamiento
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Cargar información principal
            cursor.execute('SELECT * FROM entrenamientos WHERE id = ?', (entrenamiento_id,))
            entrenamiento = cursor.fetchone()
            
            if not entrenamiento:
                raise ValueError(f"No se encontró el entrenamiento con ID {entrenamiento_id}")
            
            # Cargar configuración del modelo
            cursor.execute('SELECT * FROM configuracion_modelo WHERE entrenamiento_id = ?', 
                          (entrenamiento_id,))
            config = cursor.fetchone()
            
            # Cargar métricas
            cursor.execute('SELECT * FROM metricas WHERE entrenamiento_id = ?', 
                          (entrenamiento_id,))
            metricas = cursor.fetchall()
            
            # Cargar estadísticas
            cursor.execute('SELECT estadisticas_json FROM estadisticas_dataset WHERE entrenamiento_id = ?', 
                          (entrenamiento_id,))
            estadisticas = cursor.fetchone()
            
            # Construir objeto de respuesta
            resultado = {
                'info': {
                    'id': entrenamiento[0],
                    'nombre': entrenamiento[1],
                    'dataset_nombre': entrenamiento[2],
                    'fecha_creacion': entrenamiento[3],
                    'num_patrones': entrenamiento[4],
                    'num_entradas': entrenamiento[5],
                    'num_salidas': entrenamiento[6],
                    'num_centros': entrenamiento[7],
                    'porcentaje_entrenamiento': entrenamiento[8],
                    'funcion_activacion': entrenamiento[9],
                    'error_optimo': entrenamiento[10],
                    'descripcion': entrenamiento[11]
                },
                'modelo': {
                    'centros': pickle.loads(config[2]),
                    'pesos': pickle.loads(config[3]),
                    'scaler': pickle.loads(config[4]),
                    'label_encoder': pickle.loads(config[5])
                },
                'metricas': {
                    'entrenamiento': {
                        'EG': metricas[0][3],
                        'MAE': metricas[0][4],
                        'RMSE': metricas[0][5],
                        'Converge': bool(metricas[0][6])
                    },
                    'prueba': {
                        'EG': metricas[1][3],
                        'MAE': metricas[1][4],
                        'RMSE': metricas[1][5]
                    }
                },
                'estadisticas': json.loads(estadisticas[0]) if estadisticas else {}
            }
            
            return resultado
            
        finally:
            conn.close()
    
    def listar_entrenamientos(self):
        """
        Lista todos los entrenamientos guardados
        
        Returns:
            Lista de diccionarios con información de cada entrenamiento
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, nombre, dataset_nombre, fecha_creacion, num_centros, 
                   porcentaje_entrenamiento 
            FROM entrenamientos 
            ORDER BY fecha_creacion DESC
        ''')
        
        entrenamientos = []
        for row in cursor.fetchall():
            entrenamientos.append({
                'id': row[0],
                'nombre': row[1],
                'dataset': row[2],
                'fecha': row[3],
                'num_centros': row[4],
                'split': f"{row[5]*100:.0f}%"
            })
        
        conn.close()
        return entrenamientos
    
    def eliminar_entrenamiento(self, entrenamiento_id):
        """Elimina un entrenamiento y todos sus datos asociados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM estadisticas_dataset WHERE entrenamiento_id = ?', 
                          (entrenamiento_id,))
            cursor.execute('DELETE FROM metricas WHERE entrenamiento_id = ?', 
                          (entrenamiento_id,))
            cursor.execute('DELETE FROM configuracion_modelo WHERE entrenamiento_id = ?', 
                          (entrenamiento_id,))
            cursor.execute('DELETE FROM entrenamientos WHERE id = ?', 
                          (entrenamiento_id,))
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error al eliminar entrenamiento: {str(e)}")
        finally:
            conn.close()
    
    def exportar_modelo(self, entrenamiento_id, ruta_destino):
        """
        Exporta un modelo entrenado a un archivo pickle
        
        Args:
            entrenamiento_id: ID del entrenamiento
            ruta_destino: Ruta donde guardar el archivo
        """
        entrenamiento = self.cargar_entrenamiento(entrenamiento_id)
        
        modelo_export = {
            'info': entrenamiento['info'],
            'modelo': entrenamiento['modelo'],
            'metricas': entrenamiento['metricas']
        }
        
        with open(ruta_destino, 'wb') as f:
            pickle.dump(modelo_export, f)
        
        return True
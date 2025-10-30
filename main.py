"""
Aplicación GUI para Red Neuronal RBF
Interfaz completa con Tkinter
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from datetime import datetime

# Importar módulos propios
from data_handler import DataHandler
from rbf_model import RBFNeuralNetwork
from storage_manager import StorageManager

class RBFApp:
    def __init__(self, root):
        """Inicializa la aplicación"""
        self.root = root
        self.root.title("Red Neuronal RBF - Sistema Completo")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Componentes
        self.data_handler = DataHandler()
        self.rbf_model = None
        self.storage = StorageManager()
        
        # Variables
        self.dataset_cargado = False
        self.modelo_entrenado = False
        self.modelo_cargado_id = None
        self.modelo_cargado = None
        self.ruta_dataset = tk.StringVar()
        self.columna_salida = tk.StringVar()
        self.num_centros = tk.IntVar(value=5)
        self.error_optimo = tk.DoubleVar(value=0.1)
        self.porcentaje_train = tk.DoubleVar(value=70.0)
        
        # Crear interfaz
        self.crear_interfaz()
        
        # Redireccionar print a consola
        import sys
        sys.stdout = TextRedirector(self.console_text)
        
        # Actualizar combo de modelos al iniciar
        self.root.after(100, self.actualizar_combo_modelos)
        
    def crear_interfaz(self):
        """Crea todos los componentes de la interfaz"""
        
        # Frame principal con división
        main_container = ttk.PanedWindow(self.root, orient='horizontal')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Panel izquierdo: Notebook (pestañas)
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill='both', expand=True)
        
        # Pestaña 1: Carga y Preprocesamiento
        self.tab_datos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_datos, text=" Datos")
        self.crear_tab_datos()
        
        # Pestaña 2: Configuración y Entrenamiento
        self.tab_entrenamiento = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text=" Entrenamiento")
        self.crear_tab_entrenamiento()
        
        # Pestaña 3: Evaluación y Visualización
        self.tab_evaluacion = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_evaluacion, text=" Evaluación")
        self.crear_tab_evaluacion()
        
        # Pestaña 4: Modelos Guardados
        self.tab_modelos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_modelos, text=" Modelos")
        self.crear_tab_modelos()
        
        # Pestaña 5: Usar Modelo (Predicciones)
        self.tab_prediccion = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_prediccion, text=" Predicción")
        self.crear_tab_prediccion()
        
        # Panel derecho: Consola de salida
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        self.crear_consola(right_panel)
        
    def crear_tab_datos(self):
        """Crea la pestaña de carga y preprocesamiento de datos"""
        
        # Frame principal con scrollbar
        canvas = tk.Canvas(self.tab_datos)
        scrollbar = ttk.Scrollbar(self.tab_datos, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # PASO 1: Carga de Dataset
        paso1 = ttk.LabelFrame(scrollable_frame, text="Paso 1: Cargar Dataset", padding="10")
        paso1.pack(fill='x', pady=(10, 10), padx=10)
        
        ttk.Label(paso1, text="Archivo:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(paso1, textvariable=self.ruta_dataset, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(paso1, text="Examinar...", command=self.cargar_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        # Información del dataset
        self.info_dataset = scrolledtext.ScrolledText(paso1, height=6, width=70, state='disabled')
        self.info_dataset.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        
        # PASO 2: Configuración de preprocesamiento
        paso2 = ttk.LabelFrame(scrollable_frame, text="Paso 2: Configurar Preprocesamiento", padding="10")
        paso2.pack(fill='x', pady=(0, 10), padx=10)
        
        ttk.Label(paso2, text="Columna de Salida:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.combo_columnas = ttk.Combobox(paso2, textvariable=self.columna_salida, width=30, state='readonly')
        self.combo_columnas.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Checkbutton(paso2, text="Normalizar entradas", variable=tk.BooleanVar(value=True)).grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Button(paso2, text=" Preprocesar Datos", command=self.preprocesar_datos, 
                  style='Accent.TButton', width=25).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Estadísticas
        self.estadisticas_text = scrolledtext.ScrolledText(paso2, height=6, width=70, state='disabled')
        self.estadisticas_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # PASO 3: División de datos
        paso3 = ttk.LabelFrame(scrollable_frame, text="Paso 3: Dividir Datos", padding="10")
        paso3.pack(fill='x', pady=(0, 10), padx=10)
        
        ttk.Label(paso3, text="% Entrenamiento:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Scale(paso3, from_=50, to=90, variable=self.porcentaje_train, orient='horizontal', length=250).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(paso3, textvariable=self.porcentaje_train, width=5).grid(row=0, column=2, padx=5, pady=5)
        
        # Botón de dividir más visible
        btn_dividir = ttk.Button(paso3, text=" DIVIDIR DATASET", command=self.dividir_datos, 
                                style='Accent.TButton', width=30)
        btn_dividir.grid(row=1, column=0, columnspan=3, pady=15)
        
        self.division_info = tk.Label(paso3, text="", font=('Arial', 10), fg='green', wraplength=600, justify='left')
        self.division_info.grid(row=2, column=0, columnspan=3, pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def crear_tab_entrenamiento(self):
        """Crea la pestaña de configuración y entrenamiento"""
        
        main_frame = ttk.Frame(self.tab_entrenamiento, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Configuración RBF
        config_frame = ttk.LabelFrame(main_frame, text="Configuración de la Red RBF", padding="10")
        config_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(config_frame, text="Número de Centros Radiales:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=2, to=50, textvariable=self.num_centros, width=10).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(config_frame, text="(Más centros = menor error)", font=('Arial', 8, 'italic'), foreground='#666').grid(row=0, column=2, sticky='w', padx=5)
        
        ttk.Label(config_frame, text="Error de Aproximación Óptimo:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.error_optimo, width=10).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(config_frame, text="(Clasif: 0.3-0.5, Regr: 0.1-0.3)", font=('Arial', 8, 'italic'), foreground='#666').grid(row=1, column=2, sticky='w', padx=5)
        
        ttk.Label(config_frame, text="Función de Activación:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Label(config_frame, text="FA(d) = d² × ln(d)", font=('Courier', 10, 'bold')).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Botón de configuración automática
        btn_auto = ttk.Button(config_frame, text=" Configuración Automática", 
                             command=self.configuracion_automatica)
        btn_auto.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Botón de entrenamiento
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        self.btn_entrenar = ttk.Button(btn_frame, text=" INICIAR ENTRENAMIENTO", 
                                       command=self.iniciar_entrenamiento,
                                       style='Accent.TButton', width=30)
        self.btn_entrenar.pack(side='left', padx=5)
        
        ttk.Button(btn_frame, text=" Limpiar Todo", 
                  command=self.limpiar_todo,
                  width=20).pack(side='left', padx=5)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress.pack(pady=10)
        
        # Resultados del entrenamiento
        resultados_frame = ttk.LabelFrame(main_frame, text="Resultados del Entrenamiento", padding="10")
        resultados_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        self.resultados_text = scrolledtext.ScrolledText(resultados_frame, height=15, width=80, state='disabled')
        self.resultados_text.pack(fill='both', expand=True, padx=5, pady=5)
        
    def crear_tab_evaluacion(self):
        """Crea la pestaña de evaluación y visualización"""
        
        # Frame principal con scrollbar
        canvas = tk.Canvas(self.tab_evaluacion)
        scrollbar = ttk.Scrollbar(self.tab_evaluacion, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Métricas
        metricas_frame = ttk.LabelFrame(main_frame, text=" Métricas del Modelo", padding="10")
        metricas_frame.pack(fill='x', pady=(0, 10))
        
        # Tabla de métricas
        columns = ('Conjunto', 'EG', 'MAE', 'RMSE', 'Convergencia')
        self.tree_metricas = ttk.Treeview(metricas_frame, columns=columns, show='headings', height=3)
        
        for col in columns:
            self.tree_metricas.heading(col, text=col)
            self.tree_metricas.column(col, width=120, anchor='center')
        
        self.tree_metricas.pack(fill='x', padx=5, pady=5)
        
        # Visualizaciones
        visual_frame = ttk.LabelFrame(main_frame, text=" Visualizaciones", padding="10")
        visual_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Botones de control
        btn_frame = ttk.Frame(visual_frame)
        btn_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(btn_frame, text=" Generar Gráficos", 
                  command=self.generar_y_mostrar_graficos).pack(side='left', padx=5)
        ttk.Button(btn_frame, text=" Abrir Carpeta", 
                  command=self.abrir_resultados).pack(side='left', padx=5)
        
        # Frame para mostrar imágenes
        self.imagenes_frame = ttk.Frame(visual_frame)
        self.imagenes_frame.pack(fill='both', expand=True)
        
        # Guardar modelo
        guardar_frame = ttk.LabelFrame(main_frame, text=" Guardar Modelo Entrenado", padding="10")
        guardar_frame.pack(fill='x', pady=(10, 0))
        
        info_label = ttk.Label(guardar_frame, text=" Los modelos se guardan en: database/rbf_trainings.db", 
                              font=('Arial', 9, 'italic'), foreground='#666')
        info_label.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky='w')
        
        self.nombre_modelo = tk.StringVar(value="Modelo_RBF_1")
        ttk.Label(guardar_frame, text="Nombre del modelo:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(guardar_frame, textvariable=self.nombre_modelo, width=30).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(guardar_frame, text=" Guardar Modelo", 
                  command=self.guardar_modelo, style='Accent.TButton').grid(row=1, column=2, padx=5, pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def crear_tab_modelos(self):
        """Crea la pestaña de modelos guardados"""
        
        main_frame = ttk.Frame(self.tab_modelos, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Información de ubicación
        info_frame = ttk.LabelFrame(main_frame, text="📍 Ubicación de Almacenamiento", padding="10")
        info_frame.pack(fill='x', pady=(0, 10))
        
        db_path = os.path.abspath('database/rbf_trainings.db')
        ttk.Label(info_frame, text=f"Base de datos:", font=('Arial', 9, 'bold')).pack(anchor='w')
        
        path_frame = ttk.Frame(info_frame)
        path_frame.pack(fill='x', pady=5)
        
        path_entry = ttk.Entry(path_frame, width=70)
        path_entry.insert(0, db_path)
        path_entry.config(state='readonly')
        path_entry.pack(side='left', padx=(0, 5))
        
        ttk.Button(path_frame, text=" Abrir", command=lambda: self.abrir_carpeta('database')).pack(side='left')
        
        ttk.Label(info_frame, text="Los gráficos se guardan en: resultados/graficos/", 
                 font=('Arial', 9, 'italic'), foreground='#666').pack(anchor='w', pady=(5, 0))
        
        # Controles
        controles_frame = ttk.Frame(main_frame)
        controles_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(controles_frame, text=" Actualizar Lista", 
                  command=self.actualizar_lista_modelos).pack(side='left', padx=5)
        ttk.Button(controles_frame, text=" Cargar Modelo", 
                  command=self.cargar_modelo_seleccionado).pack(side='left', padx=5)
        ttk.Button(controles_frame, text=" Eliminar Modelo", 
                  command=self.eliminar_modelo_seleccionado).pack(side='left', padx=5)
        ttk.Button(controles_frame, text=" Exportar", 
                  command=self.exportar_modelo_seleccionado).pack(side='left', padx=5)
        
        # Lista de modelos
        list_frame = ttk.LabelFrame(main_frame, text="Modelos Entrenados", padding="5")
        list_frame.pack(fill='both', expand=True)
        
        columns = ('ID', 'Nombre', 'Dataset', 'Fecha', 'Centros', 'Split')
        self.tree_modelos = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.tree_modelos.heading(col, text=col)
            width = 50 if col == 'ID' else 150
            self.tree_modelos.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree_modelos.yview)
        self.tree_modelos.configure(yscrollcommand=scrollbar.set)
        
        self.tree_modelos.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Cargar lista inicial
        self.actualizar_lista_modelos()
    
    def crear_tab_prediccion(self):
        """Crea la pestaña para usar modelos entrenados"""
        
        # Frame principal con scrollbar
        canvas = tk.Canvas(self.tab_prediccion)
        scrollbar = ttk.Scrollbar(self.tab_prediccion, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # PASO 1: Cargar modelo
        paso1 = ttk.LabelFrame(main_frame, text="Paso 1: Cargar Modelo Entrenado", padding="10")
        paso1.pack(fill='x', pady=(0, 10))
        
        # Lista de modelos disponibles
        ttk.Label(paso1, text="Seleccione un modelo:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 5))
        
        # Combo con modelos
        self.combo_modelos_pred = ttk.Combobox(paso1, state='readonly', width=50)
        self.combo_modelos_pred.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        
        btn_frame_1 = ttk.Frame(paso1)
        btn_frame_1.grid(row=1, column=1, padx=5)
        
        ttk.Button(btn_frame_1, text=" Actualizar", 
                  command=self.actualizar_combo_modelos).pack(side='left', padx=2)
        ttk.Button(btn_frame_1, text=" Cargar", 
                  command=self.cargar_modelo_para_prediccion,
                  style='Accent.TButton').pack(side='left', padx=2)
        
        # Información del modelo cargado
        self.info_modelo_cargado = scrolledtext.ScrolledText(paso1, height=8, width=80, state='disabled')
        self.info_modelo_cargado.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        # PASO 2: Ingresar datos para predicción
        paso2 = ttk.LabelFrame(main_frame, text="Paso 2: Ingresar Datos para Predicción", padding="10")
        paso2.pack(fill='x', pady=(0, 10))
        
        # Opciones de entrada
        ttk.Label(paso2, text="Método de entrada:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 10))
        
        self.metodo_entrada = tk.StringVar(value="manual")
        ttk.Radiobutton(paso2, text="Entrada Manual", variable=self.metodo_entrada, 
                       value="manual", command=self.cambiar_metodo_entrada).grid(row=1, column=0, sticky='w', padx=20)
        ttk.Radiobutton(paso2, text="Desde Archivo CSV", variable=self.metodo_entrada, 
                       value="archivo", command=self.cambiar_metodo_entrada).grid(row=2, column=0, sticky='w', padx=20)
        
        # Frame para entrada manual
        self.frame_manual = ttk.Frame(paso2)
        self.frame_manual.grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        
        self.entrada_manual_widgets = []
        
        # Frame para archivo
        self.frame_archivo = ttk.Frame(paso2)
        self.frame_archivo.grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)
        
        self.archivo_prediccion = tk.StringVar()
        ttk.Label(self.frame_archivo, text="Archivo:").pack(side='left', padx=5)
        ttk.Entry(self.frame_archivo, textvariable=self.archivo_prediccion, width=50).pack(side='left', padx=5)
        ttk.Button(self.frame_archivo, text="Examinar...", 
                  command=self.cargar_archivo_prediccion).pack(side='left', padx=5)
        
        self.frame_archivo.pack_forget()  # Ocultar inicialmente
        
        # Botón de predicción
        btn_predecir = ttk.Button(paso2, text=" REALIZAR PREDICCIÓN", 
                                 command=self.realizar_prediccion,
                                 style='Accent.TButton', width=30)
        btn_predecir.grid(row=5, column=0, columnspan=2, pady=20)
        
        # PASO 3: Resultados
        paso3 = ttk.LabelFrame(main_frame, text="Paso 3: Resultados de la Predicción", padding="10")
        paso3.pack(fill='both', expand=True, pady=(0, 10))
        
        # Frame de resultados
        self.resultados_prediccion = scrolledtext.ScrolledText(paso3, height=12, width=80, state='disabled')
        self.resultados_prediccion.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Botones de exportación
        export_frame = ttk.Frame(paso3)
        export_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(export_frame, text=" Exportar Resultados", 
                  command=self.exportar_resultados_prediccion).pack(side='left', padx=5)
        ttk.Button(export_frame, text=" Limpiar", 
                  command=lambda: self.mostrar_en_text(self.resultados_prediccion, "")).pack(side='left', padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def actualizar_combo_modelos(self):
        """Actualiza el combo de modelos disponibles"""
        try:
            modelos = self.storage.listar_entrenamientos()
            valores = [f"ID {m['id']}: {m['nombre']} ({m['dataset']})" for m in modelos]
            self.combo_modelos_pred['values'] = valores
            
            if valores:
                self.combo_modelos_pred.current(0)
                print(f" {len(modelos)} modelos disponibles")
            else:
                print(" No hay modelos guardados")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos:\n{str(e)}")
    
    def cargar_modelo_para_prediccion(self):
        """Carga un modelo seleccionado para hacer predicciones"""
        seleccion = self.combo_modelos_pred.get()
        
        if not seleccion:
            messagebox.showwarning("Advertencia", "Debe seleccionar un modelo")
            return
        
        try:
            # Extraer ID del modelo
            modelo_id = int(seleccion.split(":")[0].replace("ID ", ""))
            
            print(f"\n{'='*60}")
            print(f"CARGANDO MODELO ID: {modelo_id}")
            print(f"{'='*60}")
            
            # Cargar datos del modelo
            datos_modelo = self.storage.cargar_entrenamiento(modelo_id)
            
            # Recrear el modelo RBF
            self.modelo_cargado = RBFNeuralNetwork(
                num_centros=datos_modelo['info']['num_centros'],
                error_optimo=datos_modelo['info']['error_optimo']
            )
            
            # Restaurar centros y pesos
            self.modelo_cargado.centros = datos_modelo['modelo']['centros']
            self.modelo_cargado.pesos = datos_modelo['modelo']['pesos']
            
            # Guardar información adicional
            self.modelo_cargado_id = modelo_id
            self.modelo_cargado_data = datos_modelo
            
            # Mostrar información del modelo
            info_text = f"""
╔═══════════════════════════════════════════════════════════════╗
║                   MODELO CARGADO EXITOSAMENTE                 ║
╚═══════════════════════════════════════════════════════════════╝

 INFORMACIÓN GENERAL:
  • ID: {datos_modelo['info']['id']}
  • Nombre: {datos_modelo['info']['nombre']}
  • Dataset: {datos_modelo['info']['dataset_nombre']}
  • Fecha: {datos_modelo['info']['fecha_creacion']}

 CONFIGURACIÓN:
  • Centros Radiales: {datos_modelo['info']['num_centros']}
  • Entradas: {datos_modelo['info']['num_entradas']}
  • Salidas: {datos_modelo['info']['num_salidas']}
  • Error Óptimo: {datos_modelo['info']['error_optimo']}
  • Split: {datos_modelo['info']['porcentaje_entrenamiento']*100:.0f}% / {(1-datos_modelo['info']['porcentaje_entrenamiento'])*100:.0f}%

 MÉTRICAS DE DESEMPEÑO:
  
  ENTRENAMIENTO:
    • EG:   {datos_modelo['metricas']['entrenamiento']['EG']:.6f}
    • MAE:  {datos_modelo['metricas']['entrenamiento']['MAE']:.6f}
    • RMSE: {datos_modelo['metricas']['entrenamiento']['RMSE']:.6f}
    • Converge: {'✓ SÍ' if datos_modelo['metricas']['entrenamiento']['Converge'] else '✗ NO'}
  
  PRUEBA:
    • EG:   {datos_modelo['metricas']['prueba']['EG']:.6f}
    • MAE:  {datos_modelo['metricas']['prueba']['MAE']:.6f}
    • RMSE: {datos_modelo['metricas']['prueba']['RMSE']:.6f}

 El modelo está listo para realizar predicciones.
   Complete los datos de entrada en el Paso 2.
"""
            
            self.mostrar_en_text(self.info_modelo_cargado, info_text)
            
            # Crear campos de entrada manual basados en el número de entradas
            self.crear_campos_entrada_manual(datos_modelo['info']['num_entradas'])
            
            print(f" Modelo cargado correctamente")
            print(f" Entradas esperadas: {datos_modelo['info']['num_entradas']}")
            print(f"{'='*60}\n")
            
            messagebox.showinfo("Éxito", f"Modelo '{datos_modelo['info']['nombre']}' cargado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelo:\n{str(e)}")
            print(f"✗ Error: {str(e)}")
    
    def crear_campos_entrada_manual(self, num_entradas):
        """Crea campos de entrada manual basados en el número de entradas del modelo"""
        # Limpiar widgets anteriores
        for widget in self.frame_manual.winfo_children():
            widget.destroy()
        
        self.entrada_manual_widgets = []
        
        ttk.Label(self.frame_manual, text=f"Ingrese {num_entradas} valores (separados por coma o uno por línea):", 
                 font=('Arial', 9, 'bold')).pack(anchor='w', pady=(0, 5))
        
        # Opción 1: Entrada en una línea
        frame_linea = ttk.Frame(self.frame_manual)
        frame_linea.pack(fill='x', pady=5)
        
        ttk.Label(frame_linea, text="Valores:").pack(side='left', padx=5)
        self.entrada_linea = ttk.Entry(frame_linea, width=60)
        self.entrada_linea.pack(side='left', padx=5)
        ttk.Label(frame_linea, text="(ej: 1.2, 3.4, 5.6)", font=('Arial', 8, 'italic')).pack(side='left')
        
        # Opción 2: Entradas individuales
        ttk.Label(self.frame_manual, text="O ingrese valores individuales:", 
                 font=('Arial', 9)).pack(anchor='w', pady=(10, 5))
        
        frame_grid = ttk.Frame(self.frame_manual)
        frame_grid.pack(fill='both', expand=True)
        
        # Crear campos individuales
        cols = 3  # Número de columnas
        for i in range(num_entradas):
            row = i // cols
            col = i % cols
            
            frame_campo = ttk.Frame(frame_grid)
            frame_campo.grid(row=row, column=col, padx=5, pady=3, sticky='ew')
            
            ttk.Label(frame_campo, text=f"X{i+1}:", width=5).pack(side='left')
            entry = ttk.Entry(frame_campo, width=15)
            entry.pack(side='left', padx=5)
            
            self.entrada_manual_widgets.append(entry)
        
        # Configurar peso de columnas
        for i in range(cols):
            frame_grid.columnconfigure(i, weight=1)
    
    def cambiar_metodo_entrada(self):
        """Cambia entre método manual y archivo"""
        if self.metodo_entrada.get() == "manual":
            self.frame_archivo.pack_forget()
            self.frame_manual.pack(fill='both', expand=True)
        else:
            self.frame_manual.pack_forget()
            self.frame_archivo.pack(fill='x')
    
    def cargar_archivo_prediccion(self):
        """Carga un archivo CSV para hacer predicciones masivas"""
        filename = filedialog.askopenfilename(
            title="Seleccionar Archivo",
            filetypes=[("CSV files", "*.csv"), ("Todos", "*.*")]
        )
        
        if filename:
            self.archivo_prediccion.set(filename)
            print(f" Archivo seleccionado: {os.path.basename(filename)}")
    
    def realizar_prediccion(self):
        """Realiza predicción con el modelo cargado"""
        if self.modelo_cargado is None:
            messagebox.showwarning("Advertencia", "Debe cargar un modelo primero")
            return
        
        try:
            import numpy as np
            
            print(f"\n{'='*60}")
            print("REALIZANDO PREDICCIÓN")
            print(f"{'='*60}")
            
            # Obtener datos de entrada
            if self.metodo_entrada.get() == "manual":
                datos_entrada = self.obtener_datos_entrada_manual()
            else:
                datos_entrada = self.obtener_datos_desde_archivo()
            
            if datos_entrada is None:
                return
            
            print(f"✓ Datos de entrada obtenidos: {datos_entrada.shape}")
            
            # Normalizar datos usando el scaler del modelo original
            scaler = self.modelo_cargado_data['modelo']['scaler']
            datos_normalizados = scaler.transform(datos_entrada)
            
            print(f"✓ Datos normalizados")
            
            # Realizar predicción
            predicciones = self.modelo_cargado.predecir(datos_normalizados)
            
            print(f"✓ Predicción realizada")
            print(f"{'='*60}\n")
            
            # Decodificar si es clasificación
            label_encoder = self.modelo_cargado_data['modelo'].get('label_encoder')
            
            # Mostrar resultados
            self.mostrar_resultados_prediccion(datos_entrada, predicciones, label_encoder)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar predicción:\n{str(e)}")
            print(f"✗ Error: {str(e)}")
    
    def obtener_datos_entrada_manual(self):
        """Obtiene datos de entrada manual"""
        import numpy as np
        
        # Intentar primero desde la entrada en línea
        linea = self.entrada_linea.get().strip()
        
        if linea:
            try:
                valores = [float(x.strip()) for x in linea.replace(',', ' ').split()]
                return np.array([valores])
            except ValueError:
                messagebox.showerror("Error", "Los valores deben ser números válidos")
                return None
        
        # Si no, obtener de campos individuales
        valores = []
        for entry in self.entrada_manual_widgets:
            valor = entry.get().strip()
            if not valor:
                messagebox.showwarning("Advertencia", "Debe llenar todos los campos")
                return None
            try:
                valores.append(float(valor))
            except ValueError:
                messagebox.showerror("Error", f"'{valor}' no es un número válido")
                return None
        
        return np.array([valores])
    
    def obtener_datos_desde_archivo(self):
        """Obtiene datos desde archivo CSV"""
        import pandas as pd
        import numpy as np
        
        archivo = self.archivo_prediccion.get()
        
        if not archivo:
            messagebox.showwarning("Advertencia", "Debe seleccionar un archivo")
            return None
        
        try:
            df = pd.read_csv(archivo)
            print(f"✓ Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
            
            # Convertir a numpy array
            return df.values.astype(float)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al leer archivo:\n{str(e)}")
            return None
    
    def mostrar_resultados_prediccion(self, entrada, predicciones, label_encoder=None):
        """Muestra los resultados de la predicción"""
        import numpy as np
        
        resultado_text = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    RESULTADOS DE PREDICCIÓN                   ║
╚═══════════════════════════════════════════════════════════════╝

 CANTIDAD: {len(predicciones)} predicción(es) realizada(s)

"""
        
        # Si es clasificación, decodificar
        if label_encoder is not None:
            try:
                # Convertir predicciones a índices de clase
                pred_indices = np.round(predicciones).astype(int).flatten()
                pred_indices = np.clip(pred_indices, 0, len(label_encoder.classes_) - 1)
                pred_clases = label_encoder.inverse_transform(pred_indices)
                
                resultado_text += "🏷️ TIPO: Clasificación\n\n"
                resultado_text += "─" * 65 + "\n"
                resultado_text += f"{'#':<5} {'ENTRADA':<40} {'PREDICCIÓN':<20}\n"
                resultado_text += "─" * 65 + "\n"
                
                for i, (ent, pred) in enumerate(zip(entrada, pred_clases)):
                    entrada_str = ", ".join([f"{v:.3f}" for v in ent[:5]])  # Primeros 5 valores
                    if len(ent) > 5:
                        entrada_str += "..."
                    resultado_text += f"{i+1:<5} {entrada_str:<40} {pred:<20}\n"
                
            except Exception as e:
                print(f"Error al decodificar: {e}")
                resultado_text += self._formato_regresion(entrada, predicciones)
        else:
            resultado_text += " TIPO: Regresión\n\n"
            resultado_text += self._formato_regresion(entrada, predicciones)
        
        resultado_text += "\n" + "─" * 65 + "\n"
        resultado_text += f"\n Predicción completada exitosamente"
        
        self.mostrar_en_text(self.resultados_prediccion, resultado_text)
        
        # Guardar para exportar
        self.ultima_prediccion = {
            'entrada': entrada,
            'prediccion': predicciones,
            'label_encoder': label_encoder
        }
    
    def _formato_regresion(self, entrada, predicciones):
        """Formato para resultados de regresión"""
        texto = "─" * 65 + "\n"
        texto += f"{'#':<5} {'ENTRADA':<40} {'PREDICCIÓN':<15}\n"
        texto += "─" * 65 + "\n"
        
        for i, (ent, pred) in enumerate(zip(entrada, predicciones.flatten())):
            entrada_str = ", ".join([f"{v:.3f}" for v in ent[:5]])
            if len(ent) > 5:
                entrada_str += "..."
            texto += f"{i+1:<5} {entrada_str:<40} {pred:<15.6f}\n"
        
        return texto
    
    def exportar_resultados_prediccion(self):
        """Exporta los resultados de predicción a CSV"""
        if not hasattr(self, 'ultima_prediccion'):
            messagebox.showwarning("Advertencia", "No hay predicciones para exportar")
            return
        
        try:
            import pandas as pd
            
            filename = filedialog.asksaveasfilename(
                title="Exportar Resultados",
                defaultextension=".csv",
                initialfile=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                filetypes=[("CSV files", "*.csv"), ("Todos", "*.*")]
            )
            
            if filename:
                entrada = self.ultima_prediccion['entrada']
                prediccion = self.ultima_prediccion['prediccion']
                label_encoder = self.ultima_prediccion['label_encoder']
                
                # Crear DataFrame
                df_dict = {}
                
                # Agregar entradas
                for i in range(entrada.shape[1]):
                    df_dict[f'X{i+1}'] = entrada[:, i]
                
                # Agregar predicciones
                if label_encoder is not None:
                    import numpy as np
                    pred_indices = np.round(prediccion).astype(int).flatten()
                    pred_indices = np.clip(pred_indices, 0, len(label_encoder.classes_) - 1)
                    df_dict['Prediccion'] = label_encoder.inverse_transform(pred_indices)
                    df_dict['Prediccion_Valor'] = prediccion.flatten()
                else:
                    df_dict['Prediccion'] = prediccion.flatten()
                
                df = pd.DataFrame(df_dict)
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("Éxito", f"Resultados exportados a:\n{filename}")
                print(f"✓ Resultados exportados: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar:\n{str(e)}")
        """Abre una carpeta específica"""
        import platform
        import subprocess
        
        path = os.path.abspath(carpeta)
        
        # Crear carpeta si no existe
        os.makedirs(path, exist_ok=True)
        
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    
    def exportar_modelo_seleccionado(self):
        """Exporta el modelo seleccionado a un archivo"""
        seleccion = self.tree_modelos.selection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Debe seleccionar un modelo")
            return
        
        try:
            item = self.tree_modelos.item(seleccion[0])
            modelo_id = item['values'][0]
            nombre = item['values'][1]
            
            # Solicitar ubicación
            filename = filedialog.asksaveasfilename(
                title="Exportar Modelo",
                defaultextension=".pkl",
                initialfile=f"{nombre}.pkl",
                filetypes=[("Pickle files", "*.pkl"), ("Todos", "*.*")]
            )
            
            if filename:
                self.storage.exportar_modelo(modelo_id, filename)
                messagebox.showinfo("Éxito", f"Modelo exportado a:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar modelo:\n{str(e)}")
        
    def crear_consola(self, parent):
        """Crea la consola de salida en el panel lateral derecho"""
        console_frame = ttk.LabelFrame(parent, text="📟 Consola de Salida", padding="5")
        console_frame.pack(fill='both', expand=True)
        
        self.console_text = scrolledtext.ScrolledText(
            console_frame, 
            height=40, 
            state='disabled', 
            bg='#1e1e1e',  # Fondo oscuro
            fg='#00ff00',  # Texto verde fosforescente
            font=('Consolas', 9),
            insertbackground='white',
            wrap='word'
        )
        self.console_text.pack(fill='both', expand=True)
        
        # Botón para limpiar consola
        btn_frame = ttk.Frame(console_frame)
        btn_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(btn_frame, text=" Limpiar", 
                  command=self.limpiar_consola, 
                  width=12).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=" Guardar Log", 
                  command=self.guardar_log, 
                  width=12).pack(side='left', padx=2)
    
    def limpiar_consola(self):
        """Limpia el contenido de la consola"""
        self.console_text.config(state='normal')
        self.console_text.delete('1.0', tk.END)
        self.console_text.config(state='disabled')
        print("Consola limpiada")
    
    def guardar_log(self):
        """Guarda el contenido de la consola en un archivo"""
        try:
            contenido = self.console_text.get('1.0', tk.END)
            
            filename = filedialog.asksaveasfilename(
                title="Guardar Log",
                defaultextension=".txt",
                initialfile=f"log_rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                filetypes=[("Text files", "*.txt"), ("Todos", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(contenido)
                messagebox.showinfo("Éxito", f"Log guardado en:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar log:\n{str(e)}")
        
    def limpiar_todo(self):
        """Limpia todos los datos y reinicia la aplicación"""
        respuesta = messagebox.askyesno(
            "Confirmar Limpieza",
            "¿Está seguro de que desea limpiar todo?\n\n"
            "Esto borrará:\n"
            "• Dataset cargado\n"
            "• Datos preprocesados\n"
            "• Modelo entrenado\n"
            "• Resultados y gráficos\n\n"
            "Los modelos guardados en la base de datos NO se eliminarán."
        )
        
        if not respuesta:
            return
        
        try:
            # Reiniciar manejador de datos
            self.data_handler = DataHandler()
            
            # Reiniciar modelo
            self.rbf_model = None
            self.modelo_entrenado = False
            self.modelo_cargado = None
            self.modelo_cargado_id = None
            
            # Limpiar variables
            self.dataset_cargado = False
            self.ruta_dataset.set("")
            self.columna_salida.set("")
            self.num_centros.set(5)
            self.error_optimo.set(0.1)
            self.porcentaje_train.set(70.0)
            
            # Limpiar interfaz - Pestaña Datos
            self.mostrar_en_text(self.info_dataset, "")
            self.mostrar_en_text(self.estadisticas_text, "")
            self.division_info.config(text="")
            self.combo_columnas['values'] = []
            
            # Limpiar interfaz - Pestaña Entrenamiento
            self.mostrar_en_text(self.resultados_text, "")
            self.progress.stop()
            self.btn_entrenar.config(state='normal')
            
            # Limpiar interfaz - Pestaña Evaluación
            self.tree_metricas.delete(*self.tree_metricas.get_children())
            for widget in self.imagenes_frame.winfo_children():
                widget.destroy()
            
            # Limpiar interfaz - Pestaña Predicción
            if hasattr(self, 'info_modelo_cargado'):
                self.mostrar_en_text(self.info_modelo_cargado, "")
                self.mostrar_en_text(self.resultados_prediccion, "")
                self.archivo_prediccion.set("")
                self.entrada_linea.delete(0, tk.END)
                for entry in self.entrada_manual_widgets:
                    entry.delete(0, tk.END)
            
            # Limpiar consola
            self.limpiar_consola()
            
            print("="*60)
            print(" SISTEMA LIMPIADO CORRECTAMENTE")
            print("="*60)
            print("\n Puede cargar un nuevo dataset en la pestaña 'Datos'")
            print(" Los modelos guardados siguen disponibles en 'Modelos'\n")
            
            # Volver a la primera pestaña
            self.notebook.select(0)
            
            messagebox.showinfo("Éxito", "Sistema limpiado correctamente.\n\nPuede comenzar un nuevo entrenamiento.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al limpiar:\n{str(e)}")
    
    def configuracion_automatica(self):
        """Configura automáticamente los parámetros según el dataset"""
        try:
            if not hasattr(self.data_handler, 'estadisticas') or not self.data_handler.estadisticas:
                messagebox.showwarning("Advertencia", "Debe preprocesar los datos primero")
                return
            
            stats = self.data_handler.estadisticas
            num_patrones = self.data_handler.dataset_info['num_patrones']
            num_entradas = stats['num_entradas']
            es_clasificacion = stats.get('es_clasificacion', False)
            
            # ===== CÁLCULO MEJORADO DE CENTROS =====
            # Fórmula más conservadora para evitar sobreajuste
            # Base: entre 10-20% del número de patrones
            centros_base = int(num_patrones * 0.15)  # 15% de patrones
            
            # Ajustar por número de entradas (más entradas = más centros necesarios)
            if num_entradas <= 3:
                factor_entradas = 0.8
            elif num_entradas <= 5:
                factor_entradas = 1.0
            elif num_entradas <= 10:
                factor_entradas = 1.2
            else:
                factor_entradas = 1.5
            
            centros_ajustados = int(centros_base * factor_entradas)
            
            # Límites razonables
            centros_optimos = max(8, min(centros_ajustados, 25))
            
            # ===== CÁLCULO MEJORADO DE ERROR ÓPTIMO =====
            if es_clasificacion:
                # Clasificación: basado en número de clases
                num_clases = stats.get('num_clases', 3)
                
                if num_clases == 2:
                    error_optimo = 0.35  # Binaria: más fácil
                elif num_clases == 3:
                    error_optimo = 0.45  # 3 clases: moderado
                else:
                    error_optimo = 0.55  # Multiclase: más difícil
                
                tipo = f"Clasificación ({num_clases} clases)"
                
            else:
                # Regresión: basado en variabilidad de los datos
                if 'rango_y' in stats:
                    rango_y = stats['rango_y']['max'] - stats['rango_y']['min']
                    std_y = stats['rango_y']['std']
                    
                    # Error óptimo = 8-12% del rango (más realista)
                    error_base = rango_y * 0.10
                    
                    # Ajustar por desviación estándar (más variabilidad = más error aceptable)
                    if std_y / rango_y > 0.3:  # Alta variabilidad
                        error_optimo = error_base * 1.5
                    elif std_y / rango_y > 0.2:  # Variabilidad media
                        error_optimo = error_base * 1.2
                    else:  # Baja variabilidad
                        error_optimo = error_base
                    
                    # Limitar entre 0.15 y 2.0
                    error_optimo = max(0.15, min(error_optimo, 2.0))
                    error_optimo = round(error_optimo, 3)
                else:
                    error_optimo = 0.4
                
                tipo = "Regresión"
            
            # Aplicar configuración
            self.num_centros.set(centros_optimos)
            self.error_optimo.set(error_optimo)
            
            # Construir mensaje explicativo
            mensaje = f"""
╔═══════════════════════════════════════════════════════════╗
║         CONFIGURACIÓN AUTOMÁTICA MEJORADA                 ║
╚═══════════════════════════════════════════════════════════╝

 ANÁLISIS DEL DATASET:
  • Tipo: {tipo}
  • Patrones totales: {num_patrones}
  • Entradas: {num_entradas}
"""
            
            if es_clasificacion:
                mensaje += f"  • Distribución: {stats.get('distribucion_clases', {})}\n"
            else:
                if 'rango_y' in stats:
                    mensaje += f"  • Rango salida: [{stats['rango_y']['min']:.3f}, {stats['rango_y']['max']:.3f}]\n"
                    mensaje += f"  • Desv. Est.: {stats['rango_y']['std']:.3f}\n"
            
            mensaje += f"""
 CONFIGURACIÓN APLICADA:

   Centros Radiales: {centros_optimos}
      Cálculo: {centros_base} (15% de {num_patrones}) × {factor_entradas:.1f} (factor entradas)
     → Equilibrio entre complejidad y generalización
     
   Error Óptimo: {error_optimo}
"""
            
            if es_clasificacion:
                mensaje += f"     Basado en: {num_clases} clases de clasificación\n"
            else:
                if 'rango_y' in stats:
                    porcentaje = (error_optimo / rango_y) * 100
                    mensaje += f"     Basado en: ~{porcentaje:.1f}% del rango de salida\n"
            
            mensaje += f"""
JUSTIFICACIÓN:

  ✓ Centros conservadores para evitar sobreajuste
  ✓ Error óptimo realista para datos reales
  ✓ Ajustado según complejidad del problema

EXPECTATIVA DE CONVERGENCIA:

  Con esta configuración:
  • Probabilidad de convergencia: ~80-90%
  • Si no converge: Aumentar centros en pasos de 5
  • Error de prueba esperado: ±{error_optimo * 1.2:.3f}

NOTA: Puede ajustar manualmente si lo desea.
"""
            
            messagebox.showinfo("Configuración Automática", mensaje)
            
            print(f"\n{'='*60}")
            print("CONFIGURACIÓN AUTOMÁTICA APLICADA")
            print(f"{'='*60}")
            print(f"  • Tipo: {tipo}")
            print(f"  • Patrones: {num_patrones}, Entradas: {num_entradas}")
            print(f"  • Centros: {centros_optimos}")
            print(f"  • Error Óptimo: {error_optimo}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en configuración automática:\n{str(e)}")
    
    # ===== FUNCIONES DE CARGA Y PREPROCESAMIENTO =====
    
    def cargar_dataset(self):
        """Carga un dataset desde archivo"""
        filename = filedialog.askopenfilename(
            title="Seleccionar Dataset",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("Todos", "*.*")],
            initialdir="datasets"
        )
        
        if filename:
            try:
                self.ruta_dataset.set(filename)
                info = self.data_handler.cargar_dataset(filename)
                
                # Mostrar información
                self.mostrar_en_text(self.info_dataset, f"""
Dataset cargado exitosamente:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nombre: {info['nombre']}
Patrones totales: {info['num_patrones']}
Columnas: {info['num_columnas']}

Columnas disponibles:
{chr(10).join(f"  • {col}" for col in info['columnas'])}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                """)
                
                # Actualizar combo de columnas
                self.combo_columnas['values'] = info['columnas']
                if info['columnas']:
                    self.combo_columnas.current(len(info['columnas']) - 1)
                
                self.dataset_cargado = True
                messagebox.showinfo("Éxito", "Dataset cargado correctamente")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar dataset:\n{str(e)}")
    
    def preprocesar_datos(self):
        """Preprocesa los datos"""
        if not self.dataset_cargado:
            messagebox.showwarning("Advertencia", "Debe cargar un dataset primero")
            return
        
        if not self.columna_salida.get():
            messagebox.showwarning("Advertencia", "Debe seleccionar una columna de salida")
            return
        
        try:
            stats = self.data_handler.preprocesar_datos(self.columna_salida.get(), normalizar=True)
            
            # Mostrar estadísticas
            texto = f"""
Preprocesamiento completado:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Variables de entrada: {stats['num_entradas']}
Variables de salida: {stats['num_salidas']}
Tipo de problema: {'Clasificación' if stats['es_clasificacion'] else 'Regresión'}

Rango de X (normalizado):
  Min: {stats['rango_X']['min']:.4f}
  Max: {stats['rango_X']['max']:.4f}
  Media: {stats['rango_X']['media']:.4f}
  Desv. Est.: {stats['rango_X']['std']:.4f}
"""
            
            if stats['es_clasificacion']:
                texto += f"\nClases detectadas: {stats['num_clases']}\n"
                texto += "Distribución:\n"
                for clase, count in stats['distribucion_clases'].items():
                    texto += f"  • Clase {clase}: {count} muestras\n"
            else:
                texto += f"\nRango de Y:\n"
                texto += f"  Min: {stats['rango_y']['min']:.4f}\n"
                texto += f"  Max: {stats['rango_y']['max']:.4f}\n"
            
            texto += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            self.mostrar_en_text(self.estadisticas_text, texto)
            messagebox.showinfo("Éxito", "Preprocesamiento completado")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en preprocesamiento:\n{str(e)}")
    
    def dividir_datos(self):
        """Divide los datos en entrenamiento y prueba"""
        try:
            porcentaje = self.porcentaje_train.get() / 100
            info = self.data_handler.dividir_datos(porcentaje_entrenamiento=porcentaje)
            
            texto = f"""✓ Datos divididos correctamente:
  • Entrenamiento: {info['patrones_entrenamiento']} patrones ({info['porcentaje_entrenamiento']*100:.0f}%)
  • Prueba: {info['patrones_prueba']} patrones ({info['porcentaje_prueba']*100:.0f}%)"""
            
            self.division_info.config(text=texto)
            messagebox.showinfo("Éxito", "Datos divididos correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al dividir datos:\n{str(e)}")
    
    # ===== FUNCIONES DE ENTRENAMIENTO =====
    
    def iniciar_entrenamiento(self):
        """Inicia el entrenamiento en un hilo separado"""
        if not self.dataset_cargado:
            messagebox.showwarning("Advertencia", "Debe cargar y preprocesar datos primero")
            return
        
        # Deshabilitar botón
        self.btn_entrenar.config(state='disabled')
        self.progress.start()
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=self.entrenar_modelo)
        thread.daemon = True
        thread.start()
    
    def entrenar_modelo(self):
        """Entrena el modelo RBF"""
        error_msg = None
        try:
            # Crear modelo
            self.rbf_model = RBFNeuralNetwork(
                num_centros=self.num_centros.get(),
                error_optimo=self.error_optimo.get()
            )
            
            # Obtener datos
            X_train, y_train = self.data_handler.get_datos_entrenamiento()
            X_test, y_test = self.data_handler.get_datos_prueba()
            
            # Entrenar
            self.metricas_train = self.rbf_model.entrenar(X_train, y_train)
            
            # Evaluar
            self.metricas_test = self.rbf_model.evaluar(X_test, y_test)
            
            # Actualizar GUI
            self.root.after(0, self.mostrar_resultados_entrenamiento)
            self.modelo_entrenado = True
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Error en entrenamiento:\n{msg}"))
        finally:
            self.root.after(0, self.finalizar_entrenamiento)
    
    def mostrar_resultados_entrenamiento(self):
        """Muestra los resultados en la interfaz"""
        
        # Determinar estado de convergencia
        converge = self.metricas_train['Converge']
        eg_train = self.metricas_train['EG']
        error_optimo = self.error_optimo.get()
        
        # Calcular qué tan cerca está de converger
        porcentaje_error = (eg_train / error_optimo) * 100
        
        if converge:
            estado_conv = "SÍ - EXCELENTE"
            color_conv = "verde"
            recomendacion = "El modelo ha convergido exitosamente."
        elif eg_train <= error_optimo * 1.5:
            estado_conv = " CASI - MUY CERCA"
            color_conv = "amarillo"
            recomendacion = f"Está a solo {(eg_train - error_optimo):.4f} de converger. Intente aumentar centros a {self.num_centros.get() + 3}."
        else:
            estado_conv = "NO"
            color_conv = "rojo"
            centros_sugeridos = min(self.num_centros.get() + 5, 30)
            error_sugerido = round(eg_train * 1.2, 3)
            recomendacion = f"Sugerencias:\n  • Aumentar centros a {centros_sugeridos}\n  • O ajustar error óptimo a {error_sugerido}"
        
        texto = f"""
{'='*60}
RESULTADOS DEL ENTRENAMIENTO
{'='*60}

ENTRENAMIENTO:
  • EG (Error General):        {eg_train:.6f}
  • MAE (Error Abs. Medio):    {self.metricas_train['MAE']:.6f}
  • RMSE (Raíz Error Cuad.):   {self.metricas_train['RMSE']:.6f}
  • Convergencia:              {estado_conv}

PRUEBA:
  • EG (Error General):        {self.metricas_test['EG']:.6f}
  • MAE (Error Abs. Medio):    {self.metricas_test['MAE']:.6f}
  • RMSE (Raíz Error Cuad.):   {self.metricas_test['RMSE']:.6f}

{'='*60}
ANÁLISIS DE CONVERGENCIA
{'='*60}

Error Objetivo:              {error_optimo}
Error Alcanzado:             {eg_train:.6f}
Diferencia:                  {abs(eg_train - error_optimo):.6f}
Porcentaje del Objetivo:     {porcentaje_error:.1f}%

Estado:                      {estado_conv}

💡 Recomendación:
{recomendacion}

{'='*60}
"""
        self.mostrar_en_text(self.resultados_text, texto)
        
        # Actualizar tabla de métricas con información de convergencia mejorada
        self.tree_metricas.delete(*self.tree_metricas.get_children())
        
        # Entrenamiento
        conv_text = "SÍ" if converge else f"NO ({porcentaje_error:.0f}%)"
        self.tree_metricas.insert('', 'end', values=(
            'Entrenamiento',
            f"{eg_train:.6f}",
            f"{self.metricas_train['MAE']:.6f}",
            f"{self.metricas_train['RMSE']:.6f}",
            conv_text
        ), tags=('converge' if converge else 'no_converge',))
        
        # Prueba
        self.tree_metricas.insert('', 'end', values=(
            'Prueba',
            f"{self.metricas_test['EG']:.6f}",
            f"{self.metricas_test['MAE']:.6f}",
            f"{self.metricas_test['RMSE']:.6f}",
            '—'
        ))
        
        # Configurar colores
        self.tree_metricas.tag_configure('converge', background='#d4edda')
        self.tree_metricas.tag_configure('no_converge', background='#f8d7da')
        
        # Cambiar a pestaña de evaluación
        self.notebook.select(self.tab_evaluacion)
    
    def finalizar_entrenamiento(self):
        """Finaliza el proceso de entrenamiento"""
        self.progress.stop()
        self.btn_entrenar.config(state='normal')
    
    # ===== FUNCIONES DE VISUALIZACIÓN Y PERSISTENCIA =====
    
    def generar_y_mostrar_graficos(self):
        """Genera gráficos del modelo y los muestra en la interfaz"""
        if not self.modelo_entrenado:
            messagebox.showwarning("Advertencia", "Debe entrenar el modelo primero")
            return
        
        try:
            X_train, y_train = self.data_handler.get_datos_entrenamiento()
            X_test, y_test = self.data_handler.get_datos_prueba()
            
            # Generar gráficos en carpeta
            self.rbf_model.generar_graficos(
                X_train, y_train,
                X_test, y_test,
                self.metricas_train,
                self.metricas_test
            )
            
            # Mostrar gráficos en la interfaz
            self.mostrar_graficos_en_gui()
            
            messagebox.showinfo("Éxito", "Gráficos generados y guardados en: resultados/graficos/")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráficos:\n{str(e)}")
    
    def mostrar_graficos_en_gui(self):
        """Muestra los gráficos generados en la interfaz"""
        try:
            from PIL import Image, ImageTk
            
            # Limpiar frame anterior
            for widget in self.imagenes_frame.winfo_children():
                widget.destroy()
            
            # Rutas de las imágenes
            img_paths = [
                'resultados/graficos/metricas_comparacion.png',
                'resultados/graficos/dispersion_predicciones.png'
            ]
            
            # Crear frame con scroll
            canvas = tk.Canvas(self.imagenes_frame, bg='white')
            scrollbar = ttk.Scrollbar(self.imagenes_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Cargar y mostrar imágenes
            for i, img_path in enumerate(img_paths):
                if os.path.exists(img_path):
                    # Cargar imagen
                    img = Image.open(img_path)
                    
                    # Redimensionar si es muy grande
                    max_width = 900
                    if img.width > max_width:
                        ratio = max_width / img.width
                        new_size = (max_width, int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Convertir para Tkinter
                    photo = ImageTk.PhotoImage(img)
                    
                    # Crear label y agregar imagen
                    label = ttk.Label(scrollable_frame, image=photo)
                    label.image = photo  # Mantener referencia
                    label.pack(pady=10, padx=10)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            print("✓ Gráficos mostrados en la interfaz")
            
        except ImportError:
            messagebox.showwarning("Advertencia", 
                                 "Instale Pillow para ver imágenes en la interfaz:\npip install Pillow")
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar gráficos:\n{str(e)}")
    
    def generar_graficos(self):
        """Genera gráficos del modelo (solo guarda, sin mostrar)"""
        if not self.modelo_entrenado:
            messagebox.showwarning("Advertencia", "Debe entrenar el modelo primero")
            return
        
        try:
            X_train, y_train = self.data_handler.get_datos_entrenamiento()
            X_test, y_test = self.data_handler.get_datos_prueba()
            
            self.rbf_model.generar_graficos(
                X_train, y_train,
                X_test, y_test,
                self.metricas_train,
                self.metricas_test
            )
            
            messagebox.showinfo("Éxito", "Gráficos generados en: resultados/graficos/")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráficos:\n{str(e)}")
    
    def abrir_resultados(self):
        """Abre la carpeta de resultados"""
        import platform
        import subprocess
        
        path = os.path.abspath("resultados")
        
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    
    def guardar_modelo(self):
        """Guarda el modelo entrenado"""
        if not self.modelo_entrenado:
            messagebox.showwarning("Advertencia", "Debe entrenar el modelo primero")
            return
        
        try:
            nombre = self.nombre_modelo.get()
            
            # Preparar datos
            modelo_data = {
                'centros': self.rbf_model.centros,
                'pesos': self.rbf_model.pesos,
                'scaler': self.data_handler.get_scaler(),
                'label_encoder': self.data_handler.get_label_encoder()
            }
            
            config = {
                'num_centros': self.num_centros.get(),
                'porcentaje_entrenamiento': self.porcentaje_train.get() / 100,
                'funcion_activacion': 'd² × ln(d)',
                'error_optimo': self.error_optimo.get()
            }
            
            # Guardar en base de datos
            entrenamiento_id = self.storage.guardar_entrenamiento(
                nombre=nombre,
                dataset_info=self.data_handler.get_dataset_info(),
                config=config,
                modelo_data=modelo_data,
                metricas_train=self.metricas_train,
                metricas_test=self.metricas_test,
                estadisticas=self.data_handler.get_estadisticas(),
                descripcion="Modelo entrenado desde la aplicación"
            )
            
            messagebox.showinfo("Éxito", f"Modelo guardado con ID: {entrenamiento_id}")
            self.actualizar_lista_modelos()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar modelo:\n{str(e)}")
    
    def actualizar_lista_modelos(self):
        """Actualiza la lista de modelos guardados"""
        self.tree_modelos.delete(*self.tree_modelos.get_children())
        
        try:
            modelos = self.storage.listar_entrenamientos()
            for modelo in modelos:
                self.tree_modelos.insert('', 'end', values=(
                    modelo['id'],
                    modelo['nombre'],
                    modelo['dataset'],
                    modelo['fecha'],
                    modelo['num_centros'],
                    modelo['split']
                ))
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos:\n{str(e)}")
    
    def cargar_modelo_seleccionado(self):
        """Carga el modelo seleccionado y muestra información detallada"""
        seleccion = self.tree_modelos.selection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Debe seleccionar un modelo")
            return
        
        try:
            item = self.tree_modelos.item(seleccion[0])
            modelo_id = item['values'][0]
            
            datos = self.storage.cargar_entrenamiento(modelo_id)
            
            info_completa = f"""
╔═══════════════════════════════════════════════════════════════╗
║                   INFORMACIÓN DEL MODELO                      ║
╚═══════════════════════════════════════════════════════════════╝

ID: {datos['info']['id']}
Nombre: {datos['info']['nombre']}
Dataset: {datos['info']['dataset_nombre']}
Fecha: {datos['info']['fecha_creacion']}

CONFIGURACIÓN:
  • Centros: {datos['info']['num_centros']}
  • Entradas: {datos['info']['num_entradas']}
  • Salidas: {datos['info']['num_salidas']}
  • Error Óptimo: {datos['info']['error_optimo']}
  • Split: {datos['info']['porcentaje_entrenamiento']*100:.0f}% / {(1-datos['info']['porcentaje_entrenamiento'])*100:.0f}%

MÉTRICAS:
  Entrenamiento:
    EG:   {datos['metricas']['entrenamiento']['EG']:.6f}
    MAE:  {datos['metricas']['entrenamiento']['MAE']:.6f}
    RMSE: {datos['metricas']['entrenamiento']['RMSE']:.6f}
    
  Prueba:
    EG:   {datos['metricas']['prueba']['EG']:.6f}
    MAE:  {datos['metricas']['prueba']['MAE']:.6f}
    RMSE: {datos['metricas']['prueba']['RMSE']:.6f}

╔═══════════════════════════════════════════════════════════════╗
║ ¿Desea usar este modelo para predicciones?                    ║
║ Vaya a la pestaña "Predicción" y cárgelo desde allí           ║
╚═══════════════════════════════════════════════════════════════╝
"""
            
            messagebox.showinfo("Modelo Cargado", info_completa)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelo:\n{str(e)}")
    
    def eliminar_modelo_seleccionado(self):
        """Elimina el modelo seleccionado"""
        seleccion = self.tree_modelos.selection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Debe seleccionar un modelo")
            return
        
        if not messagebox.askyesno("Confirmar", "¿Está seguro de eliminar este modelo?"):
            return
        
        try:
            item = self.tree_modelos.item(seleccion[0])
            modelo_id = item['values'][0]
            
            self.storage.eliminar_entrenamiento(modelo_id)
            self.actualizar_lista_modelos()
            messagebox.showinfo("Éxito", "Modelo eliminado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al eliminar modelo:\n{str(e)}")
    

    
    def mostrar_en_text(self, widget, texto):
        """Muestra texto en un widget ScrolledText"""
        widget.config(state='normal')
        widget.delete('1.0', tk.END)
        widget.insert('1.0', texto)
        widget.config(state='disabled')


class TextRedirector:
    """Redirige la salida de print a un widget de texto"""
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.config(state='normal')
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.config(state='disabled')
        
    def flush(self):
        pass


def main():
    """Función principal"""
    root = tk.Tk()
    
    # Configurar estilo
    style = ttk.Style()
    style.theme_use('clam')
    
    # Colores personalizados
    style.configure('Accent.TButton', foreground='white', background='#2196F3', 
                   font=('Arial', 10, 'bold'), padding=10)
    
    app = RBFApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
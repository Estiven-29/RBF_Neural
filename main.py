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
        self.notebook.add(self.tab_datos, text="📁 Datos")
        self.crear_tab_datos()
        
        # Pestaña 2: Configuración y Entrenamiento
        self.tab_entrenamiento = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text="🧠 Entrenamiento")
        self.crear_tab_entrenamiento()
        
        # Pestaña 3: Evaluación y Visualización
        self.tab_evaluacion = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_evaluacion, text="📊 Evaluación")
        self.crear_tab_evaluacion()
        
        # Pestaña 4: Modelos Guardados
        self.tab_modelos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_modelos, text="💾 Modelos")
        self.crear_tab_modelos()
        
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
        
        ttk.Button(paso2, text="✓ Preprocesar Datos", command=self.preprocesar_datos, 
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
        btn_dividir = ttk.Button(paso3, text="▶ DIVIDIR DATASET", command=self.dividir_datos, 
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
        
        ttk.Label(config_frame, text="Error de Aproximación Óptimo:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.error_optimo, width=10).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(config_frame, text="Función de Activación:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Label(config_frame, text="FA(d) = d² × ln(d)", font=('Courier', 10, 'bold')).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Botón de entrenamiento
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        self.btn_entrenar = ttk.Button(btn_frame, text="🚀 INICIAR ENTRENAMIENTO", 
                                       command=self.iniciar_entrenamiento,
                                       style='Accent.TButton', width=30)
        self.btn_entrenar.pack()
        
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
        metricas_frame = ttk.LabelFrame(main_frame, text="📊 Métricas del Modelo", padding="10")
        metricas_frame.pack(fill='x', pady=(0, 10))
        
        # Tabla de métricas
        columns = ('Conjunto', 'EG', 'MAE', 'RMSE', 'Convergencia')
        self.tree_metricas = ttk.Treeview(metricas_frame, columns=columns, show='headings', height=3)
        
        for col in columns:
            self.tree_metricas.heading(col, text=col)
            self.tree_metricas.column(col, width=120, anchor='center')
        
        self.tree_metricas.pack(fill='x', padx=5, pady=5)
        
        # Visualizaciones
        visual_frame = ttk.LabelFrame(main_frame, text="📈 Visualizaciones", padding="10")
        visual_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Botones de control
        btn_frame = ttk.Frame(visual_frame)
        btn_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(btn_frame, text="🎨 Generar Gráficos", 
                  command=self.generar_y_mostrar_graficos).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📁 Abrir Carpeta", 
                  command=self.abrir_resultados).pack(side='left', padx=5)
        
        # Frame para mostrar imágenes
        self.imagenes_frame = ttk.Frame(visual_frame)
        self.imagenes_frame.pack(fill='both', expand=True)
        
        # Guardar modelo
        guardar_frame = ttk.LabelFrame(main_frame, text="💾 Guardar Modelo Entrenado", padding="10")
        guardar_frame.pack(fill='x', pady=(10, 0))
        
        info_label = ttk.Label(guardar_frame, text="📍 Los modelos se guardan en: database/rbf_trainings.db", 
                              font=('Arial', 9, 'italic'), foreground='#666')
        info_label.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky='w')
        
        self.nombre_modelo = tk.StringVar(value="Modelo_RBF_1")
        ttk.Label(guardar_frame, text="Nombre del modelo:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(guardar_frame, textvariable=self.nombre_modelo, width=30).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(guardar_frame, text="💾 Guardar Modelo", 
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
        
        ttk.Button(path_frame, text="📂 Abrir", command=lambda: self.abrir_carpeta('database')).pack(side='left')
        
        ttk.Label(info_frame, text="Los gráficos se guardan en: resultados/graficos/", 
                 font=('Arial', 9, 'italic'), foreground='#666').pack(anchor='w', pady=(5, 0))
        
        # Controles
        controles_frame = ttk.Frame(main_frame)
        controles_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(controles_frame, text="🔄 Actualizar Lista", 
                  command=self.actualizar_lista_modelos).pack(side='left', padx=5)
        ttk.Button(controles_frame, text="📂 Cargar Modelo", 
                  command=self.cargar_modelo_seleccionado).pack(side='left', padx=5)
        ttk.Button(controles_frame, text="🗑️ Eliminar Modelo", 
                  command=self.eliminar_modelo_seleccionado).pack(side='left', padx=5)
        ttk.Button(controles_frame, text="📤 Exportar", 
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
    
    def abrir_carpeta(self, carpeta):
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
        
        ttk.Button(btn_frame, text="🗑️ Limpiar", 
                  command=self.limpiar_consola, 
                  width=12).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="💾 Guardar Log", 
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
        texto = f"""
{'='*60}
RESULTADOS DEL ENTRENAMIENTO
{'='*60}

ENTRENAMIENTO:
  • EG (Error General):        {self.metricas_train['EG']:.6f}
  • MAE (Error Abs. Medio):    {self.metricas_train['MAE']:.6f}
  • RMSE (Raíz Error Cuad.):   {self.metricas_train['RMSE']:.6f}
  • Convergencia:              {'✓ SÍ' if self.metricas_train['Converge'] else '✗ NO'}

PRUEBA:
  • EG (Error General):        {self.metricas_test['EG']:.6f}
  • MAE (Error Abs. Medio):    {self.metricas_test['MAE']:.6f}
  • RMSE (Raíz Error Cuad.):   {self.metricas_test['RMSE']:.6f}

{'='*60}
"""
        self.mostrar_en_text(self.resultados_text, texto)
        
        # Actualizar tabla de métricas
        self.tree_metricas.delete(*self.tree_metricas.get_children())
        self.tree_metricas.insert('', 'end', values=(
            'Entrenamiento',
            f"{self.metricas_train['EG']:.6f}",
            f"{self.metricas_train['MAE']:.6f}",
            f"{self.metricas_train['RMSE']:.6f}",
            '✓' if self.metricas_train['Converge'] else '✗'
        ))
        self.tree_metricas.insert('', 'end', values=(
            'Prueba',
            f"{self.metricas_test['EG']:.6f}",
            f"{self.metricas_test['MAE']:.6f}",
            f"{self.metricas_test['RMSE']:.6f}",
            '—'
        ))
        
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
        """Carga el modelo seleccionado"""
        seleccion = self.tree_modelos.selection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Debe seleccionar un modelo")
            return
        
        try:
            item = self.tree_modelos.item(seleccion[0])
            modelo_id = item['values'][0]
            
            datos = self.storage.cargar_entrenamiento(modelo_id)
            messagebox.showinfo("Modelo Cargado", 
                              f"Modelo: {datos['info']['nombre']}\n"
                              f"Dataset: {datos['info']['dataset_nombre']}\n"
                              f"EG Entrenamiento: {datos['metricas']['entrenamiento']['EG']:.6f}\n"
                              f"EG Prueba: {datos['metricas']['prueba']['EG']:.6f}")
            
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
    
    # ===== UTILIDADES =====
    
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
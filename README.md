# TFM


**Este repositorio contiene el código necesario para implementar un algoritmo de detección de aletas de orcas a partir de imágenes de avistamientos**

**CONTENIDO:** 

o	Requirements_code.txt: Librerías necesarias para el código

o	1_Entrenamiento_YOLO8.ipynb: Código entrenamiento modelo basado en YOLO

o	data.yaml: Archivo para entrenamiento modelo con YOLO

o	2_Anotación_recorte_YOLO.ipynb: Código aplicación modelo YOLO a imágenes

o	3_Crear_train_test_val_det_lado.ipynb: Código de creación dataset para uso en entrenamiento modelo ResNet

o	4_Detección_lado.ipynb: Código entrenamiento modelo basado en aprendizaje por transferencia – ResNet

o	5_Aplicación_detección_lado.ipynb: Código para aplicar modelo entrenado de detección de lado en nuevas imágenes

o	6_Creación_datasets_siamesas.ipynb: Código para dividir las imágenes de cada carpeta de individuo en train (80%) y test (20%).

o	7_Siamese_Contrastive.ipynb: Código para el entrenamiento del modelo de redes siamesas. 

o	8_Triplet_Loss.ipynb: Código para el entrenamiento del modelo de redes basadas en tripletes.

o	App_shiny_apyder2.py: Código para implementar la aplicación en Shiny

o	Requirements_app.txt:  para la aplicación en Shiny

o	best8.pt : Modelo entrenado basado en YOLO


'''

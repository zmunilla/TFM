# TFM


**Este repositorio contiene el código necesario para implementar un algoritmo de detección de aletas de orcas a partir de imágenes de avistamientos**

**CONTENIDO:** 

o	Requirements_code.txt: Librerías necesarias para el código

o	1_Entrenamiento_YOLO8_PEC4.ipynb: Código entrenamiento modelo basado en YOLO

o	data.yaml: Archivo para entrenamiento modelo con YOLO

o	2_Detección_recorte_YOLO_PEC4.ipynb: Código aplicación modelo YOLO a imágenes

o	3_Clasificación_lado_PEC4.ipynb: Código entrenamiento modelo basado en aprendizaje por transferencia – ResNet

o	4_Aplicación_clasificación_lado_PEC4.ipynb: Código para aplicar modelo entrenado de detección de lado en nuevas imágenes

o	5_Siamese_Contrastive_PEC4.ipynb: Código para el entrenamiento del modelo de redes siamesas. 

o	6_Triplet_Loss_PEC4.ipynb: Código para el entrenamiento del modelo de redes basadas en tripletes.

o	App_redef.py: Código para implementar la aplicación en Shiny

o	Requirements_app.txt:  para la aplicación en Shiny

o	cat_emb_left_DRIVE.pkl : Catalogo de embeddings con links de imágenes a Google Drive (left)

o	cat_emb_right_DRIVE.pkl : Catalogo de embeddings con links de imágenes a Google Drive (right)

o	catalog_triplet_left_final2.pkl : Catalogo de embeddings con rutas de imagenes locales (left)

o	catalog_triplet_right_final2.pkl : Catalogo de embeddings con rutas de imagenes locales (right)

o	drive_image_map_left.json : Archivo .json con rutas de las imágenes en Google Drive (left)

o	drive_image_map_right.json : Archivo .json con rutas de las imágenes en Google Drive (right)

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 15:28:09 2025

@author: zaida
"""

import nest_asyncio
from shiny import App, render, ui
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from ultralytics import YOLO
import cv2

nest_asyncio.apply()

#ruta al modelo entrenado de YOLO

model1_path = os.path.join(os.path.dirname(__file__), 'best8.pt')

# Cargar el modelo YOLO
model1 = YOLO(model1_path)

#fuerzo el valor de la confianza y de la iou o solapamiento

model1.conf = 0.45
model1.iou = 0.5

# 🌟 INTERFAZ DE USUARIO (UI)
app_ui = ui.page_fluid(
    ui.panel_title("🐳🔍 ORCAFIN-ID: Identificación de orcas"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("file1", "Sube una imagen de orcas", accept=[".png", ".jpg", ".jpeg"]),
            ui.input_action_button("detect", "Detectar aleta"),
            ui.output_text("prediction")
        ),
        ui.output_image("outputImage")
        )
)

# SERVIDOR
def server(input, output, session):
    @render.text
    def prediction():
        if input.file1() is None:
            return "Sube una imagen"
    
        
        file = input.file1()[0]
        image = Image.open(file["datapath"])

        # Almacenamiento temporal 
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file, format="PNG")
            temp_path = temp_file.name

        if input.detect():
            # Detectar con YOLO
            results = model1(temp_path)

            # Guardar la imagen procesada
            output_path = temp_path.replace(".png", "_result.png")
            results[0].save(output_path)

            # Obtener detalles de la predicción
            num_detections = len(results[0].boxes)
            conf_avg = sum(results[0].boxes.conf) / num_detections if num_detections > 0 else 0

            # Mostrar resultados
            if num_detections > 0:
                return f"✔️ {num_detections} aleta(s) detectada(s) con confianza promedio de {conf_avg:.2%}"
            else:
                return "❌ No se detectaron aletas"

            # Eliminar archivos temporales después de mostrar
            os.remove(temp_path)
            os.remove(output_path)


    @render.image
    def outputImage():
        if input.file1() is None:
            return None
        
        file = input.file1()[0]
        image = Image.open(file["datapath"])

        # Guardar temporalmente la imagen para YOLO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file, format="PNG")
            temp_path = temp_file.name

        if input.detect():
            # Detectar con YOLO
            results = model1(temp_path)

            # Guardar la imagen procesada
            output_path = temp_path.replace(".png", "_result.png")
            results[0].save(output_path)

            # Mostrar la imagen procesada
            return {"src": output_path, "width": "50%", "height": "auto"}  # Regresar la ruta en un diccionario


# 🔥 INICIAR LA APP
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()

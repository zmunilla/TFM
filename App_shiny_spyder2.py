# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 21:14:13 2025

@author: zaida
"""
import os
import shutil
import uuid
from pathlib import Path

import cv2
import nest_asyncio
from shiny import App, ui, render, reactive
from ultralytics import YOLO

nest_asyncio.apply()

BASE_DIR = Path(__file__).parent.resolve()
WWW_DIR = BASE_DIR / "www"
CROPS_DIR = WWW_DIR / "crops"
SAMPLES_DIR = WWW_DIR / "samples"
UPLOADS_DIR = WWW_DIR / "uploads"

for folder in [CROPS_DIR, SAMPLES_DIR, UPLOADS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

for f in CROPS_DIR.glob("*"):
    f.unlink()
for f in UPLOADS_DIR.glob("*"):
    f.unlink()

model = YOLO(str(BASE_DIR / "best8.pt"))
model.conf = 0.45
model.iou = 0.5

sample_images = [f.name for f in SAMPLES_DIR.glob("*.jpg")]

app_ui = ui.page_fluid(
    ui.panel_title("🐳🔍 ORCAFIN-ID: Identificación de orcas"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("file1", "Sube una imagen de orca", accept=[".jpg", ".jpeg", ".png"]),
            ui.input_select("sample_select", "O elige una imagen de ejemplo", choices=[""] + sample_images),
            ui.input_action_button("detect", "🔍 Detectar aleta", class_="btn btn-primary"),
            ui.output_text("prediction"),
        ),
        ui.output_ui("original_image"),
        ui.output_ui("crop_images")
    )
)

def server(input, output, session):
    current_image_path = reactive.Value(None)
    detection_result = reactive.Value(None)
    unique_id = reactive.Value("")

    def limpiar_directorios():
        for f in CROPS_DIR.glob("*"):
            f.unlink()
        for f in UPLOADS_DIR.glob("*"):
            f.unlink()

    @reactive.effect
    @reactive.event(input.detect)
    def run_detection():
        limpiar_directorios()
        detection_result.set(None)

        uid = uuid.uuid4().hex[:8]

        if input.sample_select():
            image_path = SAMPLES_DIR / input.sample_select()
            current_image_path.set(image_path)
            session.send_input_message("file1", {"value": None})
        elif input.file1():
            file = input.file1()[0]
            image_path = UPLOADS_DIR / f"uploaded_{uid}.jpg"
            shutil.copy(file["datapath"], image_path)
            current_image_path.set(image_path)
            session.send_input_message("sample_select", {"value": ""})
        else:
            return

        unique_id.set(uid)

        results = model(str(image_path))
        boxes = results[0].boxes.xyxy

        if len(boxes) == 0:
            detection_result.set(0)
            return

        img_cv = cv2.imread(str(image_path))
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = img_cv[y1:y2, x1:x2]
            crop_filename = f"crop_{uid}_{idx + 1}.jpg"
            cv2.imwrite(str(CROPS_DIR / crop_filename), crop)

        detection_result.set(len(boxes))

    @render.text
    def prediction():
        result = detection_result.get()
        if result is None:
            return ""
        elif result == 0:
            return "❌ No se detectaron aletas"
        else:
            return f"✔️ {result} aleta(s) detectada(s)"

    @render.ui
    def original_image():
        _ = unique_id.get()
        img_path = current_image_path.get()
        if img_path is None:
            return None
        rel_path = img_path.relative_to(WWW_DIR)
        return ui.div(
            ui.h4("Imagen original", style="font-weight: bold; margin-top: 20px;"),
            ui.img(src=str(rel_path).replace("\\", "/"),
                   style="width: 500px; margin: 10px; border: 2px solid #000;")
        )



    @render.ui
    def crop_images():
        _ = unique_id.get()
        crops = sorted(CROPS_DIR.glob("*.jpg"))
        if not crops:
            return ui.p("No hay recortes para mostrar")
        return ui.div(
            ui.h4("Aletas detectadas", style="font-weight: bold; margin-top: 30px;"),
            *[
                ui.img(
                    src=f"crops/{f.name}",
                    style="width: 300px; margin: 0 10px 20px 0; border: 1px solid #ccc;"
                    )
                for f in crops
                ]
            )
    

app = App(app_ui, server, static_assets=WWW_DIR)

if __name__ == "__main__":
    app.run()

# -*- coding: utf-8 -*-
"""
Created on Fri May 16 21:44:27 2025

@author: zaida
"""
# ==========================
# BLOQUE 1: IMPORTACIONES Y CONFIGURACIONES
# ==========================
import os
import shutil
import uuid
from pathlib import Path
import cv2
import nest_asyncio
from shiny import App, ui, render, reactive
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import functional as TF
from torchvision.models import resnet18, ResNet18_Weights

nest_asyncio.apply()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# BLOQUE 2: RUTAS Y CARPETAS
# ==========================
BASE_DIR = Path(__file__).parent.resolve()
WWW_DIR = BASE_DIR / "www"
CROPS_DIR = WWW_DIR / "crops"
SAMPLES_DIR = WWW_DIR / "samples"
UPLOADS_DIR = WWW_DIR / "uploads"


import gdown

model_side_id = "1yKhVzvl7Av0snWAT81S0K8khBdEeWpPR"
model_side_path = BASE_DIR / "model_side_weights.pth"

if not model_side_path.exists():
    print("Descargando model_side_weights.pth desde Google Drive...")
    url = f"https://drive.google.com/uc?id={model_side_id}"
    gdown.download(url, str(model_side_path), quiet=False)

'''
# Lista de archivos: nombre local => ID de Drive
drive_files = {
    "model_side_weights.pth": "16G4mYpOCu4XC-ifPBsagV6LqIsb6nAN2",
    "best8.pt": "12rIwiegBwzPtcRymS8Ry4Vuk1GQn4zmo",
    "cat_emb_left_DRIVE":  "1Qtb1KHkvEzqJsaK18pFOmhlloZD8WRY4",
    "cat_emb_right_DRIVE": "1vlDoChaZrdQNYBgUt_Alsp9811AgX6Bk",
    "best_triplet_left_R18_512_pesos": "1pJwL8o-FBiPymF5xFzOZOsxadyb9YdWb",
    "best_triplet_right_R18_512_pesos": "1CmNB5_Ut_W-utHkWB7pzXTK7XyPSDiyD"
}

# Descargar los archivos si no existen
for filename, file_id in drive_files.items():
    dest = BASE_DIR / filename
    if not dest.exists():
        print(f"Descargando {filename} desde Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest), quiet=False)
'''
for folder in [CROPS_DIR, SAMPLES_DIR, UPLOADS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

for f in CROPS_DIR.glob("*"):
    f.unlink()
for f in UPLOADS_DIR.glob("*"):
    f.unlink()

# ==========================
# BLOQUE 3: MODELO YOLO (DETECCIÓN)
# ==========================
model = YOLO(str(BASE_DIR / "best8.pt"))
model.conf = 0.55
model.iou = 0.4

# ==========================
# BLOQUE 4: MODELO DE CLASIFICACIÓN DE LADO
# ==========================
model_side = models.resnet50(pretrained=False)
num_ftrs = model_side.fc.in_features
model_side.fc = nn.Linear(num_ftrs, 2)
model_side.to(device)
model_side.load_state_dict(torch.load(str(BASE_DIR / "model_side_weights.pth"), map_location=device))
model_side.eval()

transforma_lado = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_side(image_path):
    image = Image.open(image_path).convert('RGB')
    transformed_image = transforma_lado(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_side(transformed_image)
        _, pred = torch.max(output, 1)
    return 'Left' if pred.item() == 0 else 'Right'

# ==========================
# BLOQUE 5: MODELOS SIAMESA Y PREDICCIÓN KNN
# ==========================
class ResizeWithPadding:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        target_h, target_w = self.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w))
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top
        img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)
        return img

resize_pad = ResizeWithPadding((224, 224))
transform_siamese = transforms.Compose([
    resize_pad,
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512, normalize=True):
        super(EmbeddingNet, self).__init__()
        self.normalize = normalize
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

def get_embedding(model, image_path, transform, device):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        embedding = model(img_tensor).squeeze(0)
    return embedding.cpu().numpy()

def predict_with_dynamic_knn(image_path, model, transform, catalog_pkl_path, device, k=5, metric='cosine'):
    with open(catalog_pkl_path, "rb") as f:
        catalog = pickle.load(f)
    class_ids = list(catalog.keys())
    embeddings = np.stack([catalog[c]['embedding'].numpy() for c in class_ids])
    image_paths_dict = {c: catalog[c]['image_path'] for c in class_ids}
    query_embedding = get_embedding(model, image_path, transform, device).reshape(1, -1)
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(query_embedding)
    neighbors = [class_ids[i] for i in indices[0]]
    image_matches = [image_paths_dict[class_ids[i]] for i in indices[0]]
    return list(zip(neighbors, distances[0], image_matches))


triplet_left = EmbeddingNet(embedding_dim=512, normalize=True).to(device)
triplet_right = EmbeddingNet(embedding_dim=512, normalize=True).to(device)

triplet_left.load_state_dict(torch.load(BASE_DIR / "best_triplet_left_R18_512_pesos.pth", map_location=device))
triplet_right.load_state_dict(torch.load(BASE_DIR / "best_triplet_right_R18_512_pesos.pth", map_location=device))

triplet_left.eval()
triplet_right.eval()


LEFT_CATALOG_PATH = BASE_DIR / "cat_emb_left_DRIVE.pkl"
RIGHT_CATALOG_PATH = BASE_DIR / "cat_emb_right_DRIVE.pkl"

import json

with open(BASE_DIR / "drive_image_map_right.json", "r") as f:
    image_map_right = json.load(f)

with open(BASE_DIR / "drive_image_map_left.json", "r") as f:
    image_map_left = json.load(f)

def get_drive_image_url(image_name, side):
    image_map = image_map_left if side == "Left" else image_map_right
    image_id = image_map.get(image_name)
    if image_id:
        # Cambiamos la URL al dominio correcto de imagen directa
        return f"https://drive.google.com/file/d/{image_id}/preview"
    return None

# ==========================
# BLOQUE 6: INTERFAZ DE USUARIO (UI)
# ==========================
sample_images = [f.name for f in SAMPLES_DIR.glob("*.jpg")]

app_ui = ui.page_fluid(
    ui.tags.style("""
        body {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .shiny-input-container label,
        .shiny-output-container,
        h1, h2, h3, h4, h5, h6, p {
            color: #f0f0f0;
        }
        .shiny-text-output, .shiny-text-output p {
            color: #f0f0f0 !important;
        }
        .btn.btn-primary {
            background-color: #0d6efd;
            border-color: #0d6efd;
        }
        .btn.btn-primary:hover {
            background-color: #0b5ed7;
            border-color: #0a58ca;
        }
        input[type="file"]::file-selector-button {
            color: #ffffff;
            background-color: #444;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
        }
        input[type="file"]::file-selector-button {
            background-color: #20c997;
            color: white;
            border: none;
            padding: 6px 12px;
            cursor: pointer;
        }
        input[type="file"]::file-selector-button:hover {
            background-color: #ffc107;
            cursor: pointer;
        }
    """),
   
    ui.tags.div(
        ui.tags.img(src="logo.png", style="height: 110px; margin-right: 15px;"),
        ui.tags.h1("ORCAFIN-ID: Identificación de orcas",
                   style="font-family: 'Roboto', sans-serif; font-size: 2.3em; margin: 0;"),
        style="display: flex; align-items: center; margin-bottom: 20px;"
    ),
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


# ==========================
# BLOQUE 7: SERVIDOR SHINY
# ==========================
def server(input, output, session):
    current_image_path = reactive.Value(None)
    detection_result = reactive.Value(None)
    unique_id = reactive.Value("")
    crop_sides = reactive.Value([])

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
        crop_sides.set([])
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
        sides = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = img_cv[y1:y2, x1:x2]
            crop_filename = f"crop_{uid}_{idx + 1}.jpg"
            crop_path = CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)
            side = predict_side(str(crop_path))

            if side == "Left":
                preds = predict_with_dynamic_knn(str(crop_path), triplet_left, transform_siamese, LEFT_CATALOG_PATH, device, metric='cosine')
            else:
                preds = predict_with_dynamic_knn(str(crop_path), triplet_right, transform_siamese, RIGHT_CATALOG_PATH, device, metric='cosine')
            
            sides.append((crop_filename, side, preds))

        crop_sides.set(sides)
        detection_result.set(len(sides))

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
            ui.img(src=str(rel_path).replace("\\", "/"), style="width: 500px; margin: 10px; border: 2px solid #000;")
        )


    @render.ui
    def crop_images():
        _ = unique_id.get()
        crops_info = crop_sides.get()
        if not crops_info:
            return ui.p("No hay recortes para mostrar")

        elements = [ui.h4("Aletas detectadas", style="font-weight: bold; margin-top: 30px;")]

        for idx, (filename, side, preds) in enumerate(crops_info):
            crop_img = ui.img(src=f"crops/{filename}", style="width: 300px; border: 1px solid #ccc;")
            pred_title = ui.h5("Predicción de identificación de individuo - TOP 5", style="margin-top: 15px; font-weight: bold;")

            pred_blocks = []
        
            for i, (name, dist, match_path) in enumerate(preds):
                similarity_pct = (1 - dist) * 100

                if similarity_pct >= 90:
                    color = "#28a745"
                elif similarity_pct >= 60:
                    color = "#ffc107"
                else:
                    color = "#dc3545"

                # Extraer ID de Drive de la URL completa si es válida
                if isinstance(match_path, str) and "id=" in match_path:
                    image_id = match_path.split("id=")[-1]
                    preview_url = f"https://drive.google.com/file/d/{image_id}/preview"
                    iframe = ui.tags.iframe(src=preview_url, width="320", height="240", style="border: none;")
                    pred_blocks.append(
                        ui.div(
                            iframe,
                            ui.p(f"{i+1}. {name}", style="font-weight: bold; font-size: 1.2em;"),
                            ui.p(f"Similitud: {similarity_pct:.2f}%", style=f"color: {color}; font-size: 0.9em;"),
                            style="margin-right: 20px; text-align: center;"
                        )
                    )
                else:
                    pred_blocks.append(
                        ui.div(
                            ui.p(f"{i+1}. {name} (sin imagen válida)", style="font-weight: bold; color: red;"),
                            style="text-align: center;"
                        )
                    )

            elements.append(
                ui.div(
                    ui.h5(f"Aleta {idx + 1} - Lado: {side}"),
                    crop_img,
                    pred_title,
                    ui.div(*pred_blocks, style="display: flex; flex-direction: row; flex-wrap: wrap; gap: 20px;"),
                    style="margin-bottom: 40px;"
                )
            )

        return ui.div(*elements)


# ==========================
# BLOQUE 8: EJECUCIÓN
# ==========================
app = App(app_ui, server, static_assets=WWW_DIR)

if __name__ == "__main__":
    app.run()
    
    
    
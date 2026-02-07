#!/usr/bin/env python3
"""
Validación final DPU: inyectas una imagen y obtienes las coordenadas de los objetos.

Flujo:
  1. Carga imagen (fichero o --synthetic).
  2. Ejecuta detector (YOLO y/o OpenCV caras) para obtener coordenadas de objetos.
  3. Pasa la misma imagen por el pipeline DPU en Python (capa 0 + capa 1) y guarda refs.
  4. Escribe las coordenadas (y opcionalmente JSON).

Uso (desde la raíz del proyecto):
  python run_dpu_validation.py imagen.jpg
  python run_dpu_validation.py imagen.jpg -o resultado.json
  python run_dpu_validation.py --synthetic   # imagen sintética, sin detector
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import conv_bn_leaky

INPUT_H, INPUT_W = 416, 416
LAYER0_C_OUT = 32
LAYER0_STRIDE = 2
LAYER1_C_OUT = 64
LAYER1_STRIDE = 2
LAYER1_K = 3


def load_image_as_tensor(path=None, synthetic=False):
    """Imagen -> tensor INT8 (3, 416, 416)."""
    if synthetic or path is None:
        np.random.seed(42)
        return np.random.randint(-128, 128, (3, INPUT_H, INPUT_W), dtype=np.int8)
    path = Path(path)
    if not path.exists():
        return None
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        bgr = cv2.resize(bgr, (INPUT_W, INPUT_H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = (rgb.astype(np.int32) - 128).clip(-128, 127).astype(np.int8)
        return np.transpose(img, (2, 0, 1))
    except Exception:
        pass
    try:
        from PIL import Image
        pil = Image.open(path)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = pil.resize((INPUT_W, INPUT_H), Image.BILINEAR)
        arr = np.array(pil)
        img = (arr.astype(np.int32) - 128).clip(-128, 127).astype(np.int8)
        return np.transpose(img, (2, 0, 1))
    except Exception:
        pass
    return None


def run_yolo_detection(path):
    """YOLO sobre la imagen; devuelve lista de {x1,y1,x2,y2,class,confidence}."""
    out = []
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model(str(path), verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                out.append({
                    "x1": float(xyxy[0]), "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]), "y2": float(xyxy[3]),
                    "class": cls, "confidence": round(conf, 4),
                    "source": "yolo",
                })
        return out
    except Exception:
        return out


def run_face_detection(path):
    """OpenCV caras; devuelve lista de {x1,y1,x2,y2,class,confidence}."""
    out = []
    try:
        import cv2
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        img = cv2.imread(str(path))
        if img is None:
            return out
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            out.append({
                "x1": int(x), "y1": int(y),
                "x2": int(x + w), "y2": int(y + h),
                "class": 0, "confidence": 1.0, "source": "opencv_face",
            })
        return out
    except Exception:
        return out


def get_object_coordinates(image_path, use_yolo=True, use_faces=True):
    """
    Ejecuta detectores sobre la imagen y devuelve lista de cajas.
    Cada caja: {"x1", "y1", "x2", "y2", "class", "confidence", "source"}.
    """
    path = Path(image_path)
    if not path.exists():
        return []
    boxes = []
    if use_yolo:
        boxes = run_yolo_detection(path)
    if use_faces:
        faces = run_face_detection(path)
        # Evitar duplicados muy solapados con YOLO (opcional: merge)
        for f in faces:
            boxes.append(f)
    return boxes


def run_dpu_pipeline(img_tensor, num_layers=2):
    """Capa 0 (+ capa 1) con pesos placeholder; devuelve (layer0_out, layer1_out o None)."""
    np.random.seed(123)
    w0 = np.random.randint(-30, 30, (LAYER0_C_OUT, 3, 3, 3), dtype=np.int8)
    b0 = np.random.randint(-500, 500, LAYER0_C_OUT, dtype=np.int32)
    layer0_out = conv_bn_leaky(img_tensor, w0, b0, scale=0.01, stride=LAYER0_STRIDE)
    if num_layers < 2:
        return layer0_out, None
    w1 = np.random.randint(-30, 30, (LAYER1_C_OUT, LAYER0_C_OUT, LAYER1_K, LAYER1_K), dtype=np.int8)
    b1 = np.random.randint(-500, 500, LAYER1_C_OUT, dtype=np.int32)
    layer1_out = conv_bn_leaky(layer0_out, w1, b1, scale=0.01, stride=LAYER1_STRIDE)
    return layer0_out, layer1_out


def main():
    ap = argparse.ArgumentParser(
        description="Validación DPU: imagen -> coordenadas de objetos"
    )
    ap.add_argument("image_path", nargs="?", default=None, help="Ruta a la imagen")
    ap.add_argument("--synthetic", action="store_true", help="Usar imagen sintética (no hay coordenadas reales)")
    ap.add_argument("-o", "--output", default=None, metavar="JSON", help="Guardar resultado en JSON")
    ap.add_argument("--no-detector", action="store_true", help="No ejecutar detector (solo pipeline DPU)")
    ap.add_argument("--layers", type=int, default=2, choices=(1, 2), help="Número de capas DPU (1 o 2)")
    ap.add_argument("--quiet", action="store_true", help="Solo imprimir coordenadas (una línea por objeto: x1 y1 x2 y2 class conf)")
    args = ap.parse_args()

    quiet = getattr(args, "quiet", False)
    if not quiet:
        print("=" * 60)
        print("VALIDACION DPU: Imagen -> Coordenadas de objetos")
        print("=" * 60)

    # 1) Cargar imagen
    if args.synthetic or not args.image_path:
        if not quiet:
            print("\n[1] Imagen: sintetica (3, 416, 416)")
        img_tensor = load_image_as_tensor(synthetic=True)
        image_path = None
    else:
        image_path = Path(args.image_path)
        if not quiet:
            print(f"\n[1] Imagen: {image_path}")
        img_tensor = load_image_as_tensor(path=image_path)
        if img_tensor is None:
            print("ERROR: No se pudo cargar la imagen.", file=sys.stderr)
            return 1
    if not quiet:
        print(f"     Tensor: {img_tensor.shape} INT8")

    # 2) Coordenadas de objetos (detector)
    coordinates = []
    if not args.no_detector and image_path is not None:
        if not quiet:
            print("\n[2] Detector (YOLO + OpenCV caras)")
        coordinates = get_object_coordinates(image_path)
        if not quiet:
            print(f"     Objetos detectados: {len(coordinates)}")
            for i, box in enumerate(coordinates[:10]):
                print(f"       [{i+1}] x1={box['x1']:.0f} y1={box['y1']:.0f} x2={box['x2']:.0f} y2={box['y2']:.0f} "
                      f"class={box['class']} conf={box['confidence']:.2f} ({box.get('source','')})")
            if len(coordinates) > 10:
                print(f"       ... y {len(coordinates) - 10} mas")
    else:
        if not quiet:
            if args.synthetic or not image_path:
                print("\n[2] Detector: omitido (imagen sintetica o sin path)")
            else:
                print("\n[2] Detector: omitido (--no-detector)")
            print("     Coordenadas: ninguna (inyecta una imagen real para deteccion)")

    # 3) Pipeline DPU (Python)
    sim_out = PROJECT_ROOT / "sim_out"
    if sim_out.exists() and not sim_out.is_dir():
        sim_out = PROJECT_ROOT / "image_sim_out"
    sim_out.mkdir(parents=True, exist_ok=True)
    np.save(sim_out / "image_input_layer0.npy", img_tensor)

    if not quiet:
        print(f"\n[3] Pipeline DPU (Python, {args.layers} capas)")
    layer0_out, layer1_out = run_dpu_pipeline(img_tensor, num_layers=args.layers)
    np.random.seed(123)
    w0 = np.random.randint(-30, 30, (LAYER0_C_OUT, 3, 3, 3), dtype=np.int8)
    b0 = np.random.randint(-500, 500, LAYER0_C_OUT, dtype=np.int32)
    np.save(sim_out / "layer0_weights.npy", w0)
    np.save(sim_out / "layer0_bias.npy", b0)
    np.save(sim_out / "layer0_output_ref.npy", layer0_out)
    if not quiet:
        print(f"     Capa 0: {img_tensor.shape} -> {layer0_out.shape}")
    if layer1_out is not None:
        w1 = np.random.randint(-30, 30, (LAYER1_C_OUT, LAYER0_C_OUT, LAYER1_K, LAYER1_K), dtype=np.int8)
        b1 = np.random.randint(-500, 500, LAYER1_C_OUT, dtype=np.int32)
        np.save(sim_out / "layer1_weights.npy", w1)
        np.save(sim_out / "layer1_bias.npy", b1)
        np.save(sim_out / "layer1_output_ref.npy", layer1_out)
        if not quiet:
            print(f"     Capa 1: {layer0_out.shape} -> {layer1_out.shape}")
    if not quiet:
        print(f"     Ref guardadas en: {sim_out}")

    # 4) Salida: coordenadas
    coords_text = []
    for b in coordinates:
        coords_text.append({
            "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"],
            "class": b["class"], "confidence": b["confidence"],
            "source": b.get("source", ""),
        })
    if quiet:
        for b in coords_text:
            print(f"{b['x1']:.0f} {b['y1']:.0f} {b['x2']:.0f} {b['y2']:.0f} {b['class']} {b['confidence']:.4f}")
    else:
        print("\n" + "-" * 60)
        print("COORDENADAS DE OBJETOS (x1, y1, x2, y2 = esquinas de la caja)")
        print("-" * 60)
        if not coordinates:
            print("  (ninguna - usa una imagen real y sin --no-detector para ver detecciones)")
        else:
            for i, b in enumerate(coordinates):
                print(f"  Objeto {i+1}: x1={b['x1']:.0f} y1={b['y1']:.0f} x2={b['x2']:.0f} y2={b['y2']:.0f}  class={b['class']} conf={b['confidence']:.2f}")
        print("-" * 60)
        print("Pipeline DPU (Python): OK. Para validar RTL, compara con layer0_output_ref.npy / layer1_output_ref.npy")
        print("=" * 60)

    # JSON
    result = {
        "image": str(image_path) if image_path else "synthetic",
        "tensor_shape": list(img_tensor.shape),
        "object_coordinates": coords_text,
        "num_objects": len(coords_text),
        "dpu_pipeline_layers": args.layers,
        "dpu_ref_dir": str(sim_out),
    }
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if not quiet:
            print(f"\nResultado guardado en: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

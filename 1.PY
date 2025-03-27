import cv2
import torch
import numpy as np
import pandas as pd
import requests

# Diccionario de traducción inglés-español
traducciones = {
    'person': 'persona',
    'bicycle': 'bicicleta',
    'car': 'auto',
    'motorcycle': 'motocicleta',
    'airplane': 'avión',
    'bus': 'autobús',
    'train': 'tren',
    'truck': 'camión',
    'boat': 'bote',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante',
    'stop sign': 'señal de stop',
    'parking meter': 'parquímetro',
    'bench': 'banco',
    'bird': 'pájaro',
    'cat': 'gato',
    'dog': 'perro',
    'horse': 'caballo',
    'sheep': 'oveja',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'oso',
    'zebra': 'cebra',
    'giraffe': 'jirafa',
    'backpack': 'mochila',
    'umbrella': 'paraguas',
    'handbag': 'bolso',
    'tie': 'corbata',
    'suitcase': 'maleta',
    'frisbee': 'frisbee',
    'skis': 'esquís',
    'snowboard': 'snowboard',
    'sports ball': 'pelota',
    'kite': 'cometa',
    'baseball bat': 'bate de béisbol',
    'baseball glove': 'guante de béisbol',
    'skateboard': 'patineta',
    'surfboard': 'tabla de surf',
    'tennis racket': 'raqueta de tenis',
    'bottle': 'botella',
    'wine glass': 'copa de vino',
    'cup': 'taza',
    'fork': 'tenedor',
    'knife': 'cuchillo',
    'spoon': 'cuchara',
    'bowl': 'cuenco',
    'banana': 'plátano',
    'apple': 'manzana',
    'sandwich': 'sándwich',
    'orange': 'naranja',
    'broccoli': 'brócoli',
    'carrot': 'zanahoria',
    'hot dog': 'perrito caliente',
    'pizza': 'pizza',
    'donut': 'dona',
    'cake': 'pastel',
    'chair': 'silla',
    'couch': 'sofá',
    'potted plant': 'planta en maceta',
    'bed': 'cama',
    'dining table': 'mesa de comedor',
    'toilet': 'inodoro',
    'tv': 'televisor',
    'laptop': 'portátil',
    'mouse': 'ratón',
    'remote': 'control remoto',
    'keyboard': 'teclado',
    'cell phone': 'teléfono móvil',
    'microwave': 'microondas',
    'oven': 'horno',
    'toaster': 'tostadora',
    'sink': 'fregadero',
    'refrigerator': 'refrigerador',
    'book': 'libro',
    'clock': 'reloj',
    'vase': 'florero',
    'scissors': 'tijeras',
    'teddy bear': 'oso de peluche',
    'hair drier': 'secador de pelo',
    'toothbrush': 'cepillo de dientes'
}

def check_internet():
    try:
        requests.get('https://google.com', timeout=5)
        return True
    except:
        return False

def load_yolo_model():
    if not check_internet():
        print("❌ Error: No hay conexión a Internet")
        return None
    
    try:
        print("🔍 Descargando modelo YOLOv5... (Paciencia, primera vez puede tardar)")
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            'ultralytics/yolov5', 
            'yolov5s', 
            pretrained=True,
            force_reload=True,
            trust_repo=True
        )
        print("✅ Modelo cargado correctamente")
        return model
    except Exception as e:
        print(f"❌ Error crítico al cargar el modelo: {str(e)}")
        print("💡 Soluciones posibles:")
        print("1. Verifica tu conexión a Internet")
        print("2. Ejecuta: pip install --upgrade torch torchvision")
        print("3. Intenta nuevamente")
        return None

def detect_objects(frame, model):
    if model is None:
        return frame, []
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.pandas().xyxy[0]
        objects = []
        
        for _, det in detections.iterrows():
            if det['confidence'] > 0.4:
                x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                
                # Traducir el nombre del objeto
                nombre_objeto = det['name']
                nombre_espanol = traducciones.get(nombre_objeto, nombre_objeto)
                
                # Dibujar bounding box con nombre en español
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{nombre_espanol} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                objects.append({
                    'class': nombre_espanol,
                    'confidence': float(det['confidence']),
                    'bbox': (x1, y1, x2, y2)
                })
        
        return frame, objects
    
    except Exception as e:
        print(f"⚠️ Error durante detección: {str(e)}")
        return frame, []

def main():
    print("\n" + "="*50)
    print("🚀 DETECTOR DE OBJETOS CON YOLOv5 (ESPAÑOL)")
    print("="*50 + "\n")
    
    model = load_yolo_model()
    if model is None:
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    print("\n🎥 Cámara iniciada - Presiona 'Q' para salir\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Error: No se pudo leer el frame de la cámara")
                break
                
            frame, objects = detect_objects(frame, model)
            
            # Mostrar objetos detectados en consola (en español)
            if objects:
                print("\nObjetos detectados:")
                for obj in objects:
                    print(f"- {obj['class']} (Confianza: {obj['confidence']:.2f})")
            
            cv2.imshow('Detección de Objetos - YOLOv5', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Programa terminado correctamente")

if __name__ == "__main__":
    main()
import os
import torch
from matplotlib import pyplot as plt
from PIL import Image

# Função para treinar o modelo YOLOv5
def treinar_modelo(data_path, epochs=50, batch_size=8):
    os.system(f"python train.py --img 512 --batch {batch_size} --epochs {epochs} --data {data_path} --weights yolov5s.pt")

# Função para detectar objetos usando o modelo YOLOv5 treinado
def detectar_objetos(img_paths, modelo_path='runs/train/exp/weights/best.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=modelo_path)
    for img_path in img_paths:
        results = model(img_path)
        results.print()
        results.show()
        results.save()  # Salvar resultados na pasta 'runs'

# Função para obter caminhos das imagens de uma classe específica
def get_img_paths(base_path, class_name, img_count=30):
    return [f'{base_path}/{class_name}/image_{i}.jpg' for i in range(img_count)]

# Exemplo de uso
if __name__ == "__main__":
    # Defina o caminho para o arquivo de dados e imagens para treino
    data_path = 'dataset.yaml'  # Certifique-se de criar o arquivo YAML com a configuração correta

    # Treine o modelo
    treinar_modelo(data_path)

    # Defina as classes e o caminho base para as imagens
    classes = ['cat', 'dog']
    base_path = 'D:/Git/deteccao_objetos_yolo/train/images'

    # Detecte objetos nas imagens de cada classe
    for class_name in classes:
        img_paths = get_img_paths(base_path, class_name)
        detectar_objetos(img_paths)

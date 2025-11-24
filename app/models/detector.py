import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class AIDetector:
    def __init__(self):
        # Caminho dinâmico para o modelo
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "ai_detector_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo não encontrado em: {model_path}\n"
                f"Coloque o arquivo ai_detector_model.pth dentro de app/models/"
            )

        # Carregar arquitetura
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, 2
        )

        # Pesos
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # Transformações
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.labels = ["IA", "REAL"]

    def predict(self, image: Image.Image):
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)

        probs = torch.softmax(output, dim=1)[0]

        return {
            "ai_probability": float(probs[0]),
            "real_probability": float(probs[1]),
            "predicted": self.labels[torch.argmax(probs).item()]
        }


detector = AIDetector()

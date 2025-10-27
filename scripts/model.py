"""
Архитектура нейронной сети для предсказания калорийности
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
from transformers import AutoModel


class CaloriePredictor(nn.Module):
    """Модель для предсказания калорийности блюда"""
    
    def __init__(
        self,
        num_ingredients: int = 30,
        max_ingredients: int = 30,
        image_embedding_dim: int = 512,
        ingredient_embedding_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3
    ):
        """
        Args:
            num_ingredients: количество типов ингредиентов
            max_ingredients: максимальное количество ингредиентов в блюде
            image_embedding_dim: размерность эмбеддинга изображения
            ingredient_embedding_dim: размерность эмбеддинга ингредиентов
            hidden_dims: размерности скрытых слоев
            dropout: вероятность dropout
        """
        super(CaloriePredictor, self).__init__()
        
        self.max_ingredients = max_ingredients
        
        # Извлечение признаков из изображения
        # Используем предобученный ResNet
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Фиксируем веса для быстрого обучения
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Адаптированный слой для изображения
        self.image_adapter = nn.Sequential(
            nn.Linear(2048, image_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BERT модель для обработки ингредиентов
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        # Фиксируем веса BERT для начала
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # Обработка выходных эмбеддингов BERT
        self.ingredient_processor = nn.Sequential(
            nn.Linear(768, 256),  # BERT base produces 768-dim embeddings
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, ingredient_embedding_dim),
            nn.ReLU()
        )
        
        # Обработка массы
        self.mass_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
        
        # Объединение всех признаков
        input_dim = image_embedding_dim + ingredient_embedding_dim + 32
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Финальный слой для регрессии
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion = nn.Sequential(*layers)
        
    def forward(
        self,
        image: torch.Tensor,
        ingredients_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mass: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image: тензор изображения (B, C, H, W)
            ingredients_ids: BERT токенизированные ингредиенты (B, seq_len)
            attention_mask: маска внимания для BERT (B, seq_len)
            mass: тензор массы (B, 1)
            
        Returns:
            calories: предсказанная калорийность (B, 1)
        """
        # Извлечение признаков из изображения
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        image_embedding = self.image_adapter(image_features)
        
        # Обработка ингредиентов через BERT
        bert_output = self.bert_model(
            input_ids=ingredients_ids,
            attention_mask=attention_mask
        )
        # Берем [CLS] токен (первый токен) для представления всего текста
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # (B, 768)
        ingredient_embedding = self.ingredient_processor(cls_embedding)  # (B, ingredient_embedding_dim)
        
        # Обработка массы
        mass_embedding = self.mass_encoder(mass)
        
        # Объединение признаков
        combined = torch.cat([image_embedding, ingredient_embedding, mass_embedding], dim=1)
        
        # Предсказание калорийности
        calories = self.fusion(combined)
        
        return calories
    
    def unfreeze_image_encoder(self):
        """Разблокировать обучение весов image_encoder"""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
    
    def unfreeze_bert(self):
        """Разблокировать обучение весов BERT"""
        for param in self.bert_model.parameters():
            param.requires_grad = True


def create_model(
    num_ingredients: int = 30,
    max_ingredients: int = 30,
    image_embedding_dim: int = 512,
    ingredient_embedding_dim: int = 64,
    hidden_dims: Tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.3
) -> CaloriePredictor:
    """
    Создать модель
    
    Args:
        num_ingredients: количество типов ингредиентов
        max_ingredients: максимальное количество ингредиентов в блюде
        image_embedding_dim: размерность эмбеддинга изображения
        ingredient_embedding_dim: размерность эмбеддинга ингредиентов
        hidden_dims: размерности скрытых слоев
        dropout: вероятность dropout
        
    Returns:
        CaloriePredictor: модель
    """
    model = CaloriePredictor(
        num_ingredients=num_ingredients,
        max_ingredients=max_ingredients,
        image_embedding_dim=image_embedding_dim,
        ingredient_embedding_dim=ingredient_embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )
    
    return model


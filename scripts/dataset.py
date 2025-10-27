"""
Dataset для загрузки и обработки данных о блюдах
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple
import warnings
from transformers import AutoTokenizer
warnings.filterwarnings('ignore')


class CalorieDataset(Dataset):
    """Dataset для предсказания калорийности блюд"""
    
    def __init__(
        self,
        dish_df: pd.DataFrame,
        ingredients_df: pd.DataFrame,
        images_dir: str,
        split: str = 'train',
        transform: transforms.Compose = None,
        max_ingredients: int = 30,
        tokenizer: AutoTokenizer = None
    ):
        """
        Args:
            dish_df: DataFrame с информацией о блюдах
            ingredients_df: DataFrame с ингредиентами
            images_dir: путь к директории с изображениями
            split: 'train' или 'test'
            transform: трансформации для изображений
            max_ingredients: максимальное количество ингредиентов
            tokenizer: BERT токенизатор
        """
        self.dish_df = dish_df[dish_df['split'] == split].copy()
        self.ingredients_df = ingredients_df
        self.images_dir = images_dir
        self.transform = transform
        self.max_ingredients = max_ingredients
        
        # Инициализация токенизатора
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        
        # Создаем словарь ID ингредиента -> название
        self.id_to_name = {}
        for _, row in ingredients_df.iterrows():
            self.id_to_name[f"ingr_{row['id']:010d}"] = row['ingr'].lower()
        
        self.num_ingredients = len(self.id_to_name)
        
    def __len__(self) -> int:
        return len(self.dish_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            idx: индекс элемента
            
        Returns:
            image: тензор изображения
            ingredients_ids: BERT токенизированные ингредиенты
            attention_mask: маска внимания для BERT
            mass: масса блюда
            calories: калорийность блюда
        """
        row = self.dish_df.iloc[idx]
        
        # Загрузка изображения
        dish_id = row['dish_id']
        image_path = os.path.join(self.images_dir, dish_id, 'rgb.png')
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Обработка ингредиентов - создаем текст для BERT токенизатора
        ingredient_ids = row['ingredients'].split(';')
        ingredient_names = []
        
        for ingr_id in ingredient_ids[:self.max_ingredients]:
            if ingr_id in self.id_to_name:
                ingredient_names.append(self.id_to_name[ingr_id])
        
        # Создаем строку с ингредиентами, разделенными запятыми
        ingredient_text = ', '.join(ingredient_names) if ingredient_names else ''
        
        # Токенизируем с помощью BERT
        ingredients = self.tokenizer(
            ingredient_text,
            padding='max_length',
            truncation=True,
            max_length=128,  # Максимальная длина для BERT
            return_tensors='pt'
        )
        
        # Извлекаем input_ids и attention_mask
        ingredients_ids = ingredients['input_ids'].squeeze(0)  # (seq_len,)
        attention_mask = ingredients['attention_mask'].squeeze(0)  # (seq_len,)
        
        # Масса и калорийность
        mass = torch.FloatTensor([row['total_mass']])
        calories = row['total_calories']
        
        return image, ingredients_ids, attention_mask, mass, calories


def get_default_transforms(image_size: int = 224, is_train: bool = True):
    """
    Получить трансформации для изображений
    
    Args:
        image_size: размер изображения
        is_train: флаг тренировочного режима (включает аугментации)
        
    Returns:
        transforms.Compose: трансформации
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_data(
    data_dir: str = 'nutrition/data',
    split: str = 'train'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загрузить данные из CSV файлов
    
    Args:
        data_dir: путь к директории с данными
        split: 'train' или 'test'
        
    Returns:
        dish_df, ingredients_df
    """
    dish_df = pd.read_csv(os.path.join(data_dir, 'dish.csv'))
    ingredients_df = pd.read_csv(os.path.join(data_dir, 'ingredients.csv'))
    
    return dish_df, ingredients_df


def create_ingredient_mapping(dish_df: pd.DataFrame, max_ingredients: int = 30) -> dict:
    """
    Создать словарь для маппинга ID ингредиентов в индексы
    
    Args:
        dish_df: DataFrame с информацией о блюдах
        max_ingredients: максимальное количество ингредиентов
        
    Returns:
        Словарь {ingredient_id: index}
    """
    unique_ids = set()
    for _, row in dish_df.iterrows():
        for ingr_id in row['ingredients'].split(';'):
            if ingr_id.startswith('ingr_'):
                idx_num = int(ingr_id.split('_')[-1])
                unique_ids.add(idx_num)
    
    sorted_ids = sorted(unique_ids)[:max_ingredients]
    return {ingr_id: idx for idx, ingr_id in enumerate(sorted_ids)}


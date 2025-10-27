"""
Утилиты для обучения модели
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional
import random

from scripts.dataset import CalorieDataset, get_default_transforms, load_data
from scripts.model import create_model


def set_seed(seed: int = 42):
    """Установить seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Вычислить метрики
    
    Args:
        predictions: предсказания модели
        targets: реальные значения
        
    Returns:
        Dict с метриками
    """
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - targets) / targets)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Обучение модели за одну эпоху
    
    Args:
        model: модель
        dataloader: загрузчик данных
        criterion: функция потерь
        optimizer: оптимизатор
        device: устройство (CPU/GPU)
        
    Returns:
        Dict с метриками
    """
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for images, ingredients_ids, attention_masks, masses, calories in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        ingredients_ids = ingredients_ids.to(device)
        attention_masks = attention_masks.to(device)
        masses = masses.to(device)
        calories = calories.to(device)
        
        # Forward pass
        outputs = model(images, ingredients_ids, attention_masks, masses).squeeze()
        
        # Вычисление потерь
        loss = criterion(outputs, calories)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Сохранение метрик
        running_loss += loss.item()
        all_predictions.extend(outputs.detach().cpu().numpy())
        all_targets.extend(calories.cpu().numpy())
    
    # Вычисление метрик
    avg_loss = running_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss
    
    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Валидация модели за одну эпоху
    
    Args:
        model: модель
        dataloader: загрузчик данных
        criterion: функция потерь
        device: устройство (CPU/GPU)
        
    Returns:
        Dict с метриками
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, ingredients_ids, attention_masks, masses, calories in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            ingredients_ids = ingredients_ids.to(device)
            attention_masks = attention_masks.to(device)
            masses = masses.to(device)
            calories = calories.to(device)
            
            # Forward pass
            outputs = model(images, ingredients_ids, attention_masks, masses).squeeze()
            
            # Вычисление потерь
            loss = criterion(outputs, calories)
            
            # Сохранение метрик
            running_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(calories.cpu().numpy())
    
    # Вычисление метрик
    avg_loss = running_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss
    
    return metrics


def train(config_path: str = 'config.yaml'):
    """
    Главная функция обучения модели
    
    Args:
        config_path: путь к конфигурационному файлу
    """
    # Загрузка конфигурации
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Установка seed
    set_seed(config['seed'])
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    # Загрузка данных
    print('\nЗагрузка данных...')
    dish_df, ingredients_df = load_data(config['data']['data_dir'])
    
    # Создание датасетов
    print('Создание датасетов...')
    train_transform = get_default_transforms(config['data']['image_size'], is_train=True)
    val_transform = get_default_transforms(config['data']['image_size'], is_train=False)
    
    train_dataset = CalorieDataset(
        dish_df, ingredients_df,
        images_dir=config['data']['images_dir'],
        split='train',
        transform=train_transform,
        max_ingredients=config['data']['max_ingredients']
    )
    
    val_dataset = CalorieDataset(
        dish_df, ingredients_df,
        images_dir=config['data']['images_dir'],
        split='test',
        transform=val_transform,
        max_ingredients=config['data']['max_ingredients']
    )
    
    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    print(f'Размер обучающей выборки: {len(train_dataset)}')
    print(f'Размер тестовой выборки: {len(val_dataset)}')
    print(f'Количество уникальных ингредиентов: {train_dataset.num_ingredients}')
    
    # Создание модели
    print('\nСоздание модели...')
    model = create_model(
        num_ingredients=train_dataset.num_ingredients,
        max_ingredients=config['data']['max_ingredients'],
        image_embedding_dim=config['model']['image_embedding_dim'],
        ingredient_embedding_dim=config['model']['ingredient_embedding_dim'],
        hidden_dims=tuple(config['model']['hidden_dims']),
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Планировщик
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Обучение
    print('\nНачало обучения...')
    best_val_mae = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(config['training']['num_epochs']):
        print(f'\nЭпоха {epoch + 1}/{config["training"]["num_epochs"]}')
        
        # Обучение
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, MAE: {train_metrics["MAE"]:.4f}')
        train_history.append(train_metrics)
        
        # Валидация
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        print(f'Val - Loss: {val_metrics["loss"]:.4f}, MAE: {val_metrics["MAE"]:.4f}')
        val_history.append(val_metrics)
        
        # Планировщик
        scheduler.step(val_metrics['loss'])
        
        # Сохранение лучшей модели
        if val_metrics['MAE'] < best_val_mae:
            best_val_mae = val_metrics['MAE']
            torch.save(model.state_dict(), config['training']['model_save_path'])
            print(f'Лучшая модель сохранена (MAE: {best_val_mae:.4f})')
        
        # Разблокировка весов image_encoder после нескольких эпох
        if epoch == config['training']['unfreeze_epoch'] - 1:
            print('Разблокировка весов image_encoder...')
            model.unfreeze_image_encoder()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'] / 10,
                weight_decay=config['training']['weight_decay']
            )
    
    print(f'\nОбучение завершено! Лучший MAE на валидации: {best_val_mae:.4f}')
    
    return {
        'best_val_mae': best_val_mae,
        'train_history': train_history,
        'val_history': val_history
    }


if __name__ == '__main__':
    train()


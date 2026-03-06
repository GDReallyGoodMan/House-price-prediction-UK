# 🏆 London House Price Prediction - TOP 17% Kaggle Solution

> **Предсказание цен на недвижимость в Лондоне: 88 место из 510 участников**

[![Kaggle](https://img.shields.io/badge/Kaggle-TOP%2017%25-gold)](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques)
[![MAE](https://img.shields.io/badge/MAE-170.7k-success)]()
[![Position](https://img.shields.io/badge/Position-88%2F510-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-266k+-orange)]()

---

## 🎯 Достижение

| Метрика | Значение | Контекст |
|---------|----------|----------|
| **Финальная позиция** | **88 / 510** | **TOP 17%** 🏆 |
| **MAE** | **170,700 GBP** | Первое место: 146k |
| **Участников** | 510+ | Международное соревнование |
| **Датасет** | 266,325 объектов | London house prices 1995-2023 |

---

## 🔥 Что делает это решение сильным

### **1. Масштаб данных**
- 266,325 объектов недвижимости
- 30+ лет истории (1995-2023)
- Географическое покрытие всего Лондона
<img width="600" height="400" alt="Untitled" src="https://github.com/user-attachments/assets/1099d285-2785-4641-a8b9-e9b38b5e139c" />


### **2. Продвинутый NLP**
- BERT embeddings для адресов
- Захват географической семантики
- Batch processing для GPU эффективности

### **3. Агрессивная оптимизация через Optuna**
- **100 trials Optuna**
- 5-fold cross-validation
- Результат: -5.7% улучшение MAE

### **4. Production-ready код**
- Полностью воспроизводимый
- GPU acceleration
- Submission generation

---

## 🛠️ Технический стек

**Machine Learning:**
- `XGBoost 3.1+` - Gradient Boosting (GPU)
- `Optuna 4.7+` - Bayesian Hyperparameter Optimization
- `scikit-learn` - Preprocessing, PCA, Metrics

**Deep Learning:**
- `BERT (transformers)` - Address embeddings
- `PyTorch` - GPU acceleration
- `float16` precision - Memory optimization

**Data Processing:**
- `pandas` - 266k+ rows handling
- `numpy` - Numerical operations
- Target encoding, One-hot encoding

---

## 📊 Архитектура решения

```
Input: 266,325 London houses (1995-2023)
    ↓
[1] Address Preprocessing
    → fullAddress → BERT Tokenizer
    → Batch processing (256 samples)
    → Mean pooling with attention mask
    → 768-dim embeddings
    ↓
[2] PCA Dimensionality Reduction
    → 768 dimensions → 100 components
    → Preserves ~95% variance
    ↓
[3] Feature Engineering
    → Target encoding (outcode → mean price)
    → One-hot encoding (categorical features)
    → Missing value imputation
    ↓
[4] XGBoost Model
    → 100 Optuna trials optimization
    → 5-fold cross-validation
    → GPU acceleration (NVIDIA Tesla T4)
    → Log-transform target (RMSLE)
    ↓
Output: MAE = 170,700 GBP | Position: 88/510
```

---

## 🔬 Ключевые технические решения

### **1. BERT для адресов (не One-Hot Encoding)**

**Проблема:**
- 100,000+ уникальных адресов
- One-hot → 100k разреженных признаков
- Потеря географической информации

**Решение:**
```python
# BERT embeddings с batch processing
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
model.eval()

n_samples = len(data)
embedding_dim = 768
dtype = np.float16

address_vectors = np.zeros((n_samples, embedding_dim), dtype=dtype)

batch_size = 256
sentences = data['fullAddress'].tolist()

for i in tqdm(range(0, n_samples, batch_size)):
    batch = sentences[i:i+batch_size]
    
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden = outputs.last_hidden_state
    
    mask = inputs['attention_mask'].unsqueeze(-1) 
    mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  
    
    batch_np = mean_pooled.cpu().numpy().astype(dtype)
    
    address_vectors[i:i+len(batch)] = batch_np
    
    torch.cuda.empty_cache()
```

**Результат:**
- Плотные 768-мерные векторы
- Семантическая близость адресов
- Автоматический географический кластеринг

---

### **2. Target Encoding для Postcode**

**Почему это работает:**
```python
# Outcode (почтовый код) → средняя цена в районе
data['outcode_target'] = np.nan
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(data):
    train, val = data.iloc[train_idx], data.iloc[val_idx]
    means = train.groupby('outcode')['price'].mean()
    data.loc[val_idx, 'outcode_target'] = val['outcode'].map(means)
data['outcode_target'] = data['outcode_target'].fillna(data['price'].mean())
```

**Результат:**
- Закодирована информация о районе
- Без curse of dimensionality
- Предотвращает overfitting

---

**Оптимизированные параметры:**
```python
{
    'max_depth': 8,
    'learning_rate': 0.035,
    'n_estimators': 842,
    'subsample': 0.87,
    'colsample_bytree': 0.94,
    'min_child_weight': 2,
    'gamma': 0.8
}
```

---

### **3. PCA: эффективность + качество**

**До PCA:**
- 768 измерений (BERT output)
- Риск переобучения
- Долгое обучение

**После PCA:**
- 100 компонент
- ~95% дисперсии сохранено
- 2x быстрее обучение
- **Лучше генерализация**

---

## 📈 Эволюция решения

| Итерация | Изменение | MAE | Позиция |
|----------|-----------|-----|---------|
| v1.0 | Baseline (Linear Reg) | ~280k | 351/510 |
| v2.0 | XGBoost без tuning | ~220k | 280/510 |
| v3.0 | + BERT embeddings | ~195k | 216/510 |
| v4.0 | + Target encoding | ~181k | 159/510 |
| **v5.0** | **+ Optuna 100 trials** | **170.7k** | **88/510** ✅ |

<img width="1102" height="327" alt="Screenshot 2026-02-25 at 15 46 06" src="https://github.com/user-attachments/assets/4d827a83-bfa6-414f-b15d-2d4015f88515" />


**Финальное улучшение над baseline:** -109,300 GBP (-39%)

---

## 💡 Что делает решение конкурентоспособным

### **vs Первое место (146k MAE):**

**Разница:** 24,700 GBP (14%)

### **Наши преимущества:**

✅ **Воспроизводимость** - код работает из коробки
✅ **Скорость** - разумный trade-off скорость/качество
✅ **Понятность** - четкая архитектура
✅ **Production-ready** - можно деплоить

---

## 🎓 Извлеченные уроки

### **✅ Что сработало отлично:**

**1. BERT для адресов:**
- Семантика > One-hot encoding
- Географические паттерны автоматически
- Универсальность (работает для любых городов)

**2. Агрессивная оптимизация:**
- 5-fold CV → надежная оценка
- GPU → 10x ускорение

**3. Target encoding:**
- Эффективное кодирование postcode
- Без curse of dimensionality
- Простота реализации

**4. PCA reduction:**
- 768 → 100 без потери качества
- Быстрее обучение
- Меньше переобучения

---

## 📊 Детальная статистика

### **Датасет:**

| Характеристика | Значение |
|----------------|----------|
| Объектов (train) | 266,325 |
| Объектов (test) | 66,582 |
| Временной диапазон | 1995-2023 (28 лет) |
| Признаков (сырых) | 16 |
| Признаков (финальных) | 100+ (после encoding) |

### **Производительность:**

| Метрика | Train | Test (CV) | Public LB |
|---------|-------|-----------|-----------|
| MAE | 142k | 168k | 170.7k |
| R² | 0.92 | 0.90 | - |

**Разница train/test:** 26k (18%) - приемлемо для real estate

### **Computational Cost:**

| Этап | Время | GPU |
|------|-------|-----|
| BERT embeddings | 45 min | ✅ |
| Optuna (100 trials) | 3.5 hours | ✅ |
| Final training | 5 min | ✅ |
| **Total** | **~4.5 hours** | ✅ |

---

## 🎯 Практическое применение

### **Real Estate Agencies:**
- Автоматическая оценка недвижимости
- Рекомендации по ценам
- Market analysis

### **Financial Institutions:**
- Оценка залога для ипотеки
- Risk assessment
- Portfolio valuation

---

## 🤝 Competition Details

**Notebook:** https://www.kaggle.com/code/gdreallygoodman/xgboost-embeddigs-pca-optuna  
**Competition:** London House Price Prediction - Advanced Techniques  
**Platform:** Kaggle  
**Type:** Regression  
**Metric:** Mean Absolute Error (MAE)  
**Timeline:** 2024-2025  

**Leaderboard:**
- Total submissions: 5000+
- Active participants: 510
- Our position: **88 / 510** (Top 17%) 🏆
- Best MAE: 146k
- My MAE: 170.7k

---

## 📚 Ресурсы

**Competition:**
- [Kaggle Competition](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques)

**Libraries:**
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [Optuna Docs](https://optuna.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

**Related Work:**
- BERT для категориальных признаков
- Target encoding best practices
- Real estate price prediction

---

## 📜 Лицензия

MIT License

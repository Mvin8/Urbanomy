# Использование ParetoScenarioSelector

## Обзор

`ParetoScenarioSelector` — это инструмент для выбора оптимального варианта застройки на основе мнений трёх агентов (администрация, жители, инвесторы) с использованием LangChain и LangGraph.

## Установка зависимостей

```bash
pip install langchain langchain-core langgraph pydantic pandas
```

## Быстрый старт

### 1. Инициализация

```python
from urbanomy.methods.land_value_modeling.pareto_llm_selector import ParetoScenarioSelector
from langchain_community.llms import Ollama  # или другой LLM

# Инициализировать LLM
llm = Ollama(model="llama3", base_url="http://localhost:11434")

# Создать селектор
selector = ParetoScenarioSelector(
    llm=llm,
    debug=False,
    additional_context="Анализируем варианты развития территории в Гатчине"
)
```

### 2. Подготовка сценариев

Сценарии могут быть в формате:
- **DataFrame** (pandas)
- **Список словарей**
- **Список ParetoScenario объектов**

```python
import pandas as pd

scenarios = pd.DataFrame([
    {
        "scenario_id": "1_residential",
        "title": "Жилой комплекс",
        "summary": "Высокоплотная жилая застройка с социальной инфраструктурой",
        "residential": 0.7,
        "business": 0.1,
        "recreation": 0.2,
        "fsi": 3.0,
        "gsi": 0.45,
        "population": 10000,
        "land_value_gain": 50_000_000,
        "investor_npv": 30_000_000,
    },
    {
        "scenario_id": "2_mixed",
        "title": "Смешанное использование",
        "summary": "Сбалансированное сочетание жилья, офисов и парков",
        "residential": 0.5,
        "business": 0.3,
        "recreation": 0.2,
        "fsi": 2.0,
        "gsi": 0.35,
        "population": 6000,
        "land_value_gain": 45_000_000,
        "investor_npv": 35_000_000,
    },
    {
        "scenario_id": "3_office",
        "title": "Деловой центр",
        "summary": "Коммерческий комплекс с офисами и сервисами",
        "residential": 0.2,
        "business": 0.6,
        "recreation": 0.2,
        "fsi": 2.5,
        "gsi": 0.40,
        "population": 2000,
        "land_value_gain": 60_000_000,
        "investor_npv": 45_000_000,
    }
])
```

### 3. Выбор лучшего варианта

```python
# Простой выбор только ID
winner_id = selector.select_best(scenarios)
print(f"Победитель: {winner_id}")

# Получить полный анализ
result = selector.get_full_analysis(scenarios)
print(result)

# Вывести красивый отчет
report = selector.print_analysis(scenarios)
print(report)
```

## Примеры использования

### Пример 1: Анализ вариантов развития квартала

```python
from langchain_community.llms import Ollama

# Инициализация
llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.5)
selector = ParetoScenarioSelector(
    llm=llm,
    additional_context="Участок в центре Гатчины, площадь 2 гектара"
)

# Варианты развития
scenarios = [
    {
        "scenario_id": "opt1",
        "title": "Культурный центр",
        "summary": "Музей, театр, библиотека с общественными зелёными зонами",
        "pros": ["Улучшает культурную жизнь", "Привлекает туристов", "Создаёт рабочие места"],
        "cons": ["Низкая коммерческая доходность", "Требует государственной поддержки"],
        "metrics": {
            "cultural_impact": 9.0,
            "economic_roi": 3.0,
            "public_benefit": 8.5,
            "job_creation": 150,
        }
    },
    {
        "scenario_id": "opt2", 
        "title": "Жилой комплекс премиум",
        "summary": "Элитные апартаменты с парковками и сервисами",
        "pros": ["Высокая доходность", "Быстрая реализация", "Социальный статус"],
        "cons": ["Экономическая сегрегация", "Нагрузка на инфраструктуру"],
        "metrics": {
            "cultural_impact": 2.0,
            "economic_roi": 8.0,
            "public_benefit": 3.0,
            "job_creation": 80,
        }
    }
]

# Анализ
winner = selector.select_best(scenarios)
print(f"Рекомендуемый вариант: {winner}")

# Подробный отчет
print(selector.print_analysis(scenarios))
```

### Пример 2: Голосование агентов

```python
# Получить результаты голосования
result = selector.get_full_analysis(scenarios)

print("Голосование агентов:")
for vote in result.votes:
    print(f"\n{vote.stakeholder}:")
    print(f"  Выбран: {vote.selected_scenario_id}")
    print(f"  Уверенность: {vote.confidence:.0%}")
    print(f"  Ранжирование: {vote.ranking}")
    print(f"  Обоснование: {vote.rationale}")
    
print(f"\nИтоговое ранжирование: {result.ranking}")
print(f"Метод решения: {result.aggregation_method}")
```

### Пример 3: Дополнительные вопросы

```python
# Задать вопрос о победителе
qa_result = selector.ask_question(
    "Какие риски у выбранного варианта?",
    scenarios
)

print(qa_result.answer)
```

## Структура результатов

### ParetoVotingResult

```python
{
    "winner_scenario_id": str,           # ID выбранного варианта
    "winner_scenario": ParetoScenario,   # Полная информация победителя
    "votes": List[StakeholderVote],      # Голоса всех трёх агентов
    "ranking": List[str],                # Ранжирование по популярности
    "aggregation_method": str,           # Способ агрегации (Borda, majority vote и т.д.)
    "aggregate_confidence": float,       # Уверенность в результате [0..1]
}
```

### StakeholderVote

```python
{
    "stakeholder": str,                  # "administration" | "residents" | "investor"
    "selected_scenario_id": str,         # Выбранный ID
    "ranking": List[str],                # Ранжирование от лучшего к худшему
    "rationale": str,                    # Обоснование выбора
    "main_tradeoffs": List[str],        # Основные компромиссы
    "metrics_considered": List[str],     # Какие метрики использованы
    "confidence": float,                 # Уверенность [0..1]
}
```

## Расширенные параметры

### Параметры инициализации

```python
selector = ParetoScenarioSelector(
    llm=llm,
    debug=True,                          # Выводить отладочную информацию
    additional_context="Дополнительный контекст для агентов"
)
```

### Параметры анализа

```python
result = selector.get_full_analysis(
    scenarios,
    max_retries=2,                       # Максимум переделок при ошибках
    stakeholder_briefs={                 # Кастомные инструкции для агентов
        "administration": "Фокусируйся на...",
        "residents": "Думай о...",
        "investor": "Анализируй..."
    }
)
```

## Роли агентов

### Администрация (Administration)
Смотрит на:
- Общественную пользу
- Бюджетную устойчивость
- Налоговую базу
- Рабочие места
- Транспортную связность
- Реализуемость
- Социальную инфраструктуру
- Экологические риски
- Политическую приемлемость

### Жители (Residents)
Смотрят на:
- Качество жизни
- Доступность жилья и сервисов
- Шум и загрязнение
- Экологию
- Озеленение
- Безопасность
- Walkability
- Нагрузку на инфраструктуру
- Повседневный комфорт

### Инвесторы (Investor)
Смотрят на:
- Доходность
- Скорость реализации
- Спрос на рынке
- Ликвидность
- CAPEX/OPEX
- Риски
- Простоту согласований
- Коммерческий потенциал
- Вероятность окупаемости

## Интеграция с существующим кодом

```python
from urbanomy.methods.land_value_modeling.ga_mc_optimizer import DistrictProblem
from urbanomy.methods.land_value_modeling.pareto_llm_selector import ParetoScenarioSelector

# Получить Парето-фронт из оптимизатора
res = minimize(problem, algorithm, ('n_gen', 50), verbose=False)

# Подготовить сценарии
scenarios = pd.DataFrame([
    {
        "scenario_id": f"solution_{i}",
        "title": f"Решение {i}",
        "summary": f"Вариант {i} из Парето-фронта",
        "land_value_gain": -res.F[i][0],  # Первая целевая функция
        "investor_npv": -res.F[i][1],     # Вторая целевая функция
        **res.X[i]  # Параметры решения
    }
    for i in range(len(res.X))
])

# Анализ
selector = ParetoScenarioSelector(llm=llm)
winner = selector.select_best(scenarios)
print(f"Оптимальный вариант из Парето-фронта: {winner}")
```

## Советы по использованию

1. **Уточняйте контекст**: Добавляйте `additional_context` с информацией о территории, её истории, ограничениях.

2. **Метрики важны**: Включайте в DataFrame реальные числовые метрики — агенты их анализируют.

3. **Описания сценариев**: Хорошее описание в `summary` помогает агентам лучше понять вариант.

4. **Достаточно сценариев**: 3-7 вариантов обычно достаточно, более 10 может замедлить анализ.

5. **Температура LLM**: Используйте `temperature=0.3-0.5` для более детерминированных результатов.

## Теория: Парето-фронт и голосование

Парето-фронт — это множество решений, где улучшение одной целевой функции приводит к ухудшению другой. Селектор использует голосование трёх агентов для выбора из этого множества наиболее сбалансированного решения.

Метод агрегации:
1. Каждый агент ранжирует варианты по своим критериям
2. Применяется метод Борда (Borda count)
3. При равенстве используется уверенность агентов (confidence tie-breaker)
4. Результат: вариант с максимальным согласием между интересантами

## Решение проблем

### Ошибка: "Ollama connection refused"
Убедитесь, что сервис Ollama запущен:
```bash
ollama serve
```

### LLM возвращает невалидный JSON
Увеличьте `max_retries` или понизьте `temperature`:
```python
result = selector.get_full_analysis(scenarios, max_retries=3)
```

### Агенты выбирают разные варианты
Это нормально — это означает реальный компромисс между интересами. Селектор выбирает наиболее сбалансированный вариант.

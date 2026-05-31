# ServerBalancer

**Прототип балансировщика нагрузки**

Курс: Теория игр | Трек: Программная инженерия | Уровень: 3 курс

Автор: Соколов Глеб Константинович | Группа: ИТ-5 | Руководитель: Шилина Алла Владимировна

---

## Описание

Python-приложение, моделирующее пул серверов как набор очередей массового
обслуживания **M/M/1** и распределяющее входящий поток запросов методом
**линейного программирования** (min-max загрузки) так, чтобы минимизировать
время отклика бутылочного горлышка и не допустить перегрузки ни одного узла.
Сравнивается с наивными стратегиями (равномерный сплит / round-robin) и с
теоретическим оптимумом по среднему времени отклика.

## Математическая модель

| Компонент | Описание | Реализация |
|---|---|---|
| СМО (M/M/1) | Пуассоновский вход λᵢ, эксп. обслуживание μᵢ, расчёт ρ, P_wait, Lq, W | `core/queue_math.py` |
| Оптимизация (ЛП) | min z при λᵢ ≤ z·μᵢ, Σλᵢ = Λ, z ≤ ρ_max | `core/balancer.py` |
| Эталоны | равномерный/round-robin + выпуклый оптимум по W | `core/baselines.py` |
| Оценка качества | агрегирование метрик по системе | `core/evaluate.py` |
| Валидация (имитация) | независимая проверка формул на SimPy | `core/simulation.py` |

## Технологический стек

- **Ядро:** NumPy, SciPy, **PuLP** (CBC-солвер)
- **Имитация:** SimPy
- **Интерфейс:** **FastAPI + uvicorn** (REST), Rich (CLI)
- **Инфраструктура:** Pydantic (валидация), YAML (конфиги), Docker
- **SE-практики:** pytest, pytest-cov (покрытие 95%)

## Быстрый старт

```bash
https://github.com/GoDL1ghT/game-theory.git
cd game-theory/lab4
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Расчёт распределения (CLI)
python main.py --config config/scenario.yaml

# Эксперименты и графики
python experiments.py

# REST-сервис
uvicorn api.app:app --host 0.0.0.0 --port 8000
# затем: curl http://localhost:8000/health
```

Через Docker:

```bash
docker compose up --build
curl http://localhost:8000/health
```

## Структура проекта

```
lab4/
├── config/scenario.yaml      # пул серверов, поток, SLA
├── core/
│   ├── queue_math.py         # метрики M/M/1 (M/M/c как обобщение)
│   ├── balancer.py           # ЛП min-max на PuLP
│   ├── baselines.py          # эталоны + выпуклый оптимум
│   ├── evaluate.py           # агрегированные метрики
│   ├── models.py             # Pydantic-валидация
│   └── simulation.py         # имитация M/M/1 на SimPy
├── api/app.py                # FastAPI: /health, /balance
├── tests/                    # pytest (unit + integration)
├── experiments.py            # сценарии + графики
├── main.py                   # CLI (Rich)
├── Dockerfile, docker-compose.yml
├── requirements.txt
└── report.md                 # отчёт
```

## Тестирование

```bash
pytest tests/ -v --cov=core --cov=api --cov-report=term-missing
```

Минимальное покрытие ≥ 60%; фактическое — **95%**. Включены unit-тесты формул
СМО против аналитических значений, тесты корректности ЛП и интеграционные тесты
REST-эндпоинтов и имитационной модели.

## REST API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | health-check для контейнера |
| POST | `/balance` | распределение трафика и метрики СМО |


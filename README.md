# trade_exp_12

Исследовательская FastAPI-платформа для загрузки данных Bybit, визуальной проверки признаков и поиска состояний криптовалютного рынка.

## Требования

- Git
- Python 3.12 или 3.13
- Доступ к порту 8031

## Первичное развёртывание на Linux-сервере

```bash
cd ~
mkdir -p apps
cd apps
git clone https://github.com/vadfe/trade_exp_12.git
cd trade_exp_12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m ruff check .
python -m uvicorn trade_exp_12.main:app --host 0.0.0.0 --port 8031
```

Интерфейс будет доступен по адресу `http://SERVER_IP:8031`.

## Обновление сервера после коммита

```bash
cd ~/apps/trade_exp_12
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m ruff check .
python -m uvicorn trade_exp_12.main:app --host 0.0.0.0 --port 8031
```

Перед обновлением остановите старый процесс через `Ctrl+C`. Если приложение управляется systemd, вместо последней команды перезапустите соответствующий service.

Стартовые raw/processed datasets и SQLite-реестр хранятся в Git, поэтому после `git clone` или `git pull` повторная загрузка этих данных не требуется. Новые версии данных также попадут в следующий коммит, если явно добавить их в Git. Каталог `artifacts/` не хранится в репозитории.

## Первичный запуск на Windows

```powershell
git clone https://github.com/vadfe/trade_exp_12.git
cd trade_exp_12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m ruff check .
python -m trade_exp_12.main
```

Интерфейс: `http://127.0.0.1:8031`.

## Повторный запуск на Windows

```powershell
cd C:\path\to\trade_exp_12
.\.venv\Scripts\Activate.ps1
python -m trade_exp_12.main
```

## Эксперименты с моделью

Через веб-интерфейс: перейдите на страницу **Эксперименты** (`/experiments`), выберите параметры и нажмите «Запустить». Обучение происходит в фоновом режиме; прогресс отображается на странице задач.

Результаты сохраняются в `data/experiments/` (не попадают в Git). Каждый запуск содержит:
- `report.json` — полный отчёт с метриками, confusion matrix и историей эпох;
- `best_model.pt` — веса лучшей модели (по val accuracy).

Страница эксперимента отображает:
- графики loss/accuracy по эпохам;
- точность по классам (up / down / undefined);
- confusion matrix;
- гиперпараметры.

### Установка PyTorch

На CPU-сервере:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Если на сервере есть CUDA GPU:
```bash
pip install torch
```

## Структура зависимостей

- `requirements.txt` — runtime-зависимости сервера.
- `requirements-dev.txt` — runtime + pytest, respx и ruff.
- `pyproject.toml` — метаданные Python-пакета и допустимые диапазоны зависимостей.

Виртуальное окружение нельзя копировать между машинами или версиями Python. На каждой машине `.venv` создаётся заново.

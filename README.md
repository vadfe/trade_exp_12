# trade_exp_12

Исследовательская FastAPI-платформа для загрузки данных Bybit, визуальной проверки признаков и поиска состояний криптовалютного рынка.

## Требования

- Git
- Python 3.12 или 3.13
- Доступ к порту 8031

## Первичное развёртывание на Linux-сервере

```bash
cd /opt
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
cd /opt/trade_exp_12
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m ruff check .
python -m uvicorn trade_exp_12.main:app --host 0.0.0.0 --port 8031
```

Перед обновлением остановите старый процесс через `Ctrl+C`. Если приложение управляется systemd, вместо последней команды перезапустите соответствующий service.

Каталоги `data/` и `artifacts/` не хранятся в Git. `git pull` их не удаляет, поэтому загруженные свечи и processed datasets остаются на сервере.

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

## Структура зависимостей

- `requirements.txt` — runtime-зависимости сервера.
- `requirements-dev.txt` — runtime + pytest, respx и ruff.
- `pyproject.toml` — метаданные Python-пакета и допустимые диапазоны зависимостей.

Виртуальное окружение нельзя копировать между машинами или версиями Python. На каждой машине `.venv` создаётся заново.

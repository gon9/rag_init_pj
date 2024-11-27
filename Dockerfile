FROM python:3.12-slim-bookworm

WORKDIR /app

# Poetryのインストール
RUN pip install --no-cache-dir poetry

# 依存関係の定義ファイルをコピー
COPY pyproject.toml poetry.lock ./

# 依存関係のインストール
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

# アプリケーションのコピー
COPY . .

# 環境変数を設定
ENV PYTHONUNBUFFERED=1

CMD ["poetry", "run", "python", "src/app.py"]
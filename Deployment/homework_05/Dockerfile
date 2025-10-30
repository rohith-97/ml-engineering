FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
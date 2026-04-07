FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install \
      "numpy" \
      "pandas>=2.2.3" \
      "geopandas>=1.0.0,<2.0.0" \
      "shapely" \
      "catboost" \
      "libpysal" \
      "blocksnet==1.0.0.a9" && \
    pip install -e . --no-deps

EXPOSE 8765

CMD ["python", "-m", "urbanomy.qgis_land_value_service", "--config", "/config/service.json", "--host", "0.0.0.0", "--port", "8765"]

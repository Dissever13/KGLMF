FROM python:3.11-slim
RUN pip install --no-cache-dir pytest numpy scipy pandas sympy scikit-learn matplotlib networkx requests
ENV PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp
WORKDIR /tmp

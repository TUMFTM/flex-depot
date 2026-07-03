FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Pyomo resolves solver "gurobi" to the gurobipy-based LP file interface,
# so the pip package is sufficient (no full Gurobi installation needed).
# The gurobipy version must be covered by the license on the workstation.
# Keep the list below in sync with [project.dependencies] in pyproject.toml.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "pandas>=2.0" \
        "numpy>=1.24" \
        "pyomo>=6.6" \
        "tqdm>=4.67.1" \
        "plotly>=5.18" \
        "tzdata>=2024.1" \
        "openpyxl>=3.1.3" \
        "toml>=0.10.2" \
        "pydantic>=2.13.3" \
        "pydantic-settings>=2.14.0" \
        "gurobipy==13.0.2"

CMD ["python", "-c", "print('flex-depot image ready')"]

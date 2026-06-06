# Contributing to KANQAS-NISQ

## Development Setup

```bash
# Clone and install
git clone https://github.com/Aqasch/KANQAS_code.git
cd KANQAS_code
pip install -r requirements-dev.txt
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Follow PEP8 with 120-character line limit
- Use type hints for all function signatures
- Add docstrings following NumPy/Google style
- Run `ruff check .` before committing

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run smoke tests only
pytest tests/ -k "smoke" -v

# Run with coverage
pytest tests/ --cov=agents --cov=chemistry --cov=hardware
```

## Pull Request Process

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with clear description

## Adding New Molecules

Add new molecules in `chemistry/molecule.py` using qiskit-nature's PySCFDriver.

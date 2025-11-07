# 🤝 Contributing to Prediksi Cuaca Mikro

Terima kasih atas minat Anda untuk berkontribusi pada proyek **Prediksi Cuaca Mikro**! Kami sangat menghargai setiap kontribusi, baik berupa perbaikan bug, penambahan fitur, atau peningkatan dokumentasi.

## 📋 Daftar Isi

- [Code of Conduct](#code-of-conduct)
- [Cara Berkontribusi](#cara-berkontribusi)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)

---

## 📜 Code of Conduct

Proyek ini mengadopsi [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Dengan berpartisipasi, Anda diharapkan untuk mematuhi kode etik ini.

### Prinsip Utama:

- ✅ Bersikap ramah dan inklusif
- ✅ Menghormati pendapat dan pengalaman yang berbeda
- ✅ Menerima kritik konstruktif dengan lapang dada
- ✅ Fokus pada apa yang terbaik untuk komunitas
- ✅ Menunjukkan empati terhadap anggota komunitas lainnya

---

## 🎯 Cara Berkontribusi

### 1. Melaporkan Bug 🐛

Jika Anda menemukan bug, silakan buat issue dengan:

- **Judul yang jelas dan deskriptif**
- **Langkah-langkah untuk mereproduksi bug**
- **Hasil yang diharapkan vs yang terjadi**
- **Screenshot** (jika relevan)
- **Environment details** (OS, Python version, dll)

**Template Bug Report:**
```markdown
## Deskripsi Bug
[Jelaskan bug dengan jelas]

## Langkah Reproduksi
1. Jalankan '...'
2. Klik pada '...'
3. Scroll ke '...'
4. Lihat error

## Hasil yang Diharapkan
[Apa yang seharusnya terjadi]

## Hasil Aktual
[Apa yang sebenarnya terjadi]

## Screenshot
[Tambahkan screenshot jika membantu]

## Environment
- OS: [e.g., Windows 11]
- Python: [e.g., 3.10.5]
- Browser: [jika relevan]
```

### 2. Mengusulkan Fitur Baru ✨

Jika Anda memiliki ide untuk fitur baru:

- **Buat issue dengan label "enhancement"**
- **Jelaskan masalah yang ingin dipecahkan**
- **Deskripsikan solusi yang Anda usulkan**
- **Pertimbangkan alternatif yang sudah Anda pikirkan**

**Template Feature Request:**
```markdown
## Fitur yang Diusulkan
[Deskripsi singkat fitur]

## Motivasi
[Mengapa fitur ini dibutuhkan?]

## Deskripsi Detail
[Penjelasan lengkap tentang fitur]

## Alternatif yang Dipertimbangkan
[Solusi alternatif lain yang sudah dipikirkan]

## Informasi Tambahan
[Konteks, screenshot, atau informasi lainnya]
```

### 3. Memperbaiki Dokumentasi 📚

Dokumentasi yang baik sangat penting! Kontribusi untuk:

- Memperbaiki typo atau kesalahan grammar
- Menambahkan contoh penggunaan
- Memperjelas instruksi yang membingungkan
- Menambahkan dokumentasi untuk fitur baru

### 4. Berkontribusi Kode 💻

Lihat section [Pull Request Process](#pull-request-process) di bawah.

---

## 🛠️ Development Setup

### 1. Fork Repository

Klik tombol "Fork" di bagian kanan atas halaman GitHub.

### 2. Clone Repository

```bash
git clone https://github.com/your-username/prediksi-cuaca-mikro.git
cd prediksi-cuaca-mikro
```

### 3. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 5. Setup Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

### 6. Buat Branch Baru

```bash
git checkout -b feature/nama-fitur-anda
# atau
git checkout -b fix/nama-bug-yang-diperbaiki
```

---

## 🔄 Pull Request Process

### 1. Update Code Anda

Pastikan code Anda:

- ✅ Mengikuti [Coding Standards](#coding-standards)
- ✅ Memiliki test yang memadai
- ✅ Tidak merusak test yang sudah ada
- ✅ Terdokumentasi dengan baik

### 2. Run Tests

```bash
# Run semua tests
pytest

# Run dengan coverage
pytest --cov=src tests/

# Run linting
flake8 src/
black --check src/
mypy src/
```

### 3. Commit Changes

```bash
git add .
git commit -m "feat: menambahkan fitur prediksi real-time"
```

Lihat [Commit Message Guidelines](#commit-message-guidelines) untuk format yang benar.

### 4. Push ke Fork Anda

```bash
git push origin feature/nama-fitur-anda
```

### 5. Buat Pull Request

- Buka repository fork Anda di GitHub
- Klik "Compare & pull request"
- Isi template PR dengan lengkap
- Submit PR

### 6. Review Process

- Maintainer akan me-review PR Anda
- Lakukan perubahan yang diminta (jika ada)
- Setelah approved, PR akan di-merge

---

## 📏 Coding Standards

### Python Style Guide

Ikuti **PEP 8** style guide untuk Python:

```python
# ✅ Good
def calculate_weather_score(temperature, humidity):
    """
    Calculate weather score based on temperature and humidity.
    
    Args:
        temperature (float): Temperature in Celsius
        humidity (float): Humidity percentage
    
    Returns:
        float: Weather score
    """
    score = (temperature * 0.6) + (humidity * 0.4)
    return score


# ❌ Bad
def CalcScore(t,h):
    s=(t*0.6)+(h*0.4)
    return s
```

### Code Formatting

Gunakan **Black** untuk formatting:

```bash
# Format semua file Python
black src/

# Check tanpa mengubah file
black --check src/
```

### Linting

Gunakan **flake8** untuk linting:

```bash
flake8 src/ --max-line-length=88 --ignore=E203,W503
```

### Type Hints

Gunakan type hints untuk fungsi:

```python
def predict_weather(temperature: float, humidity: float) -> str:
    """Predict weather condition."""
    # implementation
    return "Cerah"
```

### Docstrings

Gunakan Google style docstrings:

```python
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest model for weather prediction.
    
    Args:
        X_train: Training features with shape (n_samples, n_features)
        y_train: Training labels with shape (n_samples,)
    
    Returns:
        Trained Random Forest classifier
    
    Raises:
        ValueError: If training data is empty
    
    Example:
        >>> model = train_model(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    if len(X_train) == 0:
        raise ValueError("Training data cannot be empty")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
```

---

## 📝 Commit Message Guidelines

Gunakan **Conventional Commits** format:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: Fitur baru
- `fix`: Bug fix
- `docs`: Perubahan dokumentasi
- `style`: Format code (tidak mengubah fungsi)
- `refactor`: Refactoring code
- `test`: Menambah atau memperbaiki tests
- `chore`: Maintenance tasks

### Contoh

```bash
# Fitur baru
feat(model): tambahkan support untuk XGBoost

# Bug fix
fix(preprocess): perbaiki handling missing values

# Dokumentasi
docs(readme): update installation instructions

# Refactoring
refactor(utils): simplify data loading function

# Tests
test(model): tambahkan unit tests untuk prediction
```

### Best Practices

- ✅ Gunakan present tense ("add" bukan "added")
- ✅ Gunakan imperative mood ("move" bukan "moves")
- ✅ Jangan capitalize huruf pertama
- ✅ Jangan gunakan titik di akhir subject
- ✅ Limit subject ke 50 karakter
- ✅ Wrap body di 72 karakter
- ✅ Jelaskan "what" dan "why" bukan "how"

---

## 🧪 Testing Guidelines

### Unit Tests

```python
import pytest
from src.predict import predict_weather

def test_predict_weather_cerah():
    """Test weather prediction for sunny conditions."""
    result = predict_weather(
        temperature=30.0,
        humidity=45.0,
        pressure=1016.0
    )
    assert result == "Cerah"

def test_predict_weather_hujan():
    """Test weather prediction for rainy conditions."""
    result = predict_weather(
        temperature=24.0,
        humidity=80.0,
        pressure=1008.0,
        rainfall=3.5
    )
    assert result == "Hujan"

def test_predict_weather_invalid_input():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError):
        predict_weather(temperature=-100)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### Test Coverage

Minimal test coverage: **80%**

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# Open coverage report
# Windows: start htmlcov/index.html
# Mac: open htmlcov/index.html
# Linux: xdg-open htmlcov/index.html
```

---

## 📚 Documentation Guidelines

### Code Comments

```python
# ✅ Good: Explain WHY, not WHAT
# Use exponential decay to give more weight to recent data
weight = np.exp(-0.1 * time_diff)

# ❌ Bad: State the obvious
# Multiply weight by 0.1
weight = weight * 0.1
```

### README Updates

Jika menambahkan fitur baru:

1. Update **Features** section
2. Tambahkan ke **Usage** section dengan contoh
3. Update **API Documentation** jika relevan
4. Tambahkan screenshot jika diperlukan

### Jupyter Notebooks

- Tambahkan markdown cells untuk penjelasan
- Bersihkan output sebelum commit
- Gunakan heading yang terstruktur (H1, H2, H3)
- Tambahkan visualisasi untuk memperjelas

---

## 🎨 Branch Naming Convention

```bash
# Features
feature/add-weather-api
feature/improve-model-accuracy

# Bug fixes
fix/handle-missing-data
fix/prediction-error

# Documentation
docs/update-readme
docs/add-api-docs

# Refactoring
refactor/simplify-preprocessing
refactor/optimize-training

# Tests
test/add-unit-tests
test/improve-coverage
```

---

## 🏷️ Issue Labels

| Label | Deskripsi |
|-------|-----------|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Improvements to documentation |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `question` | Further information requested |
| `wontfix` | This will not be worked on |
| `duplicate` | This issue already exists |

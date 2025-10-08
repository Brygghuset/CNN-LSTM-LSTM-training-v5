# Bidrag till Master POC Preprocessing v5.0

Tack för ditt intresse av att bidra till detta projekt! Vi välkomnar alla typer av bidrag.

## 🚀 Sätt att Bidra

### 🐛 Buggar
- Rapportera buggar genom att skapa en [Issue](https://github.com/Brygghuset/CNN-LSTM-LSTM-training-v5/issues)
- Inkludera detaljerad beskrivning, steg för att reproducera och förväntat beteende
- Bifoga relevanta loggar och skärmdumpar

### ✨ Nya Funktioner
- Skapa en Issue för att diskutera nya funktioner innan implementation
- Beskriv användningsfallet och fördelarna
- Vänta på godkännande innan du börjar koda

### 📚 Dokumentation
- Förbättra README.md
- Lägg till exempel och tutorials
- Översätt dokumentation till andra språk
- Korrigera stavfel och grammatik

### 🧪 Tester
- Lägg till nya tester för befintlig funktionalitet
- Förbättra test coverage
- Skapa integrationstester
- Lägg till performance benchmarks

## 🔧 Utvecklingsmiljö

### Setup
```bash
# Klona repository
git clone https://github.com/Brygghuset/CNN-LSTM-LSTM-training-v5.git
cd CNN-LSTM-LSTM-training-v5

# Skapa virtuell miljö
python -m venv venv
source venv/bin/activate  # På Windows: venv\Scripts\activate

# Installera dependencies
pip install -r requirements.txt

# Installera development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### Kodstandarder

#### Python
- Följ PEP 8
- Använd type hints
- Skriv docstrings för alla funktioner
- Maximal radlängd: 88 tecken (Black standard)

#### Git
- Använd beskrivande commit-meddelanden
- En commit per logisk ändring
- Skapa feature branches för större ändringar

#### Tester
- Skriv tester för all ny funktionalitet
- Sträva efter >90% test coverage
- Använd pytest för alla tester

## 📋 Pull Request Process

### 1. Förberedelse
```bash
# Skapa en ny branch
git checkout -b feature/din-funktion

# Gör dina ändringar
# ... kod ...

# Commita ändringar
git add .
git commit -m "feat: lägg till ny funktion X"
```

### 2. Kvalitetskontroll
```bash
# Kör tester
python -m pytest tests/

# Kontrollera kodkvalitet
black src/ tests/
flake8 src/ tests/
mypy src/

# Kontrollera test coverage
python -m pytest --cov=src tests/
```

### 3. Skapa Pull Request
- Beskriv ändringarna tydligt
- Länka till relaterade Issues
- Inkludera skärmdumpar om UI-ändringar
- Uppdatera dokumentation om nödvändigt

### 4. Review Process
- Vänta på review från maintainers
- Adressera feedback
- Håll PR:en uppdaterad med main branch

## 🏗️ Projektstruktur

```
src/
├── data/                    # Data processing modules
├── utils/                   # Utility functions
├── master_poc_preprocessing_v5.py  # Main entry point
└── checkpoint_manager.py    # Checkpoint system

tests/
├── unit_tests/              # Unit tests
├── integration_tests/       # Integration tests
└── end_to_end_tests/        # End-to-end tests

docs/                        # Documentation
configs/                     # Configuration files
scripts/                     # Execution scripts
```

## 🧪 Testning

### Kör alla tester
```bash
python -m pytest tests/
```

### Kör specifika tester
```bash
# Unit tests
python -m pytest tests/unit_tests/

# Integration tests
python -m pytest tests/integration_tests/

# End-to-end tests
python -m pytest tests/end_to_end_tests/
```

### Test coverage
```bash
python -m pytest --cov=src --cov-report=html tests/
```

## 📝 Commit Meddelanden

Använd [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: lägg till ny funktion
fix: korrigera bugg
docs: uppdatera dokumentation
test: lägg till tester
refactor: refaktorera kod
perf: förbättra prestanda
chore: underhållsåtgärder
```

## 🔍 Code Review Checklist

### Innan du skickar PR
- [ ] Alla tester passerar
- [ ] Kod följer projektets standarder
- [ ] Dokumentation är uppdaterad
- [ ] Commit-meddelanden är beskrivande
- [ ] Inga känsliga data är exponerade

### Under review
- [ ] Adressera alla kommentarer
- [ ] Håll PR:en uppdaterad
- [ ] Testa ändringarna lokalt
- [ ] Verifiera att CI/CD passerar

## 🐛 Bug Report Template

```markdown
## Beskrivning
Kort beskrivning av buggen.

## Steg för att Reproducera
1. Gå till '...'
2. Klicka på '...'
3. Scrolla ner till '...'
4. Se fel

## Förväntat Beteende
Beskriv vad som borde hända.

## Faktiskt Beteende
Beskriv vad som faktiskt händer.

## Miljö
- OS: [e.g. macOS 14.0]
- Python: [e.g. 3.13]
- Version: [e.g. v5.0]

## Ytterligare Kontext
Lägg till annan kontext om problemet här.
```

## ✨ Feature Request Template

```markdown
## Funktion
Beskriv funktionen du vill ha.

## Problem
Vilket problem löser denna funktion?

## Lösning
Beskriv din föreslagna lösning.

## Alternativ
Beskriv alternativa lösningar du har övervägt.

## Ytterligare Kontext
Lägg till annan kontext eller skärmdumpar om funktionsförfrågan här.
```

## 📞 Kontakt

- **Issues**: [GitHub Issues](https://github.com/Brygghuset/CNN-LSTM-LSTM-training-v5/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Brygghuset/CNN-LSTM-LSTM-training-v5/discussions)

## 📄 Licens

Genom att bidra till detta projekt godkänner du att dina bidrag licensieras under [MIT License](LICENSE).

---

Tack för ditt bidrag! 🎉

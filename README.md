# Fruit Classification - MobileNetV2 Transfer Learning

Dieses Projekt verwendet **MobileNetV2 Transfer Learning** zur Klassifikation von 9 verschiedenen Obstsorten.

---

## 🚀 Schnellstart

### Voraussetzungen

1. **Python 3.11** (oder neuer) muss installiert sein --> getestet auf 3.11.9 und 3.13.7
2. **Abhängigkeiten installieren** (falls noch nicht vorhanden):
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn pillow
   ```

   > **Hinweis:** Das Projekt enthält bereits ein vorkonfiguriertes `venv/`-Verzeichnis. Falls Sie dieses nutzen möchten, aktivieren Sie es mit `.\venv\Scripts\activate` (Windows) oder `source venv/bin/activate` (Linux/Mac).

### Projekt verwenden

#### Option A: Nur Vorhersagen (schnell)
Wenn das Modell bereits trainiert ist (`model_output/best_fruit_classifier_cnn.keras` existiert):

1. Öffnen Sie `Fruit_Prediction_Only.ipynb` in Jupyter Notebook oder VS Code
2. Legen Sie Ihre Obstbilder in den `Obst`-Ordner
3. Führen Sie alle Zellen aus

#### Option B: Modell selbst trainieren
1. Stellen Sie sicher, dass Trainingsdaten im `Bilder/Training/` und `Bilder/Test/` Ordner vorhanden sind
2. Öffnen Sie `Fruit_Classification_CNN_Complete.ipynb`
3. Führen Sie alle Zellen der Reihe nach aus
4. Das Training dauert ca. 10-30 Minuten (je nach Hardware)

---

## ⚠️ Wichtig: Worauf Sie achten müssen

### 1. Bildformat für optimale Ergebnisse
Das Modell wurde auf dem **Fruits-360 Dataset** trainiert. Für beste Ergebnisse sollten Ihre Bilder folgende Eigenschaften haben:

| ✅ Optimal | ❌ Problematisch |
|-----------|------------------|
| Weißer/heller Hintergrund | Komplexe Hintergründe |
| Einzelne, ganze Frucht | Aufgeschnittene Früchte |
| Frucht zentriert | Mehrere Früchte im Bild |
| Keine Wasserzeichen | Shutterstock/Stock-Photo Watermarks |
| Gute Beleuchtung | Schatten oder Überbelichtung |

### 2. Trainingsdaten-Struktur
Die Trainingsbilder müssen in folgender Struktur vorliegen:
```
Bilder/
├── Training/
│   ├── Apple */       # Ordner die mit "Apple" beginnen
│   ├── Banana */
│   ├── Cherry */
│   └── ...
└── Test/
    ├── Apple */
    ├── Banana */
    └── ...
```

### 3. GPU-Nutzung (empfohlen)
- Mit GPU: Training in ~10 Minuten
- Ohne GPU (nur CPU): Training in ~30+ Minuten
- TensorFlow erkennt CUDA-fähige GPUs automatisch

### 4. Speicherplatz
- Trainingsdaten: ~500 MB (gefiltert)
- Trainiertes Modell: ~6 MB
- Temporäre Daten: ~500 MB im `filtered_data/` Ordner

---

## 📊 Modell-Architektur

### **MobileNetV2 Transfer Learning**

Das Modell besteht aus zwei Teilen:

#### **1. Feature Extractor: MobileNetV2 (eingefroren)**
- Vortrainiert auf **ImageNet** (1.4 Millionen echte Fotos, 1000 Klassen)
- Erkennt bereits allgemeine visuelle Features (Kanten, Texturen, Formen, Farben)
- Layer sind **eingefroren** (werden nicht mittrainiert)
- Verwendet Global Average Pooling am Ende

#### **2. Custom Classifier (trainierbar)**
- **BatchNormalization** → Stabilisierung
- **Dropout (0.3)** → Overfitting-Schutz
- **Dense (256, ReLU)** → Feature-Kombination
- **BatchNormalization** → Stabilisierung
- **Dropout (0.3)** → Overfitting-Schutz
- **Dense (9, Softmax)** → Output für 9 Obstklassen

---

### **Modell-Statistiken:**
- **Basismodell:** MobileNetV2 (vortrainiert auf ImageNet)
- **Trainierbare Parameter:** nur der Classifier-Teil (~330K)
- **Nicht-trainierbare Parameter:** ~2.2M (eingefrorene MobileNetV2-Weights)
- **Architektur-Typ:** Transfer Learning
- **Preprocessing:** MobileNetV2 `preprocess_input` (skaliert Pixel auf [-1, 1])

**Vorteile gegenüber Custom CNN:**
- ✅ Schnelleres Training (nur Classifier wird trainiert)
- ✅ Bessere Generalisierung durch vortrainierte Features
- ✅ Robuster gegen Störungen (Watermarks, verschiedene Hintergründe)
- ✅ Weniger Overfitting bei kleinen Datensätzen

---

## 🧠 Wichtige Konzepte

### **1. Transfer Learning** 🔄
**Was ist das?**
- Ein **vortrainiertes Modell** (MobileNetV2, trainiert auf ImageNet) wird als Basis verwendet
- Das Modell hat bereits gelernt, **allgemeine visuelle Features** zu erkennen
- Wir frieren diese Layer ein und trainieren nur einen neuen **Classifier** für unsere Obstklassen

**Vorteile:**
- Viel **weniger Trainingsdaten** nötig
- **Schnelleres** Training
- **Bessere** Generalisierung
- **Robuster** gegen Variationen in den Bildern

---

### **2. MobileNetV2** 📱
**Was ist das?**
- Ein **effizientes** CNN, entwickelt von Google
- Vortrainiert auf **ImageNet** (1.4M Bilder, 1000 Klassen)
- Verwendet **Depthwise Separable Convolutions** für weniger Parameter
- Ideal für **mobile und eingebettete Anwendungen**

---

### **3. BatchNormalization** ⚖️
**Was macht es?**
- **Normalisiert** die Werte zwischen den Layern
- Macht das Training **stabiler und schneller**
- Verhindert, dass Werte zu groß oder zu klein werden

---

### **4. Dropout** 🎲
**Was macht es?**
- Schaltet **zufällig** einige Neuronen während des Trainings aus
- Verhindert **Overfitting** (dass das Modell die Trainingsdaten auswendig lernt)
- In diesem Modell: 0.3 = 30% der Neuronen werden ausgeschaltet

---

### **5. Dense (Fully Connected Layer)** 🔗
**Was macht er?**
- **Klassischer neuronaler Layer** - jedes Neuron ist mit allen vorherigen verbunden
- Kombiniert alle gelernten Features zu einer Entscheidung

---

### **6. Softmax (Aktivierungsfunktion im Output)** 📊
**Was macht sie?**
- Wandelt die 9 Output-Werte in **Wahrscheinlichkeiten** um (0-100%)
- Alle Wahrscheinlichkeiten zusammen ergeben **100%**

**Beispiel:**
```
Rohwerte:           Nach Softmax:
Apple:    2.5  →    Apple:      85%
Banana:   0.3  →    Banana:     10%
Cherry:  -1.2  →    Cherry:      2%
...                 ...
                    Summe:     100%
```

---

## 🔄 Datenfluss durch das Modell

```
Bild (100×100×3)
    ↓
MobileNetV2 preprocess_input (skaliert auf [-1, 1])
    ↓
MobileNetV2 Feature Extractor (eingefroren)
    ↓ (1280 Features via Global Average Pooling)
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense(256, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense(9) + Softmax
    ↓
[Apple: 85%, Banana: 10%, Cherry: 2%, ...]
```

---

## 📝 Komponenten Zusammenfassung

| Komponente | Funktion |
|-----------|----------|
| **MobileNetV2** | Vortrainierter Feature Extractor |
| **BatchNorm** | Stabilisierung |
| **Dropout** | Overfitting-Schutz |
| **Dense** | Entscheidungsfindung |
| **Softmax** | Wahrscheinlichkeiten |

---

## 🍎 Klassifizierte Obstsorten

Das Modell kann folgende 9 Obstsorten erkennen:

1. Apple (Apfel)
2. Banana (Banane)
3. Cherry (Kirsche)
4. Kiwi
5. Lemon (Zitrone)
6. Orange
7. Peach (Pfirsich)
8. Strawberry (Erdbeere)
9. Tomato (Tomate)

---

## 📁 Projektstruktur

```
ML/
├── Fruit_Classification_CNN_Complete.ipynb  # Vollständiges Training
├── Fruit_Prediction_Only.ipynb              # Nur Vorhersagen (lädt trainiertes Modell)
├── model_output/                            # Gespeicherte Modelle
│   ├── best_fruit_classifier_cnn.keras      # Bestes trainiertes Modell
│   └── label_mapping_cnn.json               # Label-Mapping (Index → Klassenname)
├── Bilder/                                  # Trainingsdaten
│   ├── Training/                            # Trainingsbilder
│   └── Test/                                # Testbilder
├── diagnose_model.py                        # Modell-Diagnose-Skript
├── analyze_apple_image.py                   # Bild-Analyse-Skript
└── README.md                                # Diese Datei
```

---

## 🚀 Verwendung

### Training (vollständig):
Öffnen Sie `Fruit_Classification_CNN_Complete.ipynb` und führen Sie alle Zellen aus.

### Nur Vorhersagen (schnell):
Öffnen Sie `Fruit_Prediction_Only.ipynb` - lädt das bereits trainierte Modell in Sekunden!

---

## ⚠️ Wichtige Hinweise

Das Modell funktioniert am besten mit Bildern, die:
- **Weißen Hintergrund** haben
- **100×100 Pixel** groß sind (oder automatisch skaliert werden)
- **Zentriertes Obst** zeigen
- **Gute Beleuchtung** haben

Bilder mit anderem Hintergrund oder Stil können zu falschen Vorhersagen führen!


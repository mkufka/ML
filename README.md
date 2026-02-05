# Fruit Classification CNN - Dokumentation

Dieses Projekt verwendet ein Convolutional Neural Network (CNN) zur Klassifikation von 9 verschiedenen Obstsorten.

---

## 🚀 Schnellstart

### Voraussetzungen

1. **Python 3.11** (oder neuer) muss installiert sein --> getestet auf 3.11.9 und 3.13.7
2. **Virtual Environment** erstellen und aktivieren:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Abhängigkeiten installieren:**
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn pillow
   ```

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

### **Gesamt: 27 Layer**

Aufgeteilt in:

#### **1. Convolutional Blocks (4 Blöcke):**
- **Block 1:** 2× Conv2D (32 Filter) + 2× BatchNorm + MaxPooling + Dropout = **6 Layer**
- **Block 2:** 2× Conv2D (64 Filter) + 2× BatchNorm + MaxPooling + Dropout = **6 Layer**
- **Block 3:** 2× Conv2D (128 Filter) + 2× BatchNorm + MaxPooling + Dropout = **6 Layer**
- **Block 4:** 2× Conv2D (256 Filter) + 2× BatchNorm + MaxPooling + Dropout = **6 Layer**

**Convolutional Teil: 24 Layer**

#### **2. Fully Connected Teil:**
- **GlobalAveragePooling2D** = **1 Layer**
- **Dense (512) + BatchNorm + Dropout** = **3 Layer**
- **Dense (256) + BatchNorm + Dropout** = **3 Layer**
- **Output Dense (9 Klassen) mit Softmax** = **1 Layer**

**Dense Teil: 8 Layer**

---

### **Modell-Statistiken:**
- **Insgesamt: 27 Layer**
- **Trainierbare Parameter: 1.440.937** (ca. 1,4 Millionen)
- **Architektur-Typ:** Custom CNN (Convolutional Neural Network)
- **4 Convolutional Blocks** mit steigender Filterzahl (32 → 64 → 128 → 256)
- **2 Dense Hidden Layers** (512 → 256)
- **Output Layer** mit 9 Neuronen (für 9 Obstklassen)

Das ist ein **mittelgroßes CNN** - nicht zu klein (würde underfitting verursachen), nicht zu groß (würde overfitting verursachen). Perfekt für 9 Obstklassen! 🍎🍌🍊

---

## 🧠 Layer-Typen Erklärt

### **1. Conv2D (Convolutional Layer)** 🔍
**Was macht er?**
- Erkennt **Muster und Features** im Bild (z.B. Kanten, Farben, Texturen)
- Verwendet kleine **Filter** (3×3 Pixel), die über das Bild "gleiten"
- Frühe Layer erkennen einfache Muster (Kanten), tiefe Layer erkennen komplexe Muster (Formen, Objekte)

**In diesem Modell:**
- Block 1: 32 Filter (erkennt 32 verschiedene einfache Muster)
- Block 2: 64 Filter (erkennt 64 komplexere Muster)
- Block 3: 128 Filter
- Block 4: 256 Filter (erkennt sehr komplexe Features wie "Apfelform" oder "Bananenkrümmung")

**Beispiel:** Ein Filter könnte spezialisiert sein auf "rote runde Formen" → Apfel!

---

### **2. BatchNormalization** ⚖️
**Was macht er?**
- **Normalisiert** die Werte zwischen den Layern
- Macht das Training **stabiler und schneller**
- Verhindert, dass Werte zu groß oder zu klein werden

**Analogie:** Wie ein Thermostat, der die Temperatur konstant hält, damit nichts überhitzt oder einfriert.

---

### **3. MaxPooling2D** 📉
**Was macht er?**
- **Verkleinert** das Bild (z.B. von 100×100 auf 50×50)
- Nimmt nur die **wichtigsten Informationen** (Maximum aus jedem 2×2 Bereich)
- Reduziert Rechenaufwand und macht das Modell robuster

**Beispiel:** 
```
Vorher (4×4):     Nachher (2×2):
[1 3 2 4]         [3 8]
[2 1 5 8]    →    [9 7]
[6 9 3 2]
[4 7 1 5]
```
Nimmt jeweils das Maximum aus jedem 2×2 Block.

---

### **4. Dropout** 🎲
**Was macht er?**
- Schaltet **zufällig** einige Neuronen während des Trainings aus (z.B. 25% oder 50%)
- Verhindert **Overfitting** (dass das Modell die Trainingsdaten auswendig lernt)
- Zwingt das Modell, robuster zu werden

**Analogie:** Wie ein Fußballteam, das auch mit 10 statt 11 Spielern trainiert, damit es flexibler wird.

**In diesem Modell:**
- 0.25 = 25% der Neuronen werden ausgeschaltet (in Conv-Blöcken)
- 0.5 = 50% der Neuronen werden ausgeschaltet (in Dense-Layern)

---

### **5. GlobalAveragePooling2D** 🌐
**Was macht er?**
- Nimmt den **Durchschnitt** aller Werte in jedem Feature-Map
- Wandelt z.B. 256 Feature-Maps (6×6 Pixel) in 256 einzelne Zahlen um
- Reduziert massiv die Parameter-Anzahl

**Beispiel:**
```
Feature-Map (6×6):        Ergebnis:
[1 2 3 4 5 6]
[2 3 4 5 6 7]       →     Durchschnitt = 4.5
[3 4 5 6 7 8]
...
```

---

### **6. Dense (Fully Connected Layer)** 🔗
**Was macht er?**
- **Klassischer neuronaler Layer** - jedes Neuron ist mit allen vorherigen verbunden
- Kombiniert alle gelernten Features zu einer Entscheidung
- Die letzten Dense-Layer "denken" über die Features nach

**In diesem Modell:**
- Dense(512): 512 Neuronen kombinieren Features
- Dense(256): 256 Neuronen verfeinern die Entscheidung
- Dense(9): **Output-Layer** - 9 Neuronen (eine pro Obstsorte)

---

### **7. Softmax (Aktivierungsfunktion im Output)** 📊
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
[Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout]  ← Block 1
    ↓ (50×50×32)
[Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout]  ← Block 2
    ↓ (25×25×64)
[Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout]  ← Block 3
    ↓ (12×12×128)
[Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout]  ← Block 4
    ↓ (6×6×256)
GlobalAveragePooling2D
    ↓ (256 Zahlen)
Dense(512) → BatchNorm → Dropout
    ↓ (512 Zahlen)
Dense(256) → BatchNorm → Dropout
    ↓ (256 Zahlen)
Dense(9) + Softmax
    ↓
[Apple: 85%, Banana: 10%, Cherry: 2%, ...]
```

---

## 📝 Layer-Typen Zusammenfassung

| Layer-Typ | Funktion |
|-----------|----------|
| **Conv2D** | Mustererkennung |
| **BatchNorm** | Stabilisierung |
| **MaxPooling** | Verkleinerung |
| **Dropout** | Overfitting-Schutz |
| **GlobalAveragePooling** | Komprimierung |
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


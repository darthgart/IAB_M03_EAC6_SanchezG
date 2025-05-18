# Projecte de Clustering amb KMeans i PCA

Aquest projecte realitza una anàlisi de clustering sobre un conjunt de dades generat artificialment, aplicant tècniques com KMeans i reducció de dimensionalitat amb PCA.

## 📁 Estructura del projecte
```bash
IAB_M03_EAC6_SanchezG/
├── src/
│ ├── functions.py # Funcions del projecte
│ └── main.py # Script principal
├── tests/
│ └── test_main.py # Testos del projecte
├── doc/ # Documentació generada (p. ex. amb pdoc)
├── captures/ # Imatges generades pel projecte
├── requirements.txt # Dependències del projecte
├── LICENSE # Llicència del projecte
├── README.md # Aquest fitxer
└── llegir.txt # Informació addicional (si cal)
```
---

## ⚙️ Instal·lació del projecte

1. Clona aquest repositori:

```bash
git clone https://github.com/usuari/projecte.git
cd projecte
```

2. Crea un entorn virtual:

```bash
python -m venv env
source env/bin/activate
```

3. Instal·la les dependències:

```bash
pip install -r requirements.txt
```

## ▶️ Execució del projecte

Executa el fixer principal:

```bash
python src/main.py
```

Això generarà les gràfiques dels clústers i les gurardarà automàticament a la carpeta **captures**

## ✅ Comprovació de l'anàlisi estàtic

Per comprvar la qualitat del codi amb pylint:

```bash
pylint src/
```

## 🧾 Generació de la documentació

La documentació de les funcions es pot generar amb pdoc:

```bash
pdoc src/functions.py --html --output-dir doc/
```

## 🧪 Comprovació dels tests

Executa els tests amb pytest des de la carpeta arrel:

```bash
pytest
```

📄 Llicència

Aquest projecte està sota la llicència MIT. Consulta el fitxer **LICENSE** per a més informació.

# IR_Image_Classification_System_DEMO

## Descrizione del Progetto

Questo progetto è una demo di un sistema di classificazione e retrieval di immagini IR (infrarosso) per veicoli militari. Utilizza una rete neurale basata su ResNet18, fine-tunata in Google Colab (link al notebook: https://colab.research.google.com/drive/11ksPxgEH4Hmqi_2sDPvsAkuIeYK-ucHy?usp=sharing ), per estrarre embeddings dalle immagini. Successivamente, viene effettuata una similarity search utilizzando ChromaDB per trovare le immagini più simili a una query selezionata.

Il sistema permette di:

- Classificare veicoli militari da immagini IR
- Effettuare retrieval basato su similarità visiva
- Visualizzare risultati attraverso un'interfaccia web Streamlit

## Requisiti

- Python 3.8+
- Ambiente virtuale (venv)

## Installazione

1. Clona il repository:

   ```bash
   git clone https://github.com/GGCIRILLO/IR_Image_Classification_System_DEMO.git
   cd IR_Image_Classification_System_DEMO
   ```

2. Crea un ambiente virtuale:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
   ```

3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

## Lancio dell'App

Per avviare l'applicazione web:

```bash
streamlit run app.py
```

L'app sarà disponibile su http://localhost:8501. Seleziona un'immagine di test e clicca "Esegui Query" per vedere i risultati della similarity search.

## Struttura del Progetto

- `app.py`: Applicazione principale Streamlit
- `requirements.txt`: Dipendenze Python
- `best_weights.pth`: Pesi del modello allenato
- `classes.json`: Classi dei veicoli
- `data/`: Directory con i dati (immagini processate)
- `vector_db`: Directory per il database vettoriale ChromaDB

## Note

Assicurati che tutti i file di dati e modelli siano presenti prima di lanciare l'app. Per l'addestramento del modello, consulta il notebook Colab collegato.

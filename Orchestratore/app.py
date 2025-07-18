# ==============================================================================
# 1. IMPORTAZIONI E CONFIGURAZIONE INIZIALE
# ==============================================================================
import os
import requests
import chromadb
import uuid
import io
from flask import Flask, request, jsonify
from pypdf import PdfReader

# Inizializza l'applicazione web Flask
app = Flask(__name__)
# Garantisce che i caratteri speciali (es. accenti) vengano resi correttamente nelle risposte JSON
app.config['JSON_AS_ASCII'] = False

# --- Configurazione dei Servizi Dipendenti ---
# Legge gli indirizzi degli altri microservizi dalle variabili d'ambiente.
# Se una variabile non è impostata, usa un valore di default (utile per test locali).
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-service:8080/create-embedding")
LLM_API_URL = os.getenv("LLM_API_URL", "http://gemma-api:8080/generate")

# Nome della collection in ChromaDB. Una collection è come una tabella in un database relazionale.
COLLECTION_NAME = "rag_session_collection"

# --- Inizializzazione del Database Vettoriale in-memory ---
# Creiamo un'istanza di ChromaDB che vive solo nella RAM di questo servizio.
# Questo evita la necessità di un database separato e di storage persistente.
# Perfetto per sessioni di chat singole con un solo documento.
print("Inizializzazione di ChromaDB in-memory...")
chroma_client = chromadb.Client()

# Per assicurare che ogni sessione parta da zero, tentiamo di eliminare la collection
# precedente all'avvio e ne creiamo una nuova, vuota.
try:
    chroma_client.delete_collection(name=COLLECTION_NAME)
except Exception:
    # Ignoriamo l'errore se la collection non esiste, è normale al primo avvio.
    pass
collection = chroma_client.create_collection(name=COLLECTION_NAME)
print(f"Database vettoriale in-memory e collection '{COLLECTION_NAME}' pronti. ✅")


# ==============================================================================
# 2. ENDPOINT PER L'INDICIZZAZIONE DEL PDF
# ==============================================================================
# Questo endpoint riceve un file PDF, lo processa e lo indicizza nella memoria di ChromaDB.
@app.route('/index-pdf', methods=['POST'])
def index_pdf():
    # Usiamo 'global collection' per poter modificare la variabile definita all'esterno della funzione.
    global collection

    # Controlla se un file è stato effettivamente inviato nella richiesta.
    if 'file' not in request.files:
        return jsonify({"error": "Nessun file trovato nella richiesta."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nome del file vuoto."}), 400

    try:
        # --- Reset della Memoria ---
        # Dato che gestiamo un solo documento alla volta, ogni nuovo upload
        # cancella la collection precedente e ne crea una nuova.
        print(f"Reset della memoria per il nuovo file: {file.filename}")
        chroma_client.delete_collection(name=COLLECTION_NAME)
        collection = chroma_client.create_collection(name=COLLECTION_NAME)

        # --- Estrazione e Chunking del Testo dal PDF ---
        # Leggiamo il file PDF in memoria per l'elaborazione.
        pdf_file = io.BytesIO(file.read())
        reader = PdfReader(pdf_file)
        
        # Estraiamo tutto il testo da tutte le pagine e lo uniamo in un'unica stringa.
        all_text = ""
        for page in reader.pages:
            all_text += page.extract_text() + "\n"
        
        # Dividiamo il testo in paragrafi. I paragrafi sono solitamente separati da
        # doppie interruzioni di riga. Rimuoviamo anche eventuali spazi bianchi e paragrafi vuoti.
        paragraphs = [p.strip() for p in all_text.split('\n\n') if p.strip()]
        
        # --- Logica di Chunking con Sovrapposizione (Overlap) ---
        # Parametri per definire come vengono creati i chunk di testo.
        CHUNK_SIZE = 3    # Ogni chunk conterrà 3 paragrafi.
        CHUNK_OVERLAP = 1 # I chunk si sovrapporranno di 1 paragrafo.

        chunks = []
        # Calcoliamo il "passo" (stride) con cui scorreremo la lista dei paragrafi.
        stride = CHUNK_SIZE - CHUNK_OVERLAP
        
        for i in range(0, len(paragraphs), stride):
            # Creiamo un chunk prendendo 'CHUNK_SIZE' paragrafi a partire dall'indice corrente 'i'.
            chunk = "\n\n".join(paragraphs[i : i + CHUNK_SIZE])
            chunks.append(chunk)
            # Usciamo dal ciclo se il prossimo passo ci porterebbe fuori dalla lista.
            if i + stride >= len(paragraphs):
                break
        
        print(f"File diviso in {len(chunks)} chunk sovrapposti.")

        if not chunks:
            return jsonify({"message": "Il PDF non contiene testo estraibile."})

        # --- Creazione degli Embedding e Salvataggio ---
        # Chiamiamo il nostro servizio di embedding per convertire tutti i chunk in vettori.
        response = requests.post(EMBEDDING_API_URL, json={"text": chunks})
        response.raise_for_status() # Controlla se la chiamata API ha avuto successo.
        embeddings = response.json()['embedding']
        
        # Generiamo ID unici per ogni chunk.
        ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
        
        # Aggiungiamo i dati (testi, embedding, id) alla collection di ChromaDB.
        collection.add(embeddings=embeddings, documents=chunks, ids=ids)
        
        return jsonify({"message": f"File '{file.filename}' indicizzato in memoria. {len(chunks)} chunk processati."})

    except Exception as e:
        print(f"Errore durante l'indicizzazione: {e}")
        return jsonify({"error": f"Errore interno del server durante l'indicizzazione: {e}"}), 500


# ==============================================================================
# 3. ENDPOINT PER LA CHAT CON LOGICA RAG
# ==============================================================================
# Questo endpoint riceve una domanda, la usa per cercare contesto nel database
# e poi chiama l'LLM per generare una risposta basata sul contesto.
@app.route('/chat', methods=['POST'])
def chat():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Richiesta JSON non valida."}), 400

    messages = json_data.get('messages')
    if not messages:
        return jsonify({"error": "La chiave 'messages' è obbligatoria."}), 400

    # --- NUOVA LOGICA: ACCETTA PARAMETRI DAL FRONTEND ---
    max_tokens = json_data.get('max_tokens', 512) # Usa il valore dal frontend o un default
    user_system_prompt = json_data.get('system_prompt', "Sei un assistente utile.") # Usa il prompt dal frontend o un default
    # ----------------------------------------------------

    if collection.count() == 0:
        return jsonify({"response": "Per favore, carica e indicizza un documento PDF prima di iniziare a chattare."})

    user_question = messages[-1]['content']

    try:
        # 1. Crea embedding per la domanda
        response = requests.post(EMBEDDING_API_URL, json={"text": user_question})
        response.raise_for_status()
        query_embedding = response.json()['embedding']
        
        # 2. Cerca il contesto in ChromaDB
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = "\n---\n".join(results['documents'][0])
        
        # 3. Costruisci il prompt finale, combinando il prompt di sistema dell'utente con il contesto RAG
        final_system_prompt = f"{user_system_prompt}\n\nUsa il seguente contesto per formulare la tua risposta:\n\nCONTESTO:\n{context}"
        
        final_messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        # 4. Chiama l'API LLM passando anche il max_tokens configurato
        llm_payload = {"messages": final_messages, "max_tokens": max_tokens}
        llm_response = requests.post(LLM_API_URL, json=llm_payload)
        llm_response.raise_for_status()
        
        return jsonify(llm_response.json())

    except Exception as e:
        print(f"Errore durante la chat: {e}")
        return jsonify({"error": f"Errore interno del server durante la chat: {e}"}), 500
    
# ==============================================================================
# 4. AVVIO DELL'APPLICAZIONE
# ==============================================================================
if __name__ == '__main__':
    # Questo blocco viene eseguito solo se avvii lo script con "python orchestrator_app.py".
    # In produzione su OpenShift, il server Gunicorn avvierà l'app.
    app.run(host='0.0.0.0', port=8080)
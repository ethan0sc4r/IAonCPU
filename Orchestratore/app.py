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

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Leggi la configurazione dei servizi dipendenti dalle variabili d'ambiente
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-service:8080/create-embedding")
LLM_API_URL = os.getenv("LLM_API_URL", "http://llm-api-service:8080/generate") 
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb-service")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

try:
    print(f"Connessione a ChromaDB su {CHROMA_HOST}:{CHROMA_PORT}...")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print("Connessione a ChromaDB riuscita. ✅")
except Exception as e:
    print(f"ERRORE: Impossibile connettersi a ChromaDB. {e}")
    chroma_client = None


# ==============================================================================
# 2. ENDPOINT PER L'INDICIZZAZIONE DEL PDF
# ==============================================================================
@app.route('/index-pdf', methods=['POST'])
def index_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "Nessun file trovato."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nome del file vuoto."}), 400

    try:
        # --- MODIFICA CHIAVE: Crea una collection con un nome unico ---
        collection_name = f"doc-{uuid.uuid4()}"
        print(f"Creazione di una nuova collection: {collection_name}")
        collection = chroma_client.create_collection(name=collection_name)
        # -----------------------------------------------------------

        pdf_file = io.BytesIO(file.read())
        reader = PdfReader(pdf_file)
        
        all_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        paragraphs = [p.strip() for p in all_text.split('\n\n') if p.strip()]
        
        CHUNK_SIZE, CHUNK_OVERLAP = 3, 1
        chunks = []
        stride = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(paragraphs), stride):
            chunk = "\n\n".join(paragraphs[i:i + CHUNK_SIZE])
            chunks.append(chunk)
            if i + stride >= len(paragraphs): break
        
        if not chunks:
            return jsonify({"message": "Il PDF non contiene testo estraibile."})

        response = requests.post(EMBEDDING_API_URL, json={"text": chunks})
        response.raise_for_status()
        embeddings = response.json()['embedding']
        
        ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
        collection.add(embeddings=embeddings, documents=chunks, ids=ids)
        
        # --- MODIFICA CHIAVE: Restituisce il nome della collection creata ---
        return jsonify({
            "message": f"File '{file.filename}' indicizzato con successo in una sessione isolata.",
            "collection_name": collection_name
        })

    except Exception as e:
        print(f"Errore durante l'indicizzazione: {e}")
        return jsonify({"error": f"Errore interno del server: {e}"}), 500


# ==============================================================================
# 3. ENDPOINT PER LA CHAT
# ==============================================================================
@app.route('/chat', methods=['POST'])
def chat():
    json_data = request.get_json()
    messages = json_data.get('messages')
    max_tokens = json_data.get('max_tokens', 512)
    user_system_prompt = json_data.get('system_prompt', "Sei un assistente AI utile.")
    # --- MODIFICA CHIAVE: Riceve il nome della collection da usare ---
    collection_name = json_data.get('collection_name')
    # -------------------------------------------------------------

    if not messages:
        return jsonify({"error": "La chiave 'messages' è obbligatoria."}), 400

    user_question = messages[-1]['content']

    try:
        # --- MODIFICA CHIAVE: Logica condizionale basata sulla collection ---
        if collection_name:
            # CASO RAG: Usa la collection specificata
            print(f"Esecuzione chat RAG sulla collection: {collection_name}")
            collection = chroma_client.get_collection(name=collection_name)
            
            response = requests.post(EMBEDDING_API_URL, json={"text": user_question})
            query_embedding = response.json()['embedding']
            
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            context = "\n---\n".join(results['documents'][0])
            
            final_system_prompt = f"{user_system_prompt}\n\nUsa il seguente contesto per formulare la tua risposta:\n\nCONTESTO:\n{context}"
            final_messages = [{"role": "system", "content": final_system_prompt}, {"role": "user", "content": user_question}]
        else:
            # CASO GENERICO: Nessuna collection, chat diretta
            print("Esecuzione chat generica (nessuna collection specificata).")
            final_messages = [{"role": "system", "content": user_system_prompt}, {"role": "user", "content": user_question}]
        # -------------------------------------------------------------------

        llm_payload = {"messages": final_messages, "max_tokens": max_tokens}
        llm_response = requests.post(LLM_API_URL, json=llm_payload, timeout=300)
        llm_response.raise_for_status()
        
        return jsonify(llm_response.json())

    except Exception as e:
        print(f"Errore durante la chat: {e}")
        return jsonify({"error": f"Errore interno del server: {e}"}), 500

# ==============================================================================
# 4. AVVIO DELL'APPLICAZIONE
# ==============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

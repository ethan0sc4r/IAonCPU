# Importiamo le librerie necessarie
from flask import Flask, request, jsonify
from llama_cpp import Llama
import os # Necessario per leggere le variabili d'ambiente

# -----------------------------------------------------------------------------
# 1. INIZIALIZZAZIONE DELL'APPLICAZIONE FLASK
# -----------------------------------------------------------------------------
# Creiamo l'istanza dell'app web
app = Flask(__name__)

# Impostiamo la configurazione per restituire JSON con caratteri non-ASCII (es. lettere accentate)
app.config['JSON_AS_ASCII'] = False

# -----------------------------------------------------------------------------
# 2. CARICAMENTO DEL MODELLO GGUF
# -----------------------------------------------------------------------------
# Definiamo il percorso della cache che corrisponderà al volume emptyDir su OpenShift
MODEL_CACHE_PATH = "./model_cache"
os.environ['HF_HOME'] = MODEL_CACHE_PATH
os.environ['HUGGINGFACE_HUB_CACHE'] = MODEL_CACHE_PATH
os.environ['HF_HUB_CACHE'] = MODEL_CACHE_PATH
# Leggiamo il token di Hugging Face dalla variabile d'ambiente.
# Questo è il modo sicuro e corretto per gestire le credenziali in un container.
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print("ATTENZIONE: La variabile d'ambiente HUGGING_FACE_HUB_TOKEN non è stata trovata.")

print("Tentativo di caricamento del modello Llama 3 GGUF...")

try:
    # Inizializziamo il modello usando Llama.from_pretrained
    llm = Llama.from_pretrained(
        # Repository e file verificati su Hugging Face
        repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2",
        filename="Meta-Llama-3-8B-Instruct-v2.Q4_K_M.gguf",
        
        # Passiamo esplicitamente il token per l'autenticazione
        hf_token=hf_token,
        
        # Parametri fondamentali per l'esecuzione
        n_gpu_layers=0,      # <-- Forza l'esecuzione al 100% sulla CPU
        n_ctx=4096,          # <-- Imposta la dimensione massima del contesto
        cache_dir=MODEL_CACHE_PATH, # <-- Specifica dove salvare/cercare il modello
        verbose=True         # <-- Abilita log dettagliati al caricamento
    )
    print("Modello Llama 3 GGUF caricato con successo. ✅")

except Exception as e:
    print(f"ERRORE CRITICO durante il caricamento del modello: {e}")
    llm = None

# -----------------------------------------------------------------------------
# 3. DEFINIZIONE DELL'ENDPOINT API
# -----------------------------------------------------------------------------
@app.route('/generate', methods=['POST'])
def generate_text():
    # Se il modello non è stato caricato, restituisce un errore
    if llm is None:
        return jsonify({"error": "Il modello non è disponibile a causa di un errore di caricamento."}), 503

    # Legge i dati JSON inviati nella richiesta
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Richiesta JSON non valida o vuota."}), 400

    # Estrae la lista di messaggi dal corpo della richiesta (obbligatoria)
    messages = json_data.get('messages')
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "La chiave 'messages' è obbligatoria e deve essere una lista."}), 400

    # Estrae il numero massimo di token, con un valore di default se non fornito
    max_tokens = json_data.get('max_tokens', 512)

    try:
        # Genera la risposta usando il metodo create_chat_completion
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )
        
        # Estrae il contenuto del messaggio di risposta
        cleaned_response = response['choices'][0]['message']['content'].strip()

        # Restituisce la risposta in formato JSON
        return jsonify({"response": cleaned_response})

    except Exception as e:
        print(f"Errore durante la generazione del testo: {e}")
        return jsonify({"error": "Errore interno del server durante la generazione."}), 500

# -----------------------------------------------------------------------------
# 4. AVVIO DEL SERVER (usato solo per test locali)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Questo blocco viene eseguito solo se avvii lo script con "python app.py".
    # In produzione su OpenShift, Gunicorn avvierà l'app, ignorando questo blocco.
    app.run(host='0.0.0.0', port=8080)
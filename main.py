# Importiamo le librerie necessarie
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os # Importiamo 'os' per leggere le variabili d'ambiente

# -----------------------------------------------------------------------------
# 1. INIZIALIZZAZIONE DELL'APPLICAZIONE FLASK
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Imposta la codifica JSON per supportare correttamente i caratteri UTF-8 (es. lettere accentate)
app.config['JSON_AS_ASCII'] = False

# -----------------------------------------------------------------------------
# 2. CARICAMENTO DEL MODELLO E DEL TOKENIZER
#
#    Questa sezione viene eseguita una sola volta all'avvio dell'applicazione.
# -----------------------------------------------------------------------------

# Definiamo un percorso dedicato per la cache del modello, che verrà montato come volume emptyDir
MODEL_CACHE_PATH = "./model_cache" 
model_id = "google/gemma-2-2b-it"  # Modello instruction-tuned di Gemma 2
dtype = torch.float32  # Tipo di dato ottimale per la CPU

print(f"Tentativo di caricamento del modello '{model_id}'...")
print(f"La cache del modello verrà salvata e cercata in: {MODEL_CACHE_PATH}")

try:
    # Carichiamo il tokenizer, specificando dove salvarlo/cercarlo
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_PATH)
    
    # Carichiamo il modello, forzando l'uso della CPU e specificando la cache
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",  # Forza l'uso della CPU
        cache_dir=MODEL_CACHE_PATH 
    )
    print("Modello caricato con successo su CPU. ✅")

except Exception as e:
    print(f"ERRORE CRITICO durante il caricamento del modello: {e}")
    # Se il modello non si carica, l'applicazione non può funzionare.
    model = None
    tokenizer = None

# -----------------------------------------------------------------------------
# 3. DEFINIZIONE DELL'ENDPOINT API
# -----------------------------------------------------------------------------
@app.route('/generate', methods=['POST'])
def generate_text():
    # Controlliamo se il modello è stato caricato correttamente all'avvio
    if model is None or tokenizer is None:
        return jsonify({"error": "Modello non disponibile a causa di un errore di caricamento."}), 503

    # Estraiamo i dati JSON inviati nella richiesta
    json_data = request.get_json()

    # Verifichiamo che sia stato fornito un 'prompt'
    if not json_data or 'prompt' not in json_data:
        return jsonify({"error": "Prompt non fornito nel corpo della richiesta."}), 400

    prompt = json_data['prompt']
    
    try:
        # Prepariamo l'input per il modello nel formato di chat richiesto da Gemma
        chat = [
            { "role": "user", "content": prompt },
        ]
        input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # Convertiamo il testo in token (numeri) e li spostiamo sulla CPU
        input_ids = tokenizer(input_text, return_tensors="pt").to("cpu")

        # Generiamo la risposta usando il modello, limitando la lunghezza
        outputs = model.generate(**input_ids, max_new_tokens=250)
        
        # Riconvertiamo i token di output in testo leggibile
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Puliamo la risposta per restituire solo il testo generato dal modello,
        # rimuovendo il prompt iniziale che viene incluso nell'output.
        cleaned_response = response_text[len(input_text):].strip()

        # Restituiamo la risposta in formato JSON leggibile
        return jsonify({"response": cleaned_response})

    except Exception as e:
        print(f"Errore durante la generazione del testo: {e}")
        return jsonify({"error": "Errore interno del server durante la generazione del testo."}), 500

# -----------------------------------------------------------------------------
# 4. AVVIO DEL SERVER (usato solo per test locali con 'python app.py')
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Avviamo il server Flask sulla porta 8080, accessibile da tutta la rete
    # In produzione su OpenShift, questo blocco non viene eseguito; si usa Gunicorn.
    app.run(host='0.0.0.0', port=8080)
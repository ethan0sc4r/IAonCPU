# Importiamo le librerie necessarie
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from PIL import Image
import base64
import io

# -----------------------------------------------------------------------------
# 1. INIZIALIZZAZIONE DELL'APPLICAZIONE E DEL MODELLO
# -----------------------------------------------------------------------------
app = Flask(__name__)
 
# Definiamo il percorso che useremo per la cache del modello.
# Questo percorso deve corrispondere al volume emptyDir che configurerai su OpenShift.
MODEL_CACHE_PATH = "/app//model_cache"

# Carica il modello CLIP, specificando esplicitamente la cartella della cache.
# Questa operazione scaricherà il modello la prima volta e verrà eseguita
# una sola volta all'avvio del container.
print("Caricamento del modello di embedding CLIP in corso...")
try:
    model = SentenceTransformer(
        'clip-ViT-B-32',
        cache_folder=MODEL_CACHE_PATH  # <-- Configurazione esplicita della cache
    )
    print("Modello CLIP caricato con successo. ✅")
except Exception as e:
    print(f"ERRORE CRITICO durante il caricamento del modello CLIP: {e}")
    model = None

# -----------------------------------------------------------------------------
# 2. DEFINIZIONE DELL'ENDPOINT PER CREARE GLI EMBEDDING
# -----------------------------------------------------------------------------
@app.route('/create-embedding', methods=['POST'])
def create_embedding():
    # Se il modello non è stato caricato, restituisce un errore
    if model is None:
        return jsonify({"error": "Modello di embedding non disponibile."}), 503

    # Legge i dati JSON inviati nella richiesta
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Richiesta JSON non valida o vuota."}), 400

    embedding = None

    # Caso 1: La richiesta contiene una chiave "text"
    if 'text' in json_data:
        text = json_data['text']
        print(f"Creazione embedding per il testo: '{text[:50]}...'")
        embedding = model.encode(text)

    # Caso 2: La richiesta contiene una chiave "image" con dati in Base64
    elif 'image' in json_data:
        base64_image_string = json_data['image']
        print("Creazione embedding per un'immagine...")
        try:
            # Decodifica la stringa Base64 in dati binari
            image_data = base64.b64decode(base64_image_string)
            # Apre l'immagine usando la libreria Pillow
            image = Image.open(io.BytesIO(image_data))
            # Crea l'embedding per l'immagine
            embedding = model.encode(image)
        except Exception as e:
            return jsonify({"error": f"Errore nel processare l'immagine Base64: {e}"}), 400
    
    # Se non viene fornito né testo né immagine
    else:
        return jsonify({"error": "La richiesta deve contenere una chiave 'text' o 'image'."}), 400

    # Converte il risultato (un array NumPy) in una lista Python per la risposta JSON
    return jsonify({"embedding": embedding.tolist()})

# -----------------------------------------------------------------------------
# 3. AVVIO DEL SERVER (usato solo per test locali)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Questo blocco viene eseguito solo se avvii lo script con "python embed_app.py".
    # In produzione su OpenShift, Gunicorn avvierà l'app, ignorando questo blocco.
    app.run(host='0.0.0.0', port=8080)
import streamlit as st
import requests
import os

# --- 1. CONFIGURAZIONE ---
# Legge l'URL dell'API Orchestrator da una variabile d'ambiente per flessibilità in OpenShift.
# Se non la trova, usa un valore di default per i test locali.
ORCHESTRATOR_API_URL = os.getenv("ORCHESTRATOR_API_URL", "http://orchestrator-service:8080")

# Imposta la configurazione della pagina Streamlit (titolo, layout, etc.)
st.set_page_config(page_title="Chat RAG Multi-Sessione", layout="wide")
st.title("📄 Chat RAG Multi-Sessione")

# --- 2. GESTIONE DELLO STATO DELLA SESSIONE ---
# Inizializza le variabili nello stato della sessione di Streamlit se non esistono già.
# Questo permette di mantenere i dati (cronologia chat, parametri) tra le interazioni dell'utente.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ciao! Scegli un PDF dalla sidebar per iniziare una nuova sessione di chat."}]
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 512
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Sei un assistente esperto. Rispondi alla domanda dell'utente in modo chiaro e conciso, basandoti esclusivamente sul contesto fornito."
# Traccia la collection attiva per questa specifica sessione utente.
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# --- 3. SIDEBAR DI CONFIGURAZIONE E UPLOAD ---
with st.sidebar:
    st.header("Configurazione")
    
    st.subheader("Carica un Documento")
    # Widget per l'upload del file.
    uploaded_file = st.file_uploader("Scegli un file PDF per avviare l'indicizzazione", type="pdf", key="file_uploader")

    # Logica di indicizzazione automatica
    if uploaded_file is not None and uploaded_file.id != st.session_state.get("last_uploaded_file_id"):
        st.info(f"Nuovo file rilevato: {uploaded_file.name}")
        with st.spinner('Indicizzazione in corso... Potrebbe richiedere tempo.'):
            try:
                # Prepara il file per l'invio tramite una richiesta POST multipart/form-data.
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                # Chiama l'API per indicizzare il file.
                response = requests.post(f"{ORCHESTRATOR_API_URL}/index-pdf", files=files, timeout=300)
                
                if response.status_code == 200:
                    response_data = response.json()
                    st.success(response_data.get('message', 'File indicizzato!'))
                    
                    # Salva il nome della collection e l'ID del file nella sessione corrente.
                    st.session_state.collection_name = response_data.get('collection_name')
                    st.session_state.last_uploaded_file_id = uploaded_file.id
                    
                    # Resetta la cronologia della chat per iniziare una nuova conversazione.
                    st.session_state.messages = [{"role": "assistant", "content": f"Ho letto il documento '{uploaded_file.name}'. Ora puoi farmi delle domande."}]
                    st.rerun() # Forza un refresh dell'interfaccia per mostrare subito il messaggio.
                else:
                    st.error(f"Errore dal server: {response.json().get('error', 'Errore sconosciuto')}")
            except Exception as e:
                st.error(f"Errore di connessione all'API: {e}")

    st.divider()

    # Sezione per configurare i parametri dell'LLM, salvati nello stato della sessione.
    st.subheader("Parametri LLM")
    st.session_state.max_tokens = st.slider("Max Token di Risposta", 256, 4096, st.session_state.max_tokens, 256)
    st.session_state.system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=250)

# --- 4. INTERFACCIA DI CHAT PRINCIPALE ---
# Mostra tutti i messaggi salvati nella cronologia della sessione corrente.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accetta e processa il nuovo input dell'utente.
if prompt := st.chat_input("Fai una domanda..."):
    # Aggiunge e visualizza il messaggio dell'utente.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara e invia la richiesta all'assistente.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Sto elaborando la tua richiesta..."):
            try:
                # Costruisce il payload per l'API di chat.
                # Include i parametri configurati e il nome della collection attiva.
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": st.session_state.max_tokens,
                    "system_prompt": st.session_state.system_prompt,
                    "collection_name": st.session_state.collection_name # Sarà None se nessun file è stato caricato.
                }
                
                # Chiama l'endpoint di chat dell'orchestratore.
                response = requests.post(f"{ORCHESTRATOR_API_URL}/chat", json=payload, timeout=120)
                
                if response.status_code == 200:
                    assistant_response = response.json().get('response', 'Risposta non valida.')
                    message_placeholder.markdown(assistant_response)
                    # Aggiunge la risposta dell'assistente alla cronologia.
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    error_msg = f"Errore dal server: {response.json().get('error', 'Errore sconosciuto')}"
                    message_placeholder.markdown(error_msg)
            except Exception as e:
                error_msg = f"Errore di connessione all'API: {e}"
                message_placeholder.markdown(error_msg)
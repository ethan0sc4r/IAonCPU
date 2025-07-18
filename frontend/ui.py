import streamlit as st
import requests
import os

# --- 1. CONFIGURAZIONE ---
# Legge l'URL dell'API Orchestrator da una variabile d'ambiente per flessibilità in OpenShift
ORCHESTRATOR_API_URL = os.getenv("ORCHESTRATOR_API_URL", "http://127.0.0.1:8080")

# Imposta la configurazione della pagina Streamlit
st.set_page_config(page_title="Chat RAG", layout="wide")
st.title("📄 Chat RAG con i Tuoi Documenti")

# --- 2. GESTIONE DELLO STATO DELLA SESSIONE ---
# Inizializza la cronologia della chat e i parametri se non esistono già.
# Questo permette di mantenere i dati tra le interazioni dell'utente.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ciao! Carica un PDF e configura i parametri per iniziare."}]
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 512
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Sei un assistente esperto. Rispondi alla domanda dell'utente in modo chiaro e conciso, basandoti esclusivamente sul contesto fornito."

# --- 3. SIDEBAR DI CONFIGURAZIONE ---
with st.sidebar:
    st.header("Configurazione")
    
    # Sezione per l'upload del file PDF
    st.subheader("Carica un Documento")
    uploaded_file = st.file_uploader("Scegli un file PDF", type="pdf")
    
    if st.button("Indicizza il File"):
        if uploaded_file is not None:
            with st.spinner('Indicizzazione in corso...'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                    response = requests.post(f"{ORCHESTRATOR_API_URL}/index-pdf", files=files, timeout=300) # Timeout lungo per l'indicizzazione
                    if response.status_code == 200:
                        st.success(response.json().get('message', 'File indicizzato!'))
                        # Resetta la chat quando un nuovo file viene indicizzato
                        st.session_state.messages = [{"role": "assistant", "content": f"Ho letto il documento '{uploaded_file.name}'. Ora puoi farmi delle domande."}]
                    else:
                        st.error(f"Errore dal server: {response.json().get('error', 'Errore sconosciuto')}")
                except Exception as e:
                    st.error(f"Errore di connessione all'API: {e}")
        else:
            st.warning("Per favore, carica un file prima.")

    st.divider()

    # Sezione per configurare i parametri dell'LLM
    st.subheader("Parametri LLM")
    st.session_state.max_tokens = st.slider(
        "Max Token di Risposta", 
        min_value=256, 
        max_value=4096, 
        value=st.session_state.max_tokens, 
        step=256
    )
    st.session_state.system_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.system_prompt, 
        height=250
    )

# --- 4. INTERFACCIA DI CHAT PRINCIPALE ---
# Mostra i messaggi della cronologia
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accetta e processa il nuovo input dell'utente
if prompt := st.chat_input("Fai una domanda sul documento..."):
    # Aggiungi e mostra il messaggio dell'utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara e invia la richiesta all'assistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Sto elaborando la tua richiesta..."):
            try:
                # Costruisci il payload per l'API con i parametri configurati
                payload = {
                    "messages": [{"role": "user", "content": prompt}], # Invia solo l'ultima domanda
                    "max_tokens": st.session_state.max_tokens,
                    "system_prompt": st.session_state.system_prompt
                }
                
                response = requests.post(f"{ORCHESTRATOR_API_URL}/chat", json=payload, timeout=120)
                
                if response.status_code == 200:
                    assistant_response = response.json().get('response', 'Risposta non valida.')
                    message_placeholder.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    error_msg = f"Errore dal server: {response.json().get('error', 'Errore sconosciuto')}"
                    message_placeholder.markdown(error_msg)
            except Exception as e:
                error_msg = f"Errore di connessione all'API: {e}"
                message_placeholder.markdown(error_msg)
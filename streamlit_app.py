import streamlit as st
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")


client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

SEARCH_COLUMNS = [
    "Product/Plan",
    "Storage",
    "Device Limits",
    "User Type",
    "Included Apps/Services",
    "Key Features",
    "Security Features",
    "Target Use Case"
]

# --- Получение эмбеддингов батчами ---
def get_embeddings_batch(texts, batch_size=128, max_retries=3, sleep_sec=1.5):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        safe_chunk = [t if (t and t.strip()) else "[EMPTY]" for t in chunk]

        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_DEPLOYMENT,
                    input=safe_chunk
                )
                chunk_embs = [np.array(item.embedding, dtype=np.float32).tolist() for item in resp.data]
                embeddings.extend(chunk_embs)
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError(f"Embedding batch failed: {e}")
                time.sleep(sleep_sec)
    return embeddings

def build_row_texts(df, available_cols):
    texts = []
    for _, row in df.iterrows():
        parts = [str(row[c]).strip() for c in available_cols]
        text = " ".join(p for p in parts if p)
        texts.append(text if text else "[EMPTY]")
    return texts

def compute_and_save_embeddings(src_path, out_path="data_with_embeddings.pkl"):
    if src_path.endswith(".csv"):
        df = pd.read_csv(src_path)
    else:
        df = pd.read_excel(src_path, engine="openpyxl")
    df = df.fillna("")

    available_cols = [c for c in SEARCH_COLUMNS if c in df.columns and df[c].notna().any()]
    texts = build_row_texts(df, available_cols)

    embeddings = get_embeddings_batch(texts, batch_size=128)

    dims = {len(e) for e in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dims}")

    df["embedding"] = [np.array(e, dtype=np.float32).tolist() for e in embeddings]
    df.to_pickle(out_path)
    return df, available_cols

def load_embeddings(out_path="data_with_embeddings.pkl"):
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Embeddings file not found: {out_path}")
    df = pd.read_pickle(out_path)
    dims = {len(e) for e in df["embedding"]}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions in saved file: {dims}")
    return df

def cosine_sim_matrix(query_emb, matrix):
    q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(q, matrix)[0]
    return sims

# --- Извлечение ключевых слов ---
def extract_keywords(user_query):
    prompt = f"""
    Ты — ассистент по подбору лицензий.
    Извлеки из текста только названия сервисов/функций и ключевые слова.
    Запрос: "{user_query}"
    Верни список слов через запятую, без пояснений.
    """
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return [kw.strip() for kw in resp.choices[0].message.content.split(",") if kw.strip()]

# --- Генерация ответа ---
def generate_dynamic_response(keywords, results, answers):
    table_str = results.head(5).to_json(orient="records", force_ascii=False)
    prompt = f"""
    Ты — помощник по подбору лицензий.

    Ключевые слова: {', '.join(keywords)}
    Ответы пользователя: {answers}

    Найденные строки:
    {table_str}

    Если информации недостаточно — задай один уточняющий вопрос.
    Если достаточно — выбери одну лицензию и объясни, почему она подходит.
    Будь дружелюбным и очень кратким.
    пропиши максимум 2 предложения.
    """
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# --- Интерфейс Streamlit ---
st.title("Чатбот по подбору лицензий (эмбеддинги V3)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "answers" not in st.session_state:
    st.session_state.answers = {}

# --- Загрузка данных ---
if not os.path.exists("data_with_embeddings.pkl"):
    df, available_cols = compute_and_save_embeddings("2.csv", "data_with_embeddings.pkl")
else:
    df = load_embeddings("data_with_embeddings.pkl")
    available_cols = [c for c in SEARCH_COLUMNS if c in df.columns]

emb_matrix = np.vstack(df["embedding"].apply(np.array).values).astype(np.float32)

# История сообщений
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ввод пользователя
user_input = st.chat_input("Введите запрос или ответ...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.results.empty:
        # Первый запрос
        st.session_state.keywords = extract_keywords(user_input)
        q_emb = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=user_input).data[0].embedding
        sims = cosine_sim_matrix(q_emb, emb_matrix)
        df["__similarity"] = sims
        results = df.sort_values("__similarity", ascending=False).head(10)
        st.session_state.results = results
        if not results.empty:
            st.success(f"Найдено совпадений: {len(results)}")
            with st.expander("Показать найденные лицензии"):
                st.dataframe(results)
        else:
            st.warning("Ничего не найдено. Попробуйте уточнить запрос.")
    else:
        # Уточнение
        step = f"step_{len(st.session_state.answers)+1}"
        st.session_state.answers[step] = user_input

    # Ответ бота
    bot_reply = generate_dynamic_response(
        st.session_state.keywords,
        st.session_state.results,
        st.session_state.answers
    )
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
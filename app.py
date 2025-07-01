import streamlit as st
import lyricsgenius
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.chains.llm import LLMChain  # Still works but deprecated, optional to switch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- SETUP ---

# Load API keys from Streamlit secrets
GENIUS_API_KEY = st.secrets["GENIUS_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Genius client
genius = lyricsgenius.Genius(GENIUS_API_KEY, remove_section_headers=True)

# OpenAI LLM (GPT-4.1)
llm = ChatOpenAI(
    model="gpt-4-1106-preview",  # GPT-4.1 model
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7
)

# Chains
lyrics_prompt = PromptTemplate.from_template("Summarize the following lyrics:\n\n{lyrics}")
lyrics_chain = LLMChain(llm=llm, prompt=lyrics_prompt, output_key="summary")

genre_prompt = PromptTemplate.from_template(
    "Identify the genre of the song based on this summary. Give a single word output:\n\n{summary}"
)
genre_chain = LLMChain(llm=llm, prompt=genre_prompt, output_key="genre")

overall_chain = SequentialChain(
    chains=[lyrics_chain, genre_chain],
    input_variables=["lyrics"],
    output_variables=["summary", "genre"],
    verbose=False
)

# --- STREAMLIT UI ---

st.set_page_config(page_title="🎵 Lyrics Genre Classifier", layout="centered")
st.title("🎵 Lyrics Genre Classifier with OpenAI + Genius")
st.markdown("Get song lyrics, summarize them, and identify the genre using **GPT-4.1** from **OpenAI**.")

song_name = st.text_input("Enter Song Name", value="Bulleya")
artist_name = st.text_input("Enter Artist Name", value="Arijit Singh")

if st.button("Analyze"):
    with st.spinner("Fetching lyrics and analyzing..."):
        try:
            song_obj = genius.search_song(song_name, artist_name)
            lyrics = song_obj.lyrics if song_obj and song_obj.lyrics else "Lyrics not found."

            result = overall_chain.invoke({"lyrics": lyrics})

            st.subheader("🎤 Original Lyrics")
            st.text_area("", lyrics, height=300)

            st.subheader("💡 Summary")
            st.write(result["summary"])

            st.subheader("🎵 Predicted Genre")
            st.success(result["genre"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")

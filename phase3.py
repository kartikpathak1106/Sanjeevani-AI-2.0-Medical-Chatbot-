import os
import base64
import re
import inspect
from tavily import TavilyClient
import tempfile
import streamlit as st
import base64
def clean_ai_response(text):
     if not text: return text 
     cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
     return cleaned.strip()

def render_svg(path, width=220):
    with open(path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    b64 = base64.b64encode(
        svg_content.encode("utf-8")
    ).decode("utf-8")

    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" style="display:block;margin:0 auto;"/>',
        unsafe_allow_html=True
    )


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from brain_of_the_doctor import encode_image, analyze_image_with_query
from pateintsvoice import transcribe_with_groq
from doctorvoice import text_to_speech_with_gtts

import database as db
from safety import check_safety, add_disclaimer
from document_ingest import add_pdf_to_vectorstore

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


st.set_page_config(
    page_title="Sanjeevani AI 2.0",
    layout="wide"
)


db.init_db()


DB_FAISS_PATH = "vectorstore/db_faiss"


VISION_SYSTEM_PROMPT = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise (max 3 sentences). No preamble, start your answer right away please.
The patient's question / voice note (if any) is: """

WELLNESS_SYSTEM_PROMPT = """You are Sanjeevani AI's caring wellness companion.
Look VERY carefully at this photo of the person's face — their eyes, expression, and overall energy. 
Be balanced and fair in your observation.
Do NOT default to saying "tired" just because you are unsure — only say someone looks tired if there are CLEAR visible signs like heavy/half-closed eyelids, dark circles, or a visibly drained expression. Most people in a normal photo look perfectly fine — your default assumption should be that they look okay/neutral/good unless something clearly stands out as tired. 
Respond warmly in 4 to 5 sentences: - If they look fresh, alert, neutral, or generally fine — say so warmly and positively. - ONLY if there are clear, obvious signs of tiredness, gently mention it and suggest resting, water, or a short break.
 Do NOT mention any disease, medical condition, or diagnosis. 
This is only a friendly, honest energy/mood check based on appearance. Speak directly and warmly, like a caring friend giving an honest, balanced observation."""

TEXT_PROMPT_TEMPLATE = """ You are Sanjeevani AI, a medical information assistant.
Answer using ONLY the provided medical context below.
If the context does NOT contain enough information to answer the question, respond with EXACTLY this single word and nothing else: NEED_WEB_SEARCH Otherwise, give a clear medium-length answer (5 to 8 sentences), covering meaning, causes/symptoms, basic treatment/care, and when to see a doctor — whatever is relevant. Do not mention context/documents.
Match the user's language: - Hindi → Hindi - Hinglish → simple Hinglish - English → English 
Context: {context} Question: {question} Answer: """


CHAT_INPUT_SUPPORTS_FILES = (
    "accept_file" in inspect.signature(st.chat_input).parameters
)
BG_PATH = "assets/medical_bg.png"

with open(BG_PATH, "rb") as f:
    BG_B64 = base64.b64encode(f.read()).decode()


DARK_UI_CSS = f"""
<style>



.stApp {{
    background: transparent !important;
    color: #f5f5f5 !important;
}}

/* Actual Streamlit application container */
[data-testid="stAppViewContainer"] {{
    background: transparent !important;
    position: relative !important;
}}
[data-testid="stHeader"] {{
    background: transparent !important;
}}

[data-testid="stToolbar"] {{
    background: transparent !important;
}}
/* Background image layer */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;

    background-image: url("data:image/png;base64,{BG_B64}") !important;

    background-size: 100% 100% !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;

    z-index: 0 !important;
    pointer-events: none !important;
}}
/* Actual main area */
[data-testid="stMain"] {{
    background: transparent !important;
    background-color: transparent !important;
    position: relative !important;
    z-index: 1 !important;
}}

/* Main content container */
[data-testid="stMainBlockContainer"] {{
    background: transparent !important;
    background-color: transparent !important;
}}

/* Older Streamlit main selector */
.main {{
    background: transparent !important;
    background-color: transparent !important;
}}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {{
    background-color: #11161d !important;
    z-index: 100 !important;
}}

section[data-testid="stSidebar"] * {{
    color: #e5e7eb !important;
}}
/* Sidebar scrolling */
section[data-testid="stSidebar"] > div:first-child {{
    height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}}


/* =====================================================
   CHAT MESSAGES
   ===================================================== */

.stChatMessage {{
    background-color: #151a22 !important;
    color: #f5f5f5 !important;
    border-radius: 14px !important;
}}

.stChatMessage p,
.stChatMessage span,
.stChatMessage div {{
    color: #f5f5f5 !important;
}}


/* =====================================================
   CHAT INPUT
   ===================================================== */

[data-testid="stChatInput"] {{
    background-color: #1b2029 !important;
    border: 1px solid #303744 !important;
    border-radius: 14px !important;
}}

[data-testid="stChatInput"] textarea {{
    background-color: #1b2029 !important;
    color: #f5f5f5 !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: #8b95a5 !important;
}}


/* =====================================================
   NORMAL INPUTS
   ===================================================== */

input {{
    background-color: #151a22 !important;
    color: #f5f5f5 !important;
    border: 1px solid #303744 !important;
}}

textarea {{
    background-color: #151a22 !important;
    color: #f5f5f5 !important;
}}


/* =====================================================
   BUTTONS
   ===================================================== */

button {{
    color: #f5f5f5 !important;
    background-color: #151a22 !important;
    border: 1px solid #303744 !important;
}}

button:hover {{
    background-color: #202733 !important;
}}


/* =====================================================
   HEADINGS
   ===================================================== */

h1, h2, h3, h4, h5, h6 {{
    color: #f5f5f5 !important;
}}


/* =====================================================
   NORMAL TEXT
   ===================================================== */

p, label {{
    color: #d1d5db !important;
}}


/* =====================================================
   EXPANDER / SOURCES
   ===================================================== */

[data-testid="stExpander"] {{
    background-color: #11161d !important;
    border: 1px solid #303744 !important;
}}

[data-testid="stExpander"] * {{
    color: #e5e7eb !important;
}}


/* =====================================================
   FILE UPLOADER
   ===================================================== */

[data-testid="stFileUploader"] {{
    background-color: #151a22 !important;
}}

[data-testid="stFileUploader"] * {{
    color: #e5e7eb !important;
}}


/* =====================================================
   POPOVER
   ===================================================== */

[data-testid="stPopover"] {{
    background-color: #11161d !important;
}}

[data-testid="stPopover"] * {{
    color: #e5e7eb !important;
}}


/* =====================================================
   GREETING
   ===================================================== */

.sanjeevani-greeting {{
    text-align: center;
    padding-top: 55px;
    padding-bottom: 35px;
}}

.sanjeevani-greeting-icon {{
    width: 100px;
    height: 100px;
    margin: 0 auto 18px auto;
}}

.sanjeevani-greeting-icon img {{
    width: 100px;
    height: 100px;
    object-fit: contain;
}}

.sanjeevani-greeting-title {{
    font-size: 30px;
    font-weight: 700;
    color: #f5f5f5 !important;
    margin-bottom: 6px;
}}

.sanjeevani-greeting-question {{
    font-size: 27px;
    font-weight: 600;
    color: #e5e7eb !important;
    margin-bottom: 10px;
}}

.sanjeevani-greeting-sub {{
    font-size: 16px;
    color: #8b95a5 !important;
}}


/* =====================================================
   CAMERA
   ===================================================== */

[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {{
    max-width: 100% !important;
    width: 100% !important;
    height: auto !important;
    object-fit: cover;
}}


/* =====================================================
   POPOVER BODY
   ===================================================== */

div[data-testid="stPopoverBody"] {{
    max-width: 340px !important;
    overflow: hidden !important;
}}

</style>
"""


@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_vectorstore():

    if "vectorstore" not in st.session_state:

        embedding_model = get_embedding_model()

        st.session_state.vectorstore = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    return st.session_state.vectorstore


def set_custom_prompt(template):

    return PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )


def get_qa_chain(vectorstore):

    return RetrievalQA.from_chain_type(

        llm=ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=0.2,
            max_tokens=1500,
            groq_api_key=os.environ.get(
                "GROQ_API_KEY"
            ),
        ),

        chain_type="stuff",

        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 5
            }
        ),

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": set_custom_prompt(
                TEXT_PROMPT_TEMPLATE
            )
        },
    )
def web_search(query):
    client = TavilyClient(
        api_key=os.environ.get("TAVILY_API_KEY")
    )

    results = client.search(
        query,
        max_results=4
    )

    combined = ""

    for r in results.get("results", []):
        combined += f"- {r['title']}: {r['content']}\n"

    return combined


def answer_from_web(question, search_text):
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.2,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )

    prompt = (
        "You are Sanjeevani AI. Using the following live web search results, "
        "answer the medical question in simple language and medium length "
        "(5 to 8 sentences). "
        "then continue normally.\n\n"
        f"Search results:\n{search_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)

    return response.content


def auth_page():

    render_svg(
        "assets/logo_full_heart_dark.svg",
        width=360
    )
    def set_page_background(image_path): 
        with open(image_path, "rb") as f: img_bytes = f.read() 
        b64 = base64.b64encode(img_bytes).decode("utf-8") 
        st.markdown( f""" <style> [data-testid="stAppViewContainer"] {{ background-image: linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)), url("data:image/png;base64,{b64}"); background-size: cover; background-position: center; background-attachment: fixed; }} </style> """, unsafe_allow_html=True )

    st.markdown(
        """
        <h4 style='text-align:center;color:white;'>
        Your Personal Health Assistant
        </h4>
        """,
        unsafe_allow_html=True
    )

    tab_login, tab_register = st.tabs(
        [
            "Login",
            "Register"
        ]
    )

    with tab_login:

        with st.form("login_form"):

            username = st.text_input(
                "Username",
                key="login_username"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password"
            )

            submitted = st.form_submit_button(
                "Login"
            )

            if submitted:

                user_id = db.authenticate_user(
                    username,
                    password
                )

                if user_id:

                    st.session_state.user_id = user_id

                    st.session_state.username = username

                    st.session_state.current_session_id = (
                        db.create_session(
                            user_id,
                            title="New Chat"
                        )
                    )

                    st.session_state.messages = []
                    

                    st.rerun()

                else:

                    st.error(
                        "oops! Wrong username or Password"
                    )

    with tab_register:

        with st.form("register_form"):

            new_username = st.text_input(
                "Choose a username",
                key="reg_username"
            )

            new_password = st.text_input(
                "Choose a password",
                type="password",
                key="reg_password"
            )

            submitted = st.form_submit_button(
                "Register"
            )

            if submitted:

                success, message = db.create_user(
                    new_username,
                    new_password
                )

                if success:

                    st.success(
                        message
                    )

                else:

                    st.error(
                        message
                    )


def render_message(role, content):
    avatar = "assets/avatar_logo.png" if role == "assistant" else None

    st.chat_message(
        role,
        avatar=avatar
    ).markdown(content)


def handle_user_turn( text, image_file=None, 
                     source_label="text", 
                     mode="medical"
                       ):

    session_id = (
        st.session_state.current_session_id
    )

    display_text = (
        text
        if text
        else
        "(photo attached, no question typed)"
    )

    prefix = (
        "mic"
        if source_label == "voice"
        else ""
    )

    user_display = (
        f"{prefix}{display_text}"
    )

    if len(st.session_state.messages) == 0:

        chat_title = (
            text or display_text
        )[:40]

        db.rename_session(
            session_id,
            chat_title
        )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_display,
            "image": image_file
        }
    )

    db.save_message(
        session_id,
        "user",
        user_display
    )

    render_message(
        "user",
        user_display
    )

    if image_file is not None:

        st.image(
            image_file,
            width=200
        )

    is_safe, crisis_message = check_safety(
        text or ""
    )

    if not is_safe:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": crisis_message,
                "image": None
            }
        )

        db.save_message(
            session_id,
            "assistant",
            crisis_message
        )

        render_message(
            "assistant",
            crisis_message
        )

        return

    sources_to_show = []

    audio_path = None

    try:

        with st.spinner(
            "Sanjeevani AI is thinking..."
        ):

            if image_file is not None:

                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(
                        getattr(
                            image_file,
                            "name",
                            "img.jpg"
                        )
                    )[1] or ".jpg",
                    delete=False
                ) as tmp_img:

                    tmp_img.write(
                        image_file.getvalue()
                    )

                    image_path = tmp_img.name

                active_prompt = ( WELLNESS_SYSTEM_PROMPT if mode == "wellness" else VISION_SYSTEM_PROMPT ) 
                query = ( active_prompt + ( text if text else "No specific question, just check the photo." ) )
                

                answer = analyze_image_with_query(
                    query=query,
                    encoded_image=encode_image(
                        image_path
                    ),
                    model="qwen/qwen3.8-27b",
                )
                answer = clean_ai_response(answer)

            else:

                vectorstore = get_vectorstore()

                qa_chain = get_qa_chain(
                    vectorstore
                )

                response = qa_chain.invoke(
                    {
                        "query": text
                    }
                )

                answer = response["result"]
                answer = clean_ai_response(answer)
                if answer.strip() == "NEED_WEB_SEARCH":
                  with st.spinner("🌐 Searching the internet..."): search_text = web_search(text) 
                  answer = answer_from_web(text, search_text) 
                  sources_to_show = ["🌐 Live web search (not from uploaded documents)"] 
                else:
                     for doc in response.get(
                     "source_documents",
                      []
                ):

                      src = doc.metadata.get(
                        "source",
                        "unknown"
                    )

                      page = doc.metadata.get(
                        "page",
                        "?"
                    )

                      label = (
                        f"{os.path.basename(src)} "
                        f"(page {page})"
                    )

                     if label not in sources_to_show:

                        sources_to_show.append(
                            label
                        )
        final_answer = add_disclaimer(
            answer
        )

        if mode != "wellness":
            with st.spinner(
                "🔊 Generating voice response..."
            ):

                audio_path = os.path.join(
                    tempfile.gettempdir(),
                    f"sanjeevani_voice_{session_id}_{len(st.session_state.messages)}.mp3"
                )

                audio_path = text_to_speech_with_gtts(
                    answer,
                    output_filepath=audio_path
                )


        # -------------------------------------------------
        # Save assistant message INCLUDING audio_path
        # -------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": final_answer,
            "image": None,
            "sources": sources_to_show,
            "audio_path": audio_path
        }

        st.session_state.messages.append(
            assistant_message
        )

        db.save_message(
            session_id,
            "assistant",
            final_answer
        )

        render_message(
            "assistant",
            final_answer
        )

        if sources_to_show:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources_to_show:

                    st.write(
                        "-",
                        source
                    )

        # -------------------------------------------------
        # SHOW VOICE RESPONSE
        # -------------------------------------------------

        if audio_path and os.path.exists(audio_path):

            st.audio(
                audio_path,
                format="audio/mp3"
            )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )


def render_sidebar():

    with st.sidebar:

        st.image(
            "assets/sidebar_logo.png",
            use_container_width=True
        )

        st.markdown(
            f"### {st.session_state.username}"
        )

        if st.button(
    "Logout",
    key="logout_button"
):

            for key in [
                "user_id",
                "username",
                "current_session_id",
                "messages",
                "vectorstore"
            ]:

                st.session_state.pop(
                    key,
                    None
                )

            st.rerun()

        st.divider()

        if st.button(
            "➕ New Chat",
             use_container_width=True,
             key="new_chat_button" 
                 ):

            st.session_state.current_session_id = (
                db.create_session(
                    st.session_state.user_id,
                    title="New Chat"
                )
            )

            st.session_state.messages = []

            st.rerun()

        st.divider()

        if st.button(
            "➕ New Chat",
            use_container_width=True
        ):

            st.session_state.current_session_id = (
                db.create_session(
                    st.session_state.user_id,
                    title="New Chat"
                )
            )

            st.session_state.messages = []

            st.rerun()

        st.markdown(
            "#### 💬 Chat History"
        )

        sessions = db.get_sessions(
            st.session_state.user_id
        )

        for s in sessions:

            history_check = db.get_messages(
                s["id"]
            )

            if len(history_check) == 0:
                continue

            label = (
                s["title"]
                if s["title"]
                else
                "New Chat"
            )

            # -------------------------------------------------
            # CHAT HISTORY + DELETE BUTTON
            # -------------------------------------------------

            col_chat, col_delete = st.columns(
                [5, 1],
                gap="small"
            )

            with col_chat:

                if st.button(
                    label[:30],
                    key=f"session_{s['id']}",
                    use_container_width=True
                ):

                    st.session_state.current_session_id = (
                        s["id"]
                    )

                    history = db.get_messages(
                        s["id"]
                    )

                    st.session_state.messages = [
                        {
                            "role": m["role"],
                            "content": m["content"],
                            "image": None
                        }
                        for m in history
                    ]

                    st.rerun()

            with col_delete:

                if st.button(
                    "🗑️",
                    key=f"delete_session_{s['id']}",
                    help="Delete this chat"
                ):

                    db.delete_session(
                        s["id"]
                    )

                    # If currently opened chat is deleted,
                    # create a fresh new chat
                    if (
                        st.session_state.current_session_id
                        == s["id"]
                    ):

                        st.session_state.current_session_id = (
                            db.create_session(
                                st.session_state.user_id,
                                title="New Chat"
                            )
                        )

                        st.session_state.messages = []

                    st.rerun()

        st.divider()

        st.markdown(
            "####  Upload Medical Document"
        )

        uploaded_pdf = st.file_uploader(
            "Add a PDF to the knowledge base",
            type=["pdf"],
            key="pdf_uploader"
        )

        if uploaded_pdf is not None:

            if st.button(
                "Add to Knowledge Base"
            ):

                with st.spinner(
                    "Reading and indexing document..."
                ):

                    vectorstore = get_vectorstore()

                    num_chunks = (
                        add_pdf_to_vectorstore(
                            uploaded_pdf,
                            vectorstore,
                            get_embedding_model()
                        )
                    )

                st.success(
                    f"Added! ({num_chunks} chunks indexed from {uploaded_pdf.name})"
                )

def set_page_background(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

def main():

    if "user_id" not in st.session_state:

        auth_page()

        return

    if "current_session_id" not in st.session_state:

        st.session_state.current_session_id = (
            db.create_session(
                st.session_state.user_id,
                title="New Chat"
            )
        )

    if "messages" not in st.session_state:

        st.session_state.messages = []

    render_sidebar()

    st.markdown(
        DARK_UI_CSS,
        unsafe_allow_html=True
    )

    has_messages = (
        len(st.session_state.messages) > 0
    )

    if not has_messages:

        st.markdown(
            """<div class="sanjeevani-greeting">
<img src="data:image/svg+xml;base64,"""
            + base64.b64encode(
                open(
                    "assets/logo_icon_heart_white.svg",
                    "rb"
                ).read()
            ).decode()
            + """"
style="width:90px;display:block;margin:2px auto 5px;">
<div style="text-align:center;font-size:30px;font-weight:700;margin-bottom:8px;">
<span style="color:#3b82f6;">Doctor</span>
<span style="color:#22c55e;"> Sanjeevani</span>
</div>
<div style="text-align:center;font-size:25px;">What's on your mind today?</div>
<div style="text-align:center;font-size:21px;color:white;;">
    Ask about symptoms, upload a photo, or just talk — I'm listening.
</div>
</div>""",
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # RENDER ALL SAVED MESSAGES
    # -------------------------------------------------

    for msg in st.session_state.messages:

        render_message(
            msg["role"],
            msg["content"]
        )

        if msg.get("image") is not None:

            st.image(
                msg["image"],
                width=200
            )

        if msg.get("sources"):

            with st.expander(
                "📚 Sources"
            ):

                for source in msg["sources"]:

                    st.write(
                        "-",
                        source
                    )

        # -------------------------------------------------
        # IMPORTANT: AUDIO WILL SURVIVE RERUN
        # -------------------------------------------------

        saved_audio = msg.get(
            "audio_path"
        )

        if (
            saved_audio
            and os.path.exists(saved_audio)
        ):

            st.audio(
                saved_audio,
                format="audio/mp3"
            )

    col_input, col_mic, col_wellness = st.columns( [8, 1, 1] )

    with col_mic:

        with st.popover(
            "mic",
            use_container_width=True
        ):

            st.caption(
                "Record your question"
            )

            audio_value = st.audio_input(
                "Speak",
                label_visibility="collapsed"
            )

            if audio_value is not None:

                if st.button(
                    "Send",
                    use_container_width=True
                ):

                    with st.spinner(
                        "🎧 Transcribing..."
                    ):

                        with tempfile.NamedTemporaryFile(
                            suffix=".wav",
                            delete=False
                        ) as tmp_audio:

                            tmp_audio.write(
                                audio_value.getvalue()
                            )

                            audio_path = tmp_audio.name

                        transcribed_text = (
                            transcribe_with_groq(
                                audio_filepath=audio_path,
                                stt_model="whisper-large-v3",
                                groq_api_key=os.environ.get(
                                    "GROQ_API_KEY"
                                )
                            )
                        )

                    handle_user_turn(
                        transcribed_text,
                        image_file=None,
                        source_label="voice"
                    )

    with col_wellness:
      with st.popover("CAM", use_container_width=True):
        st.caption("Quick wellness check")

        wellness_photo = st.camera_input(
            "Show me your face",
            label_visibility="collapsed"
        )

        if wellness_photo is not None:
            if st.button(
                "Let's see...",
                use_container_width=True
            ):
                handle_user_turn(
                    "Let me check your wellness.?",
                    image_file=wellness_photo,
                    source_label="text",
                    mode="wellness"
                )

                #st.rerun()
    with col_input:

        if CHAT_INPUT_SUPPORTS_FILES:

            prompt = st.chat_input(
                "Ask anything",
                accept_file=True,
                file_type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "webp"
                ]
            )

            if prompt:

                prompt_text = (
                    prompt.text
                    if hasattr(prompt, "text")
                    else
                    prompt.get("text")
                )

                files = (
                    prompt.files
                    if hasattr(prompt, "files")
                    else
                    prompt.get("files")
                )

                image_file = (
                    files[0]
                    if files
                    else
                    None
                )

                if prompt_text or image_file:

                    handle_user_turn(
                        prompt_text,
                        image_file=image_file,
                        source_label="text"
                    )

                    st.rerun()

        else:

            prompt_text = st.chat_input(
                "Ask anything"
            )

            if prompt_text:

                handle_user_turn(
                    prompt_text,
                    image_file=None,
                    source_label="text"
                )


if __name__ == "__main__":
    main()
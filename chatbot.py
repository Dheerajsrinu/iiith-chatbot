import streamlit as st
import uuid

from app.backend.db import (
    init_db,
    create_thread,
    get_messages_by_thread,
    get_threads_by_user
)
from app.backend.chat_service import run_chat_stream
from app.backend.model_loader import load_models
from app.helper import save_images
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.types import Interrupt

from app.backend.db import save_message, is_waiting_for_review
from langgraph.types import Command

from views.auth_view import render_auth_view
# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Checkout Chatbot",
    layout="wide"
)

if "user_id" not in st.session_state:
    render_auth_view()
    st.stop()
# -------------------------------------------------
# Load models ONCE
# -------------------------------------------------
load_models()

# -------------------------------------------------
# Init DB
# -------------------------------------------------
init_db()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("Conversations")

# if "user_id" not in st.session_state:
#     st.session_state.user_id = str(uuid.uuid4())
# if "user_id" not in st.session_state:
#     st.switch_page("pages/auth.py")

st.sidebar.divider()

# if st.sidebar.button("🚪 Logout"):
#     st.session_state.clear()
#     st.rerun()
_, menu_col = st.columns([9.0, 0.2])

with menu_col:
    with st.popover("⋮"):
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

user_id = st.session_state.user_id
st.sidebar.caption(f"User ID: {user_id}")

# ---- Image Upload (STABLE LOCATION) ----
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.sidebar.markdown("### 📎 Attach Images")
uploaded_images = st.sidebar.file_uploader(
    "Upload image(s)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"image_uploader_{st.session_state.uploader_key}"
)

st.sidebar.divider()

# ---- Threads ----
threads = get_threads_by_user(user_id)

if st.sidebar.button("➕ New Chat"):
    count=0
    if threads:
        count = len(threads)
    thread_id = create_thread(user_id=user_id, title=f"Custom Chat - {count}")
    st.session_state.thread_id = thread_id
    st.rerun()

for thread_id, title, db_user_id in threads:
    if user_id == db_user_id:
        if st.sidebar.button(title or str(thread_id), key=str(thread_id)):
            st.session_state.thread_id = str(thread_id)
            st.rerun()

# -------------------------------------------------
# Main Chat
# -------------------------------------------------
if "thread_id" not in st.session_state:
    st.info("Start a new chat or select one from the left.")
    st.stop()

st.title("Checkout Chatbot")



# -------------------------------------------------
# Render chat history
# -------------------------------------------------
messages = get_messages_by_thread(st.session_state.thread_id)

for role, content in messages:
    with st.chat_message(role):
        st.write(content)

# -------------------------------------------------
# Chat Input (NATIVE – DO NOT MOVE)
# -------------------------------------------------

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "awaiting_interrupt" not in st.session_state:
    st.session_state.awaiting_interrupt = False

if st.session_state.awaiting_interrupt:
    with st.chat_message("assistant"):
        st.markdown(f"⚠️ **{st.session_state.pending_interrupt}**")
user_input = st.chat_input("Type your message")

# -------------------------------------------------
# Handle submit
# -------------------------------------------------
if "seen_message_ids" not in st.session_state:
    st.session_state.seen_message_ids = set()


if user_input:

    image_paths = []
    user_message_parts = []
    if st.session_state.awaiting_interrupt:
        st.session_state.awaiting_interrupt = False
        st.session_state.pending_interrupt = None
    # -----------------------------
    # Attach images (if any)
    # -----------------------------
    if uploaded_images:
        image_bytes = [img.read() for img in uploaded_images]
        filenames = [img.name for img in uploaded_images]
        image_paths = save_images(image_bytes, filenames)

        user_message_parts.append(
            "📎 **Attached files:**\n"
            + "\n".join([f"- {name}" for name in filenames])
        )

    user_message_parts.append(user_input)
    full_user_message = "\n\n".join(user_message_parts)

    # -----------------------------
    # ONE user bubble
    # -----------------------------
    with st.chat_message("user"):
        st.markdown(full_user_message)

    # -----------------------------
    # Assistant streaming + tools
    # -----------------------------
    with st.chat_message("assistant"):

        status_holder = {"box": None}
        final_text_chunks = []

        def ai_stream():
            thread_id = st.session_state.thread_id
            is_review = is_waiting_for_review(thread_id)

            # -----------------------------------
            # 🔁 RESUME FROM INTERRUPT USING COMMAND
            # -----------------------------------
            if is_review:
                with st.chat_message("user"):
                    st.write(user_input)

                save_message(
                    st.session_state.thread_id,
                    "user",
                    user_input
                )

                # user_input here is the yes/no entered by the user
                cmd = Command(
                    resume=True,
                    update={
                        "messages": [HumanMessage(content=user_input)]
                    }
                )

                stream = run_chat_stream(
                    thread_id=thread_id,
                    command=cmd
                )

            # -----------------------------------
            # ▶️ NORMAL FLOW (NEW USER MESSAGE)
            # -----------------------------------
            else:
                stream = run_chat_stream(
                    thread_id=thread_id,
                    user_input=user_input,
                    images_list=image_paths
                )

            # -----------------------------------
            # STREAM HANDLING
            # -----------------------------------
            for event in stream:
                # ----------------------------
                # ⛔ INTERRUPT
                # ----------------------------
                if "__interrupt__" in event:
                    interrupt_obj = event["__interrupt__"][0]
                    question = interrupt_obj.value.get("question", "")

                    # Store interrupt state (NOT UI)
                    st.session_state.pending_interrupt = question
                    st.session_state.awaiting_interrupt = True

                    return  # ⛔ stop streaming

                # ----------------------------
                # 🧠 MESSAGES
                # ----------------------------
                if "messages" in event:
                    for msg in event["messages"]:
                        # Skip messages already shown
                        if msg.id in st.session_state.seen_message_ids:
                            continue

                        st.session_state.seen_message_ids.add(msg.id)

                        # 🔧 Tool message
                        if isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, "name", "tool")
                            st.status(f"🔧 Using `{tool_name}` …")

                        # 🤖 Assistant message
                        elif isinstance(msg, AIMessage):
                            yield msg.content



        assistant_text = st.write_stream(ai_stream())

        # -------- Finalize tool box --------
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )

    # -----------------------------
    # Persist final assistant message
    # -----------------------------
    if assistant_text:
        save_message(
            st.session_state.thread_id,
            "assistant",
            assistant_text
        )
    st.session_state.uploader_key += 1
    st.rerun()
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
from app.ui.styles import (
    apply_custom_styles, 
    render_sidebar_logo, 
    render_user_profile_bottom, 
    render_header_compact,
    render_image_preview_card,
    COLORS
)

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Retail Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Apply custom styles
# -------------------------------------------------
apply_custom_styles()

# -------------------------------------------------
# Auth check
# -------------------------------------------------
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
# Session state initialization
# -------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "pending_images" not in st.session_state:
    st.session_state.pending_images = []
if "show_image_confirm" not in st.session_state:
    st.session_state.show_image_confirm = False
if "confirmed_images" not in st.session_state:
    st.session_state.confirmed_images = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    # Logo at top
    render_sidebar_logo()
    
    st.markdown("")  # Spacing
    
    # ----- Navigation -----
    st.markdown("##### Navigation")
    
    # Only show Orders button (we're already on Chat page)
    if st.button("📦  My Orders", use_container_width=True, key="nav_orders"):
        st.switch_page("pages/orders_dashboard.py")
    
    st.divider()
    
    # ----- Conversations -----
    st.markdown("##### Conversations")
    
    user_id = st.session_state.user_id
    threads = get_threads_by_user(user_id)
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        count = len(threads) if threads else 0
        thread_id = create_thread(user_id=user_id, title=f"Chat {count + 1}")
        st.session_state.thread_id = thread_id
        st.session_state.pending_images = []
        st.session_state.confirmed_images = []
        st.session_state.show_image_confirm = False
        st.rerun()
    
    # Thread list
    if threads:
        for thread_id, title, db_user_id in threads:
            if user_id == db_user_id:
                is_active = st.session_state.get("thread_id") == str(thread_id)
                icon = "🟢" if is_active else "💬"
                
                if st.button(f"{icon} {title or 'Untitled'}", key=str(thread_id), use_container_width=True):
                    st.session_state.thread_id = str(thread_id)
                    st.session_state.pending_images = []
                    st.session_state.confirmed_images = []
                    st.session_state.show_image_confirm = False
                    st.rerun()
    else:
        st.caption("No conversations yet")
    
    # Spacer to push user profile to bottom
    st.markdown('<div style="flex: 1; min-height: 100px;"></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Logout button
    if st.button("🚪 Sign Out", use_container_width=True, key="logout"):
        st.session_state.clear()
        st.rerun()
    
    # User profile at bottom
    user_email = st.session_state.get("user_email", "User")
    render_user_profile_bottom(user_email)

# -------------------------------------------------
# Main Chat Area
# -------------------------------------------------
if "thread_id" not in st.session_state:
    # Welcome screen
    render_header_compact("Retail Checkout Assistant", "AI-Powered Product Analysis")
    
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 16px;
        margin: 1rem 0;
    ">
        <div style="font-size: 3.5rem; margin-bottom: 0.75rem;">🛒</div>
        <h2 style="color: #1a365d; margin-bottom: 0.75rem; font-size: 1.5rem;">Start Your Retail Analysis</h2>
        <p style="color: #718096; font-size: 1rem; max-width: 450px; margin: 0 auto;">
            Upload shelf images to detect products, count items, and analyze inventory using AI.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("##### What I can help you with")
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("🔍", "Detect Shelves", "Identify shelf structures"),
        ("📦", "Count Products", "Count product objects"),
        ("📊", "Empty Space", "Calculate empty %"),
        ("🏷️", "Recognize", "Identify products"),
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
                border: 1px solid #e2e8f0;
                height: 100%;
            ">
                <div style="font-size: 1.75rem; margin-bottom: 0.25rem;">{icon}</div>
                <div style="font-weight: 600; color: #1a365d; font-size: 0.9rem;">{title}</div>
                <div style="font-size: 0.75rem; color: #718096;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    st.info("👈 **Start a new chat** from the sidebar to begin!")
    st.stop()

# -------------------------------------------------
# Active Chat View
# -------------------------------------------------
render_header_compact("Retail Assistant", "Chat Session")

# -------------------------------------------------
# Render chat history
# -------------------------------------------------
messages = get_messages_by_thread(st.session_state.thread_id)

chat_container = st.container()
with chat_container:
    for role, content in messages:
        with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🤖"):
            st.markdown(content)

# -------------------------------------------------
# Image Upload Section (Above chat input)
# -------------------------------------------------
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "awaiting_interrupt" not in st.session_state:
    st.session_state.awaiting_interrupt = False

# Show interrupt message if awaiting
if st.session_state.awaiting_interrupt:
    with st.chat_message("assistant", avatar="🤖"):
        st.warning(f"⚠️ **Action Required:** {st.session_state.pending_interrupt}")
        st.caption("Please respond to continue...")

# -------------------------------------------------
# Image Upload & Preview Section (Hidden during interrupt)
# -------------------------------------------------
uploaded_images = None

# Only show file uploader when NOT awaiting interrupt
if not st.session_state.awaiting_interrupt:
    upload_col, spacer = st.columns([1, 3])

    with upload_col:
        uploaded_images = st.file_uploader(
            "📎 Attach Images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"image_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed"
        )

    # Handle newly uploaded images
    if uploaded_images and uploaded_images != st.session_state.pending_images:
        st.session_state.pending_images = uploaded_images
        st.session_state.show_image_confirm = True
        st.session_state.confirmed_images = []

    # Show image preview and confirmation
    if st.session_state.show_image_confirm and st.session_state.pending_images:
        st.markdown("---")
        st.markdown("##### 📷 Image Preview")
        st.caption("Review attached images before sending your message")
        
        # Image preview grid
        num_images = len(st.session_state.pending_images)
        cols = st.columns(min(num_images, 4))
        
        for idx, img in enumerate(st.session_state.pending_images[:4]):
            with cols[idx % 4]:
                st.image(img, use_container_width=True)
                st.caption(f"📄 {img.name}")
        
        if num_images > 4:
            st.caption(f"...and {num_images - 4} more image(s)")
        
        # Confirmation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Confirm", type="primary", use_container_width=True):
                st.session_state.confirmed_images = st.session_state.pending_images
                st.session_state.show_image_confirm = False
                st.rerun()
        
        with col2:
            if st.button("❌ Clear", use_container_width=True):
                st.session_state.pending_images = []
                st.session_state.confirmed_images = []
                st.session_state.show_image_confirm = False
                st.session_state.uploader_key += 1
                st.rerun()
        
        st.markdown("---")

    # Show confirmed images indicator
    if st.session_state.confirmed_images and not st.session_state.show_image_confirm:
        st.markdown(render_image_preview_card(len(st.session_state.confirmed_images)), unsafe_allow_html=True)

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
user_input = st.chat_input(
    "Type your message..." + (" (images attached)" if st.session_state.confirmed_images else ""),
    key="chat_input",
    disabled=st.session_state.is_processing
)

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
    # Process confirmed images
    # -----------------------------
    if st.session_state.confirmed_images:
        # Reset file positions for reading
        for img in st.session_state.confirmed_images:
            img.seek(0)
        
        image_bytes = [img.read() for img in st.session_state.confirmed_images]
        filenames = [img.name for img in st.session_state.confirmed_images]
        image_paths = save_images(image_bytes, filenames)
        
        user_message_parts.append(
            "📎 **Attached images:**\n"
            + "\n".join([f"- `{name}`" for name in filenames])
        )

    user_message_parts.append(user_input)
    full_user_message = "\n\n".join(user_message_parts)

    # -----------------------------
    # User message bubble
    # -----------------------------
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(full_user_message)

    # -----------------------------
    # Assistant streaming + tools (with loader)
    # -----------------------------
    with st.chat_message("assistant", avatar="🤖"):
        
        def ai_stream():
            thread_id = st.session_state.thread_id
            is_review = is_waiting_for_review(thread_id)

            # -----------------------------------
            # 🔁 RESUME FROM INTERRUPT
            # -----------------------------------
            if is_review:
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.write(user_input)

                save_message(
                    st.session_state.thread_id,
                    "user",
                    user_input
                )

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
            # ▶️ NORMAL FLOW
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
                # ⛔ INTERRUPT
                if "__interrupt__" in event:
                    interrupt_obj = event["__interrupt__"][0]
                    question = interrupt_obj.value.get("question", "")
                    st.session_state.pending_interrupt = question
                    st.session_state.awaiting_interrupt = True
                    return

                # 🧠 MESSAGES
                if "messages" in event:
                    for msg in event["messages"]:
                        if msg.id in st.session_state.seen_message_ids:
                            continue

                        st.session_state.seen_message_ids.add(msg.id)

                        # 🔧 Tool message - Show loading status
                        if isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, "name", "tool")
                            with st.status(f"🔧 Running `{tool_name}`...", expanded=True) as status:
                                st.write(f"Analyzing with **{tool_name}**...")
                                st.write("This may take a moment...")
                                status.update(label=f"✅ `{tool_name}` complete", state="complete", expanded=False)

                        # 🤖 Assistant message
                        elif isinstance(msg, AIMessage):
                            yield msg.content

        # Show processing indicator
        with st.spinner("🔄 Processing your request..."):
            assistant_text = st.write_stream(ai_stream())

    # -----------------------------
    # Persist final assistant message
    # -----------------------------
    if assistant_text:
        save_message(
            st.session_state.thread_id,
            "assistant",
            assistant_text
        )
    
    # Clear confirmed images after sending
    st.session_state.confirmed_images = []
    st.session_state.pending_images = []
    st.session_state.uploader_key += 1
    st.rerun()

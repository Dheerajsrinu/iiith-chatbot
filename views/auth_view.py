import streamlit as st
from app.backend.db import create_user, authenticate_user, get_user_by_email


def render_auth_styles():
    """Apply authentication page specific styles"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Remove top padding */
        .main .block-container {
            padding-top: 2rem !important;
        }
        
        .auth-container {
            max-width: 400px;
            margin: 0 auto;
        }
        
        .auth-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .auth-logo {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem;
            font-size: 2rem;
            box-shadow: 0 8px 30px rgba(26, 54, 93, 0.25);
        }
        
        .auth-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #1a365d;
            margin: 0 0 0.25rem 0;
        }
        
        .auth-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: #718096;
            margin: 0;
        }
        
        /* Custom button styling */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            border: none;
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            transition: all 0.3s ease;
            margin-top: 0.75rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(26, 54, 93, 0.3);
        }
        
        /* Input styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #1a365d;
            box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
        }
        
        /* Label styling */
        .stTextInput > label,
        .stTextArea > label {
            font-weight: 500;
            color: #2d3748;
            font-size: 0.85rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: #f7fafc;
            border-radius: 10px;
            padding: 3px;
            margin-bottom: 1.25rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            color: #718096;
        }
        
        .stTabs [aria-selected="true"] {
            background: white;
            color: #1a365d;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        
        /* Hide sidebar on auth page */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 10px;
            border-left-width: 4px;
            font-size: 0.9rem;
        }
        
        .auth-footer {
            text-align: center;
            margin-top: 1.5rem;
            font-size: 0.75rem;
            color: #a0aec0;
        }
    </style>
    """, unsafe_allow_html=True)


def render_auth_view():
    """Render the authentication view with modern styling"""
    
    # Apply custom styles
    render_auth_styles()
    
    # Already logged in → continue app
    if "user_id" in st.session_state:
        st.rerun()
    
    # Center the auth form
    col1, col2, col3 = st.columns([1.2, 1.5, 1.2])
    
    with col2:
        # Header with logo
        st.markdown("""
        <div class="auth-header">
            <div class="auth-logo">🛒</div>
            <h1 class="auth-title">Retail Assistant</h1>
            <p class="auth-subtitle">AI-Powered Checkout System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auth tabs
        tab_login, tab_signup = st.tabs(["🔐 Sign In", "✨ Create Account"])
        
        # -----------------------
        # LOGIN TAB
        # -----------------------
        with tab_login:
            st.caption("Enter your credentials to continue")
            
            email = st.text_input(
                "Email",
                placeholder="you@example.com",
                key="login_email"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
                key="login_password"
            )
            
            if st.button("Sign In", key="login_btn", type="primary"):
                if not email or not password:
                    st.error("Please fill in all fields")
                else:
                    with st.spinner("Signing in..."):
                        user = authenticate_user(email, password)
                        if user:
                            st.session_state.user_id = user["user_id"]
                            st.session_state.user_email = user["email"]
                            st.session_state.user_name = user.get("name", email.split('@')[0])
                            st.success(f"✅ Welcome back, {st.session_state.user_name}!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
        
        # -----------------------
        # SIGN UP TAB
        # -----------------------
        with tab_signup:
            st.caption("Create a new account to get started")
            
            # Name and Age in two columns
            col_name, col_age = st.columns([2, 1])
            with col_name:
                name = st.text_input(
                    "Full Name",
                    placeholder="John Doe",
                    key="signup_name"
                )
            with col_age:
                age = st.number_input(
                    "Age",
                    min_value=13,
                    max_value=120,
                    value=25,
                    key="signup_age"
                )
            
            email = st.text_input(
                "Email",
                placeholder="you@example.com",
                key="signup_email"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Min 6 characters",
                key="signup_password"
            )
            address = st.text_area(
                "Delivery Address",
                placeholder="House No, Street, City, State",
                key="signup_address",
                height=80
            )
            pincode = st.text_input(
                "Pincode",
                placeholder="6-digit code",
                max_chars=6,
                key="signup_pincode"
            )
            
            if st.button("Create Account", key="signup_btn", type="primary"):
                # Validation
                if not name or not email or not password or not address or not pincode:
                    st.error("⚠️ All fields are required")
                elif len(name) < 2:
                    st.error("⚠️ Name must be at least 2 characters")
                elif not pincode.isdigit() or len(pincode) != 6:
                    st.error("⚠️ Enter a valid 6-digit pincode")
                elif get_user_by_email(email):
                    st.error("⚠️ Email already registered")
                elif len(password) < 6:
                    st.error("⚠️ Password too short")
                else:
                    with st.spinner("Creating account..."):
                        user_id = create_user(
                            email=email,
                            password=password,
                            name=name,
                            age=int(age),
                            address=address,
                            pincode=int(pincode)
                        )
                        st.session_state.user_id = user_id
                        st.session_state.user_email = email
                        st.session_state.user_name = name
                        st.success("🎉 Account created!")
                        st.balloons()
                        st.rerun()
        
        # Footer
        st.markdown("""
        <div class="auth-footer">
            <p>Capstone Project 2025</p>
        </div>
        """, unsafe_allow_html=True)

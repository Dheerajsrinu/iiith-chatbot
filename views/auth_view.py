import streamlit as st
from app.backend.db import create_user, authenticate_user, get_user_by_email

def render_auth_view():
    col1, col2, col3 = st.columns([1, 2, 1])

    # Already logged in → continue app
    if "user_id" in st.session_state:
        st.rerun()

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
    with col2:
        # -----------------------
        # LOGIN
        # -----------------------
        with tab_login:
            st.subheader("Login")

            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login"):
                user = authenticate_user(email, password)

                if user:
                    st.session_state.user_id = user["user_id"]
                    st.session_state.user_email = user["email"]
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        # -----------------------
        # SIGN UP
        # -----------------------
        with tab_signup:
            st.subheader("Create Account")

            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")

            address = st.text_area(
                "Address",
                placeholder="House / Street / City / State",
                key="signup_address"
            )

            pincode = st.text_input(
                "Pincode",
                max_chars=6,
                key="signup_pincode"
            )

            if st.button("Sign Up"):
                # -----------------------
                # Validation
                # -----------------------
                if not email or not password or not address or not pincode:
                    st.error("All fields are required")
                    return

                if not pincode.isdigit() or len(pincode) != 6:
                    st.error("Pincode must be a 6-digit number")
                    return

                if get_user_by_email(email):
                    st.error("User already exists")
                    return

                # -----------------------
                # Create user
                # -----------------------
                user_id = create_user(
                    email=email,
                    password=password,
                    address=address,
                    pincode=pincode
                )

                st.session_state.user_id = user_id
                st.session_state.user_email = email
                st.success("Account created successfully")
                st.rerun()

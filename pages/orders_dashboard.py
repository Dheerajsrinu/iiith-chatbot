import streamlit as st
from views.auth_view import render_auth_view
from app.backend.db import init_db, get_orders_by_user

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="My Orders",
    layout="wide"
)

# -------------------------------------------------
# Auth check
# -------------------------------------------------
if "user_id" not in st.session_state:
    render_auth_view()
    st.stop()

user_id = st.session_state.user_id

# -------------------------------------------------
# Init DB
# -------------------------------------------------
init_db()

st.title("📦 My Orders")

orders = get_orders_by_user(user_id)

if not orders:
    st.info("You have not placed any orders yet.")
    st.stop()

# -------------------------------------------------
# Metrics
# -------------------------------------------------
st.subheader("📊 Overview")
st.metric("Total Orders", len(orders))

st.divider()

# -------------------------------------------------
# Orders table
# -------------------------------------------------
table_data = []

for order_id, products, created_at in orders:
    total_items = sum(products.values())

    table_data.append({
        "Order ID": str(order_id),
        "Items": total_items,
        "Created At": created_at.strftime("%Y-%m-%d %H:%M")
    })

st.dataframe(
    table_data,
    use_container_width=True,
    hide_index=True
)

st.divider()

# -------------------------------------------------
# Order details
# -------------------------------------------------
st.subheader("🧾 Order Details")

for order_id, products, created_at in orders:
    with st.expander(f"🛒 Order {order_id} — {created_at.strftime('%Y-%m-%d %H:%M')}"):
        for product_name, quantity in products.items():
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{product_name}**")
            col2.markdown(f"x {quantity}")

        st.markdown("---")
        st.markdown(f"**Total items:** `{sum(products.values())}`")

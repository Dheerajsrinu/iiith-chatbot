import uuid
import psycopg
from app.config import DB_CONFIG
from app.backend.security import hash_password, verify_password

def get_connection():
    return psycopg.connect(**DB_CONFIG)

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            # cur.execute("""CREATE DATABASE chatbot_db;""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id UUID PRIMARY KEY,
                    title TEXT,
                    user_id TEXT,
                    is_waiting_for_review BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                        
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    thread_id UUID NOT NULL,
                    role TEXT CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
                ON chat_messages(thread_id);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id UUID PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    address TEXT NOT NULL,
                    pincode INT NOT NULL,
                    age INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Add name and age columns if they don't exist (migration for existing tables)
            cur.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='users' AND column_name='name') THEN
                        ALTER TABLE users ADD COLUMN name TEXT DEFAULT '';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='users' AND column_name='age') THEN
                        ALTER TABLE users ADD COLUMN age INT DEFAULT 0;
                    END IF;
                END $$;
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    products JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_orders_thread
                        FOREIGN KEY (user_id)
                        REFERENCES chat_threads(thread_id)
                        ON DELETE CASCADE
                );
            """)
            conn.commit()

def create_thread(user_id, title="Custom Chat"):
    thread_id = str(uuid.uuid4())
    user_id = str(user_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_threads (thread_id, title, user_id) VALUES (%s, %s, %s)",
                (thread_id, title, user_id)
            )
            conn.commit()
    return thread_id

def get_all_threads():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT thread_id, title
                FROM chat_threads
                ORDER BY created_at DESC
            """)
            return cur.fetchall()

def get_threads_by_user(user_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT thread_id, title, user_id FROM chat_threads WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            return cur.fetchall()

def save_message(thread_id: str, role: str, content: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (thread_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (thread_id, role, content)
            )
            conn.commit()

def get_messages_by_thread(thread_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE thread_id = %s
                ORDER BY created_at
                """,
                (thread_id,)
            )
            return cur.fetchall()

def clear_waiting_for_review(thread_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                UPDATE chat_threads
                SET is_waiting_for_review = FALSE,
                    updated_at = NOW()
                    WHERE thread_id = %s
            """
            cur.execute(query, (thread_id,))

def mark_waiting_for_review(thread_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                UPDATE chat_threads
                SET is_waiting_for_review = TRUE,
                    updated_at = NOW()
                    WHERE thread_id = %s
            """
            cur.execute(query, (thread_id,))

def is_waiting_for_review(thread_id: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT is_waiting_for_review
                FROM chat_threads
                WHERE thread_id = %s
                """,
                (thread_id,)
            )

            row = cur.fetchone()

            if row is None:
                return False
            
            return bool(row[0])

def create_user(email: str, password: str, name: str, age: int, address: str, pincode: int):
    user_id = str(uuid.uuid4())
    pwd_plain = password  # no hashing

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, email, password_hash, name, age, address, pincode)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, email, pwd_plain, name, age, address, pincode)
            )
            conn.commit()

    return user_id


def get_user_by_email(email: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, email, password_hash, name
                FROM users
                WHERE email = %s
                """,
                (email,)
            )
            return cur.fetchone()


def authenticate_user(email: str, password: str):
    row = get_user_by_email(email)

    if not row:
        return None

    user_id, email, stored_password, name = row

    # plain-text comparison
    if password != stored_password:
        return None

    return {
        "user_id": str(user_id),
        "email": email,
        "name": name or email.split('@')[0]  # Fallback to email prefix if no name
    }

def create_order(user_id: str, products: dict):
    order_id = str(uuid.uuid4())

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO orders (id, user_id, products)
                VALUES (%s, %s, %s)
                """,
                (order_id, user_id, psycopg.types.json.Json(products))
            )
            conn.commit()

    return order_id

def get_orders_by_user(user_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.id, o.products, o.created_at
                FROM orders o 
                join chat_threads ct on ct.thread_id = o.user_id
                WHERE ct.user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
            return cur.fetchall()

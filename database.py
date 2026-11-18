"""
Simple local SQLite database for Sanjeevani AI 2.0.
Handles: user accounts, chat sessions, and chat history.
"""

import sqlite3
import hashlib
import secrets
import uuid
from datetime import datetime
from contextlib import contextmanager


DB_PATH = "sanjeevani.db"


@contextmanager
def get_connection():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()

    finally:
        conn.close()


def init_db():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        """)


# --------------------------------------------------
# Password hashing
# --------------------------------------------------

def _hash_password(password, salt=None):

    if salt is None:
        salt = secrets.token_hex(16)

    pwd_hash = hashlib.sha256(
        (salt + password).encode("utf-8")
    ).hexdigest()

    return pwd_hash, salt


# --------------------------------------------------
# User management
# --------------------------------------------------

def create_user(username, password):

    username = username.strip()

    if not username or not password:
        return False, "Username aur password khali nahi ho sakte."

    if len(password) < 4:
        return False, "Password kam se kam 4 characters ka hona chahiye."

    pwd_hash, salt = _hash_password(password)

    try:

        with get_connection() as conn:

            conn.execute(
                """
                INSERT INTO users
                (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    pwd_hash,
                    salt,
                    datetime.utcnow().isoformat()
                ),
            )

        return True, "Account ban gaya! Ab login karo."

    except sqlite3.IntegrityError:

        return False, "Ye username pehle se registered hai."


def authenticate_user(username, password):

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT id, password_hash, salt
            FROM users
            WHERE username = ?
            """,
            (username.strip(),)
        ).fetchone()

    if row is None:
        return None

    pwd_hash, _ = _hash_password(
        password,
        salt=row["salt"]
    )

    if pwd_hash == row["password_hash"]:
        return row["id"]

    return None


# --------------------------------------------------
# Chat sessions
# --------------------------------------------------

def create_session(user_id, title="New Chat"):

    session_id = str(uuid.uuid4())

    with get_connection() as conn:

        conn.execute(
            """
            INSERT INTO chat_sessions
            (id, user_id, title, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                title,
                datetime.utcnow().isoformat()
            ),
        )

    return session_id


def rename_session(session_id, new_title):

    with get_connection() as conn:

        conn.execute(
            """
            UPDATE chat_sessions
            SET title = ?
            WHERE id = ?
            """,
            (
                new_title,
                session_id
            )
        )


def get_sessions(user_id):

    """
    Sirf wahi chats return hongi jinke andar
    kam se kam ek message hai.

    Isse empty 'New Chat' entries sidebar mein
    show nahi hongi.
    """

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                s.id,
                s.title,
                s.created_at
            FROM chat_sessions s
            WHERE s.user_id = ?
            AND EXISTS (
                SELECT 1
                FROM chat_messages m
                WHERE m.session_id = s.id
            )
            ORDER BY s.created_at DESC
            """,
            (user_id,)
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def delete_session(session_id):

    """
    Complete chat delete karta hai:
    1. Chat messages
    2. Chat session
    """

    with get_connection() as conn:

        conn.execute(
            """
            DELETE FROM chat_messages
            WHERE session_id = ?
            """,
            (session_id,)
        )

        conn.execute(
            """
            DELETE FROM chat_sessions
            WHERE id = ?
            """,
            (session_id,)
        )


# --------------------------------------------------
# Messages
# --------------------------------------------------

def save_message(session_id, role, content):

    with get_connection() as conn:

        conn.execute(
            """
            INSERT INTO chat_messages
            (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                datetime.utcnow().isoformat()
            ),
        )


def get_messages(session_id):

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                role,
                content,
                created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]
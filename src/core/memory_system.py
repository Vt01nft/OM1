"""
Long-Term Memory System for OpenMind OM1
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongTermMemory:
    """Persistent memory system for OM1 agents."""
    
    def __init__(self, db_path: str = "data/om1_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Memory system initialized at {self.db_path}")
    
    def _init_database(self):
        """Create database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_memory(self, role: str, content: str, agent_id: str, session_id: str):
        """Add a new memory."""
        timestamp = datetime.now().timestamp()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memories (agent_id, session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (agent_id, session_id, role, content, timestamp))
            conn.commit()
        logger.info(f"Added memory for {agent_id}")


if __name__ == "__main__":
    print("=== OM1 Long-Term Memory System Demo ===\n")
    memory = LongTermMemory()
    memory.add_memory("user", "Hello, Spot!", "spot_demo", "test_session_1")
    memory.add_memory("assistant", "Hello! How can I help?", "spot_demo", "test_session_1")
    print("\n✅ Demo complete! Memory system is working.")
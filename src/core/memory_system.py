"""
Long-term Memory System for OM1
Addresses issues #856 and #880 - Adds persistent conversational memory
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    timestamp: str
    role: str  # 'user' or 'assistant'
    content: str
    agent_id: str
    session_id: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None


class LongTermMemory:
    """
    Persistent memory system for OM1 agents.
    Stores conversations, retrieves relevant context, and manages memory lifecycle.
    """
    
    def __init__(self, db_path: str = "data/memory.db", max_context_entries: int = 10):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_context_entries = max_context_entries
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT,
                importance_score REAL DEFAULT 0.5
            )
        """)
        
        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_session 
            ON memories(agent_id, session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON memories(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def add_memory(
        self, 
        role: str, 
        content: str, 
        agent_id: str, 
        session_id: str,
        metadata: Optional[Dict] = None,
        importance_score: float = 0.5
    ) -> str:
        """Add a new memory entry"""
        memory_id = self._generate_id(content, session_id)
        timestamp = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, timestamp, role, content, agent_id, session_id, metadata, importance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            timestamp,
            role,
            content,
            agent_id,
            session_id,
            json.dumps(metadata) if metadata else None,
            importance_score
        ))
        
        conn.commit()
        conn.close()
        
        return memory_id
    
    def get_recent_memories(
        self, 
        agent_id: str, 
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Retrieve recent memories for an agent/session"""
        if limit is None:
            limit = self.max_context_entries
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            query = """
                SELECT id, timestamp, role, content, agent_id, session_id, metadata
                FROM memories
                WHERE agent_id = ? AND session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (agent_id, session_id, limit))
        else:
            query = """
                SELECT id, timestamp, role, content, agent_id, session_id, metadata
                FROM memories
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (agent_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        memories = []
        for row in rows:
            metadata = json.loads(row[6]) if row[6] else None
            memories.append(MemoryEntry(
                id=row[0],
                timestamp=row[1],
                role=row[2],
                content=row[3],
                agent_id=row[4],
                session_id=row[5],
                metadata=metadata
            ))
        
        return list(reversed(memories))
    
    def get_context_window(
        self, 
        agent_id: str, 
        session_id: str,
        include_past_sessions: bool = True
    ) -> str:
        """Generate a formatted context window for the LLM"""
        current_memories = self.get_recent_memories(agent_id, session_id, limit=5)
        
        context_parts = ["=== Current Conversation ==="]
        for mem in current_memories:
            context_parts.append(f"{mem.role.upper()}: {mem.content}")
        
        if include_past_sessions:
            past_memories = self.get_recent_memories(agent_id, limit=5)
            if past_memories:
                context_parts.append("\n=== Relevant Past Context ===")
                for mem in past_memories[:3]:
                    if mem.session_id != session_id:
                        context_parts.append(f"{mem.role.upper()}: {mem.content}")
        
        return "\n".join(context_parts)
    
    def _generate_id(self, content: str, session_id: str) -> str:
        """Generate a unique ID for a memory"""
        unique_str = f"{content}{session_id}{datetime.utcnow().isoformat()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]
    
    def get_statistics(self, agent_id: str) -> Dict:
        """Get memory statistics for an agent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_memories,
                COUNT(DISTINCT session_id) as total_sessions,
                MIN(timestamp) as first_memory,
                MAX(timestamp) as last_memory
            FROM memories
            WHERE agent_id = ?
        """, (agent_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            "total_memories": row[0],
            "total_sessions": row[1],
            "first_memory": row[2],
            "last_memory": row[3]
        }


if __name__ == "__main__":
    memory = LongTermMemory()
    session = "test_session_001"
    memory.add_memory("user", "Hello!", "spot", session)
    print("Memory system works!")
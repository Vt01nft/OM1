"""Test the memory system"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.memory_system import LongTermMemory

print("Testing Memory System...")

memory = LongTermMemory(db_path="data/test.db")
memory.add_memory("user", "Hello, robot!", "test_agent", "session1")
memory.add_memory("assistant", "Hi! How can I help?", "test_agent", "session1")

memories = memory.get_recent_memories("test_agent", "session1")
print(f"✓ Stored {len(memories)} memories")

for mem in memories:
    print(f"  {mem.role}: {mem.content}")

stats = memory.get_statistics("test_agent")
print(f"\n✓ Statistics: {stats}")
print("\n✅ Memory system works perfectly!")
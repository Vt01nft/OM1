import sys
sys.path.insert(0, 'src')

import types
mock = types.ModuleType('providers.singleton')
def singleton(cls): return cls
mock.singleton = singleton
sys.modules['providers.singleton'] = mock

with open('src/providers/voice_assistant_provider.py', 'r') as f:
    code = f.read()
code = code.replace('from .singleton import singleton', 'from providers.singleton import singleton')
exec(code)

assistant = VoiceAssistantProvider()

# Test command parsing
commands = [
    'check my balance',
    'send 0.5 SOL to alice',
    'pay 1.2 solana to bob',
    'transfer 0.01 sol',
    'what is my balance',
    'help',
    'quit',
]

print('Testing command parsing:')
for cmd in commands:
    result = assistant.parse_command(cmd)
    print(f'  "{cmd}" -> {result}')

print()
assistant.speak('Voice assistant is ready!')
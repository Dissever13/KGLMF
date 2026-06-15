from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Agent:
    id: str
    role: str
    system_prompt: str
    history: List[Dict[str, str]] = field(default_factory=list)

    def add_to_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_history_text(self, max_tokens: int = 2000) -> str:
        text = ""
        for msg in self.history[-20:]:
            text += f"{msg['role']}: {msg['content']}\n"
            if len(text) > max_tokens:
                break
        return text

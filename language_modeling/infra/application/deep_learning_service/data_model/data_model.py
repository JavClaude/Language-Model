from pydantic import BaseModel


class TextInput(BaseModel):
    seed_str: str
    maximum_sequence_length: int
    top_k_word: int


class TextOutput(BaseModel):
    seed_str: str
    generated_text: str

import re
def clean_text(text:str):
    text = text.lower()
    text = re.sub(r"\s+"," ",text)
    text = re.sub(r"[^a-zA-Z0-9.,!?]","",text)
    return text.strip()

def chunk_text(text:str,chunk_size:int = 300):
    words = text.split()
    chunks = []
    for i in range(0,len(words),300):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
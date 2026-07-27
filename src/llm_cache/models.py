from typing import TypedDict

class CacheEntry(TypedDict):
    entry_id:str
    query: str
    thread_id: str
    response: str
    timestamp: int
    hits: int

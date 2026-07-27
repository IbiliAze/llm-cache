import hashlib


def fingerprint(query: str, **params: object) -> str:
  normalized = ' '.join(query.lower().split())
  material = repr((normalized, sorted(params.items())))
  return hashlib.sha256(material.encode()).hexdigest()

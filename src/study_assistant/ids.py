import hashlib
import uuid


def stable_id(*parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:32]
    return str(uuid.UUID(hex=digest))

import random
import string

def random_file(prefix: str = "", extension: str = "wav") -> str:
    prefix = f"{prefix}-" if prefix else prefix
    return f"{prefix}{random_str()}.{extension}"

def random_str(size: int = 16) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=size))
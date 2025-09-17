import os
import hashlib
import logging


def validate_file(file_path: str, max_file_size: int = 100 * 1024 * 1024) -> bool:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            raise ValueError(f"File too large: {file_size} bytes")

        if file_size == 0:
            raise ValueError("File is empty")

        valid_extensions = ['.csv', '.json', '.xlsx', '.txt']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported file type: {file_path}")

        return True
    except Exception as exc:
        logging.getLogger(__name__).error(f"File validation failed: {str(exc)}")
        return False


def calculate_file_hash(file_path: str) -> str | None:
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as exc:
        logging.getLogger(__name__).error(f"Hash calculation failed: {str(exc)}")
        return None



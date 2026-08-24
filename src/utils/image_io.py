import os
from typing import List
from src.utils.exceptions import InvalidInputError

def validate_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if the given filename has an allowed extension."""
    if '.' not in filename:
        raise InvalidInputError("File does not have an extension.")
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        raise InvalidInputError(f"Unsupported file extension. Allowed: {', '.join(allowed_extensions)}")
    
    return True

def validate_file_size(file_content: bytes, max_size_mb: int) -> bool:
    """Check if the file size is within the allowed limit."""
    size_in_bytes = len(file_content)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if size_in_bytes == 0:
        raise InvalidInputError("File is empty.")
        
    if size_in_bytes > max_size_bytes:
        raise InvalidInputError(f"File size exceeds the {max_size_mb}MB limit.")
        
    return True

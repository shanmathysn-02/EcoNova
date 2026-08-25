from werkzeug.datastructures import FileStorage
from src.utils.exceptions import InvalidInputError
from src.utils.image_io import validate_extension, validate_file_size

def validate_image_upload(file: FileStorage, allowed_extensions: list[str], max_size_mb: int) -> bool:
    """
    Validates an uploaded image file for Phase 2 readiness.
    Checks if file exists, is not empty, has a valid extension, and is within size limits.
    """
    if not file or not file.filename:
        raise InvalidInputError("No file provided.")
    
    validate_extension(file.filename, allowed_extensions)
    
    file_content = file.read()
    validate_file_size(file_content, max_size_mb)
    
    # Reset file pointer after reading so it can be used later
    file.seek(0)
    
    return True

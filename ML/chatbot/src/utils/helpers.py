import base64
import mimetypes
from pathlib import Path
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import NoSuchTableError

def encode_image_to_base64(image_path: Path) -> tuple[str | None, str | None]:
    """Reads an image file, encodes it to base64, and determines MIME type."""
    try:
        if not image_path.is_file():
            print(f"Error: Image file not found at {image_path}")
            return None, None

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image'):
            print(f"Error: Invalid image MIME type: {mime_type} for {image_path}")
            return None, None

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string, mime_type

    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None, None

def table_exists(inspector: Inspector, table_name: str, schema: str | None = None) -> bool:
    """Checks if a table exists using the inspector."""
    try:
        # inspector.get_columns() raises NoSuchTableError if table doesn't exist
        inspector.get_columns(table_name, schema=schema)
        return True
    except NoSuchTableError:
        return False
    except Exception as e:
        print(f"Error checking if table {table_name} exists: {e}")
        return False # Assume false on other errors

# Add any other general utility functions here
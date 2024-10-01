import os
import json
import PyPDF2

def load_document(file_path):
    """Load a text document from the given file path."""
    file_path = os.path.normpath(file_path)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    # Attempt to open and read the document
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin1') as file:
                content = file.read()
            return content
        except Exception as e:
            raise Exception(f"An error occurred while reading the file with 'latin1' encoding: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

def load_pdf_document(file_path):
    """Load a PDF document from the given file path."""
    file_path = os.path.normpath(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + ' '
        return text.strip()
    except Exception as e:
        raise Exception(f"An error occurred while reading the PDF file: {e}")

def load_json_document(file_path):
    """Load a JSON document from the given file path."""
    file_path = os.path.normpath(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data)  # Convert JSON object to string for processing
    except Exception as e:
        raise Exception(f"An error occurred while reading the JSON file: {e}")

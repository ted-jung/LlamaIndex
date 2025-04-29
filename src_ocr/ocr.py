# ===========================================================================
# Ollama OCR
# Created: 29, Apr 2025
# Updated: 29, Apr 2025
# Writer: Ted, Jung
# Description:
#   Extract all text from the image and pdf Using Ollama OCR
#   output format: markdown, text, json, structured, key_value, table
# ===========================================================================

from ollama_ocr import OCRProcessor

ocr = OCRProcessor(
    model_name="llama3.2-vision:11b", 
    base_url="http://localhost:11434/api/generate",)

result = ocr.process_image(
    image_path="/Users/tedj/Desktop/uber.png", # path to your pdf files "path/to/your/file.pdf"
    format_type="markdown",  # Options: markdown, text, json, structured, key_value
    custom_prompt="Extract all text, from top to bottom as it looks like.", # Optional custom prompt
    language="Korean" # Specify the language of the text (New! 🆕)
)
print(result)
import base64
from pathlib import Path

def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    return b64_string

# Convert the image
image_path = Path(__file__).parent / 'static' / 'fruit-pattern-bg.png'
base64_data = image_to_base64(str(image_path))
print(f"data:image/png;base64,{base64_data}") 
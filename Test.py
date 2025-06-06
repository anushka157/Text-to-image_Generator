from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# Initialize the Gemini client with your API key
client = genai.Client(api_key=' ')  # Replace with your actual API key

# Get input from the user
user_prompt = input("Enter a description for the image you want to generate: ")

# input
contents = (user_prompt,)

# Generate content using Gemini
response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
)

# Process and display the response
for part in response.candidates[0].content.parts:
    if part.text is not None:
        print("Text Output:\n", part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        plt.imshow(image)
        plt.axis('off')
        plt.title("Generated Image")
        plt.show()

import base64
import io
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
    # base_url="https://fauengtrussed.fau.edu/provider/generic",
)
_MODEL = "gpt-4o"


def _img_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def generate_script(slide_image) -> str:
    """Generate a spoken presentation script by visually reading a slide image."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(slide_image)}"}},
                {"type": "text", "text": (
                    "You are writing a script for a live presentation. "
                    "Look at this slide and write a natural, conversational spoken script "
                    "as if a confident human presenter is delivering it live to an audience. "
                    "Use first-person language and speak directly to the audience. "
                    "Describe any charts, diagrams, or visuals as you would out loud. "
                    "Vary your sentence rhythm — mix short punchy sentences with longer ones. "
                    "Never use placeholder text or stage directions. "
                    "Write exactly what should be spoken, nothing else. Keep it to 4-6 sentences."
                )},
            ],
        }],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

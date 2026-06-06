import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLMOD_API_KEY")
print(api_key)
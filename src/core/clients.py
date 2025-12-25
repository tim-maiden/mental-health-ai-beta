import tiktoken
from openai import OpenAI
from supabase import create_client, Client
from src.config import OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
encoding = tiktoken.get_encoding("cl100k_base")


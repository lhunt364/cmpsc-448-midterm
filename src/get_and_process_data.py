from datasets import load_dataset

ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
df = ds.to_pandas()
# dataset has 20 models, filter that down to 4 groups, 9 models: GPT = gpt-4 + gpt-3.5-turbo, Claude = claude-v1 + claude-instant-v1, PaLM = palm-2, Llama = vicuna-13b, koala-13b, alpaca-13b, and llama-13b

import os
from google import genai
import pandas as pd
from google.genai import types

source = pd.read_csv(r'Data\Captions\gemini_prompts_other_frame_val.csv')
responses = source['response'].tolist()


client = genai.Client()

all_embeddings = []

# Process in batches of 100
batch_size = 100
for i in range(0, len(responses), batch_size):
    batch = responses[i:i+batch_size]
    print(f'processing batch {i}')
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch,
        config=types.EmbedContentConfig(task_type="CLASSIFICATION", output_dimensionality = 768),
    )
    print(result)
    # Extract embeddings from this batch
    all_embeddings.extend([embedding.values for embedding in result.embeddings])

# Convert to DataFrame (rows = embeddings, columns = dimensions)
df = pd.DataFrame(all_embeddings)
df['title'] = source['video_name']
df.to_csv('gemini_embeds_val_diff_frames.csv', index=False)

print(f"Saved {len(all_embeddings)} embeddings to CSV")

import pandas as pd
import os


df = pd.read_csv("precise_processing_results.csv")  # Replace with actual filename


base_url = "https://senseiprod.github.io/audio-storage/processed_audios/"


df['url'] = df['output_file'].apply(lambda path: base_url + os.path.basename(path))


df[['name', 'url']].to_csv("voices.csv", index=False, encoding='utf-8')

print("✅ name_url_mapping.csv generated successfully.")

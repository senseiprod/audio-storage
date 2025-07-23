import os
import csv

# Folder containing audio files
folder_path = 'processed_audios'

# Base URL prefix
base_url = 'https://senseiprod.github.io/audio-storage/processed_audios/'

# Output CSV file name
csv_filename = 'audio_files.csv'

def extract_name(filename):
    # Remove extension
    name = filename.replace('.wav', '')
    # Remove _original if present
    if name.endswith('_original'):
        name = name[:-9]  # remove last 9 chars "_original"
    return name

def main():
    files = os.listdir(folder_path)
    # Filter only wav files ending with _original or just .wav
    valid_files = [f for f in files if f.endswith('.wav') and (f.endswith('_original.wav') or not '_original' in f)]

    # Sort files alphabetically (optional)
    valid_files.sort()

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'url'])
        for filename in valid_files:
            name = extract_name(filename)
            url = base_url + filename
            writer.writerow([name, url])

    print(f'CSV file "{csv_filename}" generated successfully.')

if __name__ == "__main__":
    main()

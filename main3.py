import os
import csv
import urllib.parse

folder_path = 'processed_audios'
base_url = 'https://senseiprod.github.io/audio-storage/processed_audios/'
csv_filename = 'audio_files.csv'

def extract_name(filename):
    name = filename.replace('.wav', '')
    if name.endswith('_original'):
        name = name[:-9]
    return name

def main():
    files = os.listdir(folder_path)
    valid_files = [f for f in files if f.endswith('.wav') and (f.endswith('_original.wav') or not '_original' in f)]
    valid_files.sort()

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'url'])
        for filename in valid_files:
            name = extract_name(filename)
            # URL encode the filename part only (not the whole URL)
            encoded_filename = urllib.parse.quote(filename)
            url = base_url + encoded_filename
            writer.writerow([name, url])

    print(f'CSV file "{csv_filename}" generated successfully.')

if __name__ == "__main__":
    main()

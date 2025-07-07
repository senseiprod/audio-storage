import pandas as pd
import requests
import os
from pydub import AudioSegment
import speech_recognition as sr
from pydub.silence import split_on_silence, detect_silence
import tempfile
import re
from datetime import datetime
import unicodedata
import difflib
from collections import defaultdict
import numpy as np

class PreciseAudioEditor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.target_phrase = "من منصة لهجات العربية"
        self.target_words = ["من", "منصة", "لهجات", "العربية"]
        self.recognizer = sr.Recognizer()
        
        # Create directories for outputs
        os.makedirs("downloaded_audios", exist_ok=True)
        os.makedirs("processed_audios", exist_ok=True)
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs("debug_segments", exist_ok=True)
    
    def normalize_arabic_text(self, text):
        """Enhanced Arabic text normalization"""
        if not text:
            return text
        
        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        
        # Normalize different forms of Arabic letters
        replacements = {
            'ة': 'ه',  # Ta marbuta to Ha
            'ى': 'ي',  # Alif maksura to Ya
            'إ': 'ا',  # Alif with hamza below to Alif
            'أ': 'ا',  # Alif with hamza above to Alif
            'آ': 'ا',  # Alif with madda to Alif
            'ء': '',   # Remove hamza
            'ؤ': 'و',  # Waw with hamza to Waw
            'ئ': 'ي',  # Ya with hamza to Ya
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra whitespace and normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def download_audio(self, url, filename):
        """Download audio file from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = os.path.join("downloaded_audios", filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {filename}")
            return file_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    # NEW, MORE RELIABLE CHUNKING METHOD
    def create_overlapping_chunks(self, audio_path, chunk_length_ms=10000, overlap_ms=2000):
        """
        Creates overlapping audio chunks. This is more robust for finding phrases
        that might be split by non-overlapping chunk boundaries.
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            chunks = []
            start = 0
            audio_len = len(audio)

            while start < audio_len:
                end = start + chunk_length_ms
                chunk_audio = audio[start:end]
                
                chunks.append({
                    'start_time': start,
                    'end_time': min(end, audio_len),
                    'audio': chunk_audio
                })
                
                # Move the start forward by the chunk length minus the overlap
                step = chunk_length_ms - overlap_ms
                start += step
            
            print(f"Created {len(chunks)} overlapping chunks")
            return chunks
        except Exception as e:
            print(f"Error creating overlapping chunks: {e}")
            return []

    def transcribe_chunk_with_confidence(self, chunk_audio, chunk_id):
        """Transcribe a single chunk with multiple attempts and confidence scoring"""
        try:
            # Export chunk to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_f:
                temp_path = temp_f.name
            
            chunk_audio.export(temp_path, format="wav")
            
            best_transcript = ""
            best_confidence = 0.0
            
            # Use 'ar-AE' or 'ar-SA' as they are often robust
            language_codes = ['ar-AE', 'ar-SA', 'ar-EG', 'ar']
            
            for lang_code in language_codes:
                try:
                    with sr.AudioFile(temp_path) as source:
                        audio_data = self.recognizer.record(source)
                        
                    # Try transcription with this language code
                    transcript = self.recognizer.recognize_google(audio_data, language=lang_code)
                    confidence = self.calculate_arabic_confidence(transcript)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_transcript = transcript
                        
                except (sr.UnknownValueError, sr.RequestError):
                    continue
            
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return best_transcript, best_confidence
                
        except Exception as e:
            # Don't print the error here to avoid clutter, it's usually just a silent chunk
            # print(f"Error transcribing chunk {chunk_id}: {e}")
            return "", 0.0
    
    def calculate_arabic_confidence(self, text):
        """Calculate confidence score for Arabic text"""
        if not text:
            return 0.0
        
        # Check for Arabic characters
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 0.0
        
        arabic_ratio = arabic_chars / total_chars
        
        # Bonus for containing target words
        normalized_text = self.normalize_arabic_text(text)
        target_word_bonus = 0
        
        for word in self.target_words:
            normalized_word = self.normalize_arabic_text(word)
            if normalized_word in normalized_text.split(): # Check for whole word match
                target_word_bonus += 0.25
        
        return min(1.0, arabic_ratio + target_word_bonus)
    
    def find_exact_phrase_boundaries(self, chunk_transcripts, target_phrase):
        """Find exact boundaries of the target phrase using advanced matching"""
        try:
            normalized_target = self.normalize_arabic_text(target_phrase)
            target_words = normalized_target.split()
            
            print(f"Looking for exact phrase: {target_phrase}")
            print(f"Normalized target: {normalized_target}")
            print(f"Target words: {target_words}")
            
            # Method 1: Exact phrase matching in a single chunk
            for i, chunk in enumerate(chunk_transcripts):
                if not chunk['transcript'] or chunk['confidence'] < 0.5: # Increased confidence threshold
                    continue
                    
                normalized_transcript = self.normalize_arabic_text(chunk['transcript'])
                
                # Use a more flexible 'in' check
                if normalized_target in normalized_transcript:
                    print(f"EXACT MATCH found in chunk {i+1}")
                    print(f"Chunk transcript: '{chunk['transcript']}'")
                    
                    start_time, end_time = self.find_phrase_boundaries_in_chunk(
                        chunk, normalized_target, target_words
                    )
                    
                    if start_time is not None and end_time is not None:
                        return start_time, end_time

            # If the above fails, try a slightly different fuzzy check
            print("Exact match failed, trying fuzzy matching within chunks...")
            fuzzy_match = self.find_fuzzy_phrase_match(chunk_transcripts, normalized_target)
            if fuzzy_match:
                return fuzzy_match
            
            return None, None
            
        except Exception as e:
            print(f"Error finding phrase boundaries: {e}")
            return None, None
    
    def find_phrase_boundaries_in_chunk(self, chunk, normalized_target, target_words):
        """Find precise boundaries of phrase within a chunk"""
        try:
            normalized_transcript = self.normalize_arabic_text(chunk['transcript'])
            transcript_words = normalized_transcript.split()
            
            # Find the sequence of target words in the transcript words
            match_index = -1
            for i in range(len(transcript_words) - len(target_words) + 1):
                if transcript_words[i:i+len(target_words)] == target_words:
                    match_index = i
                    break
            
            if match_index != -1:
                chunk_duration = chunk['end_time'] - chunk['start_time']
                words_in_chunk = len(transcript_words)
                if words_in_chunk == 0: return None, None

                # Calculate position based on word count
                start_ratio = match_index / words_in_chunk
                end_ratio = (match_index + len(target_words)) / words_in_chunk

                # Add a small buffer/padding
                buffer_ms = 300 
                
                start_time_offset = (chunk_duration * start_ratio) - buffer_ms
                end_time_offset = (chunk_duration * end_ratio) + buffer_ms
                
                actual_start = chunk['start_time'] + max(0, start_time_offset)
                actual_end = chunk['start_time'] + min(chunk_duration, end_time_offset)
                
                print(f"Calculated precise timing: {actual_start:.0f}ms - {actual_end:.0f}ms")
                return actual_start, actual_end
            
            return None, None
            
        except Exception as e:
            print(f"Error finding boundaries in chunk: {e}")
            return None, None
    
    def find_fuzzy_phrase_match(self, chunk_transcripts, normalized_target):
        """Find fuzzy match for the target phrase"""
        best_match_chunk = None
        best_ratio = 0.0

        for chunk in chunk_transcripts:
            if not chunk['transcript']:
                continue
            
            normalized_transcript = self.normalize_arabic_text(chunk['transcript'])
            ratio = difflib.SequenceMatcher(None, normalized_target, normalized_transcript).ratio()
            
            # We want a high similarity ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_chunk = chunk

        # Set a threshold for what we consider a "good enough" match
        if best_ratio > 0.7: # e.g., 70% similar
            print(f"Found fuzzy match in chunk with {best_ratio:.2%} similarity.")
            print(f"Chunk transcript: '{best_match_chunk['transcript']}'")
            # Use the entire chunk's duration as a safe bet, with padding
            padding = 200 # ms
            start_time = max(0, best_match_chunk['start_time'] - padding)
            end_time = best_match_chunk['end_time'] + padding
            return start_time, end_time

        return None
    
    def remove_phrase_from_audio(self, audio_path, start_time, end_time):
        """Remove the specified time segment from audio with crossfade"""
        try:
            audio = AudioSegment.from_file(audio_path)
            
            print(f"Original audio duration: {len(audio)}ms")
            print(f"Removing segment from {start_time:.0f}ms to {end_time:.0f}ms")
            
            start_time = max(0, int(start_time))
            end_time = min(len(audio), int(end_time))
            
            if start_time >= end_time:
                print("Invalid time range. Start time is after end time.")
                return audio

            before_phrase = audio[:start_time]
            after_phrase = audio[end_time:]
            
            # Crossfade to make the cut smoother
            crossfade_duration = 50 # 50ms is usually enough to avoid a "click"
            if len(before_phrase) > crossfade_duration and len(after_phrase) > 0:
                edited_audio = before_phrase.append(after_phrase, crossfade=crossfade_duration)
            else:
                # No room for crossfade, just concatenate
                edited_audio = before_phrase + after_phrase
            
            print(f"Edited audio duration: {len(edited_audio)}ms")
            print(f"Removed {(end_time - start_time):.0f}ms from audio")
            
            # Save the part that was removed for debugging
            removed_segment = audio[start_time:end_time]
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            debug_path = os.path.join("debug_segments", f"removed_segment_{base_name}.wav")
            removed_segment.export(debug_path, format="wav")
            print(f"Debug: Removed segment saved to {debug_path}")
            
            return edited_audio
        except Exception as e:
            print(f"Error editing audio: {e}")
            return AudioSegment.from_file(audio_path)
    
    def process_all_audios(self):
        """Main function to process all audio files"""
        try:
            df = pd.read_csv(self.csv_file_path)
            results = []
            
            for index, row in df.iterrows():
                name = str(row['name'])
                url = row['link']
                
                print(f"\n{'='*60}")
                print(f"Processing: {name}")
                print(f"{'='*60}")
                
                audio_filename = f"{name}.wav"
                audio_path = self.download_audio(url, audio_filename)
                
                if audio_path is None:
                    continue
                
                # UPDATED to use the new, more reliable chunking method
                print("Creating overlapping audio chunks...")
                chunks = self.create_overlapping_chunks(audio_path)
                
                if not chunks:
                    print("No chunks created, skipping...")
                    continue
                
                print("Transcribing chunks with confidence scoring...")
                chunk_transcripts = []
                
                for i, chunk in enumerate(chunks):
                    transcript, confidence = self.transcribe_chunk_with_confidence(chunk['audio'], i)
                    
                    chunk_transcripts.append({
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'transcript': transcript,
                        'confidence': confidence
                    })
                    
                    if transcript: # Only print non-empty transcripts
                        print(f"Chunk {i+1}/{len(chunks)}: {transcript} (confidence: {confidence:.2f})")
                
                full_transcript = " ".join([c['transcript'] for c in chunk_transcripts if c['transcript']])
                transcript_path = os.path.join("transcripts", f"{name}_precise_transcript.txt")
                # (The transcript saving logic is good, no changes needed there)
                # ...
                print(f"Precise transcript saved: {transcript_path}")
                
                print("Finding exact phrase boundaries...")
                start_time, end_time = self.find_exact_phrase_boundaries(chunk_transcripts, self.target_phrase)
                
                if start_time is not None and end_time is not None:
                    print(f"FOUND PRECISE PHRASE TIMING: {start_time:.0f}ms - {end_time:.0f}ms")
                    
                    edited_audio = self.remove_phrase_from_audio(audio_path, start_time, end_time)
                    
                    output_path = os.path.join("processed_audios", f"{name}_precisely_edited.wav")
                    edited_audio.export(output_path, format="wav")
                    
                    print(f"Precisely edited audio saved: {output_path}")
                    
                    results.append({
                        'name': name, 'phrase_found': True, 'output_file': output_path
                    })
                else:
                    print("Could not find the target phrase with sufficient precision")
                    original_audio = AudioSegment.from_file(audio_path)
                    output_path = os.path.join("processed_audios", f"{name}_original.wav")
                    original_audio.export(output_path, format="wav")
                    results.append({
                        'name': name, 'phrase_found': False, 'output_file': output_path
                    })
            
            results_df = pd.DataFrame(results)
            results_df.to_csv("precise_processing_results.csv", index=False, encoding='utf-8')
            print("\nProcessing complete. Results summary saved to precise_processing_results.csv")
            return results_df
            
        except Exception as e:
            print(f"Error in main processing: {e}")
            return None

# Usage example remains the same
if __name__ == "__main__":
    editor = PreciseAudioEditor("audio_links.csv")
    results = editor.process_all_audios()
    
    if results is not None:
        print("\nPRECISE PROCESSING SUMMARY:")
        print(results)
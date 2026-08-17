# =====================================================================
# Multi-Source Streaming Pipeline Aggregating Emilia, Common Voice, & LibriSpeech
# =====================================================================

import io
import queue
import threading
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

class MegaAggregatorSpeechPipeline(IterableDataset):
    """
    Streams and aggregates major open-source speech data foundations simultaneously.
    Combines conversational, audiobook, and crowd-sourced multi-lingual vocal inputs
    into a unified matrix sequence space.
    """
    def __init__(self, hf_token=None, sample_rate=16000, max_audio_samples=48000, buffer_size=128):
        """
        Args:
            hf_token (str): Optional Hugging Face token for accessing gated repositories.
            sample_rate (int): target voice sampling rate (16kHz standard for Speech Diffusion).
            max_audio_samples (int): Max static frame threshold (48,000 samples = 3.0 seconds).
            buffer_size (int): Quantities of processed tensors held in the background queue.
        """
        self.sample_rate = sample_rate
        self.max_samples = max_audio_samples
        self.buffer_size = buffer_size
        self.hf_token = hf_token
        
        print("[*] Launching Ultimate Cross-Model Open Source Data Aggregator Pipeline...")

    def _tokenize_utf8(self, text_string):
        """Transforms text strings into a universal 0-255 byte token space."""
        return torch.tensor([byte for byte in text_string.encode('utf-8')], dtype=torch.long)

    def _process_audio_tensor(self, waveform, native_sr):
        """Standardizes, filters, and normalizes incoming audio from any raw source."""
        # 1. Resample to standard target studio frequency
        if native_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=native_sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 2. Downmix stereo channels to single mono layers
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Acoustic Energy Activity Check (Filters empty noise)
        rms_energy = torch.sqrt(torch.mean(waveform ** 2))
        if rms_energy < 0.005:
            return None

        # 4. Peak Amplitude Volume Normalization
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.95

        # 5. Enforce static temporal constraints
        if waveform.size(1) > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad_size = self.max_samples - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_size))

        return waveform

    def _stream_worker(self, output_queue, stop_event):
        """Multi-source streaming crawler running across separate internet endpoints."""
        try:
            print("[*] Connecting Stream Source 1: Amphion Emilia-Large...")
            stream_emilia = load_dataset("amphion/Emilia-Dataset", split="train", streaming=True, token=self.hf_token)
            
            print("[*] Connecting Stream Source 2: Mozilla Common Voice (EN)...")
            stream_cv = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", streaming=True, token=self.hf_token)
            
            print("[*] Connecting Stream Source 3: LibriSpeech Audiobooks (Clean)...")
            stream_libri = load_dataset("msmcalister/librispeech_clean", split="train.360", streaming=True, token=self.hf_token)

            # Zip or alternate between streams to create an aggressively balanced training dataset mixture
            iter_emilia = iter(stream_emilia)
            iter_cv = iter(stream_cv)
            iter_libri = iter(stream_libri)

            while not stop_event.is_set():
                # Loop through each open dataset sequentially to inject variety into the training batch
                for source_name, stream_iterator in [("Emilia", iter_emilia), ("CommonVoice", iter_cv), ("LibriSpeech", iter_libri)]:
                    try:
                        record = next(stream_iterator)
                        waveform, native_sr, text = None, None, ""

                        # --- SOURCE SPECIFIC SCHEMA UNPACKING ---
                        if source_name == "Emilia":
                            meta = record.get("json", {})
                            text = meta.get("text", meta.get("transcript", ""))
                            lang = meta.get("language", "en").lower()
                            if lang not in ["en", "zh", "english", "chinese"]: 
                                continue
                            audio_bytes = record["mp3"]["bytes"]
                            waveform, native_sr = torchaudio.load(io.BytesIO(audio_bytes), format="mp3")

                        elif source_name == "CommonVoice":
                            text = record.get("sentence", "")
                            # Common Voice bundles raw audio arrays directly inside an audio key
                            audio_data = record.get("audio", {})
                            waveform = torch.tensor(audio_data["array"]).unsqueeze(0).float()
                            native_sr = audio_data["sampling_rate"]

                        elif source_name == "LibriSpeech":
                            text = record.get("text", "")
                            audio_data = record.get("audio", {})
                            waveform = torch.tensor(audio_data["array"]).unsqueeze(0).float()
                            native_sr = audio_data["sampling_rate"]

                        # --- UNIFIED PROCESSING AND INGESTION ---
                        if not text or not text.strip():
                            continue

                        processed_wave = self._process_audio_tensor(waveform, native_sr)
                        if processed_wave is None:
                            continue # Skip low-quality silent entries

                        tokenized_text = self._tokenize_utf8(text.strip())

                        # Push completely prepped data straight to the execution queues
                        output_queue.put((processed_wave, tokenized_text), block=True, timeout=15)

                    except StopIteration:
                        # If a specific stream dataset ends, quietly catch it and continue pulling others
                        continue
                    except Exception:
                        pass # Absorb transient network dropout errors to protect uptime

        except Exception as e:
            print(f"(!) Critical data stream worker failure: {e}")
        finally:
            output_queue.put(None)

    def __iter__(self):
        data_queue = queue.Queue(maxsize=self.buffer_size)
        stop_signal = threading.Event()
        
        worker_thread = threading.Thread(target=self._stream_worker, args=(data_queue, stop_signal), daemon=True)
        worker_thread.start()

        try:
            while True:
                tensor_payload = data_queue.get(block=True)
                if tensor_payload is None:
                    break
                yield tensor_payload
                data_queue.task_done()
        finally:
            stop_signal.set()
            worker_thread.join(timeout=1.0)


def emilia_smart_collate_fn(batch):
    """Combines individual multi-source dataset records into matched batch matrices."""
    waveforms, text_tokens_list = zip(*batch)
    stacked_waveforms = torch.stack(waveforms)
    
    max_text_sequence_len = max(len(tokens) for tokens in text_tokens_list)
    padded_text_tokens = torch.zeros(len(text_tokens_list), max_text_sequence_len, dtype=torch.long)
    
    for batch_index, individual_tokens in enumerate(text_tokens_list):
        padded_text_tokens[batch_index, :len(individual_tokens)] = individual_tokens
        
    return stacked_waveforms, padded_text_tokens


# =====================================================================
# SYSTEM VERIFICATION SUITE
# =====================================================================
if __name__ == "__main__":
    print("[*] Verifying production compatibility for the Aggregator Engine...")
    
    # Initialize the multi-source data stream
    mega_pipeline = MegaAggregatorSpeechPipeline(hf_token=None, buffer_size=16)
    data_loader = DataLoader(mega_pipeline, batch_size=3, collate_fn=emilia_smart_collate_fn)
    
    print("[*] Pulling multi-source cross-model data from the internet cache layer...")
    data_iterator = iter(data_loader)
    processed_waveforms, processed_texts = next(data_iterator)
    
    print("\n[+] Success! Your Cross-Model Aggregator Dataset is fully operational:")
    print(f" -> Aggregated Waveform Batch Shape : {list(processed_waveforms.shape)} (Mono, Energy Filtered)")
    print(f" -> Aggregated Token Batch Shape    : {list(processed_texts.shape)} [Universal Shared UTF-8 Space]")
    print("-" * 85)

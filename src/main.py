from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from emotion_detection import detect_emotion

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=8)


descriptions = ['d', 'energetic EDM', 'sad jazz']

melody, sr = torchaudio.load('./assets/bach.mp3')

wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'audio_output/{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
    

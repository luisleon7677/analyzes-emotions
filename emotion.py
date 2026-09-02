from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
import torch
import torchaudio
import torch.nn.functional as F


# ==========================================
# 1. LOAD MODEL AND PROCESSOR
# ==========================================

model_id = "Dpngtm/wav2vec2-emotion-recognition"

print("Loading model...")

model = AutoModelForAudioClassification.from_pretrained(model_id)
processor = Wav2Vec2Processor.from_pretrained(model_id)

print("Model loaded successfully.")


# ==========================================
# 2. LOAD AUDIO
# ==========================================

audio_path = "audio/angry.wav"

speech_array, sampling_rate = torchaudio.load(audio_path)

print(f"Original sampling rate: {sampling_rate}")
print(f"Audio channels: {speech_array.shape[0]}")


# ==========================================
# 3. RESAMPLE TO 16 kHz
# ==========================================

if sampling_rate != 16000:

    print("Resampling audio to 16kHz...")

    resampler = torchaudio.transforms.Resample(
        orig_freq=sampling_rate,
        new_freq=16000
    )

    speech_array = resampler(speech_array)

    sampling_rate = 16000


# ==========================================
# 4. CONVERT STEREO TO MONO
# ==========================================

if speech_array.shape[0] > 1:

    print("Converting stereo audio to mono...")

    speech_array = torch.mean(
        speech_array,
        dim=0,
        keepdim=True
    )


# ==========================================
# 5. PROCESS AUDIO
# ==========================================

inputs = processor(
    speech_array.squeeze(),
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)


# ==========================================
# 6. PREDICT
# ==========================================

with torch.no_grad():

    logits = model(**inputs).logits


# ==========================================
# 7. CONVERT LOGITS TO PROBABILITIES
# ==========================================

probabilities = F.softmax(logits, dim=-1)[0]


# ==========================================
# 8. GET EMOTION LABELS
# ==========================================

id2label = model.config.id2label


# ==========================================
# 9. DISPLAY SCORES
# ==========================================

print()
print("=" * 40)
print("EMOTION ANALYSIS")
print("=" * 40)

for emotion_id, probability in enumerate(probabilities):

    emotion = id2label[emotion_id]

    score = probability.item() * 100

    print(f"{emotion:<10}: {score:6.2f}%")


# ==========================================
# 10. GET DOMINANT EMOTION
# ==========================================

predicted_id = torch.argmax(probabilities).item()

predicted_label = id2label[predicted_id]

predicted_score = probabilities[predicted_id].item() * 100


print("=" * 40)
print(f"Dominant emotion: {predicted_label}")
print(f"Confidence:       {predicted_score:.2f}%")
print("=" * 40)

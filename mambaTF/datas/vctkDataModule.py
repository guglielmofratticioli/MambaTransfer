from datasets import load_dataset
import torchaudio
from torch.utils.data import Dataset

class VCTKDataset(Dataset):
    def __init__(self, split="train", transform=None):
        # Load the dataset from Hugging Face
        self.dataset = load_dataset("CSTR-Edinburgh/vctk", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load audio and transcription
        item = self.dataset[idx]
        audio_path = item['file']
        waveform, sample_rate = torchaudio.load(audio_path)

        # Apply transformation if any
        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "transcription": item['text']
        }
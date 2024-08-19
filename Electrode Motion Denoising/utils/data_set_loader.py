# Dataset class definition
class ECGDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals, segment_length=1500):
        self.clean_signals = clean_signals
        self.noisy_signals = noisy_signals
        self.signal_numbers = list(clean_signals.keys())
        self.segment_length = segment_length

    def __len__(self):
        # Calculate the total number of segments in the entire dataset
        total_segments = 0
        for signal_number in self.signal_numbers:
            signal_length = self.clean_signals[signal_number].size
            segments_per_signal = signal_length // self.segment_length
            total_segments += segments_per_signal * len(self.noisy_signals)  # Multiply by the number of SNR levels

        return total_segments

    def __getitem__(self, idx):
        # Calculate the total number of segments per signal (across all SNRs)
        segments_per_signal = 10  # Assuming 15000 samples per signal and 1500 samples per segment

        # Determine the signal index based on idx
        signal_idx = idx // (segments_per_signal * len(self.noisy_signals))

        # Ensure that signal_idx is within range
        if signal_idx >= len(self.signal_numbers):
            raise IndexError(f"Signal index {signal_idx} out of range for available signals")

        # Determine the segment index within that signal and SNR
        segment_in_signal_idx = idx % (segments_per_signal * len(self.noisy_signals))
        snr_idx = segment_in_signal_idx // segments_per_signal
        segment_idx = segment_in_signal_idx % segments_per_signal

        # Get the signal number based on the signal index
        signal_number = self.signal_numbers[signal_idx]
        snr = list(self.noisy_signals.keys())[snr_idx]

        # Calculate the start and end indices for the segment
        start_idx = segment_idx * self.segment_length
        end_idx = start_idx + self.segment_length

        # Check if the signal number exists in the noisy signals for the current SNR
        if signal_number in self.noisy_signals[snr]:
            # Access the clean signal array and extract the corresponding segment
            clean_signal_data = self.clean_signals[signal_number][0]  # Access the entire signal
            clean_signal_segment = clean_signal_data[start_idx:end_idx]

            # Prepare a list to store all noisy signal segments for this clean segment
            noisy_signal_segments = []

            # Extract the corresponding noisy signal segment for each copy
            for noisy_signal_copy in self.noisy_signals[snr][signal_number]:
                noisy_signal_segment = noisy_signal_copy[0][start_idx:end_idx]
                noisy_signal_segments.append(noisy_signal_segment)

            # Convert the clean segment to a tensor
            clean_signal_tensor = torch.tensor(clean_signal_segment).unsqueeze(0)  # Shape: [1, 1500]

            # Stack all noisy segments into a tensor
            noisy_signal_tensor = torch.tensor(noisy_signal_segments)  # Shape: [num_noisy_copies, 1500]

            # Add a channel dimension to both clean and noisy signals
            clean_signal_tensor = clean_signal_tensor.unsqueeze(0)  # Shape: [1, 1, 1500]
            noisy_signal_tensor = noisy_signal_tensor.unsqueeze(1)  # Shape: [num_noisy_copies, 1, 1500]

            return clean_signal_tensor, noisy_signal_tensor

        # If no corresponding noisy signal exists, return an empty tensor pair
        return torch.tensor([]), torch.tensor([])
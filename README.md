# VirtualKeyboard
Virtual Keyboard application using a trained SNN for Senior Design-BE4999

<img width="512" height="74" alt="data_pipeline" src="https://github.com/user-attachments/assets/e88e3cb0-80fa-405a-8086-64beabf794bf" />

## Abstract
The designed project is an EEG-driven virtual keyboard intended to aid those who are neurologically impaired and improve technology in non-verbal communication. The system integrates a brain–computer interface (BCI) in order to interpret electroencephalogram (EEG) signals from a database in order to find the associated character. To implement a low energy, low latency design for a virtual keyboard system, a spiking neural network (SNN) classifier was implemented with field programmable gate array (FPGA) acceleration. Using a pre-recorded EEG P300 speller dataset, we first preprocess and reduce the feature space using waveform conserving techniques. A three layer fully connected SNN [8→128→64→2] was trained to classify for P300 signal presence with a balanced test accuracy of 59.90% and a loss of 0.6629. The process of character selection is based on the calculating and summing of probabilities of P300 signal presence in each row or column, and finding the intersection of the row and column with the highest score. The trained SNN was transferred to a FPGA for further evaluation. Final character selection accuracy was calculated as 48.31 ± 22.15% on CPU and {} on FPGA, which was significantly higher than the random weight classifier baseline (p << 0.05). Latency analysis revealed a total latency of 137ms from preprocessing to character selection on CPU and {} using FPGA accelerated SNN. Energy usage for the virtual keyboard was determined to be 109.21𝜇J through FLOP analysis with a standard conversion. Comparison with a traditional artificial neural network (ANN) with similar accuracy revealed that the SNN is lower energy but higher latency.

## Dataset
This system is trained using a publicly available EEG dataset [1]. 55 participants performed RSVP and P300 speller tasks, with EEG recorded using 32 electrodes at 512 Hz [1]. Participants completed both calibration runs (training data) and test runs (evaluation data).
- SNN Training Split: 69300 for testing, 14850 for validation and testing
- Character selection split: 385 characters for validation, 1155 characters for testing

## How To Run
#### Step 1:
  Download necessary files from GitHub and navigate to directory within computer
  Download necessary .mat files from [1] and place in sub-directory datasets/Won2022_BIDS/.mat_files/
  
#### Step 2: Activate Virtual Environment
Windows:	
```powershell
python -m venv .venv
.venv\Scripts\activate
```

MacOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install Dependencies
```pip install -r requirements.txt```

#### Step 4
```python VirtualKeyboard_cmd_ln_cpu.py```

User will be prompted to enter subject number and then character to be simulated with virtual selection. The character selection process can be repeated. Enter 'clear' to remove selection history and 'quit' to exit the program



# VirtualKeyboard
A Virtual Keyboard is a brain-computer interface (BCI) that enables a user to select characters using brain activity measured through electroencephalography (EEG). The system detects neural responses associated with attention (specifically the P300 event-related potential) and converts these signals into character outputs using a trained spiking neural network (SNN).

<img width="512" height="74" alt="data_pipeline" src="https://github.com/user-attachments/assets/e88e3cb0-80fa-405a-8086-64beabf794bf" />

## Abstract
The designed project is an EEG-driven virtual keyboard intended to aid those who are neurologically impaired and improve technology in non-verbal communication. The system integrates a brain–computer interface (BCI) in order to interpret electroencephalogram (EEG) signals from a database in order to find the associated character. To implement a low energy, low latency design for a virtual keyboard system, a spiking neural network (SNN) classifier was implemented with field programmable gate array (FPGA) acceleration. Using a pre-recorded EEG P300 speller dataset, we first preprocess and reduce the feature space using waveform conserving techniques. A three layer fully connected SNN [8→128→64→2] was trained to classify for P300 signal presence with a balanced test accuracy of 59.90% and a loss of 0.6629. The process of character selection is based on the calculating and summing of probabilities of P300 signal presence in each row or column, and finding the intersection of the row and column with the highest score. The trained SNN was transferred to a FPGA for further evaluation. Final character selection accuracy was calculated as 48.31 ± 22.15% on CPU and {} on FPGA, which was significantly higher than the random weight classifier baseline (p << 0.05). Latency analysis revealed a total latency of 137ms from preprocessing to character selection on CPU and {} using FPGA accelerated SNN. Energy usage for the virtual keyboard was determined to be 109.21𝜇J through FLOP analysis with a standard conversion. Comparison with a traditional artificial neural network (ANN) with similar accuracy revealed that the SNN is lower energy but higher latency.

## Dataset
This system is trained using a publicly available EEG dataset [1]. 55 participants performed RSVP and P300 speller tasks, with EEG recorded using 32 electrodes at 512 Hz [1]. Participants completed both calibration runs (training data) and test runs (evaluation data).
- SNN Training Split: 69300 for testing, 14850 for validation and testing
- Character selection split: 385 characters for validation, 1155 characters for testing

## Preprocessing and Feature Extraction
Data was split into 600ms epochs after each flash occurs.
#### Preprocessing
- channel selection (8 channels in Parietal/Midline region)
- baseline subtraction
- linear detrending
- 0.5-15Hz bandpass filter
#### Feature Extraction
- time downsampling to 171Hz
- time window: 200-400ms from flash
- Standard scaling per channel

## Spike Encoding
Transmits continuous data into [-1, 0, 1] spikes upon change between consecutive time steps reaching the specified threshold

## SNN Architecture and Hyperparameters
Binary classification of P300/non-P300 signals
- Leaky Integrate and Fire (LIF) neurons: simulate biological neurons by integrating a continually leaking signal and firing spikes.
- Fully connected layers [8→128→64→2]

|                        |Final Design Selection           |           |
|------------------------|---------------------------------|-----------|
| Preprocessing          | Filter Range                    | 0.5-15Hz  |
| Feature Extraction     | Downsample Factor               | 3         |
|                        | Time Window                     | 200-400ms |
| Delta Encoding         | Threshold                       | 0.09      |
| LIF Neurons            | Decay Constant (beta)           | 0.95      |
|                        | Threshold                       | 1         |
| SNN Architecture       | Hidden Layer Sizes              | [128, 64] |
|                        | Dropout                         | 0.1       |
| Training               | Surrogate Gradient Constant (k) | 10        |
|                        | Batch Size                      | 256       |
|                        | Epochs                          | 100       |

## Character Selection
- sum log probabilities of positive P300 classification for each row and column
- intersection of highest probability row and column is the selected character

## How To Run
#### Step 1: Download Files
  Download necessary files from GitHub and navigate to directory within computer
  Download necessary .mat files from [1] and place in sub-directory `datasets/Won2022_BIDS/.mat_files/`
  
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
```
pip install -r requirements.txt
```

#### Step 4: Run Simulation File
```
python VirtualKeyboard_cmd_ln_cpu.py
```

User will be prompted to enter subject number and then character to be simulated with virtual selection. The character selection process can be repeated. Enter 'clear' to remove selection history and 'quit' to exit the program

## Performance Evaluation
### Character Selection accuracy
|Method            |Accuracy      |Significance  |
|------------------|--------------|--------------|
|CPU               |48.31 ± 22.15%|p = 8.85e-20  |
|FPGA              |TBA           |TBA           |
|Random Classifier |9.35 ± 8.69%  |N/A           |

### Latency and Energy
|Pipeline Stage     |CPU Latency(ms)   |CPU Energy (μJ)  |FPGA Latency (ms) |FPGA Energy (μJ)  |
|-------------------|------------------|-----------------|------------------|------------------|
|Preprocessing      |95.2              |65.20            |95.2              |65.20             |
|Feature Extraction |3.4               |4.94             |3.4               |4.94              |
|Spike Encoding     |0.3               |0.82             |0.3               |0.82              |
|SNN Classification |67.5              |38.23            |TBA               |TBA               |
|Character Selection|0.6               |0.02             |0.6               |0.02              |
|Full Pipeline      |167.0             |109.21           |TBA               |TBA               |

## References
[1] K. Won, M. Kwon, M. Ahn, and S. C. Jun, “EEG Dataset for RSVP and P300 Speller Brain-Computer Interfaces,” Scientific Data, vol. 9, no. 1, p. 388, Jul. 2022, doi: https://doi.org/10.1038/s41597-022-01509-w. 

*For a full list of all references used during the project, see the report references section

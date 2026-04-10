from pathlib import Path
import numpy as np
from pipeline_functions import *
import torch

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # create SNN model and load weights
    snn = SNNModule.createSNN(8, [128, 64], betas=[0.95, 0.95, 0.95], thresholds=[1, 1, 1])
    
    weights_path = base_dir / "model_weights/2hl12864_th072_win2_4_f3_weights.pth"
    print(weights_path)
    weights = torch.load(weights_path, weights_only=True)
    snn.load_state_dict(weights)
    snn.eval()

    running_correct_chars = ""
    running_predicted_chars = ""

    exit_code = "QUIT"

    not_valid_input = True
    quit = False

    while not_valid_input:
        # get subject
        subject = input("Enter subject number (1-55): ").upper()

        if subject == exit_code:
            quit=True
            print("Exiting program...")
            break  # Exits the while loop

        if not subject.isdigit():
            print("Invalid input. Please enter a number between 1 and 55.")
            continue  # Prompts the user again for input

        subject = int(subject)  # Convert to integer after validation

        if subject >= 1 and subject <= 55:
            not_valid_input = False  # Valid input received, exit the loop
        else:
            print("Invalid subject number. Please enter a number between 1 and 55.")

    # loop until terminated by user
    while not quit:
        not_valid_input = True  # Reset for character input
        # character selection loop
        while not_valid_input:
            # get character
            character = input("Enter character to select (A-Z): ").upper()

            if character == exit_code:
                quit=True
                print("Exiting program...")
                break  # Exits the inner while loop

            # path to .mat file
            file_path = base_dir / f"datasets/Won2022_BIDS/.mat_files/s{subject:02d}.mat"

            print(file_path)
            # load in character data
            try:
                character_data = fn_preprocess.load_subject_character_data(file_path, letter=character)
                not_valid_input = False  # Valid input received, exit the loop
            except ValueError as e:
                print(f"Error loading character data for subject {subject}, character {character}: {e} \n" +
                    "Please enter a different character")
                continue # Prompts the user again for input
        if quit:
            break  # Exits the outer while loop before processing takes place

        # preprocess character data
        X, y, time, char = fn_preprocess.preprocess_one_character(character_data)

        # feature extraction
        X, y = fn_feature_extraction.extractFeatures(X, y, k=1, factor=3, t_min=200, t_max=400, norm_type='std')

        # delta encoding
        X_spikes = delta_encoding.delta_encode(X, 0.072)

        X_spikes = torch.from_numpy(X_spikes).float()
        y = y.astype(int)

        # classify with SNN
        with torch.no_grad():
            spk_out, _ = snn(X_spikes, batch_first=True)
        
        # character selection
        selected_char, _, _, _, _ = Characterselection.p300_speller_cycle_prob(spk_out, y)

        running_correct_chars += character
        running_predicted_chars += selected_char

        print(f"Selected output: {running_predicted_chars}")
        print(f"Correct output: {running_correct_chars}")

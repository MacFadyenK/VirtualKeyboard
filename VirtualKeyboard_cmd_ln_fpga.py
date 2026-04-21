from pathlib import Path
import numpy as np
from pipeline_functions import *
import torch
import serial
import struct


def send_to_fpga(X_spikes):
    # does the data need to be int8 for FPGA?
    #binary_data = X_spikes.astype(np.int8).tobytes()
    for sample in X_spikes:
        data = sample.astype(np.int16).tobytes()
        ser.write(data)

    #ser.write(binary_data)  # Send the binary data to the FPGA

def receive_from_fpga(num_time_steps, num_output_neurons=2):
    num_bytes = num_time_steps * num_output_neurons * 2  # 2 bytes per int16
    received_data = ser.read(num_bytes)
    spk_out = np.frombuffer(received_data, dtype=np.int16).reshape(num_time_steps, num_output_neurons)
    return spk_out

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # establish connection to FPGA and initialize it
    ser = serial.Serial('COM3', 115200, timeout=1)  # Update 'COM3' to your FPGA's serial port

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
        X, y = fn_feature_extraction.extractFeatures(X, y, factor=3, t_min=200, t_max=400, norm_type='std', norm_factor=1)

        # delta encoding
        X_spikes = delta_encoding.delta_encode(X, 0.09)

        #send the spike encoded data to the FPGA for classification
        send_to_fpga(X_spikes)

        # retrieve the predicted matrix of binary classifications from the FPGA, which should be in the form of a 2D array of shape (time steps, num output neurons)
        spk_out = receive_from_fpga(num_time_steps=X_spikes.shape[1], num_output_neurons=2)
        
        # character selection
        selected_char, _, _, _, _ = Characterselection.p300_speller_cycle_prob(spk_out, y)

        running_correct_chars += character
        running_predicted_chars += selected_char

        print(f"Selected output: {running_predicted_chars}")
        print(f"Correct output: {running_correct_chars}")


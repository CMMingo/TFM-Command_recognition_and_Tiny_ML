# IMPORT PACKAGES AND LIBRARIES
import numpy as np
import random
import os
from time import time

# Audio specifics
import soundfile as sf
from pydub import AudioSegment
from librosa import load
from librosa.feature import rms, mfcc

# Coral Dev Board Specifics
#import tflite_runtime.interpreter as tflite # to make tests on laptop
from pycoral.utils import edgetpu
#from pycoral.utils import dataset
#from pycoral.adapters import common
#from pycoral.adapters import classify



# GLOBAL VARIABLES
MODEL_FILE = "./models/CNN_MFCCs_model_quantized_metadata_edgetpu.tflite"
DATA_ENG_PATH = "./data/eng"
LABELS_FILE = "./labels_ENG.txt"
COMMON_SAMPLE_RATE_LIBROSA = 22050
N_MFCC = 12
HOP_LENGTH = 512
N_FFT = 2048



# FUNCTIONS
def pad_short_audio(signal):
    pad_ms = 1000
    
    silence = AudioSegment.silent(duration=pad_ms-len(signal)+1)
    signal = signal + silence  # adding silence after the signal
    signal = signal[:COMMON_SAMPLE_RATE_LIBROSA]
    
    return signal


def extract_loudest_second(signal, sample_rate):
    results = {
        "slices": [],
        "rms": []
    }

    for slice_down in range(0, len(signal)-COMMON_SAMPLE_RATE_LIBROSA, int(COMMON_SAMPLE_RATE_LIBROSA/15)): # steps of 1470 frames
        slice_up = slice_down + COMMON_SAMPLE_RATE_LIBROSA if slice_down + COMMON_SAMPLE_RATE_LIBROSA < len(signal) else len(signal) 
        results["slices"].append([slice_down, slice_up])
        results["rms"].append(np.sum(rms(signal[slice_down:slice_up])))
    slice_down, slice_up = results["slices"][np.argmax(results["rms"])]

    out_filename = "./processed_audio.wav"
    signal = signal[slice_down:slice_up]
    sf.write(out_filename, signal, sample_rate)
    
    return signal


def preprocess_audio_file(signal, sample_rate):
    if len(signal) < COMMON_SAMPLE_RATE_LIBROSA:
        signal = pad_short_audio(signal)
    else:
        signal = extract_loudest_second(signal, sample_rate)
    
    return signal



# MAIN
if __name__ == "__main__":
    
    while True:
        wait = input("Press any key to process a test file. Cancel with Ctrl+C")

        tic1 = time()
    
        # Get labels
        with open(LABELS_FILE, "r") as f:
            labels = f.readlines()
            labels = [label.strip() for label in labels]

        # Load random audio
        filename = random.choice(os.listdir(DATA_ENG_PATH))
        file = os.path.join(DATA_ENG_PATH, filename).replace("\\", "/")
        signal, sample_rate = load(file, sr=COMMON_SAMPLE_RATE_LIBROSA)

        # Preprocess audio -> transform to standard form
        signal = preprocess_audio_file(signal, sample_rate)

        tic2 = time()

        # Load model, initializing the interpreter
        #interpreter = tflite.Interpreter(model_path=str(MODEL_FILE)) # interpreter to make tests on laptop
        interpreter = edgetpu.make_interpreter(str(MODEL_FILE), delegate=edgetpu.load_edgetpu_delegate())
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_size = input_details[0]["shape"][1]

        signal, _ = load("./processed_audio.wav", sr=COMMON_SAMPLE_RATE_LIBROSA)
        if len(signal) < input_size:
            signal.resize(input_size)

        # Extract features
        MFCCs = mfcc(signal, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
        MFCCs = MFCCs.T # transpose to keep consistency with what the CNN expects (# temporal segments, # MFCCs)

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details[0]["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            MFCCs = MFCCs / input_scale + input_zero_point

        # Prepare data to be fed into the CNN
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis].astype(input_details[0]["dtype"])

        # Make predictions
        interpreter.set_tensor(input_details[0]["index"], MFCCs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        tac2 = time()
        
        top_index = np.argmax(output[0])
        label = labels[top_index]
        score = output[0][top_index]
        
        tac1 = time()

        # Display prediction, ground truth and times
        print("\n---prediction---")
        print("Class:", label)
        print("Score:", score, "\n")
        print("---truth---")
        print(file.split("/")[-1][:-4], "\n")
        print("---times---")
        print(f"Model executed in {np.round(tac2-tic2, 5)} seconds")
        print(f"Whole process done in {np.round(tac1-tic1, 5)} seconds\n")

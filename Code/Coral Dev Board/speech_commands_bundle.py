# IMPORT PACKAGES AND LIBRARIES
import numpy as np
import random
import os
from time import time

# Audio specifics
import soundfile as sf
from pydub import AudioSegment
from librosa import load, power_to_db
from librosa.feature import rms, melspectrogram, mfcc

# Coral Dev Board Specifics
#import tflite_runtime.interpreter as tflite # to make tests on laptop
from pycoral.utils import edgetpu
#from pycoral.utils import dataset
#from pycoral.adapters import common
#from pycoral.adapters import classify



# GLOBAL VARIABLES
MODEL_FILE = [
    "./models/CNN_melspecs.tflite",
    "./models/CNN_melspecs_model_quantized_metadata.tflite",
    "./models/CNN_melspecs_model_quantized_metadata_edgetpu.tflite",
    "./models/CNN_MFCCs.tflite",
    "./models/CNN_MFCCs_model_quantized_metadata.tflite",
    "./models/CNN_MFCCs_model_quantized_metadata_edgetpu.tflite",
    "./models/CNN_raws.tflite",
    "./models/CNN_raws_model_quantized_metadata.tflite",
    "./models/CNN_raws_model_quantized_metadata_edgetpu.tflite",
    "./models/TL_model_browserfft_speech.tflite",
    "./models/TL_model_quantized_ESP_MM_metadata.tflite",
]
DATA_PATH = ["./data/eng", "./data/esp"]
LABELS_FILE = ["./labels_ENG.txt", "./labels_ESP.txt"]
COMMON_SAMPLE_RATE_LIBROSA = 22050
INPUT_SAMPLE_RATE = 44100
DOWNSAMPLED_SAMPLE_RATE = COMMON_SAMPLE_RATE_LIBROSA/5
N_MELS = 80
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


def CNN_melspecs(option):
    # Load model, initializing the interpreter
    #interpreter = tflite.Interpreter(model_path=str(MODEL_FILE[option])) # interpreter to make tests on laptop
    if option == 2:
        interpreter = edgetpu.make_interpreter(str(MODEL_FILE[option]), delegate=edgetpu.load_edgetpu_delegate())
    else:
        interpreter = edgetpu.make_interpreter(str(MODEL_FILE[option]))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]["shape"][1]

    signal, sample_rate = load("./processed_audio.wav", sr=COMMON_SAMPLE_RATE_LIBROSA)
    if len(signal) < input_size:
        signal.resize(input_size)

    # Extract features
    melspec = melspectrogram(signal, sr=sample_rate, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
    melspec_dB = power_to_db(melspec, ref=np.max)
    melspec_dB = melspec_dB.T # transpose to keep consistency with what the CNN expects (# temporal segments, # Mel bands)

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details[0]["dtype"] == np.uint8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        melspec_dB = melspec_dB / input_scale + input_zero_point

    # Prepare data to be fed into the CNN
    melspec_dB = melspec_dB[np.newaxis, ..., np.newaxis].astype(input_details[0]["dtype"])

    # Make predictions
    interpreter.set_tensor(input_details[0]["index"], melspec_dB)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    return output


def CNN_MFCCs(option):
    # Load model, initializing the interpreter
    #interpreter = tflite.Interpreter(model_path=str(MODEL_FILE[option])) # interpreter to make tests on laptop
    if option == 5:
        interpreter = edgetpu.make_interpreter(str(MODEL_FILE[option]), delegate=edgetpu.load_edgetpu_delegate())
    else:
        interpreter = edgetpu.make_interpreter(str(MODEL_FILE[option]))
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

    return output


def CNN_raws(option):
    # Load model, initializing the interpreter
    #interpreter = tflite.Interpreter(model_path=str(MODEL_FILE[option])) # interpreter to make tests on laptop
    if option == 8:
        interpreter = edgetpu.make_interpreter(str(MODEL_FILE[option]), delegate=edgetpu.load_edgetpu_delegate())
    else:
        interpreter = edgetpu.make_interpreter(str(MODEL_FILE[option]))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]["shape"][1]

    signal, _ = load("./processed_audio.wav", sr=DOWNSAMPLED_SAMPLE_RATE)
    if len(signal) < input_size:
        signal.resize(input_size)

    # Extract features
    #Not required in this case

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details[0]["dtype"] == np.uint8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        signal = signal / input_scale + input_zero_point

    # Prepare data to be fed into the CNN
    signal = signal[np.newaxis, ..., np.newaxis].astype(input_details[0]["dtype"])

    # Make predictions
    interpreter.set_tensor(input_details[0]["index"], signal)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    return output


def TL_spanish(option):
    # Load model, initializing the interpreter
    #interpreter = tflite.Interpreter(model_path=str(MODEL_FILE[option]))
    interpreter = edgetpu.make_interpreter(str(MODEL_FILE[option]))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]["shape"][1]

    signal, _ = load("./processed_audio.wav", sr=INPUT_SAMPLE_RATE)
    if len(signal) < input_size:
        signal.resize(input_size)

    # Extract features
    #Not required in this case

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details[0]["dtype"] == np.uint8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        signal = signal / input_scale + input_zero_point

    signal = np.expand_dims(signal[:input_size], axis=0).astype(input_details[0]["dtype"])

    # Make predictions
    interpreter.set_tensor(input_details[0]["index"], signal)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    return output



# MAIN
if __name__ == "__main__":
    
    msg = """Model options to choose:
\t1 - CNN model using Mel spectrograms
\t2 - CNN quantized model using Mel spectrograms
\t3 - CNN quantized model using Mel spectrograms, prepared for TPU execution
\t4 - CNN model using MFCCs
\t5 - CNN quantized model using MFCCs
\t6 - CNN quantized model using MFCCs, prepared for TPU execution
\t7 - CNN model using using raw signals
\t8 - CNN quantized model using using raw signals
\t9 - CNN quantized model using using raw signals, prepared for TPU execution
\t10 - Spanish model with TL
\t11 - Spanish quantized model with TL\n"""
    print(msg)

    options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    while True:
        option = None
        while option not in options:
            option = input("Select a model (send 'h' to see model options or press 'Ctrl+C' to stop): ")

            if str(option) in ["h", "'h'", "help"]:
                print(msg)

            try:
                option = int(option)-1
            except:
                pass

            if option not in options and str(option) not in ["h", "'h'", "help"]:
                print("Invalid option. Valids are: [1,2,3,4,5,6,7,8,9,10,11]\n")


        aux = MODEL_FILE[option].split("/")[-1]
        print(f"Using {aux} model")

        tic1 = time()
    
        # Get labels
        with open(LABELS_FILE[0 if option not in [9,10] else 1], "r") as f:
            labels = f.readlines()
            labels = [label.strip() for label in labels]

        # Load random audio
        filename = random.choice(os.listdir(DATA_PATH[0 if option not in [9,10] else 1]))
        file = os.path.join(DATA_PATH[0 if option not in [9,10] else 1], filename).replace("\\", "/")
        signal, sample_rate = load(file, sr=COMMON_SAMPLE_RATE_LIBROSA)

        # Preprocess audio -> transform to standard form
        signal = preprocess_audio_file(signal, sample_rate)

        tic2 = time()

        # Prepare and run model
        if option in [0,1,2]:
            output = CNN_melspecs(option)
        elif option in [3,4,5]:
            output = CNN_MFCCs(option)
        elif option in [6,7,8]:
            output = CNN_raws(option)
        elif option in [9,10]:
            output = TL_spanish(option)

        tac2 = time()

        top_index = np.argmax(output[0])
        label = labels[top_index]
        score = output[0][top_index]

        tac1 = time()
        
        # Display prediction, ground truth and times
        print("---prediction---")
        print("Class:", label)
        print("Score:", score, "\n")
        print("---truth---")
        print(file.split("/")[-1][:-4], "\n")
        print("---times---")
        print(f"Model executed in {np.round(tac2-tic2, 5)} seconds")
        print(f"Whole process done in {np.round(tac1-tic1, 5)} seconds\n")

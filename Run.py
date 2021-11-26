"""
CAVI (Coadaptive audio-visual instrument)
Cagri Erdem, 2020
RITMO, University of Oslo, Norway.

myo-to-osc by Charles P. Martin (2018)
myo-to-osc v2 by Qichao Lan & Cagri Erdem (2019)
Keras MDN implementation by Charles P. Martin (2018)
"""
from myo import *
import numpy as np
import datetime
import time
import os, sys
from queue import Queue
from threading import Thread
from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

import tensorflow._api.v2.compat.v1 as tf
from tensorflow._api.v2.compat.v1 import keras

# import tensorflow as tf
# from tensorflow import keras
import mdn

from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

################################################
############# GLOBAL PARAMS & VARS #############
################################################
# OSC
OUTPUT_PORT = 5666
INPUT_PORT = 5660
INPUT_IP = "127.0.0.1"
osc_client = udp_client.SimpleUDPClient("localhost", OUTPUT_PORT)
# MYO
TTY = [f for f in os.listdir("/dev") if f.startswith("tty.usbmodem")]  # only for mac
print(TTY)
# MAC = ["d0:8d:fc:7f:f5:f1", "d9:db:d4:2b:d4:17"]
MYOs = {
    "SVERM7": "d0:8d:fc:7f:f5:f1",
    "SVERM8": "ef:9d:fd:31:ea:10",
    "SVERM9": "c4:ca:8f:db:03:6b",
    "SVERM10": "d4:61:dc:9e:4d:cf",
}
MAC = MYOs["SVERM7"]
print("The auto MAC address is: ", MAC)

myo = []
generation_list = []
main_buffer = []
emg_buffer = []
acc_buffer = []

# TODO: Replace all divisions with multiplications!

################################################
################ MODEL PARAMS ##################
################################################
arm = {"R": [2, 3, 6, 7, "RIGHT"], "L": [4, 5, 7, 0, "LEFT"]}

npydir = "/Users/cagrierdem/Desktop/dat/modelv2/coadaptive/npy_mavg/"  # local
# Selected models (dict based on differently prepared datasets)
models = {
    "npy2": [
        "Coadaptive-20201114-10_14_31-32-5-7-300-100-1-1-256-0_5-1e-05.h5",  # 2-32
        "Coadaptive-20201114-10_16_36-64-5-7-300-100-1-1-256-0_5-1e-05.h5",  # 2-64
    ],
    "mavg": [
        "Coadaptive-20201122-23_18_10-16-5-7-300-100-1-1-256-0_5-1e-05.h5",  # 2-16
        "Coadaptive-20201122-23_17_46-32-5-7-300-100-1-1-256-0_5-1e-05.h5",  # 2-32
        "Coadaptive-20201122-23_18_25-64-5-7-300-100-1-1-256-0_5-1e-05.h5",  # 2-64
        "Coadaptive-20201206-00_22_10-32-5-7-300-50-1-1-256-0_2-1e-05.h5",  # 2-32 ---Usable!
        "Coadaptive-20201206-15_41_52-64-5-7-300-50-1-1-256-0_2-1e-05.h5",  # 2-64
        "Coadaptive-20201207-15_41_13-32-5-7-300-50-1-1-2048-0_2-1e-05.h5",  # 2-32 stateful ---Usable!
        "Coadaptive-20201207-11_29_34-64-5-7-300-50-1-1-1024-0_2-1e-05.h5",  # 2-64
    ],
    "raw_ds_r": [
        "Coadaptive-20201209-00_31_26-32-5-7-300-50-1-50-512-0_2-1e-05.h5",  # 2-32 regular
        "Coadaptive-20201208-15_03_30-32-5-7-300-50-1-1-512-0_2-1e-05.h5",  # 2-32 statefull
        "Coadaptive-20201208-15_00_26-32-5-7-300-50-1-1-512-0_2-1e-05.h5",  # 2-32 stateless
    ],
    "proc": [
        "Coadaptive-20201212-02_23_14-32-5-7-70-50-1-1-256-0_2-1e-05.h5",  # 2-32 (only impros)
        "Coadaptive-20201212-02_19_58-32-5-7-300-50-1-1-512-0_2-1e-05.h5",  # 2-32
        "Coadaptive-20201212-02_28_46-32-64-5-7-300-50-1-1-512-0_2-1e-05.h5",  # 2-16_32
    ],
    "proc_ds": [
        "Coadaptive-20201212-15_49_26-32-5-7-300-50-1-1-256-0_2-1e-05.h5",  # 2-32
        "Coadaptive-20201212-15_52_06-64-5-7-300-50-1-1-256-0_2-1e-05.h5",  # 2-64
        "Coadaptive-20201212-15_59_12-32-64-5-7-300-50-1-1-256-0_2-1e-05.h5",  # 2-32_64 --Best!
        "Coadaptive-20201213-00_21_06-32-64-5-7-70-50-1-1-64-0_2-1e-05.h5",  # 2-32_64 (only impros)
        "Coadaptive-20201213-00_18_04-32-5-7-70-50-1-1-64-0_2-1e-05.h5",  # 2-32 (only impros)
        "Coadaptive-20201213-00_18_46-64-5-7-300-100-1-1-128-0_2-1e-05.h5",  # 2-64 (seq100) --Usable!
        # w/ updated dataloader: (the rest above is essentially useless...)
        "Coadaptive-20201213-20_47_20-32-32-5-7-300-50-1-1-256-0_2-1e-05.h5",  # 2-32 --Best!!!
        "Coadaptive-20201213-20_43_08-32-64-5-7-300-50-1-1-256-0_2-1e-05.h5",  # 2-32_64
        "Coadaptive-20201213-20_49_03-64-64-5-7-300-100-1-1-512-0_2-1e-05.h5",  # 2-64
        "Coadaptive-20201221-18_17_55-32-32-5-7-300-50-1-1-128-0_2-1e-05.h5",  # 2-32 w/ acc_filt
    ],
    "proc_ds_single": [
        "Coadaptive-ACC-20201218-13_17_43-32-32-5-3-300-50-1-1-128-0_2-1e-05.h5",  # 2-32 only ACC
        "Coadaptive-EMG-20201218-13_15_57-32-32-5-4-300-50-1-1-128-0_2-1e-05.h5",  # 2-32 only EMG
        "Coadaptive-mixedACC-20201221-23_42_52-32-32-5-3-462-50-1-1-512-0_2-1e-05.h5",  # 2-32 only ACC (mixed w dance)
    ],
    "newest": [
        "Coadaptive-20210308-20_52_11-64-64-3-7-300-200-1-1-512-0_3-1e-05.h5",  # 2-64, seq:200, 3 mixtures
        "Coadaptive-20210309-12_07_03-64-64-5-7-300-200-1-1-512-0_3-1e-05.h5",  # 2-64, seq:200, 5 mixtures
        "Coadaptive-20210329-18_19_51-64-32-3-7-300-200-1-1-256-0_2-1e-05.h5",  # 2-64_32, seq:200, 3 mixtures
        "Coadaptive-20210312-11_06_25-32-32-5-7-300-100-1-1-512-0_2-1e-05.h5",  # LEFT, 2-32, seq:100
        "Coadaptive-20210312-14_59_26-64-32-5-7-300-100-1-1-128-0_2-1e-05.h5",  # LEFT, 2-64_32, seq:100
        "Coadaptive-LHAND-20210314-13_24_39-64-32-3-7-300-200-1-1-256-0_3-1e-05.h5",  # LEFT, 2-64_32, seq:200, 3 mixtures (looks ok!)
        "Coadaptive-LHAND-20210318-13_48_20-32-32-3-7-300-200-1-1-256-0_3-1e-05.h5",  # LEFT, 2-32, seq:200, 3 mixtures
        "Coadaptive-LHAND-20210316-13_10_03-32-64-3-7-300-200-1-1-256-0_3-1e-05.h5",  # LEFT, 2-32_64, seq:200, 3 mixtures
        "Coadaptive-20210401-14_48_38-32-32-3-7-300-200-1-1-256-0_1-1e-05.h5",  # 2-32, seq:200, 3 mixtures
        "Coadaptive-20210329-18_19_51-64-32-3-7-300-200-1-1-256-0_2-1e-05-1.h5",  # 2-64_32, seq:200, 3 mixtures, lower dropout
        "Coadaptive-20210404-13_27_57-64-64-5-7-300-200-1-1-256-0_1-1e-05.h5",  # 2-64, seq:200, 5 mixtures (looks ok!)
    ],
}

MODEL_NAME = models["proc_ds"][-4]
_MYO = arm["R"]

SCALE = 1.0
# Larger pi_temp –> sampling from different distributions at every time step
PI = 0.7  # 0.8 before
SIGMA = 0.01
SEQ_LEN = 200  # 50 –> 5 sec
_LIMIT = 1  # How many sequences will be concatenated (buffered) before streaming (5 if upsampling)
BPM = 600  # 50 Hz (3000) is optimum for continuity or 10 Hz (600) no upsampling
UPSAMPLE = False
_FS = 50  # sr

INPUT_DIMS = 7
HIDDEN_UNITS1 = 32
HIDDEN_UNITS2 = HIDDEN_UNITS1 #// 2
OUTPUT_DIMS = 7
N_MIXES = 5
STATE = True
res_states = False  # reset the model states before prediction
SEQ_BASED = True
NUM_STEPS = 250  # // 2  # for sampling from the mixture distributions (500 looks good, 300 for seq200 but min 125 for seq50)
# Preprocessing
RECTIFY = True
NORM = False
acc_buffer_len = 1000  # sr * downsample_rate (seq_len * sr//10)
emg_buffer_len = 4000  # sr * downsample_rate (seq_len * sr//10)
MAVG = True

################################################
############## HELPER FUNCTIONS ################
################################################
def rms_of(x, window_length, hop_length, df=False):
    """Clculate moving root-mean-square (RMS),
    also downsamples depending on the chosen window size."""
    if df:
        x = x.to_numpy()
    rms = []
    for ax in x.T:
        energy = np.array(
            [
                np.sum((ax[i : i + window_length] ** 2))
                for i in range(0, len(ax), hop_length)
            ]
        )
        rms.append(energy)
    rms = np.array(rms).T
    return rms


def filt_imu(array, lowcut=1, highcut=10, fs=50, order=4):

    nyq = fs / 2
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")

    new = []
    for ax in array.T:
        filt = signal.lfilter(b, a, ax)
        new.append(filt)
    new = np.array(new).T
    return new


def live_acc_filt(
    acc,
    lowcut=1,
    highcut=10,
    sr=50,
    order=4,
    btype="band",
    rect=False,
    scale=False,
    df=False,
):

    if df:
        acc = acc.to_numpy()

    nyq = sr / 2
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype=btype)

    new = []
    for ax in acc.T:
        filt = signal.lfilter(b, a, ax)
        new.append(filt)
    acc_f = np.array(new).T

    if rect:
        acc_f = np.abs(acc_f)

    if scale:
        scaler = MaxAbsScaler()
        acc_f = scaler.fit_transform(acc_f)

    if df:
        time = np.linspace(0, len(acc_f) / sr, len(acc_f))
        acc_f = pd.DataFrame(data=acc_f)
        acc_f.index = time

    return acc_f


class MovingAverage:
    """Live calculation of the moving average"""

    # TODO: add a loop to calculate multiple channels in single instance
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)


def rs_by_interp(signal, input_fs, output_fs):
    """This function is adapted from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
    which was released under LGPL."""
    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(signal.shape[0] * scale)
    resampled = []
    for ax in signal.T:
        resampled_signal = np.interp(
            np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
            np.linspace(0.0, 1.0, len(ax), endpoint=False),  # known positions
            ax,
        )
        resampled.append(resampled_signal)
    resampled = np.array(resampled).T
    return resampled


def smooth_of(x, win_size=SEQ_LEN // 2, order=3):
    """Savitzky-Golay filter for smoothing"""
    # win_size = x.shape[0] // 2
    if win_size % 2 == 0:
        win_size = win_size + 1
    new = []
    for ax in x.T:
        # window size 51, polynomial order 3
        filt = signal.savgol_filter(ax, win_size, order)
        new.append(filt)
    new = np.array(new).T
    return new


def scale_of(sig):
    scaler = MinMaxScaler()
    return scaler.fit_transform(sig)


def zero_pad(x):
    padded = []
    for arr in x.T:
        z = np.insert(arr, slice(0, None), 0)
        padded.append(z)
    padded = np.array(padded).T
    return padded


################################################
############## SIGNAL FUNCTIONS ################
################################################
def from_queue(delay=0.005):
    """Collect preprocessed EMG & ACC to send to the model"""
    global emg_buffer
    global acc_buffer
    time.sleep(delay)
    emg_back = emg_queue.get(block=True, timeout=None)
    acc_back = acc_queue.get(block=True, timeout=None)
    proc_sig = np.zeros(shape=(7,))
    proc_sig = np.concatenate((emg_back, acc_back), axis=0)
    return proc_sig.astype("float32")


def proc_emg(index):
    def handler(emg_data):
        global emg_buffer
        address = "/emg" + str(index)
        # Choose the channels used in the training dataset
        proc_emg_data = (
            emg_data[_MYO[0]],
            emg_data[_MYO[1]],
            emg_data[_MYO[2]],
            emg_data[_MYO[3]],
        )
        osc_client.send_message(address, proc_emg_data)  # Send OSC the true EMG
        # Calculate the moving RMS and downsample to 10 Hz
        emg_buffer.append(proc_emg_data)
        if len(emg_buffer) == emg_buffer_len:
            emg_arr = np.asarray(emg_buffer)
            emg_buffer = []  # reset the list
            emg_rms = rms_of(emg_arr, window_length=50, hop_length=20)
            if RECTIFY:
                emg_rms = np.abs(emg_rms)
                # Put in queue for prediction
            for e in range(len(emg_rms)):
                emg_queue.put_nowait(emg_rms[e, :])

    return handler  # should return the function so the *index can be identical


def proc_imu(index):
    def handler(quat_data, acc_data, gyro_data):
        global acc_buffer
        address = "/acc" + str(index)
        osc_client.send_message(address, acc_data)  # Send OSC the true ACC
        proc_acc_data = np.asarray(acc_data)
        if MAVG:
            # Calculate the moving average and downsample to 10 Hz
            aX = acc_mavgX.process(proc_acc_data[0])
            aY = acc_mavgY.process(proc_acc_data[1])
            aZ = acc_mavgZ.process(proc_acc_data[2])
            proc_acc_array = np.array([aX, aY, aZ])
        else:
            proc_acc_array = proc_acc_data

        if RECTIFY:
            proc_acc_array = np.abs(proc_acc_array)

        acc_buffer.append(proc_acc_array)
        if len(acc_buffer) == acc_buffer_len:
            acc_arr = np.asarray(acc_buffer)
            acc_buffer = []
            # acc_f = live_acc_filt(acc_arr, rect=True, scale=False)
            # acc_f = filt_imu(acc_arr)
            # acc_f = scale_of(acc_f)
            # acc_ds = rs_by_interp(acc_f, 50, 10)  # downsample
            acc_ds = rs_by_interp(acc_arr, 50, 10)  # downsample
            # Put in queue for prediction
            for a in range(len(acc_ds)):
                acc_queue.put_nowait(acc_ds[a, :])

    return handler


def init_myo(index, time):
    m = Myo(adapter=BT(tty="/dev/" + TTY[index], baudrate=115200), start_time=time)

    m.add_emg_handler(proc_emg(index))  # comment out these 2 lines when not using OSC
    m.add_imu_handler(proc_imu(index))

    # m.connect(address=MAC[index])
    m.connect(address=MAC)
    m.sleep_mode(1)
    m.set_mode(
        EMG_Mode.send_emg.value,
        IMU_Mode.send_data.value,
        Classifier_Mode.disabled.value,
    )
    m.vibrate(1)
    myo.append(m)


def run_myo(index):
    """Loop for running MYO"""
    try:
        print("myo {} starts at: {}".format(index, time.time() - TIME))
        while True:
            myo[index].run()
    except KeyboardInterrupt:
        # EXPORT CSV:
        # myo[index].emg_to_csv()
        # myo[index].imu_to_csv()
        myo[index].disconnect()
        print("\nDisconnected")
    finally:
        pass


def get_osc(osc_address: str, *osc_arguments,) -> None:
    """Get model and streaming parameters as OSC messages"""
    p, s, b = np.array([*osc_arguments])
    p = np.round(p, 2)  # PI
    s = np.round(s, 3)  # SIGMA
    b = int(b)  # BPM
    param_queue.put_nowait((p, s, b))


################################################
############ PREDICTION FUNCTIONS ##############
################################################
def NN(model_name, state=STATE, input_len=SEQ_LEN):
    """Build the model"""
    models_folder = "./models/"
    decoder = keras.Sequential()
    decoder.add(
        keras.layers.LSTM(
            HIDDEN_UNITS1,
            batch_input_shape=(1, input_len, INPUT_DIMS),
            return_sequences=True,
            stateful=state,
        )
    )
    decoder.add(
        keras.layers.LSTM(HIDDEN_UNITS2, return_sequences=False, stateful=state)
    )
    decoder.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))
    decoder.add(keras.layers.Activation("linear", dtype="float32"))
    decoder.load_weights(os.path.join(models_folder, model_name))
    print("\nModel loaded. Stream will start soon.")
    return decoder


def shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def shift_seq(seq, new_val):
    """Shift the array by adding val to the end"""
    new_arr = np.zeros(shape=seq.shape)
    new_arr[-1:] = new_val
    new_arr[:-1] = seq[1:]
    return new_arr


def pred_seq_based(
    model,
    input_seq,
    num_steps=NUM_STEPS,
    seq_len=SEQ_LEN,
    n_features=OUTPUT_DIMS,
    n_mixes=N_MIXES,
    pi_temp=PI,
    sigma_temp=SIGMA,
    scale=SCALE,
    verbose=False,
):
    """Predict a new sequence based on a given sequence.
    Note that num_steps is arbitrary and should be experimented based on accuracy VS latency.
    (Substantially influenced from Charles P. Martin's EMPI, adapted w/ Benedikte Wallace)"""

    Nseq = input_seq
    steps = 0
    while steps < num_steps:
        params = model.predict(Nseq.reshape(1, seq_len, n_features) * scale)
        pred = (
            mdn.sample_from_output(
                params[0], n_features, n_mixes, temp=pi_temp, sigma_temp=sigma_temp
            )
            / scale
        )
        if verbose:
            print(f"Prediction step {steps} of shape {pred.shape}")
        Nseq = shift(Nseq, -1, fill_value=pred)
        # Nseq = shift_seq(Nseq, pred)
        if verbose:
            print("Shifted seq shape:", Nseq.shape)
        steps += 1
        # print(steps)
    gen = Nseq
    return gen


def pred_sample_based(
    model,
    input_seq,
    n_features=OUTPUT_DIMS,
    n_mixtures=N_MIXES,
    pi_temp=PI,
    sigma_temp=SIGMA,
    scale=SCALE,
    num_steps=NUM_STEPS,
):
    """Condition the network and generate sequence (adapted from EMPI examples of C.P. Martin)"""

    steps = 0
    for sample in input_seq:
        params = model.predict(sample.reshape(1, 1, n_features) * scale)

        new_sample = (
            mdn.sample_from_output(
                params[0], n_features, n_mixtures, temp=pi_temp, sigma_temp=sigma_temp
            )
            / scale
        )
        output_seq = [new_sample.reshape((n_features,))]

    while steps < num_steps - 1:  # input_seq.shape[0] - 1:
        params = model.predict(new_sample.reshape(1, 1, n_features) * scale)
        new_sample = (
            mdn.sample_from_output(
                params[0], n_features, n_mixtures, temp=pi_temp, sigma_temp=sigma_temp
            )
            / scale
        )
        output = new_sample.reshape(n_features,)
        # output = proc_generated_touch(output)
        output_seq.append(output.reshape((n_features,)))
        steps += 1

    predicted_seq = np.array(output_seq)
    return predicted_seq


def prediction_loop(
    model, seq_len, limit,
):
    """Collects EMG and ACC data from their respective queues and passes into the model, 
        then collects the predictions and puts in the streaming queue."""
    global main_buffer
    global generation_list
    global PI
    global SIGMA
    global BPM
    try:
        print("Prediction loop initialized.")
        while True:
            if not param_queue.empty():
                PI, SIGMA, BPM = param_queue.get(block=False, timeout=None)
                print(f"Params changed! PI: {PI}, SIGMA: {SIGMA}, BPM: {BPM}")

            proc_signals = from_queue()
            main_buffer.append(proc_signals)
            if len(main_buffer) == seq_len:
                buffer_array = np.asarray(main_buffer)
                main_buffer = []  # reset the buffer

                if NORM:
                    buffer_array = scale_of(buffer_array)

                if res_states:
                    model.reset_states()
                else:
                    pass

                if SEQ_BASED:
                    prediction = pred_seq_based(
                        model, buffer_array, pi_temp=PI, sigma_temp=SIGMA
                    )
                    print("Predicted array:", prediction.shape)
                else:
                    prediction = pred_sample_based(
                        model, buffer_array, pi_temp=PI, sigma_temp=SIGMA
                    )

                if UPSAMPLE:
                    # prediction = zero_pad(prediction)
                    prediction = rs_by_interp(prediction, 10, _FS)
                else:
                    prediction = rs_by_interp(prediction, 10, 10)

                generation_list.append(prediction)
                print(len(generation_list))
                if len(generation_list) == limit:
                    genstack = np.concatenate(generation_list)
                    generation_list = []
                    # print(genstack.shape)
                    stream_queue.put_nowait(genstack)
                    genstack = np.zeros(shape=(SEQ_LEN * _LIMIT, OUTPUT_DIMS))

    except KeyboardInterrupt:
        print("\nNo signal loop anymore!")
    finally:
        pass


def stream():
    """Main loop for data streaming w/ main clock"""
    global BPM
    try:
        print("\nCAVI started!")
        while True:
            gen_arr = stream_queue.get(block=True, timeout=None)
            for g in range(len(gen_arr)):
                pemg = tuple(gen_arr[g, :4])
                pacc = tuple(gen_arr[g, 4:])
                print(
                    f"Prediction step {g}: EMG {np.shape(pemg)}, ACC {np.shape(pacc)}"
                )
                osc_client.send_message("/pred_emg", pemg)
                osc_client.send_message("/pred_acc", pacc)
                time.sleep(60 / BPM)
    except KeyboardInterrupt:
        pass
    finally:
        pass


################################################
################# RUNNING ######################
################################################
# Incoming OSC server:
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/params", get_osc)
server = osc_server.ThreadingOSCUDPServer((INPUT_IP, INPUT_PORT), dispatcher)
# Model:
if SEQ_BASED:
    network = NN(model_name=MODEL_NAME, input_len=SEQ_LEN)
else:
    network = NN(model_name=MODEL_NAME, input_len=1)
# Queues
emg_queue = Queue()
acc_queue = Queue()
stream_queue = Queue()
param_queue = Queue()
# MAVG instances
acc_mavgX = MovingAverage(window_size=5)
acc_mavgY = MovingAverage(window_size=5)
acc_mavgZ = MovingAverage(window_size=5)
# MYO initialization
TIME = time.time()
init_myo(0, TIME)  # single MYO
# Threads:
# 1
print("Starting MYO thread...")
print("Which MYO? -", _MYO[-1])
myo_thread = Thread(target=run_myo, args=(0,), name="myo_thread", daemon=True)
# 2
print("Starting Prediction thread...")
prediction_thread = Thread(
    target=prediction_loop,
    args=(network, SEQ_LEN, _LIMIT,),
    name="prediction_thread",
    daemon=True,
)
# 3
print("Starting Stream thread...")
stream_thread = Thread(target=stream, args=(), name="stream_thread", daemon=True)
# 4
print("Starting OSC thread...")
server_thread = Thread(
    target=server.serve_forever, name="osc_server_thread", daemon=True
)
# RUN MOFO RUN!
try:
    myo_thread.start()
    prediction_thread.start()
    stream_thread.start()
    server_thread.start()

    myo_thread.join()
    prediction_thread.join()
    stream_thread.join()
    server_thread.join()
except KeyboardInterrupt:
    print("\nThreads stopping...\n")
finally:
    print("\nCAVI stopped.\n")
# When the music's over, turn off the lights.


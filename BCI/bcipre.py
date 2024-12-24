import numpy as np
import glob
import re
import pandas as pd
from scipy.signal import butter, lfilter

def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


DataFolder = 'train/'
list_of_files = glob.glob(DataFolder + 'Data_*.csv')
list_of_files.sort()

reg = re.compile(r'\d+')
freq = 200.0
epoc_window = int(1.3 * freq)  


selected_channels = [3, 6, 8, 16, 24, 42, 54, 55, 50, 32, 22, 12, 14, 4]


X = []
User = []

for f in list_of_files:
    print(f"Processing file: {f}")
    user, session = reg.findall(f)
    sig = np.array(pd.read_csv(f))

    EEG = sig[:, 1:-2][:, selected_channels]  
    Trigger = sig[:, -1]  

    sigF = bandpass(EEG, [4.0, 30.0], freq)

    idxFeedBack = np.where(Trigger == 1)[0]  

    for idx in idxFeedBack:
        epoch = sigF[idx:idx + epoc_window, :]
        if epoch.shape[0] == epoc_window: 
            X.append(epoch)
            User.append(int(user))
X = np.array(X).transpose((0, 2, 1))  
info = np.array([User])

np.save('infos.npy', info)
np.save('epochs.npy', X)
print("Head of infos.npy:")
print(info[:5])

print("\nShape of epochs.npy:")
print(X.shape)

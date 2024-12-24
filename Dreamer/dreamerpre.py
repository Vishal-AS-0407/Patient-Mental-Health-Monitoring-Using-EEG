import scipy.io as sio
from scipy import signal, stats
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre
import cupy as cp
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

def signal_filter_gpu(signal_data, freq_range, fs=128):
    signal_cuda = cp.asarray(signal_data)
    nyquist = fs / 2
    freqs = np.array(freq_range) / nyquist  
    numtaps = 101
    h = signal.firwin(numtaps, freqs, window='hamming')
    h_cuda = cp.asarray(h)
    filtered = cp.convolve(signal_cuda, h_cuda, mode='same')
    return filtered

def shannon_entropy_cuda(signal_data):
    signal_data = cp.asarray(signal_data)
    signal_data = (signal_data - cp.min(signal_data)) / (cp.max(signal_data) - cp.min(signal_data))
    
    hist, _ = cp.histogram(signal_data, bins=100, density=True)
    hist = hist[hist > 0]
    
    return float(-cp.sum(hist * cp.log2(hist)).get())

def renyi_entropy_cuda(signal_data, alpha=2):
    signal_data = cp.asarray(signal_data)
    signal_data = (signal_data - cp.min(signal_data)) / (cp.max(signal_data) - cp.min(signal_data))
    
    hist, _ = cp.histogram(signal_data, bins=100, density=True)
    hist = hist[hist > 0]
    
    if alpha == 1:
        return shannon_entropy_cuda(signal_data)
    else:
        return float((1 / (1 - alpha)) * cp.log2(cp.sum(hist ** alpha)).get())

def calculate_psd_cuda(signal_data, fs=128):
    signal_cuda = cp.asarray(signal_data)
    n = len(signal_data)
    freqs = cp.fft.rfftfreq(n, d=1/fs)
    psd = cp.abs(cp.fft.rfft(signal_cuda))**2 / n
    return freqs, psd

def safe_divide_cuda(s, b):
    s_cuda = cp.asarray(s)
    b_cuda = cp.asarray(b)
    result = cp.zeros_like(s_cuda, dtype=float)
    mask = b_cuda != 0
    result[mask] = s_cuda[mask] / b_cuda[mask]
    result[~mask] = s_cuda[~mask]
    return cp.asnumpy(result)

def preprocessing_gpu(input_signal, feature):
    input_signal_cuda = cp.asarray(input_signal)
    
    try:
       
        filtered_data = signal_filter_gpu(input_signal_cuda, [0.5, 45])  
        filtered_theta = signal_filter_gpu(filtered_data, [4, 8])      
        filtered_alpha = signal_filter_gpu(filtered_data, [8, 13])      
        filtered_beta = signal_filter_gpu(filtered_data, [13, 30])   
        

        _, psd_theta = calculate_psd_cuda(filtered_theta)
        _, psd_alpha = calculate_psd_cuda(filtered_alpha)
        _, psd_beta = calculate_psd_cuda(filtered_beta)
        
  
        feature.extend([
            float(cp.max(psd_theta).get()),
            float(cp.max(psd_alpha).get()),
            float(cp.max(psd_beta).get())
        ])
        
        feature.extend([
            float(cp.mean(filtered_data).get()),
            float(cp.var(filtered_data).get()),
            float(cp.std(filtered_data).get()),
            float(cp.sqrt(cp.mean(filtered_data**2)).get()),
            float(cp.max(cp.abs(filtered_data)).get()),
            float(cp.min(cp.abs(filtered_data)).get()),
            float(stats.kurtosis(cp.asnumpy(filtered_data))),
            float(stats.skew(cp.asnumpy(filtered_data)))
        ])
        
       
        total_power = cp.sum(psd_theta) + cp.sum(psd_alpha) + cp.sum(psd_beta)
        band_power_ratio = float((cp.sum(psd_beta) / total_power).get()) if total_power > 0 else 0
        
      
        filtered_data_cpu = cp.asnumpy(filtered_data)
        shannon_ent = shannon_entropy_cuda(filtered_data_cpu)
        renyi_ent = renyi_entropy_cuda(filtered_data_cpu)
        
     
        feature.extend([
            band_power_ratio,
            shannon_ent,
            renyi_ent
        ])
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise
        
    return feature

if __name__ == '__main__':
    
    cp.cuda.Device(0).use()
    
    total = 0
    path = u'DREAMER.mat'
    data = sio.loadmat(path)
    print("Check 1 over over")
    
    EEG_tmp = np.zeros((23, 18, 14 * 14))
    total_iterations = 23 * 18 * 14
    
    try:
        for k in range(0, 23):
            for j in range(0, 18):
                for i in range(0, 14):
                    B, S = [], []
                    basl = data['DREAMER'][0, 0]['Data'][0, k]['EEG'][0, 0]['baseline'][0, 0][j, 0][:, i]
                    stim = data['DREAMER'][0, 0]['Data'][0, k]['EEG'][0, 0]['stimuli'][0, 0][j, 0][:, i]
                    
                    B = preprocessing_gpu(basl, B)
                    S = preprocessing_gpu(stim, S)
                    
                    Extrod = safe_divide_cuda(np.array(S), np.array(B))
                    Extrod = np.nan_to_num(Extrod, nan=1.0, posinf=1.0, neginf=1.0)
                    
                    feature_idx = i * 14
                    EEG_tmp[k, j, feature_idx:feature_idx + 14] = Extrod
                    
                    total += 1
                    progress = total/total_iterations * 100
                    bar_length = 50
                    filled_length = int(bar_length * progress // 100)
                    bar = '=' * filled_length + '-' * (bar_length - filled_length)
                    print(f'\rProgress: |{bar}| {progress:.1f}% Complete', end='')
        
        print("\nFeature extraction completed!")
        
        feature_names = ['psdtheta', 'psdalpha', 'psdbeta', 'mean', 'variance', 'std_dev', 
                        'rms', 'max_amp', 'min_amp', 'kurtosis', 'skewness',
                        'band_power_ratio', 'shannon_entropy', 'renyi_entropy']
        
        col = []
        for i in range(0, 14):
            for feature in feature_names:
                col.append(f'{feature}_ch{i+1}_un')
        
        EEG = pd.DataFrame(EEG_tmp.reshape((23 * 18, EEG_tmp.shape[2])), columns=col)
        
        scaler = pre.StandardScaler()
        for i in range(len(col)):
            EEG[col[i][:-3]] = scaler.fit_transform(EEG[[col[i]]])
        
        EEG.drop(col, axis=1, inplace=True)
        
        print("\nShape :", EEG.shape)
        print("\nSample:")
        print(EEG.head())
        EEG.to_csv('Dreamer1.csv')
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

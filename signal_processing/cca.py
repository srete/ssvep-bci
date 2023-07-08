import numpy as np
from sklearn.cross_decomposition import CCA
from matplotlib import pyplot as plt

def get_cca_reference_signals(data_len, target_freq, sampling_rate):
    reference_signals = []
    t = np.arange(0, (data_len/(sampling_rate)), step=1.0/(sampling_rate))
    reference_signals.append(np.sin(np.pi*2*target_freq*t))
    reference_signals.append(np.cos(np.pi*2*target_freq*t))
    #reference_signals.append(np.sin(np.pi*4*target_freq*t))
    #reference_signals.append(np.cos(np.pi*4*target_freq*t))
    reference_signals = np.array(reference_signals)
    
    return reference_signals

def find_correlation(eeg_multich, references):
    cca = CCA(n_components=1)
    result = np.zeros(references.shape[0])
    for f_idx in range(references.shape[0]):
        x_scores, y_scores = cca.fit_transform(eeg_multich.T, np.squeeze(references[f_idx, :, :]).T)
        corr_coef = np.corrcoef(x_scores[:, 0], y_scores[:, 0])[0, 1]
        #print(corr_coef)
        result[f_idx] = corr_coef
    return result


def cca_classify(data, reference_templates, print_corr=False):
    predicted_class = []
    labels = []
    for f_idx, blink_f in enumerate(data):
        for trial in blink_f:
            result = find_correlation(trial, reference_templates)
            predicted_class.append(np.argmax(result))
            if print_corr: print(f'Max correlation: {np.max(result)}, Min correlation: {np.min(result)}')
            labels.append(f_idx)
    return predicted_class, labels

if __name__ == '__main__':
    data = np.random.rand(200)
    ref = get_cca_reference_signals(200, 10, 200).T
    #find_correlation(1, X, Y)
    X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    cca = CCA(n_components=1)
    cca.fit(X, Y)

    X_c, Y_c = cca.transform(X, Y)
    corr_coef = np.corrcoef(X_c[:, 0], Y_c[:, 0])
    print(X_c.shape, Y_c.shape)
    print(corr_coef)
    
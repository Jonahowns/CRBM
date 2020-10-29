import numpy as np
from sklearn import preprocessing



# One Hot Encoding Sequences
enc = preprocessing.OneHotEncoder()


# Trial Seqs
seqs = np.asarray(['AGCT', 'TACC', 'CACT', 'TTCT', 'GACT', 'CCGT', 'GAAG', 'GATA'])
nseqs = np.asarray([list(x) for x in seqs])
print(nseqs)
seqs = nseqs
#seqs = nseqs.reshape(-1, 1)
print(seqs)

enc.fit(seqs)
onehotlabels = enc.transform(seqs).toarray()

print(onehotlabels)
# def dil_conv

t1 = onehotlabels[0]
t1a = t1.reshape(4, -1)

t2a = onehotlabels[1].reshape(4, -1)
print(t1a)



trial_weights = np.full((2,2), 1.0)
def convolution_kernel(weights, seq, alpha_size=4):
    ss_size = alpha_size+1
    seq_size = len(seq)
    nseq = np.roll(seq, 1)
    samplespace = np.full((ss_size, ss_size), 0.0)
    samplespace[0:seq_size, 0:alpha_size] = nseq
    samplespace[0:seq_size, alpha_size] = nseq[:, 0]
    samplespace[seq_size, 0:ss_size] = samplespace[0, 0:ss_size]

    conv_matrix = np.full((alpha_size, alpha_size), 0.0)
    # 2 x 2 Convolution
    for i in range(alpha_size):
        for j in range(alpha_size):
            sampled = samplespace[i:i+2, j:j+2]
            val = np.sum(np.multiply(sampled, weights))
            conv_matrix[i, j] = val
    return conv_matrix

# def dilated_convolution_kernel(weights, seq):
def dilated_convolution_kernel(weights, seq, alpha_size=4):
    ss_size = alpha_size+1
    seq_size = len(seq)
    nseq = np.roll(seq, 1)
    samplespace = np.full((ss_size, ss_size), 0.0)
    samplespace[0:seq_size, 0:alpha_size] = nseq
    samplespace[0:seq_size, alpha_size] = nseq[:, 0]
    samplespace[seq_size, 0:ss_size] = samplespace[0, 0:ss_size]

    conv_matrix = np.full((alpha_size-1, alpha_size-1), 0.0)
    # 2 x 2 Convolution, dilation = 1
    for i in range(alpha_size-1):
        for j in range(alpha_size-1):
            d = 1
            sampled = np.asarray([[samplespace[i][j], samplespace[i+2][j]],
                        [samplespace[i][j+2], samplespace[i+2][j+2]]])
            val = np.sum(np.multiply(sampled, weights))
            conv_matrix[i, j] = val
    return conv_matrix



ck = convolution_kernel(trial_weights, t1a)
print(ck)

ck = convolution_kernel(trial_weights, t2a)
print(ck)

ck2 = dilated_convolution_kernel(trial_weights, t1a)
print(ck2)

ck2 = dilated_convolution_kernel(trial_weights, t2a)
print(ck2)
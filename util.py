import os
import shutil
import asteroid.metrics as metrics

def getAllMetrics(mix, clean, est, sample_rate, n_src=2, metrics_list=['si_sdr', 'sdr', 'pesq']):
    """
    The funtion uses the get_metrics function of asteroid to calcultate the desired params
    inputs:
        mix: Array, mixed signal
        clean   : Array (n_srcXlen_signal), matrix with the signals of the clean speeches
        est     : Array (n_srcXlen_signal), matrix with the output of the model, estimated signals
        n_src   : Int, number of sources
        metrics : list, Desired metrix values (input-output)
    outputs:
        results : Matrix, contains the quality params in the order specify in metrics
    """
    results = np.zeros([n_src, len(metrics_list)*2])
    if n_src == 1:
        results = list(metrics.get_metrics(mix, clean, est, \
                                              sample_rate=8000, metrics_list=metrics_list).values())
    else:
        for i in range(n_src):
            results[i] = list(metrics.get_metrics(mix, clean[i], est[i], \
                                                  sample_rate=8000, metrics_list=metrics_list).values())
    return results

def mixSignals(signals, noise, SNR=5):
    """
    Mixing speech signals with noise
    input:
        signals : numpy array, It contains the speech signals (each signal is a row in the array)
        noise   : numpy array, Noise signal
        SNR     : float, desired SNR between signals and noise
    output:
        output  : numpy array, mixed signals
    """
    signals = np.dot(1/signals.max(axis=1), signals) # Normalizing signals
    signals = signals/max(signals)

    noise = noise/noise.max()
    noise = noise[:len(signals)]
    factor = 10**(SNR/10)

    signals = signals*factor

    output = signals+noise
    output = output / output.max()

    return output

def adaptDataset(rootFolder, trainingFolder, libriMixDataPath):
    """

    Params:
        rootFolder: string, path of the data part of the project
        trainingFolder: string, path of the desired training data
        libriMixDataPath: string, path with the libriMix dataset
    """
    # rootFolder = "./data"
    os.chdir(rootFolder)
    print(rootFolder)
    print(os.listdir(libriMixDataPath+"/mix_clean"))
    filesMix = os.listdir(libriMixDataPath+"/mix_clean")
    names = ["s1", "s2"]#, "s3"]
    samplesDir = {0: os.listdir(libriMixDataPath+"/s1"),
                  1: os.listdir(libriMixDataPath+"/s2")}
                  #2: os.listdir(libriMixDataPath+"/s3")}
    print(len(filesMix))
    for j, file in enumerate(filesMix):
        print(j)
        mixFolder = trainingFolder+"/"+file.split(".")[0]
        #print(mixFolder)
        if not os.path.isdir(mixFolder):
            os.makedirs(mixFolder)
        audios = file.split(".")[0]#.split("_")

        os.rename(libriMixDataPath+"/mix_clean/"+file, mixFolder+"/mixture.wav")
        for i in range(2):
            #sampleName = sample.split("-")[0]
            for s in samplesDir[i]:
                if audios in s:
                    sampleFile = audios
                    break
            #sampleFile = next((s for s in samplesDir[i] if sample in s), None)
            if os.path.exists(libriMixDataPath+"/"+names[i]+"/"+sampleFile+".wav"):
                os.rename(libriMixDataPath+"/"+names[i]+"/"+sampleFile+".wav", mixFolder+"/"+names[i]+".wav")
            else:
                shutil.rmtree(mixFolder)
                break

if __name__ == "__main__":
    root = "./asteroid/asteroid"
    librimixPath = "./notebooks/MiniLibriMix/val"
    adaptDataset(root, "./data/val", librimixPath)
import numpy
import wav2numpy


class FFT:
    @staticmethod
    def pad(d, n):
        if len(d) < n:
            r = numpy.zeros(n)
            r[:len(d)] = d
            return r
        return d

    @classmethod
    def calculate_fft(cls, wavfile, max_sample=None):
        # Read wavefile into NumPy array
        wav_data = wav2numpy.read_nparray_mono(wavfile)

        # Set to max_sample size, if requested
        if max_sample is not None:
            wav_data = cls.pad(wav_data[:max_sample], max_sample)

        wav_data = (wav_data - numpy.mean(wav_data)) / (numpy.std(wav_data) + 1e-6)

        return numpy.fft.fft(wav_data)

    @staticmethod
    def calc_convolution(fft1, fft2):
        conv = numpy.real(numpy.fft.ifft(fft1.conj() * fft2))
        idx = numpy.argmax(conv)

        return conv[idx]

    @staticmethod
    def wavs_match(file1, file2, max_sample=None, threshold=50.0):
        fft1 = FFT.calculate_fft(file1, max_sample=max_sample)
        fft2 = FFT.calculate_fft(file2, max_sample=max_sample)
        convolution = FFT.calc_convolution(fft1, fft2)

        return convolution > threshold

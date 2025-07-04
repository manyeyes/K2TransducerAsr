// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr.Model;
using SpeechFeatures;
//using KaldiNativeFbankSharp;

namespace K2TransducerAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class WavFrontend : IDisposable
    {
        private bool _disposed;
        private FrontendConfEntity _frontendConfEntity;
        OnlineFbank _onlineFbank;
        //WhisperFeatures _onlineFbank;

        public WavFrontend(FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels
                );
        }

        public float[] GetFbank(float[] samples)
        {
            float sample_rate = _frontendConfEntity.fs;
            float[] fbanks = _onlineFbank.GetFbank(samples);//or GetFbankIndoor
            return fbanks;
        }

        public void InputFinished()
        {
            _onlineFbank.InputFinished();
        }

        //public WavFrontend(FrontendConfEntity frontendConfEntity)
        //{
        //    _frontendConfEntity = frontendConfEntity;
        //    _onlineFbank = new WhisperFeatures(
        //        nMels: frontendConfEntity.n_mels,
        //        threadsNum: 5,
        //        melFiltersFilePath: null
        //        );
        //}

        //public float[] GetFbank(float[] samples)
        //{
        //    float[] tempChunk = new float[480000];
        //    Array.Copy(samples, 0, tempChunk, 0, samples.Length);
        //    tempChunk = tempChunk.Select(x => x == 0 ? -23.025850929940457F / 32768.0f : x).ToArray();
        //    float[] mel = _onlineFbank.LogMelSpectrogram(tempChunk);
        //    return mel;
        //}

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_onlineFbank != null)
                    {
                        _onlineFbank.Dispose();
                    }
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~WavFrontend()
        {
            Dispose(_disposed);
        }
    }
}

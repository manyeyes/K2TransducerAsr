﻿// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr.Model;
using KaldiNativeFbankSharp;

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

        //public void InputFinished()
        //{
        //    _onlineFbank.InputFinished();
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

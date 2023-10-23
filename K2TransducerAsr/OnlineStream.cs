// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr.Model;

namespace K2TransducerAsr
{
    public class OnlineStream
    {
        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private OnlineInputEntity _onlineInputEntity;
        private int _blank_id = 0;
        private int _unk_id = 2;
        private Int64[] _hyp;
        List<Int64> _tokens = new List<Int64>();
        List<int> _timestamps = new List<int>();
        List<List<float[]>> _states = new List<List<float[]>>();
        int _chunkLength = 0;
        int _shiftLength = 0;
        int _sampleRate = 16000;
        int _featureDim = 80;
        private static object obj = new object();
        public OnlineStream(IOnlineProj onlineProj)
        {
            if (onlineProj != null)
            {
                _states = onlineProj.GetEncoderInitStates();
                _chunkLength = onlineProj.ChunkLength;
                _shiftLength = onlineProj.ShiftLength;
                _featureDim = onlineProj.FeatureDim;
                _sampleRate = onlineProj.SampleRate;
            }
            _onlineInputEntity = new OnlineInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = _sampleRate;
            _frontendConfEntity.n_mels = _featureDim;
            _wavFrontend = new WavFrontend(_frontendConfEntity);
            _hyp = new Int64[] { _blank_id, _blank_id };

            _tokens = new List<Int64> { _blank_id, _blank_id };
        }

        public OnlineInputEntity OnlineInputEntity { get => _onlineInputEntity; set => _onlineInputEntity = value; }
        public long[] Hyp { get => _hyp; set => _hyp = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<List<float[]>> States { get => _states; set => _states = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    oLen = OnlineInputEntity.SpeechLength;
                }
                float[] features = _wavFrontend.GetFbank(samples);
                float[]? featuresTemp = new float[oLen + features.Length];
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    Array.Copy(_onlineInputEntity.Speech, 0, featuresTemp, 0, _onlineInputEntity.SpeechLength);
                }
                Array.Copy(features, 0, featuresTemp, OnlineInputEntity.SpeechLength, features.Length);
                OnlineInputEntity.Speech = featuresTemp;
                OnlineInputEntity.SpeechLength = featuresTemp.Length;
            }
        }

        // Note: chunk_length is in frames before subsampling
        public float[]? GetDecodeChunk(int chunkLength)
        {
            int featureDim = _frontendConfEntity.n_mels;
            lock (obj)
            {
                float[]? padChunk = new float[chunkLength * featureDim];
                if (chunkLength * featureDim <= _onlineInputEntity.SpeechLength)
                {
                    float[]? features = _onlineInputEntity.Speech;
                    Array.Copy(features, 0, padChunk, 0, padChunk.Length);
                    return padChunk;
                }
                else
                {
                    return null;
                }
            }
        }

        public void RemoveChunk(int shiftLength)
        {
            lock (obj)
            {
                int featureDim = _frontendConfEntity.n_mels;
                if (shiftLength * featureDim <= _onlineInputEntity.SpeechLength)
                {
                    float[]? features = _onlineInputEntity.Speech;
                    float[]? featuresTemp = new float[features.Length - shiftLength * featureDim];
                    Array.Copy(features, shiftLength * featureDim, featuresTemp, 0, featuresTemp.Length);
                    _onlineInputEntity.Speech = featuresTemp;
                    _onlineInputEntity.SpeechLength = featuresTemp.Length;
                }
            }
        }

        /// <summary>
        /// when is endpoint,determine whether it is completed
        /// </summary>
        /// <param name="isEndpoint"></param>
        /// <returns></returns>
        public bool IsFinished(bool isEndpoint = false)
        {
            int featureDim = _frontendConfEntity.n_mels;
            if (isEndpoint)
            {
                int oLen = 0;
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    oLen = OnlineInputEntity.SpeechLength;
                }
                if (oLen > 0)
                {
                    var avg = OnlineInputEntity.Speech.Average();
                    int num = OnlineInputEntity.Speech.Where(x => x != avg).ToArray().Length;
                    if (num == 0)
                    {
                        return true;
                    }
                    else
                    {
                        if (oLen <= _chunkLength * featureDim)
                        {
                            AddSamples(new float[400]);
                        }
                        return false;
                    }

                }
                else
                {
                    return true;
                }
            }
            else
            {
                return false;
            }
        }
    }
}

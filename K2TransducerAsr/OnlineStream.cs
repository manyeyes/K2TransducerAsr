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
        private OnlineCustomMetadata _onlineCustomMetadata;
        List<Int64> _tokens = new List<Int64>();
        List<int> _timestamps = new List<int>();
        List<List<float[]>> _states = new List<List<float[]>>();
        private static object obj = new object();
        public OnlineStream(OnlineCustomMetadata onlineCustomMetadata, int sampleRate = 16000, int featureDim = 80)
        {
            _onlineCustomMetadata = onlineCustomMetadata;
            _onlineInputEntity = new OnlineInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = sampleRate;
            _frontendConfEntity.n_mels = featureDim;
            _wavFrontend = new WavFrontend(_frontendConfEntity);
            _hyp = new Int64[] { _blank_id, _blank_id };
            _states = GetEncoderInitStates();
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
                }
                return padChunk;
            }
        }

        public void RemoveChunk(int shiftLength)
        {
            lock (obj)
            {
                int featureDim = _frontendConfEntity.n_mels;
                if (shiftLength * featureDim < _onlineInputEntity.SpeechLength)
                {
                    float[]? features = _onlineInputEntity.Speech;
                    float[]? featuresTemp = new float[features.Length - shiftLength * featureDim];
                    Array.Copy(features, shiftLength * featureDim, featuresTemp, 0, featuresTemp.Length);
                    _onlineInputEntity.Speech = featuresTemp;
                    _onlineInputEntity.SpeechLength = featuresTemp.Length;
                }
            }
        }


        private List<List<float[]>> GetEncoderInitStates(int batchSize = 1)
        {
            List<float[]> cached_len = new List<float[]>();
            List<float[]> cached_avg = new List<float[]>();
            List<float[]> cached_key = new List<float[]>();
            List<float[]> cached_val = new List<float[]>();
            List<float[]> cached_val2 = new List<float[]>();
            List<float[]> cached_conv1 = new List<float[]>();
            List<float[]> cached_conv2 = new List<float[]>();
            int num_encoders = _onlineCustomMetadata.Num_encoder_layers.Length;
            //TODO 改为计算尺寸
            //int cached_len_size = 0;
            //int cached_avg_size = 0;
            //int cached_key_size = 0;
            //int cached_val_size = 0;
            //int cached_val2_size = 0;
            //int cached_conv1_size = 0;
            //int cached_conv2_size = 0;
            for (int i = 0; i < num_encoders; i++)
            {

                int num_encoder_layers = _onlineCustomMetadata.Num_encoder_layers[i];
                //cached_len
                int cached_len_size = num_encoder_layers * batchSize;
                float[] cached_len_item = new float[cached_len_size];
                cached_len.Add(cached_len_item);
                //cached_avg
                int cached_avg_size = num_encoder_layers * batchSize * _onlineCustomMetadata.Encoder_dims[i];
                float[] cached_avg_item = new float[cached_avg_size];
                cached_avg.Add(cached_avg_item);
                //cached_key
                int cached_key_size = num_encoder_layers * _onlineCustomMetadata.Left_context_len[i] * batchSize * _onlineCustomMetadata.Attention_dims[i];
                float[] cached_key_item = new float[cached_key_size];
                cached_key.Add(cached_key_item);
                //cached_val
                int cached_val_size = num_encoder_layers * _onlineCustomMetadata.Left_context_len[i] * batchSize * _onlineCustomMetadata.Attention_dims[i] / 2;
                float[] cached_val_item = new float[cached_val_size];
                cached_val.Add(cached_val_item);
                //cached_val2
                int cached_val2_size = num_encoder_layers * _onlineCustomMetadata.Left_context_len[i] * batchSize * _onlineCustomMetadata.Attention_dims[i] / 2;
                float[] cached_val2_item = new float[cached_val2_size];
                cached_val2.Add(cached_val2_item);
                //cached_conv1
                int cached_conv1_size = num_encoder_layers * batchSize * _onlineCustomMetadata.Encoder_dims[i] * (_onlineCustomMetadata.Cnn_module_kernels[i] - 1);
                float[] cached_conv1_item = new float[cached_conv1_size];
                cached_conv1.Add(cached_conv1_item);
                //cached_conv2
                int cached_conv2_size = num_encoder_layers * batchSize * _onlineCustomMetadata.Encoder_dims[i] * (_onlineCustomMetadata.Cnn_module_kernels[i] - 1);
                float[] cached_conv2_item = new float[cached_conv2_size];
                cached_conv2.Add(cached_conv2_item);
            }
            List<List<float[]>> statesList = new List<List<float[]>> { cached_len, cached_avg, cached_key, cached_val, cached_val2, cached_conv1, cached_conv2 };
            return statesList;
        }
    }
}

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace K2TransducerAsr
{
    public class OnlineModel
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _joinerSession;
        private OnlineCustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _featureDim = 80;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public OnlineModel(string encoderFilePath, string decoderFilePath, string joinerFilePath, string configFilePath="", int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _joinerSession = initModel(joinerFilePath, threadsNum);

            var encoder_meta = _encoderSession.ModelMetadata.CustomMetadataMap;

            _customMetadata = new OnlineCustomMetadata();
            int decode_chunk_len;
            int.TryParse(encoder_meta["decode_chunk_len"].ToString(), out decode_chunk_len);
            _customMetadata.Decode_chunk_len = decode_chunk_len;

            int _TT;
            int.TryParse(encoder_meta["T"].ToString(), out _TT);
            _customMetadata.TT = _TT;

            _chunkLength = _TT;
            _shiftLength = decode_chunk_len;

            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("num_encoder_layers"))
            {
                _customMetadata.Num_encoder_layers = Array.ConvertAll(encoder_meta["num_encoder_layers"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }
            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("encoder_dims"))
            {
                _customMetadata.Encoder_dims = Array.ConvertAll(encoder_meta["encoder_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }
            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("attention_dims")) {
                _customMetadata.Attention_dims = Array.ConvertAll(encoder_meta["attention_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }
            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("cnn_module_kernels"))
            {
                _customMetadata.Cnn_module_kernels = Array.ConvertAll(encoder_meta["cnn_module_kernels"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }
            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("left_context_len"))
            {
                _customMetadata.Left_context_len = Array.ConvertAll(encoder_meta["left_context_len"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }

            int context_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["context_size"], out context_size);
            CustomMetadata.Context_size = context_size;
            int vocab_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["vocab_size"], out vocab_size);
            CustomMetadata.Vocab_size = vocab_size;
            int joiner_dim;
            int.TryParse(_joinerSession.ModelMetadata.CustomMetadataMap["joiner_dim"], out joiner_dim);
            _customMetadata.Joiner_dim= joiner_dim;
            //string version= _encoderSession.ModelMetadata.CustomMetadataMap["version"];
            //_customMetadata.Version = version;
            string? model_type = string.Empty;
            _encoderSession.ModelMetadata.CustomMetadataMap.TryGetValue("model_type", out model_type);
            _customMetadata.Model_type = model_type;
            string? model_author = string.Empty;
            _encoderSession.ModelMetadata.CustomMetadataMap.TryGetValue("model_author", out model_author);
            _customMetadata.Model_author = model_author;
            string? version = string.Empty;
            _encoderSession.ModelMetadata.CustomMetadataMap.TryGetValue("version", out version);
            _customMetadata.Version = version;

            //zipformer2
            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("query_head_dims"))
            {
                _customMetadata.Query_head_dims = Array.ConvertAll(encoder_meta["query_head_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }
            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("value_head_dims"))
            {
                _customMetadata.Value_head_dims = Array.ConvertAll(encoder_meta["value_head_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }
            if (_encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("num_heads"))
            {
                _customMetadata.Num_heads = Array.ConvertAll(encoder_meta["num_heads"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            }
            string? onnx_infer = string.Empty;
            _encoderSession.ModelMetadata.CustomMetadataMap.TryGetValue("onnx.infer", out onnx_infer);
            _customMetadata.Onnx_infer = onnx_infer;
            string? comment = string.Empty;
            _encoderSession.ModelMetadata.CustomMetadataMap.TryGetValue("comment", out comment);
            _customMetadata.Comment = comment;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession JoinerSession { get => _joinerSession; set => _joinerSession = value; }
        public OnlineCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }

        public InferenceSession initModel(string modelFilePath, int threadsNum = 2)
        {
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            //options.AppendExecutionProvider_DML(0);
            options.AppendExecutionProvider_CPU(0);
            //options.AppendExecutionProvider_CUDA(0);
            options.InterOpNumThreads = threadsNum;
            InferenceSession onnxSession = new InferenceSession(modelFilePath, options);
            return onnxSession;
        }

    }
}

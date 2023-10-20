﻿// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace K2TransducerAsr
{
    public class OfflineModel
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _joinerSession;
        private OfflineCustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int sos_eos_id = 1;
        private int _unk_id = 2;
        private int _featureDim = 80;

        public OfflineModel(string encoderFilePath, string decoderFilePath, string joinerFilePath, int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _joinerSession = initModel(joinerFilePath, threadsNum);

            _customMetadata = new OfflineCustomMetadata();

            int context_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["context_size"], out context_size);
            CustomMetadata.Context_size = context_size;
            int vocab_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["vocab_size"], out vocab_size);
            CustomMetadata.Vocab_size = vocab_size;

            int joiner_dim;
            int.TryParse(_joinerSession.ModelMetadata.CustomMetadataMap["joiner_dim"], out joiner_dim);
            CustomMetadata.Joiner_dim= joiner_dim;

            var encoder_meta = _encoderSession.ModelMetadata.CustomMetadataMap;
            _customMetadata.Version = encoder_meta.ContainsKey("version")? encoder_meta["version"]:"";
            _customMetadata.Model_type = encoder_meta.ContainsKey("model_type") ? encoder_meta["model_type"]:"";
            _customMetadata.Model_author = encoder_meta.ContainsKey("model_author") ? encoder_meta["model_author"]:"";
            string? comment = string.Empty;
            encoder_meta.TryGetValue("comment", out comment);
            _customMetadata.Comment = comment;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession JoinerSession { get => _joinerSession; set => _joinerSession = value; }
        public OfflineCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => sos_eos_id; set => sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
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

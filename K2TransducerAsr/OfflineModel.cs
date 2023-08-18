// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

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

        public OfflineModel(string encoderFilePath, string decoderFilePath, string joinerFilePath, int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _joinerSession = initModel(joinerFilePath, threadsNum);

            _customMetadata = new OfflineCustomMetadata();

            var encoder_meta = _encoderSession.ModelMetadata.CustomMetadataMap;

            int context_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["context_size"], out context_size);
            CustomMetadata.Context_size = context_size;
            int vocab_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["vocab_size"], out vocab_size);
            CustomMetadata.Vocab_size = vocab_size;

            int joiner_dim;
            int.TryParse(_joinerSession.ModelMetadata.CustomMetadataMap["joiner_dim"], out joiner_dim);
            CustomMetadata.Joiner_dim= joiner_dim;

            CustomMetadata.Version = _encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("version")? _encoderSession.ModelMetadata.CustomMetadataMap["version"]:"";
            CustomMetadata.Model_type = _encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("model_type") ?_encoderSession.ModelMetadata.CustomMetadataMap["model_type"]:"";
            CustomMetadata.Model_type = _encoderSession.ModelMetadata.CustomMetadataMap.ContainsKey("model_author") ? _encoderSession.ModelMetadata.CustomMetadataMap["model_author"]:"";          
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession JoinerSession { get => _joinerSession; set => _joinerSession = value; }
        public OfflineCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => sos_eos_id; set => sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }

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

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using K2TransducerAsr.Model;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;

namespace K2TransducerAsr
{
    public class OfflineRecognizer
    {
        private readonly ILogger<OfflineRecognizer> _logger;
        private FrontendConfEntity _frontendConfEntity;
        private string[] _tokens;
        private int _max_sym_per_frame = 1;
        private OfflineModel _offlineModel;
        private delegate void ForwardBatch(List<OfflineStream> streams);
        private ForwardBatch _forwardBatch;
        private delegate void Forward(OfflineStream stream);
        private Forward _forward;
        public OfflineRecognizer(string encoderFilePath, string decoderFilePath, string joinerFilePath, string tokensFilePath,
            string decodingMethod = "greedy_search", int sampleRate = 16000, int featureDim = 80, int threadsNum = 2, bool debug = false)
        {
            _offlineModel = new OfflineModel(encoderFilePath, decoderFilePath, joinerFilePath, threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);

            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = sampleRate;
            _frontendConfEntity.n_mels = featureDim;
            switch (decodingMethod)
            {
                case "greedy_search":
                    _forward = new Forward(this.ForwardGreedySearch);
                    _forwardBatch = new ForwardBatch(this.ForwardBatchGreedySearch);
                    break;
            }

            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<OfflineRecognizer>(loggerFactory);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream offlineStream = new OfflineStream(_offlineModel.CustomMetadata, sampleRate: _frontendConfEntity.fs, featureDim: _frontendConfEntity.n_mels);
            return offlineStream;
        }

        public OfflineRecognizerResultEntity GetResult(OfflineStream stream)
        {
            this._logger.LogInformation("get features begin");
            OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
            _forward.Invoke(stream);
            offlineRecognizerResultEntity = this.DecodeMulti(new List<OfflineStream>(){ stream })[0];
            return offlineRecognizerResultEntity;
        }

        public List<OfflineRecognizerResultEntity> GetResults(List<OfflineStream> streams)
        {
            this._logger.LogInformation("get features begin");
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
            _forwardBatch.Invoke(streams);
            offlineRecognizerResultEntities = this.DecodeMulti(streams);
            return offlineRecognizerResultEntities;
        }

        private EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs, int batchSize)
        {
            int featureDim = _frontendConfEntity.n_mels;
            float[] padSequence = PadSequence(modelInputs);
            var inputMeta = _offlineModel.EncoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / featureDim / batchSize, featureDim };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "x_lens")
                {
                    int[] dim = new int[] { batchSize };
                    Int64[] speech_lengths = new Int64[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        speech_lengths[i] = padSequence.Length / featureDim / batchSize;
                    }
                    var tensor = new DenseTensor<Int64>(speech_lengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _offlineModel.EncoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    encoderOutput.encoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
                    encoderOutput.encoder_out_lens = encoderResultsArray[1].AsEnumerable<Int64>().ToArray();
                }
            }
            catch (Exception ex)
            {
                //
            }
            return encoderOutput;
        }
        private DecoderOutputEntity DecoderProj(Int64[]? decoder_input, int batchSize)
        {
            int contextSize = _offlineModel.CustomMetadata.Context_size;
            DecoderOutputEntity decoderOutput = new DecoderOutputEntity();
            if (decoder_input == null)
            {
                Int64[] hyp = new Int64[] { -1, _offlineModel.Blank_id };
                decoder_input = hyp;
                if (batchSize > 1)
                {
                    decoder_input = new Int64[contextSize * batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        Array.Copy(hyp, 0, decoder_input, i * contextSize, contextSize);
                    }
                }                
            }
            var decoder_container = new List<NamedOnnxValue>();
            int[] dim = new int[] { decoder_input.Length / 2, 2 };
            var decoder_input_tensor = new DenseTensor<Int64>(decoder_input, dim, false);
            decoder_container.Add(NamedOnnxValue.CreateFromTensor<Int64>("y", decoder_input_tensor));
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
            decoderResults = _offlineModel.DecoderSession.Run(decoder_container);
            if (decoderResults != null)
            {
                var encoderResultsArray = decoderResults.ToArray();
                decoderOutput.decoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
            }
            return decoderOutput;
        }

        private JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out)
        {
            int joinerDim = _offlineModel.CustomMetadata.Joiner_dim;//512
            var inputMeta = _offlineModel.JoinerSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "encoder_out")
                {
                    int[] dim = new int[] { encoder_out.Length / joinerDim, joinerDim };
                    var tensor = new DenseTensor<float>(encoder_out, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "decoder_out")
                {
                    int[] dim = new int[] { decoder_out.Length / joinerDim, joinerDim };
                    var tensor = new DenseTensor<float>(decoder_out, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> joinerResults = null;
            joinerResults = _offlineModel.JoinerSession.Run(container);
            JoinerOutputEntity joinerOutput = new JoinerOutputEntity();
            var joinerResultsArray = joinerResults.ToArray();
            joinerOutput.Logit = joinerResultsArray[0].AsEnumerable<float>().ToArray();
            joinerOutput.Logits = joinerResultsArray[0].AsTensor<float>();
            return joinerOutput;
        }

        private void ForwardGreedySearch(OfflineStream stream)
        {
            int batchSize = 1;
            OfflineRecognizerResultEntity modelOutput = new OfflineRecognizerResultEntity();
            try
            {
                List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
                modelInputs.Add(stream.OfflineInputEntity);
                EncoderOutputEntity encoderOutput = EncoderProj(modelInputs, batchSize);
                int TT = encoderOutput.encoder_out.Length / 512;
                int t = 0;
                Int64[] hyp = new Int64[] { -1, _offlineModel.Blank_id };
                DecoderOutputEntity decoderOutput = DecoderProj(hyp, batchSize);
                float[] decoder_out = decoderOutput.decoder_out;
                
                List<Int64> hypList = new List<Int64>();
                hypList.Add(-1);
                hypList.Add(_offlineModel.Blank_id);
                List<int> timestamp = new List<int>();
                int max_sym_per_utt = 1000;
                int sym_per_frame = 0;
                int sym_per_utt = 0;
                while (t < TT && sym_per_utt < max_sym_per_utt)
                {
                    if (sym_per_frame >= _max_sym_per_frame)
                    {
                        sym_per_frame = 0;
                        t += 1;
                        continue;
                    }
                    // fmt: off
                    float[] current_encoder_out = new float[512];
                    Array.Copy(encoderOutput.encoder_out, t * 512, current_encoder_out, 0, 512);
                    // fmt: on
                    JoinerOutputEntity joinerOutput = JoinerProj(
                        current_encoder_out, decoder_out
                    );
                    Tensor<float> logits = joinerOutput.Logits;
                    List<int[]> token_nums = new List<int[]> { };
                    int itemLength = logits.Dimensions[0];
                    for (int i = 0; i < 1; i++)
                    {
                        int[] item = new int[itemLength];
                        for (int j = i * itemLength; j < (i + 1) * itemLength; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits.Dimensions[1]; k++)
                            {
                                token_num = logits[j, token_num] > logits[j, k] ? token_num : k;
                            }
                            item[j - i * itemLength] = (int)token_num;

                        }
                        token_nums.Add(item);
                    }
                    int y = token_nums[0][0];
                    if (y != _offlineModel.Blank_id && y != _offlineModel.Unk_id)
                    {
                        hypList.Add(y);
                        timestamp.Add(t);
                        Int64[] decoder_input = new Int64[] { hypList[hypList.Count - 2], hypList[hypList.Count - 1] };
                        decoder_out = DecoderProj(decoder_input, batchSize).decoder_out;
                        sym_per_utt += 1;
                        sym_per_frame += 1;
                    }
                    else
                    {
                        sym_per_frame = 0;
                        t += 1;
                    }
                }
                stream.Tokens = hypList;
                stream.Timestamps.AddRange(timestamp);
            }
            catch (Exception ex)
            {
                //
            }
        }

        private void ForwardBatchGreedySearch(List<OfflineStream> streams)
        {
            int contextSize = _offlineModel.CustomMetadata.Context_size;
            int batchSize = streams.Count;
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            foreach (OfflineStream stream in streams)
            {
                modelInputs.Add(stream.OfflineInputEntity);
            }
            try
            {
                EncoderOutputEntity encoderOutput = EncoderProj(modelInputs, batchSize);
                int TT = encoderOutput.encoder_out.Length / 512;
                Int64[] hyp = new Int64[] { -1, _offlineModel.Blank_id };
                Int64[] hyps = new Int64[contextSize * batchSize];
                for (int i = 0; i < batchSize; i++)
                {
                    Array.Copy(hyp, 0, hyps, i * contextSize, contextSize);
                }
                DecoderOutputEntity decoderOutput = DecoderProj(hyps, batchSize);
                float[] decoder_out = decoderOutput.decoder_out;
                // timestamp[i] is the frame index after subsampling
                // on which hyp[i] is decoded
                // TODO
                List<int[]> timestamp;
                int batchPerNum = TT / batchSize;
                List<Int64>[] tokens = new List<Int64>[batchSize];
                List<int>[] timestamps = new List<int>[batchSize];
                for (int t = 0; t < batchPerNum; t++)
                {
                    // fmt: off
                    float[] current_encoder_out = new float[512 * batchSize];
                    for(int b = 0;b < batchSize; b++)
                    {
                        Array.Copy(encoderOutput.encoder_out, t * 512+ (batchPerNum*512)*b, current_encoder_out, b*512, 512);
                    }
                    // fmt: on
                    JoinerOutputEntity joinerOutput = JoinerProj(
                        current_encoder_out, decoder_out
                    );
                    Tensor<float> logits = joinerOutput.Logits;
                    List<int[]> token_nums = new List<int[]> { };
                    int itemLength = logits.Dimensions[0] / batchSize;
                    for (int i = 0; i < batchSize; i++)
                    {
                        int[] item = new int[itemLength];
                        for (int j = i * itemLength; j < (i + 1) * itemLength; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits.Dimensions[1]; k++)
                            {
                                token_num = logits[j, token_num] > logits[j, k] ? token_num : k;
                            }
                            item[j - i * itemLength] = (int)token_num;

                        }
                        token_nums.Add(item);
                    }
                    bool emitted = false;
                    for (int m = 0; m < token_nums.Count; m++)
                    {
                        int y = token_nums[m][0];
                        if (tokens[m] == null)
                        {
                            tokens[m] = new List<Int64>();
                            for (int i = 0; i < batchSize; i++)
                            {
                                tokens[m].Add(_offlineModel.Blank_id);
                                tokens[m].Add(_offlineModel.Blank_id);
                            }
                        }
                        if (timestamps[m] == null)
                        {
                            timestamps[m] = new List<int>();
                            for (int i = 0; i < batchSize; i++)
                            {
                                timestamps[m].Add(0);
                                timestamps[m].Add(0);
                            }
                        }
                        if (y != _offlineModel.Blank_id && y != _offlineModel.Unk_id)
                        {
                            tokens[m].Add(y);
                            timestamps[m].Add(t);
                            emitted = true;
                        }
                        else
                        {
                        }
                    }
                    if (emitted)
                    {
                        Int64[] decoder_input = new Int64[contextSize * batchSize];
                        for (int m = 0; m < batchSize; m++)
                        {
                            Array.Copy(tokens[m].ToArray(), tokens[m].Count - contextSize, decoder_input, m * contextSize, contextSize);
                        }
                        decoder_out = DecoderProj(decoder_input, batchSize).decoder_out;
                    }

                }
                int streamIndex = 0;
                foreach (OfflineStream stream in streams)
                {
                    stream.Tokens = tokens[streamIndex];
                    stream.Timestamps.AddRange(timestamps[streamIndex]);
                    stream.RemoveSamples();
                    streamIndex++;
                }
                
            }
            catch (Exception ex)
            {
                //
            }
        }

        private List<OfflineRecognizerResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OfflineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 2)
                    {
                        break;
                    }
                    if (token == -1)
                    {
                        continue;
                    }
                    if (_tokens[token].Split(' ')[0] != "<blk>" && _tokens[token].Split(' ')[0] != "<sos/eos>" && _tokens[token].Split(' ')[0] != "<unk>")
                    {
                        if (IsChinese(_tokens[token], true))
                        {
                            text_result += _tokens[token].Split(' ')[0];
                        }
                        else
                        {
                            text_result += _tokens[token].Split(' ')[0] + "";
                        }
                    }
                }
                OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                offlineRecognizerResultEntity.text = text_result.Replace("▁", " ").ToLower();
                offlineRecognizerResultEntities.Add(offlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return offlineRecognizerResultEntities;
        }

        /// <summary>
        /// Verify if the string is in Chinese.
        /// </summary>
        /// <param name="checkedStr">The string to be verified.</param>
        /// <param name="allMatch">Is it an exact match. When the value is true,all are in Chinese; 
        /// When the value is false, only Chinese is included.
        /// </param>
        /// <returns></returns>
        private bool IsChinese(string checkedStr, bool allMatch)
        {
            string pattern;
            if (allMatch)
                pattern = @"^[\u4e00-\u9fa5]+$";
            else
                pattern = @"[\u4e00-\u9fa5]";
            if (Regex.IsMatch(checkedStr, pattern))
                return true;
            else
                return false;
        }

        private float[] PadSequence(List<OfflineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength) + 80 * 19;
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            float[,] xxx = new float[modelInputs.Count, max_speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                if (max_speech_length == modelInputs[i].SpeechLength)
                {
                    for (int j = 0; j < xxx.GetLength(1); j++)
                    {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                        xxx[i, j] = modelInputs[i].Speech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。
                    }
                    continue;
                }
                float[] nullspeech = new float[max_speech_length - modelInputs[i].SpeechLength];
                float[]? curr_speech = modelInputs[i].Speech;
                float[] padspeech = new float[max_speech_length];
                Array.Copy(curr_speech, 0, padspeech, 0, curr_speech.Length);
                for (int j = 0; j < padspeech.Length; j++)
                {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                    xxx[i, j] = padspeech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。 
                }

            }
            int s = 0;
            for (int i = 0; i < xxx.GetLength(0); i++)
            {
                for (int j = 0; j < xxx.GetLength(1); j++)
                {
                    speech[s] = xxx[i, j];
                    s++;
                }
            }
            speech = speech.Select(x => x == 0 ? -23.025850929940457F : x).ToArray();
            return speech;
        }
    }
}
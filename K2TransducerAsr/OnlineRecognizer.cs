// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;

namespace K2TransducerAsr
{
    public class OnlineRecognizer
    {

        private readonly ILogger<OnlineRecognizer> _logger;
        private FrontendConfEntity _frontendConfEntity;
        private string[] _tokens;

        private OnlineModel _onlineModel;
        private IOnlineProj? _onlineProj;

        private delegate void ForwardBatch(List<OnlineStream> streams);
        private ForwardBatch? _forwardBatch;

        public OnlineRecognizer(string encoderFilePath, string decoderFilePath, string joinerFilePath, string tokensFilePath, string configFilePath="", string decodingMethod = "greedy_search", int sampleRate = 16000, int featureDim = 80,
            int threadsNum = 2, bool debug = false, int maxActivePaths = 4, int enableEndpoint = 0)
        {
            _onlineModel = new OnlineModel(encoderFilePath, decoderFilePath, joinerFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = sampleRate;
            _frontendConfEntity.n_mels = featureDim;

            switch (_onlineModel.CustomMetadata.Model_type)
            {
                case "zipformer":
                    _onlineProj = new OnlineProjOfZipformer(_onlineModel);
                    break;
                case "zipformer2":
                    _onlineProj = new OnlineProjOfZipformer2(_onlineModel);
                    break;
            }
            switch (decodingMethod)
            {
                case "greedy_search":
                    _forwardBatch = new ForwardBatch(this.ForwardBatchGreedySearch);
                    break;
            }
            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<OnlineRecognizer>(loggerFactory);
        }

        public OnlineStream CreateOnlineStream()
        {
            OnlineStream onlineStream = new OnlineStream(_onlineProj, sampleRate: _frontendConfEntity.fs, featureDim: _frontendConfEntity.n_mels);
            return onlineStream;
        }

        public OnlineRecognizerResultEntity GetResult(OnlineStream stream)
        {
            OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
            List<OnlineStream> streams = new List<OnlineStream>();
            streams.Add(stream);
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = GetResults(streams);
            onlineRecognizerResultEntity = onlineRecognizerResultEntities[0];
            return onlineRecognizerResultEntity;
        }

        public List<OnlineRecognizerResultEntity> GetResults(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
            _forwardBatch.Invoke(streams);
            onlineRecognizerResultEntities = this.DecodeMulti(streams);
            return onlineRecognizerResultEntities;
        }

        private void ForwardBatchGreedySearch(List<OnlineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            int contextSize = _onlineModel.CustomMetadata.Context_size;
            List<OnlineInputEntity> modelInputs = new List<OnlineInputEntity>();
            List<List<List<float[]>>> stateList = new List<List<List<float[]>>>();
            List<Int64[]> hypList = new List<Int64[]>();
            List<List<Int64>> tokens = new List<List<Int64>>();
            foreach (OnlineStream stream in streams)
            {
                OnlineInputEntity onlineInputEntity = new OnlineInputEntity();
                onlineInputEntity.Speech = stream.GetDecodeChunk(_onlineModel.ChunkLength);
                if (onlineInputEntity.Speech == null)
                {
                    continue;
                }
                onlineInputEntity.SpeechLength = onlineInputEntity.Speech.Length;
                modelInputs.Add(onlineInputEntity);
                stream.RemoveChunk(_onlineModel.ShiftLength);
                hypList.Add(stream.Hyp);
                stateList.Add(stream.States);
                tokens.Add(stream.Tokens);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            int batchSize = modelInputs.Count;
            Int64[] hyps = new Int64[contextSize * batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                Array.Copy(hypList[i].ToArray(), 0, hyps, i * contextSize, contextSize);
            }
            try
            {
                List<float[]> states = new List<float[]>();
                List<List<float[]>> stackStatesList = new List<List<float[]>>();
                stackStatesList = _onlineProj.stack_states(stateList);
                EncoderOutputEntity encoderOutput = _onlineProj.EncoderProj(modelInputs, batchSize, stackStatesList);
                int joinerDim = _onlineModel.CustomMetadata.Joiner_dim;
                int TT = encoderOutput.encoder_out.Length / joinerDim;
                DecoderOutputEntity decoderOutput = _onlineProj.DecoderProj(hyps, batchSize);
                float[] decoder_out = decoderOutput.decoder_out;
                List<int[]> timestamp;
                int batchPerNum = TT / batchSize;

                List<int>[] timestamps = new List<int>[batchSize];
                for (int t = 0; t < batchPerNum; t++)
                {
                    // fmt: On
                    float[] current_encoder_out = new float[joinerDim * batchSize];
                    for (int b = 0; b < batchSize; b++)
                    {
                        Array.Copy(encoderOutput.encoder_out, t * joinerDim + (batchPerNum * joinerDim) * b, current_encoder_out, b * joinerDim, joinerDim);
                    }
                    // fmt: on
                    JoinerOutputEntity joinerOutput = _onlineProj.JoinerProj(
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
                        }
                        if (timestamps[m] == null)
                        {
                            timestamps[m] = new List<int>();
                        }
                        if (y != _onlineModel.Blank_id && y != _onlineModel.Unk_id && y != 1)
                        {
                            tokens[m].Add(y);
                            timestamps[m].Add(t);
                            emitted = true;
                        }
                        else
                        {
                            //do nothing 
                        }
                    }
                    if (emitted)
                    {
                        Int64[] decoder_input = new Int64[contextSize * batchSize];
                        for (int m = 0; m < batchSize; m++)
                        {
                            Array.Copy(tokens[m].ToArray(), tokens[m].Count - contextSize, decoder_input, m * contextSize, contextSize);
                        }
                        decoder_out = _onlineProj.DecoderProj(decoder_input, batchSize).decoder_out;
                    }

                }
                List<List<List<float[]>>> next_statesList = new List<List<List<float[]>>>();
                next_statesList = _onlineProj.unstack_states(encoderOutput.encoder_out_states);
                int streamIndex = 0;
                foreach (OnlineStream stream in streams)
                {
                    Array.Copy(tokens[streamIndex].ToArray(), tokens[streamIndex].Count - contextSize, stream.Hyp, 0, contextSize);
                    stream.Tokens = tokens[streamIndex];
                    stream.Timestamps.AddRange(timestamps[streamIndex]);
                    stream.States = next_statesList[streamIndex];
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                //
            }
        }

        private List<OnlineRecognizerResultEntity> DecodeMulti(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OnlineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 2)
                    {
                        break;
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
                OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
                onlineRecognizerResultEntity.text = text_result.Replace("▁", " ").ToLower();
                onlineRecognizerResultEntities.Add(onlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return onlineRecognizerResultEntities;
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
    }
}
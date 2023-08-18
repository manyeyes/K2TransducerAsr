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
    public class OnlineRecognizer
    {
        
        private readonly ILogger<OnlineRecognizer> _logger;
        private FrontendConfEntity _frontendConfEntity;
        private string[] _tokens;
        private OnlineModel _onlineModel;

        private delegate void ForwardBatch(List<OnlineStream> streams);
        private ForwardBatch _forwardBatch;

        public OnlineRecognizer(string encoderFilePath, string decoderFilePath, string joinerFilePath, string tokensFilePath,
            string decodingMethod = "greedy_search", int sampleRate = 16000, int featureDim = 80,
            int threadsNum = 2, bool debug = false, int maxActivePaths = 4, int enableEndpoint = 0)
        {
            _onlineModel = new OnlineModel(encoderFilePath, decoderFilePath, joinerFilePath, threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = sampleRate;
            _frontendConfEntity.n_mels = featureDim;
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
            OnlineStream onlineStream = new OnlineStream(_onlineModel.CustomMetadata, sampleRate: _frontendConfEntity.fs, featureDim: _frontendConfEntity.n_mels);
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
            //_batchSize = streams.Count;
            _forwardBatch.Invoke(streams);            
            onlineRecognizerResultEntities = this.DecodeMulti(streams);
            return onlineRecognizerResultEntities;
        }

        private EncoderOutputEntity EncoderProj(List<OnlineInputEntity> modelInputs, int batchSize, List<List<float[]>> statesList)
        {
            OnlineCustomMetadata onlineCustomMetadata = _onlineModel.CustomMetadata;
            float[] padSequence = PadSequence(modelInputs);
            var inputMeta = _onlineModel.EncoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / 80 / batchSize, 80 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

            }
            for (int i = 0; i < statesList.Count; i++)
            {
                List<float[]> items = statesList[i];
                string name = "";
                for (int m = 0; m < items.Count; m++)
                {
                    var state = items[m];

                    int[] dim = new int[1];
                    switch (i)
                    {
                        case 0:
                            name = "cached_len";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], batchSize };
                            name = name + "_" + m.ToString();
                            Int64[] state2 = state.Select(x => Convert.ToInt64(x.ToString())).ToArray();
                            var tensor_len = new DenseTensor<Int64>(state2, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor_len));
                            break;
                        case 1:
                            name = "cached_avg";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], batchSize, onlineCustomMetadata.Encoder_dims[m] };
                            name = name + "_" + m.ToString();
                            var tensor_avg = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_avg));
                            break;
                        case 2:
                            name = "cached_key";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], onlineCustomMetadata.Left_context_len[m], batchSize, onlineCustomMetadata.Attention_dims[m] };
                            name = name + "_" + m.ToString();
                            var tensor_key = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_key));
                            break;
                        case 3:
                            name = "cached_val";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], onlineCustomMetadata.Left_context_len[m], batchSize, onlineCustomMetadata.Attention_dims[m] / 2 };
                            name = name + "_" + m.ToString();
                            var tensor_val = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val));
                            break;
                        case 4:
                            name = "cached_val2";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m], onlineCustomMetadata.Left_context_len[m], batchSize, onlineCustomMetadata.Attention_dims[m] / 2 };
                            name = name + "_" + m.ToString();
                            var tensor_val2 = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val2));
                            break;
                        case 5:
                            name = "cached_conv1";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m],
                                batchSize, onlineCustomMetadata.Encoder_dims[m], onlineCustomMetadata.Cnn_module_kernels[m]-1};
                            name = name + "_" + m.ToString();
                            var tensor_conv1 = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv1));
                            break;
                        case 6:
                            name = "cached_conv2";
                            dim = new int[] { onlineCustomMetadata.Num_encoder_layers[m],
                                batchSize, onlineCustomMetadata.Encoder_dims[m], onlineCustomMetadata.Cnn_module_kernels[m]-1};
                            name = name + "_" + m.ToString();
                            var tensor_conv2 = new DenseTensor<float>(state, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv2));
                            break;
                    }
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _onlineModel.EncoderSession.Run(container);
                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    encoderOutput.encoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
                    List<float[]> states = new List<float[]>();
                    for (int i = 1; i < encoderResultsArray.Length; i++)
                    {
                        float[] item;
                        if (encoderResultsArray[i].ElementType.ToString() == typeof(System.Int64).Name)
                        {
                            item = encoderResultsArray[i].AsEnumerable<Int64>().Select(x => (float)Convert.ToDouble(x)).ToArray();
                        }
                        else
                        {
                            item = encoderResultsArray[i].AsEnumerable<float>().ToArray();
                        }
                        states.Add(item);
                    }
                    encoderOutput.encoder_out_states = states;
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
            int contextSize = _onlineModel.CustomMetadata.Context_size;
            DecoderOutputEntity decoderOutput = new DecoderOutputEntity();
            if (decoder_input == null)
            {
                Int64[] hyp = new Int64[] { -1, _onlineModel.Blank_id };
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
            int[] dim = new int[] { decoder_input.Length / contextSize, contextSize };
            var decoder_input_tensor = new DenseTensor<Int64>(decoder_input, dim, false);
            decoder_container.Add(NamedOnnxValue.CreateFromTensor<Int64>("y", decoder_input_tensor));
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
            decoderResults = _onlineModel.DecoderSession.Run(decoder_container);
            if (decoderResults != null)
            {
                var encoderResultsArray = decoderResults.ToArray();
                decoderOutput.decoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
            }
            return decoderOutput;
        }

        private JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out)
        {
            int joinerDim = _onlineModel.CustomMetadata.Joiner_dim;
            var inputMeta = _onlineModel.JoinerSession.InputMetadata;
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
            joinerResults = _onlineModel.JoinerSession.Run(container);
            JoinerOutputEntity joinerOutput = new JoinerOutputEntity();
            var joinerResultsArray = joinerResults.ToArray();
            joinerOutput.Logit = joinerResultsArray[0].AsEnumerable<float>().ToArray();
            joinerOutput.Logits = joinerResultsArray[0].AsTensor<float>();
            return joinerOutput;
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

                //method 1
                //if (_next_statesList != null)
                //{
                //    stackStatesList = stack_states_unittest(_next_statesList);
                //}
                //else
                //{
                //    stackStatesList = _onlineModel.stack_states(stateList);
                //}
                //method 2
                stackStatesList = _onlineModel.stack_states(stateList);

                EncoderOutputEntity encoderOutput = EncoderProj(modelInputs, batchSize, stackStatesList);
                int joinerDim = _onlineModel.CustomMetadata.Joiner_dim;
                int TT = encoderOutput.encoder_out.Length / joinerDim;
                DecoderOutputEntity decoderOutput = DecoderProj(hyps, batchSize);
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
                        decoder_out = DecoderProj(decoder_input, batchSize).decoder_out;
                    }

                }
                List<List<List<float[]>>> next_statesList = new List<List<List<float[]>>>();
                // method 1
                //next_statesList = _onlineModel.unstack_states(encoderOutput.encoder_out_states);
                //_next_statesList = unstack_states_unittest(encoderOutput.encoder_out_states);
                // method 2
                next_statesList = _onlineModel.unstack_states(encoderOutput.encoder_out_states);

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
        public List<List<float[]>> stack_states_unittest(List<List<List<float[]>>> stateList)
        {
            List<List<float[]>> states = new List<List<float[]>>();
            states = stateList[0];
            return states;
        }

        public List<List<List<float[]>>> unstack_states_unittest(List<float[]> stateList)
        {
            List<List<List<float[]>>> xxx = new List<List<List<float[]>>>();
            List<List<float[]>> yyy = new List<List<float[]>>();
            List<float[]> zzz = new List<float[]>();
            for (int i = 0; i < 35; i++)
            {
                zzz.Add(stateList[i]);
                if ((i + 1) % 5 == 0)
                {
                    yyy.Add(zzz);
                    zzz = new List<float[]>();
                }
            }
            xxx.Add(yyy);
            return xxx;
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
        private float[] PadSequence_unittest(List<OnlineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                float[]? curr_speech = modelInputs[i].Speech;
                Array.Copy(curr_speech, 0, speech, i* curr_speech.Length, curr_speech.Length);
            }
            speech = speech.Select(x => x == 0 ? -23.025850929940457F : x).ToArray();
            return speech;
        }

        private float[] PadSequence(List<OnlineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);
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
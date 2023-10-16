// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr.Model;
using K2TransducerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace K2TransducerAsr
{
    internal class OnlineProjOfZipformer2 : IOnlineProj
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _joinerSession;
        private OnlineCustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int sos_eos_id = 1;
        private int _unk_id = 2;

        private int _chunkLength = 39;
        private int _shiftLength = 32;
        public OnlineProjOfZipformer2(OnlineModel onlineModel)
        {
            _encoderSession = onlineModel.EncoderSession;
            _decoderSession = onlineModel.DecoderSession;
            _joinerSession = onlineModel.JoinerSession;

            var encoder_meta = _encoderSession.ModelMetadata.CustomMetadataMap;

            _customMetadata = new OnlineCustomMetadata();
            _customMetadata = onlineModel.CustomMetadata;
            _chunkLength = _customMetadata.TT;
            _shiftLength = _customMetadata.Decode_chunk_len;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession JoinerSession { get => _joinerSession; set => _joinerSession = value; }
        public OnlineCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => sos_eos_id; set => sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }

        public float[] GetEncoderInitProcessedLens(int batchSize = 1)
        {
            float[] processedLens = new float[batchSize];
            return processedLens;
        }
        public float[] GetEncoderInitEmbedStates(int batchSize = 1)
        {
            float[] embedStates = new float[batchSize * 128 * 3 * 19];
            return embedStates;
        }
        public List<List<float[]>> GetEncoderInitStates(int batchSize = 1)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            for (int i = 0; i < num_encoders; i++)
            {
                int num_encoder_layers = _customMetadata.Num_encoder_layers[i];
                int embed_dim = _customMetadata.Encoder_dims[i];
                int ds = 1;
                int num_heads = _customMetadata.Num_heads[i];
                int key_dim = _customMetadata.Query_head_dims[i] * num_heads;
                int value_dim = _customMetadata.Value_head_dims[i] * num_heads;
                int downsample_left = _customMetadata.Left_context_len[i] / ds;
                int nonlin_attn_head_dim = 3 * embed_dim / 4;
                int conv_left_pad = _customMetadata.Cnn_module_kernels[i] / 2;
                for (int layer = 0; layer < num_encoder_layers; layer++)
                {
                    //cached_key
                    int cached_key_size = downsample_left * batchSize * key_dim;
                    float[] cached_key_item = new float[cached_key_size];
                    //cached_nonlin_attn
                    int cached_nonlin_attn_size = 1 * batchSize * downsample_left * nonlin_attn_head_dim;
                    float[] cached_nonlin_attn_item = new float[cached_nonlin_attn_size];
                    //cached_val
                    int cached_val_size = downsample_left * batchSize * value_dim;
                    float[] cached_val_item = new float[cached_val_size];
                    //cached_val2
                    int cached_val2_size = downsample_left * batchSize * value_dim;
                    float[] cached_val2_item = new float[cached_val2_size];
                    //cached_conv1
                    int cached_conv1_size = batchSize * embed_dim * conv_left_pad;
                    float[] cached_conv1_item = new float[cached_conv1_size];
                    //cached_conv2
                    int cached_conv2_size = batchSize * embed_dim * conv_left_pad;
                    float[] cached_conv2_item = new float[cached_conv2_size];
                    List<float[]> states = new List<float[]>();
                    states = new List<float[]> { cached_key_item, cached_nonlin_attn_item, cached_val_item, cached_val2_item, cached_conv1_item, cached_conv2_item };
                    statesList.Add(states);
                }
            }
            float[] embed_states_item = GetEncoderInitEmbedStates(batchSize);
            List<float[]> embed_states_list = new List<float[]> { embed_states_item };
            statesList.Add(embed_states_list);
            float[] processed_lens_item = GetEncoderInitProcessedLens(batchSize);
            List<float[]> processed_lens_list = new List<float[]> { processed_lens_item };
            statesList.Add(processed_lens_list);
            return statesList;
        }

        public List<List<float[]>> stack_states(List<List<List<float[]>>> stateList)
        {
            List<List<float[]>> states = new List<List<float[]>>();
            states = stateList[0];
            return states;
        }

        public List<List<List<float[]>>> unstack_states(List<float[]> stateList)
        {
            List<List<List<float[]>>> xxx = new List<List<List<float[]>>>();
            List<List<float[]>> yyy = new List<List<float[]>>();
            List<float[]> zzz = new List<float[]>();
            for (int i = 0; i < 96; i++)
            {
                zzz.Add(stateList[i]);
                if ((i + 1) % 6 == 0)
                {
                    yyy.Add(zzz);
                    zzz = new List<float[]>();
                }
            }
            zzz = new List<float[]>();
            zzz.Add(stateList[96]);
            yyy.Add(zzz);
            zzz = new List<float[]>();
            zzz.Add(stateList[97]);
            yyy.Add(zzz);
            xxx.Add(yyy);
            return xxx;
        }

        public EncoderOutputEntity EncoderProj(List<OnlineInputEntity> modelInputs, int batchSize, List<List<float[]>> statesList)
        {
            OnlineCustomMetadata onlineCustomMetadata = _customMetadata;
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            try
            {
                
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
                int num_encoders = _customMetadata.Num_encoder_layers.Length;

                int y = 0;
                for (int i = 0; i < num_encoders; i++)
                {
                    int num_encoder_layers = _customMetadata.Num_encoder_layers[i];
                    int embed_dim = _customMetadata.Encoder_dims[i];
                    int ds = 1;//_customMetadata.Downsampling_factor[i];
                    int num_heads = _customMetadata.Num_heads[i];
                    int key_dim = _customMetadata.Query_head_dims[i] * num_heads;
                    int value_dim = _customMetadata.Value_head_dims[i] * num_heads;
                    int downsample_left = _customMetadata.Left_context_len[i] / ds;
                    int nonlin_attn_head_dim = 3 * embed_dim / 4;
                    int conv_left_pad = _customMetadata.Cnn_module_kernels[i] / 2;
                    int[] dim = new int[1];
                    string name = "";
                    for (int layer = 0; layer < num_encoder_layers; layer++)
                    {
                        List<float[]> items = statesList[y];
                        for (int m = 0; m < items.Count; m++)
                        {
                            var state = items[m];
                            switch (m)
                            {
                                case 0:
                                    name = "cached_key";
                                    dim = new int[] { downsample_left, batchSize, key_dim };
                                    name = name + "_" + y.ToString();
                                    //Int64[] state2 = state.Select(x => Convert.ToInt64(x.ToString())).ToArray();
                                    var tensor_len = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_len));
                                    break;
                                case 1:
                                    name = "cached_nonlin_attn";
                                    dim = new int[] { 1, batchSize, downsample_left, nonlin_attn_head_dim };
                                    name = name + "_" + y.ToString();
                                    var tensor_avg = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_avg));
                                    break;
                                case 2:
                                    name = "cached_val1";
                                    dim = new int[] { downsample_left, batchSize, value_dim };
                                    name = name + "_" + y.ToString();
                                    var tensor_val = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val));
                                    break;
                                case 3:
                                    name = "cached_val2";
                                    dim = new int[] { downsample_left, batchSize, value_dim };
                                    name = name + "_" + y.ToString();
                                    var tensor_val2 = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_val2));
                                    break;
                                case 4:
                                    name = "cached_conv1";
                                    dim = new int[] { batchSize, embed_dim, conv_left_pad };
                                    name = name + "_" + y.ToString();
                                    var tensor_conv1 = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv1));
                                    break;
                                case 5:
                                    name = "cached_conv2";
                                    dim = new int[] { batchSize, embed_dim, conv_left_pad };
                                    name = name + "_" + y.ToString();
                                    var tensor_conv2 = new DenseTensor<float>(state, dim, false);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor_conv2));
                                    break;
                            }
                        }

                        y++;
                    }
                }
                int[] embed_states_dim = new int[] { batchSize, 128, 3, 19 };
                float[] embed_states_value = statesList[16][0];
                var embed_states = new DenseTensor<float>(embed_states_value, embed_states_dim, false);
                container.Add(NamedOnnxValue.CreateFromTensor<float>("embed_states", embed_states));
                int[] processed_lens_dim = new int[] { batchSize };
                float[] processed_lens_value = statesList[17][0];
                Int64[] processed_lens_value2 = processed_lens_value.Select(x => Convert.ToInt64(x.ToString())).ToArray();
                var processed_lens = new DenseTensor<Int64>(processed_lens_value2, processed_lens_dim, false);
                container.Add(NamedOnnxValue.CreateFromTensor<Int64>("processed_lens", processed_lens));

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encoderSession.Run(container);

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
        public DecoderOutputEntity DecoderProj(Int64[]? decoder_input, int batchSize)
        {
            int contextSize = _customMetadata.Context_size;
            DecoderOutputEntity decoderOutput = new DecoderOutputEntity();
            if (decoder_input == null)
            {
                Int64[] hyp = new Int64[] { -1, _blank_id };
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
            decoderResults = _decoderSession.Run(decoder_container);
            if (decoderResults != null)
            {
                var encoderResultsArray = decoderResults.ToArray();
                decoderOutput.decoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
            }
            return decoderOutput;
        }

        public JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out)
        {
            int joinerDim = _customMetadata.Joiner_dim;
            var inputMeta = _joinerSession.InputMetadata;
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
            joinerResults = _joinerSession.Run(container);
            JoinerOutputEntity joinerOutput = new JoinerOutputEntity();
            var joinerResultsArray = joinerResults.ToArray();
            joinerOutput.Logit = joinerResultsArray[0].AsEnumerable<float>().ToArray();
            joinerOutput.Logits = joinerResultsArray[0].AsTensor<float>();
            return joinerOutput;
        }
    }
}

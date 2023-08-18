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
    public class OnlineModel
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
        public OnlineModel(string encoderFilePath, string decoderFilePath, string joinerFilePath, int threadsNum = 2)
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

            _chunkLength= _TT;
            _shiftLength= decode_chunk_len;

            _customMetadata.Num_encoder_layers = Array.ConvertAll(encoder_meta["num_encoder_layers"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            _customMetadata.Encoder_dims = Array.ConvertAll(encoder_meta["encoder_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            _customMetadata.Attention_dims = Array.ConvertAll(encoder_meta["attention_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            _customMetadata.Cnn_module_kernels = Array.ConvertAll(encoder_meta["cnn_module_kernels"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
            _customMetadata.Left_context_len = Array.ConvertAll(encoder_meta["left_context_len"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);

            int context_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["context_size"], out context_size);
            CustomMetadata.Context_size = context_size;
            int vocab_size;
            int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["vocab_size"], out vocab_size);
            CustomMetadata.Vocab_size = vocab_size;
            int joiner_dim;
            int.TryParse(_joinerSession.ModelMetadata.CustomMetadataMap["joiner_dim"], out joiner_dim);
            _customMetadata.Joiner_dim= joiner_dim;
            string version= _encoderSession.ModelMetadata.CustomMetadataMap["version"];
            _customMetadata.Version = version;
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

        public List<List<float[]>> stack_states(List<List<List<float[]>>> stateList)
        {

            int batch_size = stateList.Count;
            Debug.Assert(stateList[0].Count % 7 == 0, "when stack_states, state_list[0] is 7x");
            int num_encoders = stateList[0][0].Count;

            List<float[]> cached_len = new List<float[]>();
            List<float[]> cached_avg = new List<float[]>();
            List<float[]> cached_key = new List<float[]>();
            List<float[]> cached_val = new List<float[]>();
            List<float[]> cached_val2 = new List<float[]>();
            List<float[]> cached_conv1 = new List<float[]>();
            List<float[]> cached_conv2 = new List<float[]>();
            //cached_len
            List<List<float[]>> len_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                len_list.Add(stateList[n][0]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] len_avg = new float[len_list[0][i].Length * batch_size];
                int len_avg_item_length = len_list[0][i].Length;
                for (int x = 0; x < len_avg_item_length; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] len_avg_item = len_list[n][i];
                        Array.Copy(len_avg_item, x, len_avg, x * batch_size + n, 1);
                    }
                }
                cached_len.Add(len_avg);
            }
            //cached_avg
            List<List<float[]>> avg_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                avg_list.Add(stateList[n][1]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] avg = new float[avg_list[0][i].Length * batch_size];
                int avg_item_length = avg_list[0][i].Length;
                for (int x = 0; x < avg_item_length; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] avg_item = avg_list[n][i];
                        Array.Copy(avg_item, x, avg, x * batch_size + n, 1);
                    }
                }
                cached_avg.Add(avg);
            }

            //cached_key
            List<List<float[]>> key_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                key_list.Add(stateList[n][2]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] key = new float[key_list[0][i].Length * batch_size];
                int key_item_length = key_list[0][i].Length;
                int cached_key_axisnum = _customMetadata.Attention_dims[i];
                for (int x = 0; x < key_item_length / cached_key_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] key_item = key_list[n][i];
                        Array.Copy(key_item, x * cached_key_axisnum, key, (x * batch_size + n) * cached_key_axisnum, cached_key_axisnum);
                    }
                }
                cached_key.Add(key);
            }

            //cached_val
            List<List<float[]>> val_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                val_list.Add(stateList[n][3]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] val = new float[val_list[0][i].Length * batch_size];
                int val_item_length = val_list[0][i].Length;
                int cached_val_axisnum = _customMetadata.Attention_dims[i] / 2;
                for (int x = 0; x < val_item_length / cached_val_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] val_item = val_list[n][i];
                        Array.Copy(val_item, x * cached_val_axisnum, val, (x * batch_size + n) * cached_val_axisnum, cached_val_axisnum);
                    }
                }
                cached_val.Add(val);
            }

            //cached_val2
            List<List<float[]>> val2_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                val2_list.Add(stateList[n][4]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] val2 = new float[val2_list[0][i].Length * batch_size];
                int val2_item_length = val2_list[0][i].Length;
                int cached_val2_axisnum = _customMetadata.Attention_dims[i] / 2;
                for (int x = 0; x < val2_item_length / cached_val2_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] val2_item = val2_list[n][i];
                        Array.Copy(val2_item, x * cached_val2_axisnum, val2, (x * batch_size + n) * cached_val2_axisnum, cached_val2_axisnum);
                    }
                }
                cached_val2.Add(val2);
            }

            //cached_conv1
            List<List<float[]>> conv1_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                conv1_list.Add(stateList[n][5]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] conv1 = new float[conv1_list[0][i].Length * batch_size];
                int conv1_item_length = conv1_list[0][i].Length;
                int cached_conv1_axisnum = (_customMetadata.Cnn_module_kernels[i] - 1);
                for (int x = 0; x < conv1_item_length / cached_conv1_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] conv1_item = conv1_list[n][i];
                        Array.Copy(conv1_item, x * cached_conv1_axisnum, conv1, (x * batch_size + n) * cached_conv1_axisnum, cached_conv1_axisnum);
                    }
                }
                cached_conv1.Add(conv1);
            }

            //cached_conv2
            List<List<float[]>> conv2_list = new List<List<float[]>>();
            for (int n = 0; n < batch_size; n++)
            {
                conv2_list.Add(stateList[n][6]);
            }
            for (int i = 0; i < num_encoders; i++)
            {
                float[] conv2 = new float[conv2_list[0][i].Length * batch_size];
                int conv2_item_length = conv2_list[0][i].Length;
                int cached_conv2_axisnum = (_customMetadata.Cnn_module_kernels[i] - 1);
                for (int x = 0; x < conv2_item_length / cached_conv2_axisnum; x++)
                {
                    for (int n = 0; n < batch_size; n++)
                    {
                        float[] conv2_item = conv2_list[n][i];
                        Array.Copy(conv2_item, x * cached_conv2_axisnum, conv2, (x * batch_size + n) * cached_conv2_axisnum, cached_conv2_axisnum);
                    }
                }
                cached_conv2.Add(conv2);
            }

            List<List<float[]>> states = new List<List<float[]>> { cached_len, cached_avg, cached_key, cached_val, cached_val2, cached_conv1, cached_conv2 };

            return states;
        }
        public List<List<List<float[]>>> unstack_states(List<float[]> encoder_out_states)
        {
            List<List<List<float[]>>> statesList = new List<List<List<float[]>>>();
            Debug.Assert(encoder_out_states.Count % 7 == 0, "when stack_states, state_list[0] is 7x");
            int batch_size = encoder_out_states[0].Length / _customMetadata.Num_encoder_layers[0];
            int num_encoders = _customMetadata.Num_encoder_layers.Length;
            //int batch_size = states[0]
            for (int i = 0; i < batch_size; i++)
            {
                List<float[]> cached_len = new List<float[]>();
                List<float[]> cached_avg = new List<float[]>();
                List<float[]> cached_key = new List<float[]>();
                List<float[]> cached_val = new List<float[]>();
                List<float[]> cached_val2 = new List<float[]>();
                List<float[]> cached_conv1 = new List<float[]>();
                List<float[]> cached_conv2 = new List<float[]>();
                for (int j = 0; j < 7; j++)
                {
                    for (int m = 0; m < num_encoders; m++)
                    {
                        int num_encoder_layers = _customMetadata.Num_encoder_layers[m];
                        float[] item = encoder_out_states[j * num_encoders + m];
                        int n = 1;
                        switch (j)
                        {
                            case 0:
                                int cached_len_size = num_encoder_layers * n;
                                float[] cached_len_item = new float[cached_len_size];
                                for (int k = 0; k < cached_len_size; k++)
                                {
                                    Array.Copy(item, item.Length / cached_len_size * k + i, cached_len_item, k, 1);
                                }
                                cached_len.Add(cached_len_item);
                                break;
                            case 1:
                                int cached_avg_size = num_encoder_layers * n * _customMetadata.Encoder_dims[m];
                                float[] cached_avg_item = new float[cached_avg_size];
                                for (int k = 0; k < cached_avg_size; k++)
                                {
                                    Array.Copy(item, item.Length / cached_avg_size * k + i, cached_avg_item, k, 1);
                                }
                                cached_avg.Add(cached_avg_item);
                                break;
                            case 2:
                                int cached_key_axisnum = _customMetadata.Attention_dims[m];
                                int cached_key_size = num_encoder_layers * _customMetadata.Left_context_len[m] * n * _customMetadata.Attention_dims[m];
                                float[] cached_key_item = new float[cached_key_size];
                                for (int k = 0; k < cached_key_size / cached_key_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_key_size * k + i) * cached_key_axisnum, cached_key_item, k * cached_key_axisnum, cached_key_axisnum);
                                }
                                cached_key.Add(cached_key_item);
                                break;
                            case 3:
                                int cached_val_axisnum = _customMetadata.Attention_dims[m] / 2;
                                int cached_val_size = num_encoder_layers * _customMetadata.Left_context_len[m] * n * (_customMetadata.Attention_dims[m] / 2);
                                float[] cached_val_item = new float[cached_val_size];
                                for (int k = 0; k < cached_val_size / cached_val_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_val_size * k + i) * cached_val_axisnum, cached_val_item, k * cached_val_axisnum, cached_val_axisnum);
                                }
                                cached_val.Add(cached_val_item);
                                break;
                            case 4:
                                int cached_val2_axisnum = _customMetadata.Attention_dims[m] / 2;
                                int cached_val2_size = num_encoder_layers * _customMetadata.Left_context_len[m] * n * (_customMetadata.Attention_dims[m] / 2);
                                float[] cached_val2_item = new float[cached_val2_size];
                                for (int k = 0; k < cached_val2_size / cached_val2_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_val2_size * k + i) * cached_val2_axisnum, cached_val2_item, k * cached_val2_axisnum, cached_val2_axisnum);
                                }
                                cached_val2.Add(cached_val2_item);
                                break;
                            case 5:
                                int cached_conv1_axisnum = (_customMetadata.Cnn_module_kernels[m] - 1);
                                int cached_conv1_size = num_encoder_layers * n * _customMetadata.Encoder_dims[m] * (_customMetadata.Cnn_module_kernels[m] - 1);
                                float[] cached_conv1_item = new float[cached_conv1_size];
                                for (int k = 0; k < cached_conv1_size / cached_conv1_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_conv1_size * k + i) * cached_conv1_axisnum, cached_conv1_item, k * cached_conv1_axisnum, cached_conv1_axisnum);
                                }
                                cached_conv1.Add(cached_conv1_item);
                                break;
                            case 6:
                                int cached_conv2_axisnum = (_customMetadata.Cnn_module_kernels[m] - 1);
                                int cached_conv2_size = num_encoder_layers * n * _customMetadata.Encoder_dims[m] * (_customMetadata.Cnn_module_kernels[m] - 1);
                                float[] cached_conv2_item = new float[cached_conv2_size];
                                for (int k = 0; k < cached_conv2_size / cached_conv2_axisnum; k++)
                                {
                                    Array.Copy(item, (item.Length / cached_conv2_size * k + i) * cached_conv2_axisnum, cached_conv2_item, k * cached_conv2_axisnum, cached_conv2_axisnum);
                                }
                                cached_conv2.Add(cached_conv2_item);
                                break;

                        }
                    }
                }
                List<List<float[]>> states = new List<List<float[]>> { cached_len, cached_avg, cached_key, cached_val, cached_val2, cached_conv1, cached_conv2 };

                statesList.Add(states);

            }

            return statesList;
        }

    }
}

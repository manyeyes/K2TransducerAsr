// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace K2TransducerAsr.Model
{
    public class OnlineCustomMetadata
    {
        //encoder metadata
        private int[] _encoder_dims;
        private string _version;
        private string _model_type;
        private string _model_author;
        private int[] _attention_dims;
        private int _decode_chunk_len = 0;
        private int[] _num_encoder_layers;
        private int _TT = 0;
        private int[] _cnn_module_kernels;
        private int[] _left_context_len;
        //decoder custom metadata
        private int _context_size = 2;
        private int _vocab_size = 500;
        //joiner custom metadata
        private int _joiner_dim;

        public int[] Encoder_dims { get => _encoder_dims; set => _encoder_dims = value; }
        public string Version { get => _version; set => _version = value; }
        public string Model_type { get => _model_type; set => _model_type = value; }
        public string Model_author { get => _model_author; set => _model_author = value; }
        public int[] Attention_dims { get => _attention_dims; set => _attention_dims = value; }
        public int Decode_chunk_len { get => _decode_chunk_len; set => _decode_chunk_len = value; }
        public int[] Num_encoder_layers { get => _num_encoder_layers; set => _num_encoder_layers = value; }
        public int TT { get => _TT; set => _TT = value; }
        public int[] Cnn_module_kernels { get => _cnn_module_kernels; set => _cnn_module_kernels = value; }
        public int[] Left_context_len { get => _left_context_len; set => _left_context_len = value; }
        public int Context_size { get => _context_size; set => _context_size = value; }
        public int Vocab_size { get => _vocab_size; set => _vocab_size = value; }
        public int Joiner_dim { get => _joiner_dim; set => _joiner_dim = value; }
    }
}

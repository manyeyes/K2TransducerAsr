// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;

namespace K2TransducerAsr.Utils
{
    internal static class HotwordsHelper
    {
        public static List<Int64>[] NbestHotwords(List<Int64>[] token_nums, List<List<int[]>> token_nums_hotwords)
        {
            int[] hotwords1 = new int[] { 898, 2141, 401 };
            int[] hotwords2 = new int[] { 815, 2141, 3308 };
            List<int[]> hotwordsList = new List<int[]>();
            hotwordsList.Add(hotwords1);
            hotwordsList.Add(hotwords2);
            foreach (int[] hotwords in hotwordsList)
            {
                for (int i = 0; i < token_nums_hotwords.Count; i++)
                {
                    List<int[]> item_hotwords = token_nums_hotwords[i];
                    int position_0 = i;
                    int[] position_1;
                    int hotword_p = 0;
                    List<int> position = new List<int>();
                    List<int> position_words = new List<int>();
                    for (int j = 0; j < item_hotwords.Count; j++)
                    {
                        int[] item = item_hotwords[j];

                        if (hotword_p < hotwords.Length)
                        {
                            if (item.Contains(hotwords[hotword_p]))
                            {
                                position.Add(j);
                                position_words.Add(hotwords[hotword_p]);
                            }
                            else
                            {
                                position = new List<int>();
                                hotword_p = 0;
                                continue;
                            }
                            hotword_p++;
                        }
                        else
                        {
                            if (position.Count > 0)
                            {
                                position_1 = position.ToArray();
                                for (int x = 0; x < position_1.Length; x++)
                                {
                                    token_nums[position_0][position_1[x]] = position_words[x];
                                }
                            }
                            position = new List<int>();
                            hotword_p = 0;
                        }

                    }

                }
            }

            return token_nums;
        }

        public static List<int[]> NbestHotwords2(Tensor<float> logits)
        {
            int[] hotwords1 = new int[] { 815, 2141, 401 };
            int[] hotwords2 = new int[] { 815, 2141, 3308 };
            List<int[]> hotwordsList = new List<int[]>();
            hotwordsList.Add(hotwords1);
            hotwordsList.Add(hotwords2);

            List<int[]> token_nums = new List<int[]> { };
            List<List<int[]>> token_nums_hotwords = new List<List<int[]>>();
            int itemLength = logits.Dimensions[0];
            for (int i = 0; i < 1; i++)
            {
                int[] item = new int[itemLength];
                List<int[]> item_hotwords = new List<int[]>();
                for (int j = i * itemLength; j < (i + 1) * itemLength; j++)
                {
                    int token_num = 0;
                    LinkedList<int> token_num_list = new LinkedList<int>();
                    for (int k = 1; k < logits.Dimensions[1]; k++)
                    {
                        token_num = logits[j, token_num] > logits[j, k] ? token_num : k;
                        token_num_list.AddFirst(token_num);
                        if (token_num_list.Count > 10)
                        {
                            token_num_list.RemoveLast();
                        }
                    }
                    item[j - i * itemLength] = (int)token_num;
                    item_hotwords.Add(token_num_list.ToArray());

                }
                token_nums.Add(item);
                token_nums_hotwords.Add(item_hotwords);
            }
            //token_nums[0][0] = 8;
            for (int i = 0; i < token_nums_hotwords.Count; i++)
            {
                List<int[]> item_hotwords = token_nums_hotwords[i];
                int position_0 = i;
                int[] position_1;
                int hotword_p = 0;
                List<int> position = new List<int>();
                List<int> position_words = new List<int>();
                for (int j = 0; j < item_hotwords.Count; j++)
                {
                    int[] item = item_hotwords[j];
                    if (hotword_p < hotwords1.Length)
                    {
                        if (item.Contains(hotwords1[hotword_p]))
                        {
                            position.Add(j);
                            position_words.Add(hotwords1[hotword_p]);
                        }
                        else
                        {
                            position = new List<int>();
                            hotword_p = 0;
                        }
                        hotword_p++;
                    }
                    else
                    {
                        position = new List<int>();
                        hotword_p = 0;
                    }
                }
                if (position.Count > 0)
                {
                    position_1 = position.ToArray();
                    for (int x = 0; x < position_1.Length; x++)
                    {
                        token_nums[position_0][position_1[x]] = position_words[x];
                    }
                }
            }

            return token_nums;
        }

        //private int[]
    }
}
// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

namespace K2TransducerAsr.Utils
{
    internal static class HotwordsHelper
    {
        public static List<Int64>[] NbestHotwords(List<Int64>[] token_nums, List<List<int[]>> token_nums_hotwords, List<int[]> hotwordsList)
        {
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
    }
}
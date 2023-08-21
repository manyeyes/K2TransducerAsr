// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr;
using K2TransducerAsr.Model;
using NAudio.Wave;
using System.Diagnostics;
using System.IO;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        Console.WriteLine("OfflineRecognizer:");
        Console.WriteLine("batchSize:1");
        OfflineRecognizer("one");
        Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));

        Console.WriteLine("OfflineRecognizer:");
        Console.WriteLine("batchSize:>1");
        OfflineRecognizer("multi");
        Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));

        Console.WriteLine("OnlineRecognizer:");
        Console.WriteLine("batchSize:1");
        OnlineRecognizer("one");
        Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));

        Console.WriteLine("OnlineRecognizer:");
        Console.WriteLine("batchSize:>1");
        OnlineRecognizer("multi");
        Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));
    }

    private static void OfflineRecognizer(string streamDecodeMethod)
    {
        string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        string modelName = "sherpa-onnx-zipformer-small-en-2023-06-26";
        string encoderFilePath = applicationBase + "./" + modelName + "/encoder-epoch-99-avg-1.onnx";
        string decoderFilePath = applicationBase + "./" + modelName + "/decoder-epoch-99-avg-1.onnx";
        string joinerFilePath = applicationBase + "./" + modelName + "/joiner-epoch-99-avg-1.onnx";
        string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
        K2TransducerAsr.OfflineRecognizer offlineRecognizer = new K2TransducerAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);        
        List<OfflineStream> streams = new List<OfflineStream>();
        TimeSpan total_duration = new TimeSpan(0L);
        List<float[]>? samples = new List<float[]>();
        for (int i = 0; i < 2; i++)
        {
            string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", i.ToString());
            if (!File.Exists(wavFilePath))
            {
                break;
            }
            TimeSpan duration = TimeSpan.Zero;
            float[] sample = AudioHelper.GetFileSample(wavFilePath, ref duration);
            samples.Add(sample);
            total_duration += duration;
        }
        TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
        streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "multi" : streamDecodeMethod;//one ,multi
        if (streamDecodeMethod == "one")
        {
            // Non batch method
            foreach (var sample in samples)
            {
                OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                stream.AddSamples(sample);
                OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                Console.WriteLine(result.text);
                Console.WriteLine("");
            }
            // Non batch method
        }
        if (streamDecodeMethod == "multi")
        {
            //2. batch method
            foreach (var sample in samples)
            {
                OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                stream.AddSamples(sample);
                streams.Add(stream);
            }
            List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
            foreach (OfflineRecognizerResultEntity result in results)
            {
                Console.WriteLine(result.text);
                Console.WriteLine("");
            }
            // batch method
        }

        TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
        double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
        double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
        Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
        Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
        Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
        Console.WriteLine("end!");
    }

    private static void OnlineRecognizer(string streamDecodeMethod)
    {
        string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        string modelName = "sherpa-onnx-streaming-zipformer-en-2023-02-21";
        string encoderFilePath = applicationBase + "./" + modelName + "/encoder-epoch-99-avg-1.onnx";
        string decoderFilePath = applicationBase + "./" + modelName + "/decoder-epoch-99-avg-1.onnx";
        string joinerFilePath = applicationBase + "./" + modelName + "/joiner-epoch-99-avg-1.onnx";
        string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
        K2TransducerAsr.OnlineRecognizer onlineRecognizer = new K2TransducerAsr.OnlineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
        TimeSpan total_duration = TimeSpan.Zero;
        TimeSpan start_time = TimeSpan.Zero;
        TimeSpan end_time = TimeSpan.Zero;
        start_time = new TimeSpan(DateTime.Now.Ticks);
        List<List<float[]>> samplesList = new List<List<float[]>>();
        List<float[]> samples = new List<float[]>();
        int batchSize = 3; //There are 3 audio files in the example, so the maximum value is 3 
        for (int n = 0; n < batchSize; n++)
        {
            string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", n.ToString());
            if (!File.Exists(wavFilePath))
            {
                break;
            }
            TimeSpan duration = TimeSpan.Zero;
            samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration);
            samplesList.Add(samples);
            total_duration += duration;
        }
        streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "multi" : streamDecodeMethod;//one ,multi
        if (streamDecodeMethod == "one")
        {
            //one stream decode
            for (int j = 0; j < samplesList.Count; j++)
            {
                K2TransducerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                foreach (float[] samplesItem in samplesList[j])
                {
                    stream.AddSamples(samplesItem);
                    int w = (samplesItem.Length+6000)/(4000);
                    while (w>1)
                    {
                        OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
                        Console.WriteLine(result_on.text);
                        w--;
                    }
                }
            }
            // one stream decode
        }
        if (streamDecodeMethod == "multi")
        {
            // multi streams decode
            K2TransducerAsr.OnlineStream stream0 = onlineRecognizer.CreateOnlineStream();
            K2TransducerAsr.OnlineStream stream1 = onlineRecognizer.CreateOnlineStream();
            K2TransducerAsr.OnlineStream stream2 = onlineRecognizer.CreateOnlineStream();
            bool end0 = false;
            bool end1 = false;
            bool end2 = false;
            bool stopDecode = false;
            int i = 0;
            List<K2TransducerAsr.OnlineStream> streams = new List<OnlineStream>();
            while (true)
            {
                streams = new List<OnlineStream>();

                for (int j = 0; j < batchSize; j++)
                {
                    if (j == 0)
                    {
                        if (samplesList[0].Count > i)
                        {
                            stream0.AddSamples(samplesList[0][i]);
                            streams.Add(stream0);
                        }
                        else
                        {
                            if (!end0)
                            {
                                stream0.AddSamples(new float[6000]);
                                streams.Add(stream0);
                            }
                            end0 = true;
                        }
                    }
                    j++;
                    if (j == 1)
                    {
                        if (samplesList[1].Count > i)
                        {
                            stream1.AddSamples(samplesList[1][i]);
                            streams.Add(stream1);
                        }
                        else
                        {
                            if (!end1)
                            {
                                stream1.AddSamples(new float[6000]);
                                streams.Add(stream1);
                            }
                            end1 = true;
                        }
                    }
                    j++;
                    if (j == 2)
                    {
                        if (samplesList[2].Count > i)
                        {
                            stream2.AddSamples(samplesList[2][i]);
                            streams.Add(stream2);
                        }
                        else
                        {
                            if (!end2)
                            {
                                stream2.AddSamples(new float[6000]);
                                streams.Add(stream2);
                            }
                            end2 = true;
                        }
                    }
                }
                if (streams.Count == 0)
                {
                    streams.Add(stream0);
                    streams.Add(stream1);
                    streams.Add(stream2);
                }
                List<OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
                foreach (OnlineRecognizerResultEntity result in results_batch)
                {
                    Console.WriteLine(result.text);
                    Console.WriteLine("");
                }
                i++;
                if (i > 52)
                {
                    stopDecode = true;
                }
                if (end0 && end1 && end2 && stopDecode)
                {
                    break;
                }
            }
            // multi streams decode
        }


        end_time = new TimeSpan(DateTime.Now.Ticks);
        double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
        double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
        Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
        Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
        Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
        Console.WriteLine("end!");
    }
}
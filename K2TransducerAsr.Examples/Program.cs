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
        //Console.WriteLine("OfflineRecognizer:");
        //Console.WriteLine("batchSize:1");
        //OfflineRecognizer("one");
        //Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));

        Console.WriteLine("OfflineRecognizer:");
        Console.WriteLine("batchSize:>1");
        OfflineRecognizer("multi");
        Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));

        Console.WriteLine("OnlineRecognizer:");
        Console.WriteLine("batchSize:1");
        OnlineRecognizer("one");
        Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));

        //Console.WriteLine("OnlineRecognizer:");
        //Console.WriteLine("batchSize:>1");
        //OnlineRecognizer("multi");
        //Console.WriteLine(string.Concat(Enumerable.Repeat("-", 66)));
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
            samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration, chunkSize:800);
            for (int j = 0; j < 30; j++)
            {
                samples.Add(new float[400]);
            }
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
                    OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
                    Console.WriteLine(result_on.text);
                }
            }
            // one stream decode
        }
        if (streamDecodeMethod == "multi")
        {
            //multi stream decode
            List<K2TransducerAsr.OnlineStream> onlineStreams = new List<K2TransducerAsr.OnlineStream>();
            List<bool> isEndpoints = new List<bool>();
            List<bool> isEnds = new List<bool>();
            for (int num = 0; num < samplesList.Count; num++)
            {
                K2TransducerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                onlineStreams.Add(stream);
                isEndpoints.Add(false);
                isEnds.Add(false);
            }
            int i = 0;
            List<K2TransducerAsr.OnlineStream> streams = new List<K2TransducerAsr.OnlineStream>();

            while (true)
            {
                streams = new List<K2TransducerAsr.OnlineStream>();

                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (samplesList[j].Count > i && samplesList.Count > j)
                    {
                        onlineStreams[j].AddSamples(samplesList[j][i]);
                        streams.Add(onlineStreams[j]);
                        isEndpoints[0] = false;
                    }
                    else
                    {
                        streams.Add(onlineStreams[j]);
                        samplesList.Remove(samplesList[j]);
                        isEndpoints[0] = true;
                    }
                }          
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (isEndpoints[j])
                    {
                        if (onlineStreams[j].IsFinished(isEndpoints[j]))
                        {
                            isEnds[j] = true;
                        }
                        else
                        {
                            streams.Add(onlineStreams[j]);
                        }
                    }
                }
                List<K2TransducerAsr.OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
                foreach (K2TransducerAsr.OnlineRecognizerResultEntity result in results_batch)
                {
                    Console.WriteLine(result.text);
                    //Console.WriteLine("");
                }
                Console.WriteLine("");
                i++;
                bool isEnd = true;
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (!isEnds[j])
                    {
                        isEnd = false;
                        break;
                    }
                }
                if (isEnd)
                {
                    break;
                }
            }
            //multi stream decode
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
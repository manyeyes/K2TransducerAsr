// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using K2TransducerAsr;
using K2TransducerAsr.Model;
using NAudio.Wave;
using System.IO;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        string modelName = "sherpa-onnx-zipformer-small-en-2023-06-26";
        string encoderFilePath = applicationBase + "./" + modelName + "/encoder-epoch-99-avg-1.onnx";
        string decoderFilePath = applicationBase + "./" + modelName + "/decoder-epoch-99-avg-1.onnx";
        string joinerFilePath = applicationBase + "./" + modelName + "/joiner-epoch-99-avg-1.onnx";
        string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
        K2TransducerAsr.OfflineRecognizer offlineRecognizer = new K2TransducerAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath,threadsNum:2);
        List<float[]>? samples = null;
        List<OfflineStream> streams = new List<OfflineStream>();
        TimeSpan total_duration = new TimeSpan(0L);
        if (samples == null)
        {
            samples = new List<float[]>();
            for (int i = 0; i < 2; i++)
            {
                string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", i.ToString());
                if (!File.Exists(wavFilePath))
                {
                    break;
                }
                AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
                byte[] datas = new byte[_audioFileReader.Length];
                _audioFileReader.Read(datas, 0, datas.Length);
                TimeSpan duration = _audioFileReader.TotalTime;
                float[] sample = new float[datas.Length / sizeof(float)];
                Buffer.BlockCopy(datas, 0, sample, 0, datas.Length);
                samples.Add(sample);
                total_duration += duration;
            }
        }
        TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
        //1.Non batch method
        //foreach (var sample in samples)
        //{
        //    //List<float[]> temp_samples = new List<float[]>();
        //    //temp_samples.Add(sample);
        //    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
        //    stream.AddSamples(sample);
        //    OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
        //    Console.WriteLine(result.text);
        //    Console.WriteLine("");
        //}
        //2.batch method
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
        TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
        double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
        double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
        Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
        Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
        Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
        Console.WriteLine("end!");
    }
}
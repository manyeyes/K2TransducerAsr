using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class AudioHelper
{
    public static float[] GetFileSample(string wavFilePath, ref TimeSpan duration)
    {
        if (!File.Exists(wavFilePath))
        {
            return new float[1];
        }
        AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
        byte[] datas = new byte[_audioFileReader.Length];
        _audioFileReader.Read(datas, 0, datas.Length);
        duration = _audioFileReader.TotalTime;
        float[] sample = new float[datas.Length / sizeof(float)];
        Buffer.BlockCopy(datas, 0, sample, 0, datas.Length);
        return sample;
    }
    public static List<float[]> GetFileChunkSamples(string wavFilePath, ref TimeSpan duration)
    {
        List<float[]> wavdatas = new List<float[]>();
        if (!File.Exists(wavFilePath))
        {
            Trace.Assert(File.Exists(wavFilePath), "file does not exist:" + wavFilePath);
            wavdatas.Add(new float[1]);
            return wavdatas;
        }
        AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
        byte[] datas = new byte[_audioFileReader.Length];
        _audioFileReader.Read(datas);
        duration = _audioFileReader.TotalTime;

        float[] wavsdata = new float[datas.Length / sizeof(float)];
        int wavsLength = wavsdata.Length;
        Buffer.BlockCopy(datas, 0, wavsdata, 0, datas.Length);

        int chunkSize = 6000 * 20;
        int chunkNum = (int)Math.Ceiling((double)wavsLength / chunkSize);
        for (int i = 0; i < chunkNum; i++)
        {
            int offset;
            int dataCount;
            if (Math.Abs(wavsLength - i * chunkSize) > chunkSize)
            {
                offset = i * chunkSize;
                dataCount = chunkSize;
            }
            else
            {
                offset = i * chunkSize;
                dataCount = wavsLength - i * chunkSize;
            }
            float[] wavdata = new float[dataCount];
            Array.Copy(wavsdata, offset, wavdata, 0, dataCount);
            wavdatas.Add(wavdata);

        }
        return wavdatas;
    }
}

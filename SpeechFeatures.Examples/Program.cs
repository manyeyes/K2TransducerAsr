// See https://aka.ms/new-console-template for more information
using SpeechFeatures;
namespace SpeechFeatures.Examples;
internal class Program
{
    static void Main(string[]? args = null)
    {
        test_MelComputations(args);
        test_OnlineFeature(args);
        test_Rfft(args);
        test_KaldiFeature(args);
    }

    static void test_MelComputations(string[]? args = null)
    {
        // Example usage
        MelBanksOptions opts = new MelBanksOptions
        {
            numBins = 25,
            lowFreq = 20,
            highFreq = 0,
            vtlnLow = 100,
            vtlnHigh = -500,
            debugMel = false,
            htkMode = false
        };
        FrameExtractionOptions frameOpts = new FrameExtractionOptions
        {
            SampFreq = 16000
        };
        float vtlnWarpFactor = 1.0f;

        MelBanks melBanks = new MelBanks(opts, frameOpts, vtlnWarpFactor);

        float[] powerSpectrum = new float[400];
        float[] melEnergiesOut = new float[opts.numBins];

        melBanks.Compute(powerSpectrum, ref melEnergiesOut);

        Console.WriteLine($"Number of bins: {melBanks.NumBins()}");
    }

    static void test_OnlineFeature(string[]? args = null)
    {
        // Example usage
        FrameExtractionOptions opts = new FrameExtractionOptions
        {
            SampFreq = 16000,
            FrameShiftMs = 10,
            MaxFeatureVectors = 1000
        };
        FbankOptions fbankOptions = new FbankOptions();
        fbankOptions.FrameOpts = opts;
        // Assuming FbankComputer implements IFeatureComputer
        var onlineFbank = new OnlineGenericBaseFeature<FbankComputer>(new FbankComputer(fbankOptions));

        float[] waveform = new float[16000*30];
        onlineFbank.AcceptWaveform(16000, waveform, 400);

        Console.WriteLine($"Number of frames ready: {onlineFbank.NumFramesReady()}");
    }

    static void test_Rfft(string[]? args = null)
    {
        Rfft rfft = new Rfft(8);
        float[] inOutFloat = new float[8] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        List<float> inOutFloatList=new List<float>(inOutFloat);
        rfft.Compute(inOutFloatList);

        double[] inOutDouble = new double[8] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        rfft.Compute(ref inOutDouble);

        Console.WriteLine("Finished.");
    }

    static void test_KaldiFeature(string[]? args)
    {
        // 示例用法
        FbankOptions opts = KaldiFeature.GetFbankOptions(0, true, 16000, 80);
        OnlineFeature onlineFbank = KaldiFeature.GetOnlineFeature(opts);

        float[] samples = new float[400];
        KaldiFeature.AcceptWaveform(onlineFbank, 16000, samples, 400);

        KaldiFeature.InputFinished(onlineFbank);

        int numFramesReady = KaldiFeature.GetNumFramesReady(onlineFbank);

        FbankData fbankData = new FbankData();
        KaldiFeature.GetFbank(onlineFbank, 0, ref fbankData);

        FbankDatas fbankDatas = new FbankDatas();
        KaldiFeature.GetFbanks(onlineFbank, 0, ref fbankDatas);

        //List<float> frames = KaldiFeature.GetFrames(onlineFbank, 0);

        Console.WriteLine($"Number of frames ready: {numFramesReady}");
    }
}
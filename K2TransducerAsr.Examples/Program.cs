// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
namespace K2TransducerAsr.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        [STAThread]
        private static void Main()
        {
            string lang = "en";
            PrintUsage(lang);
            // The complete model path, eg: path/to/directory/modelname
            string modelBasePath = @"";// eg: path/to/directory. It is the root directory where the model is stored. If it is empty, the program root directory will be read by default.
            string defaultOfflineModelName = "k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716";
            OfflineRecognizer(streamDecodeMethod: "one", modelName: defaultOfflineModelName,modelBasePath: modelBasePath);
            string defaultOnlineModelName = "k2transducer-zipformer-ctc-small-zh-onnx-online-20250401";
            OnlineRecognizer(streamDecodeMethod: "one", modelName: defaultOnlineModelName, modelBasePath: modelBasePath);
            //suggest GC recycling (non mandatory)
            GC.Collect(); //trigger recycling
            GC.WaitForPendingFinalizers(); //waiting for the terminator to complete execution
            GC.Collect(); //recycling again
        }
        /// <summary>
        /// PrintUsage
        /// </summary>
        /// <param name="lang">en/zh</param>
        private static void PrintUsage(string lang="en")
        {
            if (lang == "en")
            {
                Console.WriteLine("\nUsage Instructions: ");
                Console.WriteLine("1. Refer to the model list in the README document and download the corresponding models to your local directory path/to/directory.");
                Console.WriteLine("2. Set the value of modelBasePath to the path of the local directory where the models are located (path/to/directory).");
                Console.WriteLine("3. Set the values of defaultOfflineModelName and defaultOnlineModelName to the model directory names (modelName), noting the distinction between model types: online/offline.\n");
            }
            else
            {
                Console.WriteLine("\n使用说明: ");
                Console.WriteLine("1.参考README文档中的模型列表，下载相应模型到本地目录 path/to/directory 。");
                Console.WriteLine("2.设置 modelBasePath 的值为模型所在的本地目录路径 path/to/directory。");
                Console.WriteLine("3.设置 defaultOfflineModelName 和 defaultOnlineModelName 的值为模型目录名（modelName），注意区分模型类型：online/offline.\n");
            }
        }
    }
}
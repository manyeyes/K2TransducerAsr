﻿namespace K2TransducerAsr.Examples
{
    internal static partial class Program
    {
        private static K2TransducerAsr.OfflineRecognizer? _offlineRecognizer;
        private static OfflineRecognizer? InitOfflineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_offlineRecognizer == null)
            {
                if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string encoderFilePath = modelBasePath + "./" + modelName + "/model.int8.onnx";
                string decoderFilePath = "";
                string joinerFilePath = "";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
                try
                {
                    string folderPath = Path.Combine(modelBasePath, modelName);
                    // 1. Check if the folder exists
                    if (!Directory.Exists(folderPath))
                    {
                        Console.WriteLine($"Error: folder does not exist - {folderPath}");
                        return null;
                    }
                    // 2. Get the complete paths of all files in the folder
                    // Optional parameters: search mode (such as "*. txt" filtering text files), whether to search subdirectories
                    string[] allFilePaths = Directory.GetFiles(folderPath);
                    foreach (string filePath in allFilePaths)
                    {
                        // Extract pure file name (including extension)
                        string fileName = Path.GetFileName(filePath);
                        //Console.WriteLine(fileName);
                        if (fileName.StartsWith("model") || fileName.StartsWith("encoder"))
                        {                            
                            if (fileName.Contains("." + modelAccuracy + "."))
                            {
                                encoderFilePath = modelBasePath + "./" + modelName + "/" + fileName;
                            }
                            else
                            {
                                if (string.IsNullOrEmpty(encoderFilePath))
                                {
                                    encoderFilePath = modelBasePath + "./" + modelName + "/" + fileName;
                                }
                            }
                        }
                        if (fileName.StartsWith("decoder"))
                        {
                            if (fileName.Contains("." + modelAccuracy + "."))
                            {
                                decoderFilePath = modelBasePath + "./" + modelName + "/" + fileName;
                            }
                            else
                            {
                                if (string.IsNullOrEmpty(decoderFilePath))
                                {
                                    decoderFilePath = modelBasePath + "./" + modelName + "/" + fileName;
                                }
                            }
                        }
                        if (fileName.StartsWith("joiner"))
                        {
                            if (fileName.Contains("." + modelAccuracy + "."))
                            {
                                joinerFilePath = modelBasePath + "./" + modelName + "/" + fileName;
                            }
                            else
                            {
                                if (string.IsNullOrEmpty(joinerFilePath))
                                {
                                    joinerFilePath = modelBasePath + "./" + modelName + "/" + fileName;
                                }
                            }
                        }
                        if (fileName.StartsWith("tokens"))
                        {
                            tokensFilePath = modelBasePath + "./" + modelName + "/" + fileName;
                        }
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _offlineRecognizer = new K2TransducerAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: threadsNum);
                    TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                    Console.WriteLine("init_models_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
                }
                catch (UnauthorizedAccessException)
                {
                    Console.WriteLine($"Error: No permission to access this folder");
                }
                catch (PathTooLongException)
                {
                    Console.WriteLine($"Error: File path too long");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error occurred: {ex.Message}");
                }
            }
            return _offlineRecognizer;
        }

        public static void OfflineRecognizer(string streamDecodeMethod, string modelName = "k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716", string modelAccuracy = "int8", int threadsNum = 2, string[]? mediaFilePaths = null,string? modelBasePath=null)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            K2TransducerAsr.OfflineRecognizer? offlineRecognizer = InitOfflineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (offlineRecognizer == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            List<OfflineStream> streams = new List<OfflineStream>();
            Console.WriteLine("Read meida Files in progress!");
            TimeSpan total_duration = new TimeSpan(0L);
            List<float[]>? samples = new List<float[]>();
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                mediaFilePaths = Directory.GetFiles(Path.Combine(modelBasePath, modelName, "test_wavs"));
            }
            foreach (string mediaFilePath in mediaFilePaths)
            {
                if (!File.Exists(mediaFilePath))
                {
                    continue;
                }
                if (Utils.AudioHelper.IsAudioByHeader(mediaFilePath))
                {
                    TimeSpan duration = TimeSpan.Zero;
                    float[] sample = Utils.AudioHelper.GetFileSample(mediaFilePath, duration: ref duration);
                    samples.Add(sample);
                    total_duration += duration;
                }
            }
            if (samples.Count == 0)
            {
                Console.WriteLine("No media file is read!");
                return;
            }
            Console.WriteLine("Automatic speech recognition in progress!");
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "multi" : streamDecodeMethod;//one ,multi
            if (streamDecodeMethod == "one")
            {
                // Non batch method
                Console.WriteLine("one stream decode results:\r\n");
                foreach (var sample in samples)
                {
                    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                    stream.AddSamples(sample);
                    K2TransducerAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                    Console.WriteLine(result.text);
                    Console.WriteLine("");
                }
                // Non batch method
            }
            if (streamDecodeMethod == "multi")
            {
                //2. batch method
                Console.WriteLine("multi stream decode results:\r\n");
                foreach (var sample in samples)
                {
                    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                    stream.AddSamples(sample);
                    streams.Add(stream);
                }
                Console.WriteLine("Recognition results:\r\n");
                List<K2TransducerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
                foreach (K2TransducerAsr.Model.OfflineRecognizerResultEntity result in results)
                {
                    Console.WriteLine(result.text);
                    Console.WriteLine("");
                }
                // batch method
            }
            if (_offlineRecognizer != null)
            {
                _offlineRecognizer.Dispose();
                _offlineRecognizer = null;                
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("recognition_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration_milliseconds:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("end!");
        }
    }
}

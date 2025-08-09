namespace K2TransducerAsr.Examples
{
    internal static partial class Program
    {
        private static K2TransducerAsr.OnlineRecognizer? _onlineRecognizer;
        private static OnlineRecognizer? InitOnlineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_onlineRecognizer == null)
            {
                if (string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string encoderFilePath = modelBasePath + "./" + modelName + "/model.int8.onnx";
                string decoderFilePath = "";
                string joinerFilePath = "";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
                try
                {
                    string folderPath = Path.Join(modelBasePath, modelName);
                    // 1. 检查文件夹是否存在
                    if (!Directory.Exists(folderPath))
                    {
                        Console.WriteLine($"错误：文件夹不存在 - {folderPath}");
                        return null;
                    }
                    // 2. 获取所有文件的文件名和目标路径（提前计算路径，避免重复拼接）
                    var fileInfos = Directory.GetFiles(folderPath)
                        .Select(filePath => new
                        {
                            FileName = Path.GetFileName(filePath),
                            // 推荐使用Path.Combine处理路径（自动适配系统分隔符）
                            TargetPath = Path.Combine(modelBasePath, modelName, Path.GetFileName(filePath))
                            // 若需严格保持原拼接方式，可替换为：
                            // TargetPath = $"{modelBasePath}/./{modelName}/{Path.GetFileName(filePath)}"
                        })
                        .ToList();

                    // 处理encoder路径（优先级：包含modelAccuracy的 > 最后一个符合前缀的）
                    var encoderCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("model") || f.FileName.StartsWith("encoder"))
                        .ToList();
                    if (encoderCandidates.Any())
                    {
                        // 优先选择包含指定modelAccuracy的文件
                        var preferredEncoder = encoderCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        encoderFilePath = preferredEncoder?.TargetPath ?? encoderCandidates.Last().TargetPath;
                    }

                    // 处理decoder路径
                    var decoderCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("decoder"))
                        .ToList();
                    if (decoderCandidates.Any())
                    {
                        var preferredDecoder = decoderCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        decoderFilePath = preferredDecoder?.TargetPath ?? decoderCandidates.Last().TargetPath;
                    }

                    // 处理joiner路径
                    var joinerCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("joiner"))
                        .ToList();
                    if (joinerCandidates.Any())
                    {
                        var preferredJoiner = joinerCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        joinerFilePath = preferredJoiner?.TargetPath ?? joinerCandidates.Last().TargetPath;
                    }

                    // 处理tokens路径（取最后一个符合前缀的）
                    tokensFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("tokens"))
                        ?.TargetPath ?? "";

                    if (string.IsNullOrEmpty(encoderFilePath) || string.IsNullOrEmpty(tokensFilePath))
                    {
                        return null;
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _onlineRecognizer = new K2TransducerAsr.OnlineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: threadsNum);
                    TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                    Console.WriteLine("init_models_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
                }
                catch (UnauthorizedAccessException)
                {
                    Console.WriteLine("错误：没有访问该文件夹的权限");
                }
                catch (PathTooLongException)
                {
                    Console.WriteLine("错误：文件路径过长");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"发生错误：{ex.Message}");
                }
            }
            return _onlineRecognizer;
        }
        public static void OnlineRecognizer(string streamDecodeMethod, string modelName = "k2transducer-zipformer-ctc-small-zh-onnx-online-20250401", string modelAccuracy = "int8", int threadsNum = 2, string[]? mediaFilePaths = null, string? modelBasePath = null)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            K2TransducerAsr.OnlineRecognizer? onlineRecognizer = InitOnlineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (onlineRecognizer == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            TimeSpan total_duration = TimeSpan.Zero;
            TimeSpan start_time = TimeSpan.Zero;
            TimeSpan end_time = TimeSpan.Zero;
            start_time = new TimeSpan(DateTime.Now.Ticks);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            List<float[]> samples = new List<float[]>();
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                mediaFilePaths = Directory.GetFiles(Path.Join(modelBasePath, modelName, "test_wavs"));
            }
            foreach (string mediaFilePath in mediaFilePaths)
            {
                if (string.IsNullOrEmpty(mediaFilePath) || !File.Exists(mediaFilePath))
                {
                    continue;
                }
                if (AudioHelper.IsAudioByHeader(mediaFilePath))
                {
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(mediaFilePath, ref duration, chunkSize: 800);
                    for (int j = 0; j < 30; j++)
                    {
                        samples.Add(new float[400]);
                    }
                    samplesList.Add(samples);
                    total_duration += duration;
                }
            }
            if (samplesList.Count == 0)
            {
                Console.WriteLine("No media file is read!");
                return;
            }
            Console.WriteLine("Automatic speech recognition in progress!");
            streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "multi" : streamDecodeMethod;//one ,multi
            if (streamDecodeMethod == "one")
            {
                //one stream decode
                Console.WriteLine("one stream decode results:\r\n");
                for (int j = 0; j < samplesList.Count; j++)
                {
                    K2TransducerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                    foreach (float[] samplesItem in samplesList[j])
                    {
                        stream.AddSamples(samplesItem);
                        K2TransducerAsr.Model.OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
                        Console.WriteLine(result_on.text);
                    }
                }
                // one stream decode
            }
            if (streamDecodeMethod == "multi")
            {
                //multi stream decode
                Console.WriteLine("multi stream decode results:\r\n");
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
                    List<K2TransducerAsr.Model.OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
                    foreach (K2TransducerAsr.Model.OnlineRecognizerResultEntity result in results_batch)
                    {
                        Console.WriteLine(result.text);
                        //Console.WriteLine("");
                    }
                    Console.WriteLine("");
                    i++;
                    bool isAllFinish = true;
                    for (int j = 0; j < samplesList.Count; j++)
                    {
                        if (!isEnds[j])
                        {
                            isAllFinish = false;
                            break;
                        }
                    }
                    if (isAllFinish)
                    {
                        break;
                    }
                }
                //multi stream decode
            }
            if (_onlineRecognizer != null)
            {
                _onlineRecognizer.Dispose();
                _onlineRecognizer = null;
            }
            end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("recognition_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration_milliseconds:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("end!");

        }
    }
}


# K2TransducerAsr

##### 简介：

**K2TransducerAsr是一个使用C#编写的“语音识别”库，底层调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持框架.Net6.0+，支持跨平台编译，支持AOT编译。使用简单方便。**

##### 支持的模型（ONNX）
| 模型名称  |  类型 | 支持语言  | 下载地址  |
| ------------ | ------------ | ------------ | ------------ |
|  k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 | 非流式  | 中文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 "modelscope") |
|  k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 | 流式  | 中文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 "modelscope") |
|  k2transducer-zipformer-en-onnx-online-zengwei-20230517 | 流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-zengwei-20230517 "modelscope") |
|  k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 "modelscope") |
|  k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 | 非流式  | 泰语  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 "modelscope") |
|  k2transducer-lstm-en-onnx-online-csukuangfj-20220903 | 流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-en-onnx-online-csukuangfj-20220903 "modelscope") |
|  k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 "modelscope") |
|  k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 | 流式  | 韩文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 "modelscope") |
|  k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 "modelscope") |
|  k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 | 流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 "modelscope") |
|  k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 "modelscope") |
|  k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 | 流式  | 中文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 "modelscope") |

## 离线（非流式）模型调用方法：
###### 1.添加项目引用
using K2TransducerAsr;
using K2TransducerAsr.Model;

###### 2.模型初始化和配置
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
K2TransducerAsr.OfflineRecognizer offlineRecognizer = new K2TransducerAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```
###### 3.调用
```csharp
List<float[]> samples = new List<float[]>();
//这里省略wav文件转samples...
//具体参考示例（K2TransducerAsr.Examples）代码
//单个识别
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
    Console.WriteLine(result.text);
}
//批量识别
List<OfflineStream> streams = new List<OfflineStream>();
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
}
```
###### 4.输出结果：
单个识别
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1062.28125
total_duration:23340
rtf:0.045513335475578405
end!
```
批量识别
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1268.6875
total_duration:23340
rtf:0.05435679091688089
end!
```

## 实时（流式）模型调用方法：

###### 1.添加项目引用
using K2TransducerAsr;
using K2TransducerAsr.Model;

###### 2.模型初始化和配置
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-multi-zh-hans-onnx-online-20231212";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
K2TransducerAsr.OnlineRecognizer onlineRecognizer = new K2TransducerAsr.OnlineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```
###### 3.调用
```csharp
List<List<float[]>> samplesList = new List<List<float[]>>();
//这里省略wav文件转samples...
//以下是批处理示意代码：
//批处理
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
while (true)
{
    //......(这里省略了一些细节,具体参看示例代码)
	List<K2TransducerAsr.OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
	foreach (K2TransducerAsr.OnlineRecognizerResultEntity result in results_batch)
	{
		Console.WriteLine(result.text);
	}
	//......(这里省略了一些细节,具体参看示例代码)
}
//单处理
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
//具体参考示例（K2TransducerAsr.Examples）代码
```

###### 4.输出结果
* 中文模型测试结果:

```
OnlineRecognizer:
batchSize:1



这是
这是第一种
这是第一种第二
这是第一种第二种
这是第一种第二种叫
这是第一种第二种叫
这是第一种第二种叫
这是第一种第二种叫呃
这是第一种第二种叫呃与
这是第一种第二种叫呃与 always
这是第一种第二种叫呃与 always always
这是第一种第二种叫呃与 always always什么
这是第一种第二种叫呃与 always always什么意思
是
是不是
是不是
是不是平凡
是不是平凡的啊
是不是平凡的啊不认
是不是平凡的啊不认识
是不是平凡的啊不认识记下来
是不是平凡的啊不认识记下来 f
是不是平凡的啊不认识记下来 frequent
是不是平凡的啊不认识记下来 frequently
是不是平凡的啊不认识记下来 frequently频
是不是平凡的啊不认识记下来 frequently频繁
是不是平凡的啊不认识记下来 frequently频繁的
是不是平凡的啊不认识记下来 frequently频繁的
elapsed_milliseconds:2070.546875
total_duration:9790
rtf:0.21149610572012256
end!
```

* 英文模型测试结果:

```




 after

 after early

 after early

 after early nightfa

 after early nightfall the ye

 after early nightfall the yellow la

 after early nightfall the yellow lamps

 after early nightfall the yellow lamps would light

 after early nightfall the yellow lamps would light up

 after early nightfall the yellow lamps would light up here

 after early nightfall the yellow lamps would light up here and

 after early nightfall the yellow lamps would light up here and there

 after early nightfall the yellow lamps would light up here and there the squa

 after early nightfall the yellow lamps would light up here and there the squalid

 after early nightfall the yellow lamps would light up here and there the squalid quar

 after early nightfall the yellow lamps would light up here and there the squalid quarter of

 after early nightfall the yellow lamps would light up here and there the squalid quarter of the bro

 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothel

 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

elapsed_milliseconds:1088.890625
total_duration:6625
rtf:0.16436084905660378
end!
```

###### 相关工程：
* 语音端点检测，解决长音频合理切分的问题，项目地址：[AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 
* 文本标点预测，解决识别结果没有标点的问题，项目地址：[AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")

###### 其他说明：

测试用例：K2TransducerAsr.Examples。
测试CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
支持平台：
Windows 7 SP1或更高版本,
macOS 10.13 (High Sierra) 或更高版本,ios等，
Linux 发行版（需要特定的依赖关系，详见.NET 6支持的Linux发行版列表），
Android（Android 5.0 (API 21) 或更高版本）。

参考
----------
[1] https://github.com/k2-fsa/icefall

[2] https://github.com/naudio/NAudio

# K2TransducerAsr
c# library for decoding K2 transducer Models，used in speech recognition (ASR)

## OfflineRecognizer
##### test model
sherpa-onnx-zipformer-en-2023-04-01

download addr：https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-04-01

sherpa-onnx-zipformer-small-en-2023-06-26

download addr：https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-small-en-2023-06-26

sherpa-onnx-zipformer-large-en-2023-06-26

download addr：https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-large-en-2023-06-26

##### test result:
no batch
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1062.28125
total_duration:23340
rtf:0.045513335475578405
end!
```
batch
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1268.6875
total_duration:23340
rtf:0.05435679091688089
end!
```

## OnlineRecognizer
##### test model
###### zh-en-model:
sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

download addr：https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

##### zh-en-model test result:
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

###### en-model:
sherpa-onnx-streaming-zipformer-en-2023-02-21

download addr：https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21

sherpa-onnx-streaming-zipformer-en-2023-06-21

download addr：https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21

sherpa-onnx-streaming-zipformer-en-2023-06-26

download addr：https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26

##### en-model test result:
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

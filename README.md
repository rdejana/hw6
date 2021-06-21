# HW 6: SSD-based Object Detection

Simple HW6 building blocks

## Prereqs
This assumes you've trained a custom object detection model using the instructions from https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md.  Take note of where you've installed the jetson_inference directory; this will be referred to as $JI. 

Copy eval.py and d2.py to $JI/python/training/detection/ssd.  All commands will be run this this directory.

Edit eval.py, updating the paths for `fileName` and `f2` to match your system.

Edit d2.py, updating the path for `input2` to match your system.

## Native Pytorch

To run, edit the following to fit your system and model to use:
```
python3 eval.py --net mb1-ssd --dataset 'data/fruit' --label_file 'models/fruit/labels.txt' --dataset_type=open_images --trained_model 'models/fruit/mb1-ssd-Epoch-9-Loss-4.919363430012828.pth'

```
I see the following results:

```
Inference time:  2.0207223892211914
Time consumed in working:  2.2561280727386475
Inference time:  0.027497529983520508
Time consumed in working:  2.4578309059143066
Inference time:  0.027361154556274414
Time consumed in working:  0.25310540199279785
Inference time:  0.02722001075744629
Time consumed in working:  0.45259952545166016
Inference time:  0.026554346084594727
Time consumed in working:  0.2525975704193115
Inference time:  0.026655912399291992
Time consumed in working:  0.45127344131469727
Inference time:  0.02639174461364746
Time consumed in working:  0.25308895111083984
Inference time:  0.02667069435119629
Time consumed in working:  0.45123744010925293
Inference time:  0.02712726593017578
Time consumed in working:  0.25193023681640625
Inference time:  0.02669692039489746
Time consumed in working:  0.4501521587371826
Inference time:  0.027019977569580078
Time consumed in working:  0.2622356414794922
Inference time:  0.027068138122558594
Time consumed in working:  0.46024084091186523
```



## TensorRT
Run the following, adjusting to fit your system.

```
python3 ./d2.py --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt           --input-blob=input_0 --output-cvg=scores --output-bbox=boxes             "/home/rdejana/jetson-inference/python/training/detection/ssd/data/fruit/test/0eb2a83700bd54d8.jpg" a.jpg
```

My output is:
```
...
Time consumed in working on image 1:  0.026487350463867188
Time consumed in working on image 2:  0.003718137741088867
Time consumed in working on image 1:  0.004147768020629883
Time consumed in working on image 2:  0.0032782554626464844
Time consumed in working on image 1:  0.003370523452758789
Time consumed in working on image 2:  0.0033092498779296875
Time consumed in working on image 1:  0.003236055374145508
Time consumed in working on image 2:  0.003385782241821289
Time consumed in working on image 1:  0.003281116485595703
Time consumed in working on image 2:  0.003265380859375
Time consumed in working on image 1:  0.0032682418823242188
Time consumed in working on image 2:  0.003385305404663086
```



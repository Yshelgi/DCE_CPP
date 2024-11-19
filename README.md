# TensorRT Zero-DCE_extension


This project leverages TensorRT(V10.x) to implement the Zero-DCE_extension inference.
For further insights into Zero-DCE, please refer to the following resources https://github.com/Li-Chongyi/Zero-DCE_extension/tree/main

## Export ONNX

I made some minor enhancements to the original model, experimented with various fitting curve functions and network modules, and then exported it to `ONNX` format.

```python
import torch
import onnx
from model import infer


model = infer()
model.eval()

ckpts = torch.load('./new_ckpts/Epoch129_111.091.pth')
model.load_state_dict(ckpts)

x = torch.rand(1,3,512,512)
input_name = "images"
output_name = "output"

torch.onnx.export(model,x,"zero_DCE_plus.onnx",opset_version=13,
                  input_names=[input_name],output_names=[output_name],
                  dynamic_axes={
                      input_name:{0:'batch_size'},
                      output_name:{0:'batch_size'},
                  }
)
```

If you use the originally provided weights, you can still refer to this export code.

## Build TensorRT Engine File

Convert the exported `ONNX` model to a `TensorRT` engine file.

`trtexec --onnx=zero_DCE_plus.onnx --saveEngine=DCE_plus_fp16.engine --fp16`

## Result 

<center class="half">
    <img src="https://s2.loli.net/2024/11/19/uakIBL4Qb5x3zco.jpg" width=300/>
    <img src="https://s2.loli.net/2024/11/19/fI6ZJnADWvxb58O.jpg" width=300/>
    <img src="https://s2.loli.net/2024/11/19/NcJG3aEf2tKDkBv.png" width=300/>
    <img src="https://s2.loli.net/2024/11/19/ASlOEWsfpnFk5MC.jpg" width=300/>
    <img src="https://s2.loli.net/2024/11/19/YfOGzDg7ZI9yLu2.png" width=300/>
    <img src="https://s2.loli.net/2024/11/19/vIxc7RktyT6w5dZ.jpg" width=300/>
</center>

For more result images, please refer to the `result` folder. The original test images can be found in the `test_imgs` folder.



​	






import torch
from rendering.manager import *

print(torch.cuda.is_available())


t = torch.Tensor([1.0, 2.0, 3.1415962])
t = t.to(torch.device('cuda:0'))
print(t.data_ptr())

image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=True)

buffer = presenter.create_buffer(3*4, BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC, MemoryProperty.GPU)

presenter.copy_gpu_pointer_to_buffer(t.data_ptr(), buffer, 4*3)

with presenter.get_compute() as man:
    man.gpu_to_cpu(buffer)

a = buffer.as_numpy()

print (t)
print (a)
print (buffer.create_gpu_tensor())

a[0] = 2.718

buffer.write(a)

with presenter.get_compute() as man:
    man.cpu_to_gpu(buffer)

presenter.copy_buffer_to_gpu_pointer(t.data_ptr(), buffer, 4*3)

print (t)
print (a)
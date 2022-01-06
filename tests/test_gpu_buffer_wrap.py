import torch
from rendering.manager import *

print(torch.cuda.is_available())


t = torch.zeros(3, device=torch.device('cuda:0'))
t[0] = t[0].item()

t2 = torch.ones(3, device=torch.device('cuda:0'))
t2[0] = t2[0].item()

image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=True)

presenter.copy_on_the_gpu(t2.storage().data_ptr(), t.storage().data_ptr(), 4*3)
t[0] = 2.0
presenter.copy_on_the_gpu(t.storage().data_ptr(), t2.storage().data_ptr(), 4*3)
# presenter.copy_buffer_to_gpu_pointer(t2.storage().data_ptr(), buffer)

print(t2)
print(t)

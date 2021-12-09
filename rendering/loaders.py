import manager
import vkw
import pywavefront

class ImageLoader:
    def __init__(self, device: manager.DeviceManager):
        self.manager = manager

    def load_file(self, path):
        pass


class ObjLoader:
    def __init__(self, device: manager.DeviceManager):
        self.manager = manager

    def load_file(self, path):
        obj = pywavefront.Wavefront(path)
        return obj



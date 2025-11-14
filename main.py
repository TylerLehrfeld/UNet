from neural_net import NeuralNet
from neural_net import LayerType
from neural_net import LayerDesc
UNET = NeuralNet()   # instantiate the C++ class
UNET.create([
    LayerDesc(LayerType.FC_LAYER, [2,2],[]),
    LayerDesc(LayerType.FC_LAYER, [2,2],[]),
    LayerDesc(LayerType.FC_LAYER, [2,2],[]),
    LayerDesc(LayerType.FC_LAYER, [2,2],[]),
])

structure = {"name": "LT", "size": (3,3), "options": {"option1": 1, "option2": 3}, "parents": {}}

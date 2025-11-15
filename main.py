from neural_net import NeuralNet
from neural_net import LayerType
from neural_net import LayerDesc

in_height = 128
in_width = 128
in_depth = 3

UNET = NeuralNet()   # instantiate the C++ class
UNET.create([
    LayerDesc(LayerType.CONV_LAYER,[in_height, in_width, in_depth, 1, 64, 3,1,1], []), #0
    LayerDesc(LayerType.MAX_POOL_LAYER, [2,2],[0]), #1
    LayerDesc(LayerType.CONV_LAYER, [in_height//2, in_width//2, in_depth, 64, 128, 3,1,1],[1]),#2
    LayerDesc(LayerType.MAX_POOL_LAYER, [2,2],[2]), #3
    LayerDesc(LayerType.CONV_LAYER, [in_height//4, in_width//4, in_depth, 128, 256, 3,1,1],[3]), #4
    LayerDesc(LayerType.MAX_POOL_LAYER, [2,2],[4]), #5
    LayerDesc(LayerType.CONV_LAYER, [in_height//8, in_width//8, in_depth,256, 512, 3,1,1],[5]), #6
    #parameters are h_in, w_in, c_in, c_out, stride/upscale
    LayerDesc(LayerType.UPSAMPLING_LAYER, [in_height//8, in_width//8, 512, 256, 2],[6]), #7
    #parameters are fully determined by parents. First parent is skip connection, second is convolution below 
    LayerDesc(LayerType.ATTENTION_LAYER, [],[4,6]), #8
    #parameters are fully determined by parents. First parent is from skip/attention, second is from convolution below
    LayerDesc(LayerType.CONCAT_LAYER, [],[8,7]), #9
    LayerDesc(LayerType.CONV_LAYER, [in_height//4, in_width//4, in_depth,512, 128, 3,1,1],[9]), #10
    LayerDesc(LayerType.UPSAMPLING_LAYER, [in_height//4, in_width//4, 128, 128, 2],[10]), #11
    LayerDesc(LayerType.ATTENTION_LAYER, [],[2,10]), #12 
    LayerDesc(LayerType.CONCAT_LAYER, [],[12,11]), #13
    LayerDesc(LayerType.CONV_LAYER, [in_height//2, in_width//2, in_depth,256, 64, 3,1,1],[13]), #14
    LayerDesc(LayerType.UPSAMPLING_LAYER, [in_height//2, in_width//2, 64, 64, 2],[14]), #15
    LayerDesc(LayerType.ATTENTION_LAYER, [],[0,14]), #16 
    LayerDesc(LayerType.CONCAT_LAYER, [],[16,15]), #17
    LayerDesc(LayerType.CONV_LAYER, [in_height, in_width, in_depth,128, 1, 3,1,1],[17]), #18
])



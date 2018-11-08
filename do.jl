#=
do:
- Julia version: 
- Author: anderscs
- Date: 2018-11-07
=#
using PyCall

pushfirst!(PyVector(pyimport("sys")["path"]), "")

@pyimport odin
@pyimport odin.models as models
@pyimport odin.utils.default as def

model_wrapper = models.load_model("mnist_vgg2")
#args, model_wrapper = def.default_chainer()


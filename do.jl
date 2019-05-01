#=
do:
- Julia version: 
- Author: anderscs
- Date: 2018-11-07
=#
using PyCall

pushfirst!(PyVector(pyimport("sys")["path"]), "")

models = pyimport("odin.models")
def = pyimport("odin.utils.default")
actions = pyimport("odin.actions")

model_wrapper = models.load_model("pix2pix")
#args, model_wrapper = def.default_chainer()

actions.actions["train_model"]()



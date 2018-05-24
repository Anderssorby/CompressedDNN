from odin.utils.default import default_chainer

if __name__ == "__main__":
    co, args, model_wrapper = default_chainer()

    co.calc_inter_layer_covariance(model_wrapper=model_wrapper)




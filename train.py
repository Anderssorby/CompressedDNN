from odin.utils.default import default_chainer

if __name__ == "__main__":
    co, args, model_wrapper = default_chainer()

    model_wrapper.train(args=args)

from actions import actions
from odin.utils.default import default_chainer

if __name__ == "__main__":
    args, model_wrapper = default_chainer()

    actions[args.action](model_wrapper, args=args)

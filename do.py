from odin.actions import action_map
from odin.utils.default import default_chainer

if __name__ == "__main__":
    args, model_wrapper = default_chainer()

    print("----SETUP----")
    print(args)
    print(model_wrapper)
    print("----END-SETUP----")

    action_map[args.action](model_wrapper, args=args)

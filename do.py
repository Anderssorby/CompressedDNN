from odin.actions import action_map
from odin.utils.default import default_arguments_and_behavior

if __name__ == "__main__":
    args, model_wrapper = default_arguments_and_behavior()

    print("----SETUP----")
    print(args)
    print(model_wrapper)
    print("----END-SETUP----")

    action_map[args.action](model_wrapper, args=args)

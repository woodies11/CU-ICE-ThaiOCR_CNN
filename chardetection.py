import cnnmodel
import argparse

# parse argument
parser = argparse.ArgumentParser(description='Whether to force recreate model.')
parser.add_argument(
	'-m', 
	'--model',
	default='model', 
	help='specify model to use (both JSON and h5 files must be of the given name)'
)
args = parser.parse_args()

model_name = args.model

model = cnnmodel.create_detection_model()
model.load_weights(model_name+".h5")


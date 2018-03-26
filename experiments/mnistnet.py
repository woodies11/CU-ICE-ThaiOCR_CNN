from .experiment import Experiment
from keras.models import model_from_json
import glob

def listdir_nohidden(path):
	directories = glob.glob(path+'/**/*', recursive=True)
	return [file for file in directories if os.path.isfile(file)]

class MNISTNET(Experiment):

    EXPERIMENT_NAME = "SIMPLE MNISTNET"
    EXPERIMENT_DESCRIPTION = """
    Models trained using a simple network originally intended for MNIST.
    """

    classes = [chr(i) for i in range(ord('ก'), ord('ฮ')+1)]
    classes_dict = {chr(i):[0,0] for i in range(ord('ก'), ord('ฮ')+1)}

    @staticmethod
    def _model_from_json(json, **kwargs):
        return model_from_json(json)

    @staticmethod
    def _model_name_from_parameters(batch_size, epochs, **kwargs):
        name_format = "{}-b{}-e"
        name = MNISTNET.EXPERIMENT_NAME.replace(' ', '_').lower()
        return name_format.format(name, batch_size, epochs)

    @staticmethod
    def __try_load_for_continuation(batch_size, epochs, **kwargs):
        # TODO: will implement
        return (None, batch_size, epochs, kwargs)

    @staticmethod
    def predict(model, test_sample, **kwargs):
        pred = model.predict_classes(test_sample)
        pred_class = MNISTNET.classes[pred[0]]
        return pred_class

    def evaluate(model, test_samples, kwargs):

        # { character : [count, right] }
        classes = MNISTNET.classes
        classes_dict = MNISTNET.classes_dict

        test_data_count = 0

        for class_key in test_samples:
            samples = test_samples[class_key]
            test_data_count += len(samples)
            for img in samples:
                pred = model.predict_classes(img)
                
                pred_proba = model.predict_proba(img)
                pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

                pred_class = classes[pred[0]]

                is_correct = str(pred_class) == str(character)

                classes_dict[character][0] += 1
                if is_correct:
                    correct_count += 1
                    classes_dict[character][1] += 1

        # -- end outer for --

        # prevent divide by zero so test can continue
        if test_data_count == 0:
            test_data_count = -1

        MNISTNET.general_logger.info('{}/{} correct ({})'.format(correct_count, test_data_count, correct_count/test_data_count))
        return {k:(classes_dict[k][1]/classes_dict[k][0] if classes_dict[k][0] > 0 else 0) for k in classes_dict}
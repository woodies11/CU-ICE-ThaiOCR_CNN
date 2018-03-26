import Experiment

class MNISTSET(Experiment):

    @staticmethod
    def _model_from_json(json, **kwargs):
        return model = model_from_json(json)

    @staticmethod
    def _model_name_from_parameters(**kwargs):
        batch_size = kwargs['batch_size']
        epochs = kwargs['epoch']

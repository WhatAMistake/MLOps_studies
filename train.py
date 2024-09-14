from keras.models import load_model
from tensorflow.keras.metrics import AUC


from disorder_classifier.classifier_model import DisorderCLF
from disorder_classifier.preprocess_data import DisorderData

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    data_splits = DisorderData(
        data_path=cfg.train.data_path,
        num_classes=cfg.num_classes,
        random_state=cfg.random_state
    )
    splitted_data = data_splits.get_splitted_data()

    trained_model = load_model(cfg.infer.model_path)
    trained_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc', AUC()]
        )
        
    accuracy = trained_model.evaluate(
        splitted_data['X_test'],
        splitted_data['Y_test']
        )


if __name__ == "__main__":
    main()

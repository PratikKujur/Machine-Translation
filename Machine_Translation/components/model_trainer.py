from machine_translation.constants import *

class Model_Trainer:
    def start_model_training(self,model,en_train, dec_train_input,dec_train_target,en_val, dec_val_input,dec_val_target):
        history = model.fit(
            [en_train, dec_train_input], dec_train_target,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([en_val, dec_val_input], dec_val_target)
        )
        return history
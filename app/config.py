import os
from algorithm.utils import read_json_file
import glob


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


prefix = os.path.join(os.pardir, 'ml_vol')
#prefix = os.path.join('opt','ml_vol')


RAND_SEED = 42


DATA_SCHEMA_PATH = glob.glob(os.path.join(
    prefix, "inputs", "data_config", "*.json"))[0]  # Gets the first file of json type
check_dir(os.path.join(prefix, "inputs", "data_config"))

TEXT_VECTORIZER_NAME = 'text_vectorizer.h5'

DATA_SCHEMA = read_json_file(DATA_SCHEMA_PATH)

FAILURE_PATH = os.path.join(prefix, 'outputs', 'errors')
check_dir(FAILURE_PATH)


HYPER_PARAM_PATH = glob.glob(os.path.join(
    prefix, 'model', 'model_config', "*.json"))[0]
check_dir(os.path.join(prefix, 'model', 'model_config'))


DATA_PATH = os.path.join(prefix, 'inputs', 'data')
check_dir(DATA_PATH)
check_dir(os.path.join(DATA_PATH, "training"))
check_dir(os.path.join(DATA_PATH, "testing"))


TRAIN_DATA_PATH = glob.glob(os.path.join(
    DATA_PATH, 'training', "textClassificationBaseMainInput", "*.csv"))[0]


TEST_DATA_PATH = glob.glob(os.path.join(
    DATA_PATH, 'testing', "textClassificationBaseMainInput", "*.csv"))[0]


MODEL_NAME = "tf_bidirectional_text_class"

MODEL_SAVE_PATH = os.path.join(prefix, 'model', 'artifacts')
check_dir(MODEL_SAVE_PATH)


# os.path.join("Utils","preprocess","artifacts")
PREPROCESS_ARTIFACT_PATH = MODEL_SAVE_PATH
check_dir(PREPROCESS_ARTIFACT_PATH)


SAVED_TEST_PRED_PATH = os.path.join(prefix, "outputs", "testing_outputs")
check_dir(SAVED_TEST_PRED_PATH)

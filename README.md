# Text Classification Tensorflow Bidirectional RNN Model

This code uses an embed layer to train for you use case with a bidirectional layers with tensorflow. Model also contains LIME Text Explainer which returns local explanations for predictions.

## Navigatie Code

### app:

Is where all of code used to run train, test or serve modules.

config.py: is a file to determine main configurations such as files paths used during preprocessing, training, serving.

inference_app.py: is a file that declare infer/ping and predictions to requested data happens.

#### Utils:

Utils consist of:

1- model_explain: Where text explainability using Lime happens

2- preprocess folder: has the preprocess classes to process data according to schema.

3- model_builder.py: where the Machine Learning model defined, built, and loaded.

4- predictions_handler.py: called when needed a prediction for inference or testing/predic.

5- utils.py: general functions to help such as load json files.

#### train

python file, called to start training on the required dataset and saves trained model to be called later during inference or during testing.

#### predict

python file, called to generate test.csv file to test model preformance after training.

#### serve

python file, called to generate inferences in production on your server, listens to port 8080.

## Model architecture

Model architecture can be defined using hyperparameters.json file located at app/opt/ml_vol/model/model_config.

model consist of GRU RNN layers wrapped with bidirictional layer, number of layers can be defined in hyperparameters.json file that should be in "model/model_config/" folder.

{

"epochs":10, #Defines for how long to train the model.

"num_layers":2, #Defines number of Bidirectional Layers.

"neurons_num":50, #Defines number of neurons for each layer.

"embed_lay_output":120, #Defines the output dimension of the embedding layer "Not trained".

"learning_rate":0.01 #Defines the learning rate passed to Adam optimizer.

}

Each of parameters must be passed to build the model.

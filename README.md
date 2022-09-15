Food-101 Deep Learning Classification

In this Project20 food categories in the Food-101 image recognition dataset were classified. We show that CNN and RNN, the deep learning technique for general object categorization, can solve this task accurately. We employed four different general deep learning neural network categories for the experiments: (1) Baseline CNN, (2) Custom CNN, (3) Transfer Learning for VGG16, and (4) Transfer Learning for ResNet18. Using all four network architectures, we studied the effects of training, validation, and testing. As a result, we attained great accuracy in the three general approaches, with 36.66%, 47.48%, 74.40%, and 71.98% accuracy, respectively. Lastly, the weight maps and feature activation maps were thoroughly analyzed, providing the users of the model with better hints on what’s really happening under the hood. We find that while adding extra convolutional layers, changing dropout ratio, and fine-tune hyperparameters of CNN could elevate the model performance, transfer learning by fine-tuning Vgg16 and ResNet18 could greatly improve the image recognition model accuracy and with fewer epochs.


All experiments were conducted and stored in the jupyternotebooks. 



data.py and prepare_data : Data extraction Data Preprocessing

model.py: Baseline model +  our best custom model + our best VGG configuration + our best resnet configuration

engine.py: Train the model and store the key statistics 

main.py: leave as is 



One can see the performances, experiments and their visualizations here.  

base_line.ipynb

final custom model.ipynb

resnet_fine_tune.ipynb

resnet_freeze.ipynb

vgg_fine_tuning.ipynb

vgg_freeze.ipynb

weights_analysis.ipynb

# final_project

## Datasets:

Please download the dataset from http://www.cs.umd.edu/~yogesh/datasets/digits.zip and extract it. This folder contains the dataset in the same format as need by our code.

## Training:
	
To train our method(third expirement), run

	python main.py --dataroot [path to the dataset]

This code trains and stores the trained models in result folder. Current checkpoint and the model that gives best performance on the validation set are stored.

to run a different expirement which we mentioned just import the correct trainer (i.e. trainer_third_exp) if no trianer is exist for the exp use regular trainer and make sure trainer import the correct models file (i.e. models_sec_exp).

## Evaluation:

To evaluate the trained models on the target domain (SVHN), run 

	python eval.py --dataroot [path to the dataset] --model_best False  --checkpoint_dir results
	

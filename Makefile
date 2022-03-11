.PHONY: all install_build_package building_package install_package

all: install_build_package building_package install_package

install_build_package:
	echo "*** Installing build package ***"
	pip install build

building_package:
	echo "*** Building train_language_modelling ***"
	python3 -m build .

install_package:
	echo "*** Installing package ***"
	pip install dist/language_modeling-0.0.1-py3-none-any.whl --force-reinstall

train_model:
	@echo "*** Training model"
	train_language_model --path_to_train_data data_for_modeling/train.txt --path_to_eval_data data_for_modeling/test.txt --n_epochs 1 --batch_size 2
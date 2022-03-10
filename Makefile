.PHONY: all install_build_package building_package install_package

all: install_build_package building_package install_package

install_build_package:
	echo "*** Installing build package ***"
	pip install build

building_package:
	echo "*** Building train_language_modelling ***"
	python -m build .

install_package:
	echo "*** Installing package ***"
	pip install dist/language_modeling-0.0.1-py3-none-any.whl --force-reinstall

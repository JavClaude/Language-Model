.PHONY: all build_package install_package quality-checks run-tests

all: build_package install_package quality-checks run-tests

build_package:
	@echo "*** installing the build package ***"
	pip3 install build==0.7.0
	@echo "*** Building train_language_modelling ***"
	python3 -m build .

install_package: build_package
	@echo "*** installing package ***"
	pip3 install dist/language_modeling-0.0.1-py3-none-any.whl --force-reinstall

quality-checks: install_package
	@echo "*** installing quality checks dependencies ***"
	pip3 install flake8==4.0.1 black==22.1.0

	@echo "*** running quality checks ***"
	flake8 language_modeling --config=.flake8 --ignore=E203
	black language_modeling

unit-tests: install_package
	@echo "*** installing test dependencies ***"
	pip3 install pytest==7.0.1 pytest-cov==3.0.0
	@echo "*** running unit tests ***"
	pytest --cov language_modeling
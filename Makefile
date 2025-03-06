SOURCE_DIR = my_package
TEST_DIR = tests

# code formatting and linting

lint:
	pylint ${SOURCE_DIR}

format:
	isort ${SOURCE_DIR}
	black ${SOURCE_DIR}

# installing

venv: # затем необходимо активировать: source .venv/bin/activate
	python3 -m venv .venv

install: # простая установка зависимостей без editable режима
	pip install .

install-dev: # установка зависимостей для разработки
	pip install -e '.[dev]'

install-docs: # установка зависимостей для тестирования
	pip install -e '.[docs]'

install-test: # установка зависимостей для сборки документации
	pip install -e '.[test]'

# building the package

build:
	python3 -m build .

clean:
	rm -rf ./build ./dist ./${SOURCE_DIR}.egg-info

# pypi

pypi: clean build # команда для сборки и загрузки в pypi
	python3 -m twine upload dist/*

test-pypi: clean build # команда для сборки и загрузки в тестовый pypi
	python3 -m twine upload --repository testpypi dist/*

# testing

test: # тестирование
	pytest ${TEST_DIR}

test-cov: # тестирование с выводом процента покрытия кода тестами
	pytest ${TEST_DIR} --cov
# === Constants ===

# the source directories which are being checked
SRC_DIRS = ./src ./tests

# === Package and Dependencies ===

# Upgrading pip, setuptools and wheel
.PHONY: upgrade-pip
upgrade-pip:
	@echo Upgrading pip, setuptools, and wheel ...
	python -m pip install --upgrade pip setuptools wheel

# Installing the required dependencies and building the package
.PHONY: install
install: upgrade-pip
	@echo Installing the required dependencies and building the package ...
	python -m pip install --upgrade .

.PHONY: install.dev
install.dev: upgrade-pip
	@echo Installing the required dependencies and building the package for development ...
	python -m pip install --upgrade .["dev"]

.PHONY: install.ci
install.ci: upgrade-pip
	@echo Installing the required dependencies for CI ...
	python -m pip install --upgrade .["git_ci"]

# === Source File Checks ===

# black format checking
.PHONY: check.black
check.black:
	@echo Checking code formatting with 'black' ...
	black --check --diff --color $(SRC_DIRS)

# isort import checking
.PHONY: check.isort
check.isort:
	@echo Checking import sorting with 'isort' ...
	isort --check --diff --color $(SRC_DIRS)

# pyright static type checking
.PHONY: check.pyright
check.pyright:
	@echo Checking types statically with 'pyright' ...
	pyright $(SRC_DIRS)

# mypy static type checking
.PHONY: check.mypy
check.mypy:
	@echo Checking types statically with 'mypy' ...
	mypy $(SRC_DIRS)

# pycodestyle style checking
.PHONY: check.pycodestyle
check.pycodestyle:
	@echo Checking code style with 'pycodestyle' ...
	pycodestyle $(SRC_DIRS) --max-line-length=88 --ignore=E203,W503,E704

# ruff lint checking
.PHONY: check.ruff
check.ruff:
	@echo Checking code style with 'ruff' ...
	ruff check $(SRC_DIRS)

# All checks combined
.PHONY: check
check: check.black check.isort check.pyright check.mypy check.pycodestyle check.ruff

# === Test Commands ===

# Running a selected test (serial)
.PHONY: test
test:
	@echo Running specific test with pytest ...
	pytest -k "$(TEST)" -x

# Running a selected test (parallel)
.PHONY: test-parallel
test-parallel:
	@echo Running specific test with pytest in parallel ...
	pytest -k "$(TEST)" -n="auto" -x

# Running the tests
.PHONY: tests
tests:
	@echo Running the tests with pytest ...
	pytest ./tests -n="auto" -x --no-jit

.PHONY: tests.htmlcov
tests.htmlcov:
	@echo Running the tests with HTML coverage report ...
	pytest --cov=pyscopee ./tests -n="auto" --cov-report=html -x --no-jit

.PHONY: tests.xmlcov
tests.xmlcov:
	@echo Running the tests with XML coverage report ...
	pytest --cov=pyscopee ./tests -n="auto" --cov-report=xml -x --no-jit

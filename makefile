
# Define the virtual environment directory
VENV_DIR = independentproject

# Define the requirements file
REQUIREMENTS_FILE = requirements.txt

# Define the Python interpreter to use
PYTHON = python3

# Create the virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)

# Install dependencies
.PHONY: install
install:
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS_FILE)

# Run the image processing script within the virtual environment
.PHONY: run
run:
	./run.sh

# Create the virtual environment, install dependencies, and run the script
.PHONY: all
all: venv install run

# Clean the virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)

# List makefile commands
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  venv    - Create the virtual environment"
	@echo "  install - Install dependencies"
	@echo "  run     - Run the image processing script"
	@echo "  all     - Create the virtual environment, install dependencies, and run the script"
	@echo "  clean   - Remove the virtual environment"


# machine-learning-algorithms
Implementation from scratch of some machine learning algorithms

## Dev

Utilisation de pyenv : https://github.com/pyenv/pyenv

## Version python

3.6, 3.7, 3.8, 3.9, 3.10

## Needed

> pyenv install 3.6.9 3.7.10 3.8.9 3.9.4 3.10a7
> python -m venv .venv
> . .venv/bin/activate
> pip install -r requirements.txt

## CI

> pyenv local 3.6.9 3.7.10 3.8.9 3.9.4 3.10a7
> tox

## Build

> python setup.py bdist_wheel

## Install

> python -m pip install -e .

## Publish

> twine upload -r innova dist/*

## Drop pycache execution

> find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

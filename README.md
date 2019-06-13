[![Build Status](https://travis-ci.com/Cognigraph/cognigraph.svg?branch=master)](https://travis-ci.com/Cognigraph/cognigraph)
[![Codecov](https://codecov.io/gh/Cognigraph/cognigraph/branch/master/graph/badge.svg)](https://codecov.io/gh/Cognigraph/cognigraph)
[![Azure](https://dev.azure.com/Cognigraph/cognigraph/_apis/build/status/Cognigraph.cognigraph?branchName=master)](https://dev.azure.com/Cognigraph/cognigraph/_build/latest?definitionId=1&branchName=master)
# Cognigraph

Обработка и визуализация ЭЭГ-сигналов в реальном времени.

## Инструкции по установке
1. #### Установка Miniconda. 

    Установка ПО производится через менежер пакетов и виртуальных сред conda,
    который необходимо предварительно скачать и установить, воспользовавшись
    инструкцией по [ссылке](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

    В случае, если conda уже установлена, этот шаг можно пропустить.

2. #### Установка программы

    Для установки ПО Когнигаф необходимо открыть *терминал с
    установленным менеждером пакетов conda* (для Windows -- Anaconda Prompt)
    и скопировать в него следующие команды:

    ```bash
    git clone --recursive git@github.com:Cognigraph/cognigraph.git
    cd cognigraph
    conda env create -f environment.yml
    conda activate cogni
    pip install --editable .
    ```

3. #### Запуск программы

    Запускать когниграф необходимо из терминала,
    предварительно активировав виртуальную среду `cogni` командой

    ```bash
    conda activate cogni
    ```

    Запуск программы из терминала осуществляется командой

    ```bash
    cognigraph
    ```

Руководство пользователя программой можно найти [здесь](https://cognigraph.github.io/cognigraph).

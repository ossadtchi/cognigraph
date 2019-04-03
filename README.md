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
    git clone --recursive git@github.com:dmalt/cognigraph.git
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



3. **Необходимые файлы.**

    Программа использует файлы из датасета _sample_, 
    распространяемого с пакетом _mne-python_. Чтобы не качать все файлы (датасет
    лежит на osf.io, загрузка с которого  происходит крайне медленно), можно скачать
    урезанную версию 
    [отсюда](https://drive.google.com/open?id=1D0jI_Z5EycI8JwJbYOAYdSycNGoarmP-). 
    Папку _MNE-sample-data_ из архива надо скопировать в то же место, куда бы ее 
    загрузил _mne-python_. Чтобы узнать, что это за место, не скачивая датасет, 
    нужно сделать следующее: 

    ```
    from mne.datasets import sample
    print(sample.data_path(download=False, verbose=False))
    ```
    Папку _MNE-sample-data_ из архива копируем в выведенный путь.




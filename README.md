# Bayesian inference 1, UEF - Autumn 2018

How to clone all the code and data provided to your computer:

```bash
git clone --recursive https://github.com/trungnt13/uef_bay1_2018.git
```
For Windows users, using github desktop may significantly simplify the process:
[https://desktop.github.com/](https://desktop.github.com/)

## Recommended reading

#### ["Think Bayes", Allen B. Downey](http://www.greenteapress.com/thinkbayes/html/index.html)
#### ["Pattern Recognition and Machine Learning", Christopher M. Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

## Setting up python environment using `miniconda`

#### Installing miniconda
Following the instruction and install Miniconda from this link:
[https://conda.io/miniconda.html](https://conda.io/miniconda.html)

#### Create the environment
> conda env create -f=path/to/cloned/folder/environment.yml

#### Using installed environment
For activating and using our environment:
> conda activate bay

Listing installed packages:
> conda list

Deactivating environment:
> conda deactivate

#### More tutorials for Windows users
[https://conda.io/docs/user-guide/install/windows.html#install-win-silent](https://conda.io/docs/user-guide/install/windows.html#install-win-silent)

## Using `pip`

> pip install -r path/to/cloned/folder/requirements.txt

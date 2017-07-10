#
# (c) 2011, 2012 Georgia Tech Research Corporation
# This source code is released under the New BSD license. 
# Please see http://wiki.quantsoftware.org/index.php?title=QSTK_License
# for license details.

# Created on Jan 16, 2013

# @author: Sourabh Bajaj
# @contact: sourabh@sourabhbajaj.com
# @summary: Automated Mac Installation script of QSTK
#

echo "Install Homebrew"

ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bash_profile
brew doctor

echo "Install QSTK"
echo "Installing python"
brew install python

brew tap samueljohn/python
brew tap homebrew/science

echo "virtualenv"
pip install virtualenv
pip install nose

echo "Installing gfortran"
brew install gfortran

echo "Installing numpy, scipy, matplotlib"
brew install numpy
brew install scipy
brew install matplotlib

echo "Create QSTK directory"
mkdir ~/QSTK
cd ~/QSTK
# virtualenv env --distribute --system-site-packages
# source ~/QSTK/env/bin/activate

echo "Install pandas, scikits"
pip install pandas
pip install scikits.statsmodels
pip install scikit-learn
pip install QSTK

echo "Test QSTK"
cd ~/QSTK
curl -O https://spark-public.s3.amazonaws.com/compinvesting1/QSTK-Setups/Examples.zip
unzip Examples.zip
cd Examples
python Validation.py

# echo "Deactivating"
# deactivate

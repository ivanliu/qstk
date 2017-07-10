
# Created on Sept 3, 2016

# @author: Peiji Chen
# @contact: m
# @summary: My Updated Mac Installation script of QSTK 
#

# The updated doc is at https://docs.google.com/document/d/1qOzB-aUtCLcPPLfwEkwaCIoCsVe5MV2AkDnhZwESrSU/edit#

# Homebrew has already been installed.
#echo "Installing python"
#brew install python
#echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bash_profile
#source ~/.bash_profile

# echo "Create QSTK directory"
#git clone https://github.com/QuantSoftware/QuantSoftwareToolkit.git	# original
#git clone https://peijicheyahoo@bitbucket.org/peijicheyahoo/quantsoftwaretoolkit.git  # forked and cleaned up



pip install `cat requirements.txt ` 
python setup.py install




echo "validating the installations"
echo "you should see the following last line"
echo " Everything works fine: You're all set." 
cd Examples
python Validation.py


# make sure matplotlib work in Mac
cp matplotlibrc ~/.matplotlib/matplotlibrc



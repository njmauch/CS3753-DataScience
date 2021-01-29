Windows
If you are on Python 2.7.9+ and 3.4+
Good news! Python 3.4 (released March 2014) and Python 2.7.9 (released December 2014) ship with Pip. 

If you do find that pip is not available when using Python 3.4+ or Python 2.7.9+, simply execute e.g.:

py -3 -m ensurepip

If you are using Python 2.7.8 or earlier. Manual instructions follow.

Python 2 = 2.7.8 and Python 3 = 3.3

Official instructions
Per https://pip.pypa.io/en/stable/installing/#do-i-need-to-install-pip:

Download get-pip.py, being careful to save it as a .py file rather than .txt. Then, run it from the command prompt:

python get-pip.py

You possibly need an administrator command prompt to do this. Follow Start a Command Prompt as an Administrator (Microsoft TechNet).

This installs the pip package, which (in Windows) contains ...\Scripts\pip.exe that path must be in PATH environment variable to use pip from the command line (see the second part of 'Alternative Instructions' for adding it to your PATH)

Alternative instructions

 1. Install setuptools  https://www.lfd.uci.edu/~gohlke/pythonlibs/#setuptools
 2. Install pip   https://www.lfd.uci.edu/~gohlke/pythonlibs/#pip

For me, this installed Pip at C:\Python27\Scripts\pip.exe. Find pip.exe on your computer, then add its folder (for example, C:\Python27\Scripts) to your path (Start / Edit environment variables). Now you should be able to run pip from the command line. Try installing a package:

pip install httpie

MacOS
MacOS comes with Python installed. But to make sure that you have Python installed open the terminal and run the following command.

python --version

If this command returns a version number that means Python exists. Which also means that you already have access to easy_install considering you are using macOS/OSX.

Run the following command.

sudo easy_install pip

Linux 
To install pip in Linux, run the appropriate command for your distribution as follows:

Install PIP On Debian/Ubuntu
# apt install python-pip	#python 2
# apt install python3-pip	#python 3

For this program you will need to ensure you have pandas and numpy installed using the commands below

pip install pandas
pip install numpy
pip install matplotlib
pip install scipy

Next you will need to ensure jupyter is installed

pip
If you use pip, you can install it with:

pip install jupyterlab


After that is installed, in your command prompt change the directory the nja859_hw5.ipynb file is located
Make sure that the file iris.csv is in the same directory as well.
Once there run the command.

jupyter notebook nja859_hw5.ipynb

This will open the ipynb in your browser. 
Go to the tab "Kernel" and make sure that you change your Kernel to python3

Once that is done go back to the tab "Kernel", select that then click on "Restart & Run All"

A pop up will show asking if you are sure you want to restart. Select "Restart and Run All Cells".  

This will run the script and display the results.
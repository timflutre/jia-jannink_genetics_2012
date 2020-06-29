This directory contains the C code used by Jia and Jannink (Genetics, 2012) and
was sent by Yi Jia to Timothée Flutre on 14/11/2015:
http://dx.doi.org/10.1534/genetics.112.144246

As the code uses the GNU Scientific Library (GSL), it is under the GNU Public
License (GPL):
http://www.gnu.org/software/gsl/
http://www.gnu.org/licenses/licenses.html#GPL

It is hosted online, https://github.com/timflutre/jia-jannink_genetics_2012,
where you can report issues. The tag v0.1.0 corresponds to the exact files sent
initially. All modifications were performed by T. Flutre since then. Please
contact me if interested.

Compilation (for Ubuntu):
```
sudo apt install libgsl-dev
gcc -o BayesA_multivariate BayesA_multivariate.c `gsl-config --cflags --libs`
gcc -o BayesCPi_multivariate BayesCPi_multivariate.c `gsl-config --cflags --libs`
```

Usage:
```
./BayesA_multivariate filename nind nmarkers ntraits numiter nfold cycle thin
./BayesCPi_multivariate filename nind nmarkers ntraits numiter nfold cycle thin
```

Retrieve the simulated data:
```
mkdir data; cd data
wget https://www.genetics.org/highwire/filestream/412909/field_highwire_adjunct_files/0/FileS1.zip
unzip FileS1.zip
```

Estimate the parameters:
```
mkdir results; cd results
ln -s ../data/FileS1/Barley_Data_Simulation/Default_simulation/QTL20_standard_rep1.txt .
../BayesA_multivariate QTL20_standard_rep1.txt 500 2000 2 110000 1 1 1
cat QTL20_standard_rep1.txt_MTBA1
# the accuracy of trait1 prediction is:-nan
# the accuracy of trait2 prediction is:-nan
# the nu mean is: 3.102967
# 0.100000 0.000000 
# 0.000000 0.100000 
```

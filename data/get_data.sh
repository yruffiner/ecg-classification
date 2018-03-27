wget https://physionet.org/challenge/2017/training2017.zip
unzip training2017.zip
rm training2017.zip
cd training2017
mkdir mat_files hea_files
mv *.mat mat_files
mv *.hea hea_files
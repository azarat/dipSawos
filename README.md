##HOW TO RUN PROGRAM

1. Move to the folder with file **_run.py_** via Terminal: 

    ````
    cd /home/atishche/shared/JETSTA
    ````
2. Type command in Terminal

    ``````
    python run.py
    ``````

##ABOUT PARAMETERS
###SETTING PARAMETERS

In the same directory where **_run.py_** is placed, you can find file **_input.json_**

That file can be opened with any text editor. Inside are placed structured list of parameters which should be filled.

####REQUIRED PARAMETERS

Obligatory parameters are "_Crash Start_" and "_Crash End_".

Both are integer numbers - order of crash in MatLab database.

If "_Crash Start_" and "_Crash End_" are different, i.e. 1 and 10, the program will analyse every crash from 1 to 10.

If "_Crash Start_" and "_Crash End_" are the same number, _i.e. 1 and 1_, the program will analyse only crash number 1.

####ADVANCED PARAMETERS


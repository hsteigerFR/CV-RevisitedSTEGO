@echo off
call C:\workspace\anaconda3\Scripts\activate.bat C:\workspace\anaconda3\
python main.py -cfg config.yml
cmd /k "echo Done."
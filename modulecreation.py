import os,sys
FILE_DIR= os.path.dirname(os.path.abspath(__file__))
ROOT_DIR= FILE_DIR[0:FILE_DIR.index("2021fc04746")].replace("\\","\\\\")
sys.path.append(FILE_DIR)
print(sys.path)
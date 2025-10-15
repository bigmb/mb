#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import subprocess
import sys
import shutil


if sys.version_info[:2] in [(3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14)]:
    py_requires = f"python{sys.version_info.major}.{sys.version_info.minor}"
else:
    py_requires = "python3.11"

file = os.getcwd() 


#subprocess.run(["cd",file]), check=True, stdout=subprocess.PIPE).stdout
os.system('cd ' + file)

os.system('./make_version.sh')

print("version file updated")
print('*'*100)

# subprocess.run(["git", "add", "."], check=True, stdout=subprocess.PIPE).stdout
# subprocess.run(["git", "commit", "-am", "Bug fix commit"], check=True, stdout=subprocess.PIPE).stdout

# print('git commit done')

subprocess.run(["git", "pull"], check=True, stdout=subprocess.PIPE).stdout
print('git pull done')
print('*'*100)

subprocess.run(["git", "push"], check=True, stdout=subprocess.PIPE).stdout
print('*'*100)
print('removing dist and build folders')


for folder in ['dist', 'build']:
    if os.path.exists(folder):
        shutil.rmtree(folder)
print('dist/build folders removed')
    

print('*'*100)

if subprocess.run(['uv','--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
    subprocess.run([py_requires, '-m', 'build'], check=True)
    print('*'*100)
    subprocess.run([py_requires, '-m', 'pip', 'install', os.path.join('dist', os.listdir('dist')[-1]), '--break-system-packages'], check=True)
    print('build and installed using pip')
    print('*'*100)
else:
    subprocess.run(['uv', 'build'], check=True)
    print('*'*100)
    subprocess.run(['sudo','uv', 'pip', 'install', '--system', os.path.join('dist', os.listdir('dist')[-1])], check=True)
    print('build and installed using uv')
    print('*'*100)

##Twine upload
subprocess.run([py_requires, '-m', 'twine', 'upload', 'dist/*'], check=True)

print('package installed')
print('*'*100)
# os.system(py_requires + ' -m twine upload dist/*')
There are two folders here:

data/ "Anything that has to do with data(preprocessing, loading, etc) will be in this folder."

models/ "Anything that has to with the model will be in this folder."

analysis/ "Anything that does analysis of the data, such as frequency distributions"

dist/ "Results of the models/analyses"

baseline/ "Folder where the baseline is"

main.py

the main.py is the main program that will take inputs, arguments and combining all the classes together.

Code of conduct:

You can create multiple calsses in one file, it is best to avoid creating too many files but also keep in mind that don't put two unrelated thing in one .py file as it may be confusing for other people.

Steps for working:
1. BEFORE DOING ANY CODING: make sure you're in your own branch ($git branch <name>): $git merge develop
2. If there are merge conflicts: $git commit <file_w_merge_conflict> -m "committing file because of merge conflict"
3. If there are no merge conflicts: go work and code!

Steps before finishing:
1. add worked files: $git add <filename>
2. commit work: $git commit -m "worked on ..."
3. push work to your own branch: $git push
4. change to develop branch: $git branch develop
5. merge changes from your own branch into the develop branch: $git merge <your_branch_name>

Steps for problems/see which files are changed:
$git status 



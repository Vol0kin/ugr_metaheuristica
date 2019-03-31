#!/bin/bash

files=( colposcopy ionosphere texture )
algorithms=( knn relief local )

for f in "${files[@]}"
do
  for a in "${algorithms[@]}"
  do
    python3 practica1.py $f $a
  done
done
